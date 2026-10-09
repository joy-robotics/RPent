"""Real in-tree JoyBrain/RoboTwin integration check, without a high-level planner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robots.robotwin import robot_spec
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.dashboard.events import NullDashboardEventSink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    robot_spec._add_cli_args(parser, False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.vla_backend != "joybrain":
        parser.error("This smoke test requires --vla-backend joybrain")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "in_tree_backend_integration_no_llm",
        "real_planner": False,
        "passed": False,
    }
    started = time.monotonic()
    daemons = []
    model = None
    primitive = None
    toolkit = None
    try:
        from PIL import Image

        daemons, kwargs = robot_spec._init_runtime(
            args, output, NullDashboardEventSink(), None
        )
        model = kwargs["model"]
        env = kwargs["env"]
        report["client_module"] = type(model).__module__
        report["policy_metadata"] = model.metadata
        report["instruction"] = env.get_task_language()
        report["initial_status"] = dict(env.last_info["episode_status"])
        for camera in robot_spec.ROBOTWIN_CAMERA_NAMES:
            Image.fromarray(np.asarray(env.render_camera(camera))).save(
                output / f"initial_{camera}.png"
            )
        if args.record_actions:
            from robots.robotwin.toolkit import RoboTwinToolkit
            from rpent.memory import MemoryManager
            from rpent.utils.logging import init_output_dir

            init_output_dir(output)
            toolkit = RoboTwinToolkit(
                runtime_kwargs=kwargs,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(root=output / "memory"),
                state_output_dir=output,
            )
            primitive = toolkit._primitives
        else:
            primitive = RoboTwinPrimitives(check_cancelled=lambda: None, **kwargs)
            primitive.start_recording()

        def execute(name, arguments):
            if toolkit is not None:
                return toolkit._step(name, **arguments)
            return getattr(primitive, name)(**arguments)

        atomic_args = {
            "arm": "left",
            "val": float(env.last_info["robot_state"]["left_gripper"]),
            "steps": 2,
        }
        report["atomic"] = execute("set_gripper", atomic_args)
        vla_args = {
            "chunks": 1,
            "use_length": 50,
            "prompt": "ignored input, not a rewritten policy task",
        }
        report["vla"] = execute("lingbot_act", vla_args)
        report["final_status"] = dict(env.last_info["episode_status"])
        for camera in robot_spec.ROBOTWIN_CAMERA_NAMES:
            Image.fromarray(np.asarray(env.render_camera(camera))).save(
                output / f"final_{camera}.png"
            )
        report["external_adapter_imported"] = any(
            name == "rpent_joybrain" or name.startswith("rpent_joybrain.")
            for name in sys.modules
        )
        report["passed"] = (
            report["client_module"] == "robots.robotwin.joybrain_client"
            and not report["external_adapter_imported"]
            and report["atomic"]["executed_actions"] == 2
            and report["vla"]["executed_steps"] > 0
            and report["vla"]["action_layout"] == "qpos14"
            and report["vla"]["prompt"] == report["instruction"]
        )
        if not report["passed"]:
            raise RuntimeError("Integration checks failed")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if toolkit is not None:
            try:
                toolkit.close()
            except Exception as error:
                report.update(passed=False, cleanup_error=repr(error))
            primitive = None
        if primitive is not None:
            frames = primitive.stop_recording()
            if frames:
                try:
                    import imageio.v2 as imageio

                    imageio.mimsave(output / "episode.mp4", frames, fps=20)
                except Exception as error:
                    report["video_error"] = repr(error)
        if model is not None:
            model.close()
        for daemon in reversed(daemons):
            daemon.stop()
        report["elapsed_s"] = time.monotonic() - started
        text = json.dumps(
            report,
            indent=2,
            default=lambda value: (
                value.tolist() if isinstance(value, np.ndarray) else str(value)
            ),
        )
        (output / "result.json").write_text(text + "\n")
        print(text, flush=True)
    if not report["passed"]:
        raise RuntimeError("Integration or cleanup failed; inspect result.json")


if __name__ == "__main__":
    main()
