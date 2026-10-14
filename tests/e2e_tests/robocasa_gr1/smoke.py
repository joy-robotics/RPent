"""Opt-in native GR1 component and bounded action-chain acceptance, without an LLM."""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from contextlib import contextmanager

from robots.robocasa_gr1.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.session import EnvState
from tests.e2e_tests.common import require_array, require_rgb, runtime_phase


@contextmanager
def measure(output, name, gpu):
    """Measure device-wide memory, including simulator graphics allocations."""
    stop = threading.Event()
    samples = []
    started = time.monotonic()

    def sample():
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            used, free, total = map(int, result.stdout.strip().split(","))
            samples.append(
                {
                    "elapsed_s": time.monotonic() - started,
                    "used_mib": used,
                    "free_mib": free,
                    "total_mib": total,
                }
            )
            stop.wait(0.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    record = {"phase": name, "status": "failed", "planner_calls": 0}
    try:
        yield record
        if not samples or not monitor.is_alive():
            raise RuntimeError("GPU telemetry failed; memory acceptance is incomplete")
        record["status"] = "passed"
    finally:
        stop.set()
        monitor.join(timeout=12)
        record["elapsed_s"] = time.monotonic() - started
        record["gpu_samples"] = samples
        record["peak_device_mib"] = max(
            (item["used_mib"] for item in samples), default=None
        )
        record["minimum_free_mib"] = min(
            (item["free_mib"] for item in samples), default=None
        )
        write_json_atomic(output / f"{name}.json", record)


def main():
    """Exercise existing paths without downloads, fabricated planners or long rollouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    spec = get_robot_spec()
    spec.add_cli_args(parser, False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-dir")
    parser.add_argument("--memory-profile", default="local", choices=("local",))
    args = parser.parse_args()
    spec.validate_args(args)
    if (
        args.cuda_device is None
        or not args.cuda_device.isdecimal()
        or args.vla_endpoint
    ):
        parser.error(
            "Smoke requires one explicit --cuda-device and an owned VLA server"
        )
    if args.max_episode_steps > 33:
        parser.error(
            "Smoke allows at most 33 native steps; pass --max-episode-steps 17"
        )
    config = spec.parse_config(args)
    root = config.output_dir
    root.mkdir(parents=True, exist_ok=False)
    with measure(root, "env_only", args.cuda_device) as evidence:
        with runtime_phase(spec, args, root / "env_only", {"env"}) as runtime:
            env = runtime["env"]
            observed = env.observe()
            require_rgb(observed["ego_rgb"], "ego_rgb")
            require_array(observed["state44"], "raw_state44", ndim=1, last_dim=44)
            artifacts = EnvState(root / "env_only")
            artifacts.save("initial_ego.png", observed["ego_rgb"], step=None)
            artifacts.save("initial_state44.npy", observed["state44"], step=None)
            evidence.update(
                state_dim=44,
                rgb_shape=list(observed["ego_rgb"].shape),
                task_language=observed["task_language"],
                executed_actions=0,
            )
            env._client.close()
    with measure(root, "vla_only", args.cuda_device) as evidence:
        with runtime_phase(spec, args, root / "vla_only", {"vla"}) as runtime:
            model = runtime["model"]
            started = time.monotonic()
            actions = model.infer(observed)
            require_array(actions, "actions29", ndim=2, last_dim=29)
            EnvState(root / "vla_only").save("prediction.npy", actions, step=None)
            evidence.update(
                inference_s=time.monotonic() - started,
                action_shape=list(actions.shape),
                executed_actions=0,
            )
            model.close()
    with measure(root, "combined", args.cuda_device) as evidence:
        with runtime_phase(spec, args, root / "combined", {"env", "vla"}) as runtime:
            from dataclasses import replace

            toolkit = get_toolkit(
                runtime_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                config=replace(config, output_dir=root / "combined"),
            )
            try:
                initial = toolkit.env.observe()
                started = time.monotonic()
                primitive = toolkit.execute_tool(
                    "set_joint_targets",
                    {
                        "targets": {
                            "waist": initial["joint_positions"]["waist"].tolist()
                        }
                    },
                ).result
                if primitive.get("error") or primitive.get("result", {}).get("error"):
                    raise RuntimeError(f"Native primitive failed: {primitive}")
                result = toolkit.execute_tool("gr1_act", {"chunks": 1}).result
                if result.get("error") or result.get("result", {}).get("error"):
                    raise RuntimeError(f"Real VLA action chain failed: {result}")
                if (
                    result["result"]["executed_actions"] < 1
                    or result["result"]["vla_inferences"] < 1
                ):
                    raise RuntimeError("No learned GR1 actions were executed")
                finish = toolkit.execute_tool(
                    "finish",
                    {
                        "status": "stuck",
                        "summary": "bounded real-component smoke, not an Agent evaluation",
                    },
                ).result
                evidence.update(
                    action_chain_s=time.monotonic() - started,
                    executed_actions=finish["episode_status"]["step_count"],
                    native_success=finish["success"],
                    vla_inferences=result["result"]["vla_inferences"],
                )
                if evidence["executed_actions"] < 1:
                    raise RuntimeError("No native actions were executed")
            finally:
                toolkit.close()
    write_json_atomic(
        root / "summary.json",
        {
            "status": "passed",
            "definition": "native reset/RGB, inference-only, native joint hold, real VLA action chain; no LLM",
            "agent_api_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
