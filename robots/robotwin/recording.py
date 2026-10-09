"""Durable native-action traces and recorded-evaluation boundaries."""

from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

CAMERAS = {
    "head": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON, including NumPy values, without non-finite numbers."""

    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load public run metadata, rejecting inline credential fields."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Recording metadata must be an object")

    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in {
                    "api_key",
                    "authorization",
                    "access_token",
                    "refresh_token",
                    "secret",
                }:
                    raise ValueError(
                        "Recording metadata must not contain inline credentials"
                    )
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)

    check(value)
    return value


def enabled(env: Any) -> bool:
    """Return whether this environment advertises native action recording."""
    return (
        getattr(env, "execution_capabilities", {}).get("native_action_recording")
        is True
    )


@contextmanager
def record_tool(env: Any, name: str, arguments: dict) -> Iterator[None]:
    """Bind every native action in a tool call to one trace context."""
    if not enabled(env):
        yield
        return
    env.trace("begin", name, arguments)
    error = None
    try:
        yield
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.trace("end", error)


class NativeRecorder:
    """Record the observation before each confirmed native executor call."""

    def __init__(self, root: Path, metadata: dict[str, Any]) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.context: dict | None = None
        self.action_index = 0
        self.actions_by_source: dict[str, int] = {}
        self.proposals = 0
        self.last_status = None
        self.failed = False
        write_json(
            root / "episode.json",
            {
                "schema": "rpent-robotwin-native-v1",
                "platform": "robotwin",
                "embodiment": "robotwin_aloha",
                "run_config": metadata,
                "camera_order": list(CAMERAS),
                "state_semantics": "absolute qpos targets: left joints6/gripper,right joints6/gripper",
                "time_axis": "native decision index; wall times are not physical simulation time",
                "training_eligible": False,
            },
        )
        self.summary()

    def event(self, value: dict) -> None:
        """Append provenance without buffering it until episode completion."""
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"wall_time_ns": time.time_ns(), **value},
                    default=lambda item: item.tolist(),
                    allow_nan=False,
                )
                + "\n"
            )

    def begin(self, name: str, arguments: dict) -> dict:
        """Start a non-nested tool context before any environment mutation."""
        if self.failed or self.context is not None:
            raise RuntimeError("Recorder failed or another tool is already active")
        self.context = {
            "tool_id": uuid4().hex,
            "tool": name,
            "arguments": copy.deepcopy(arguments),
            "source": "vla" if name == "lingbot_act" else name,
            "effective_vla_instruction": None,
        }
        self.event({"event": "tool_begin", **self.context})
        return dict(self.context)

    def end(self, error: str | None = None) -> dict:
        """Finish a tool context, retaining its error independently of success."""
        if self.context is None:
            raise RuntimeError("No active tool context")
        self.event({"event": "tool_end", "error": error, **self.context})
        self.context = None
        return self.summary()

    def proposal(self, instruction: str, actions: Any) -> dict:
        """Keep predictions separate from actually executed action labels."""
        if self.context is None or self.context["source"] != "vla":
            raise RuntimeError("VLA proposal outside its tool context")
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 14 or not np.isfinite(array).all():
            raise ValueError("Expected finite qpos14 proposal")
        self.context["effective_vla_instruction"] = instruction
        filename = f"proposal_{self.proposals:06d}.npy"
        np.save(self.root / filename, array, allow_pickle=False)
        self.proposals += 1
        self.event({"event": "vla_proposal", "path": filename, **self.context})
        return {"ok": True}

    @staticmethod
    def snapshot(task: Any, directory: Path, prefix: str) -> dict:
        """Capture synchronous RGB and action-target state at a call boundary."""
        from PIL import Image

        observation = task.get_obs()
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected finite native qpos14 state")
        result = {
            "qpos_target14": state.copy(),
            "camera_paths": {},
            "instruction": task.get_instruction(),
        }
        for name, native in CAMERAS.items():
            image = np.asarray(observation["observation"][native]["rgb"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid native RGB camera {native}")
            filename = f"{prefix}_{name}.png"
            Image.fromarray(image).save(directory / filename)
            result["camera_paths"][name] = filename
        return result

    def attach(self, task: Any) -> None:
        """Observe native calls without splitting RPC chunks or changing controls."""
        original = task.take_action

        def take_action(action, *args, **kwargs):
            if self.failed or self.context is None:
                raise RuntimeError("Unattributed or uncertain action refused")
            action_type = kwargs.get("action_type", args[0] if args else "qpos")
            raw = np.asarray(action, dtype=np.float64)
            if (
                action_type != "qpos"
                or raw.shape != (14,)
                or not np.isfinite(raw).all()
            ):
                raise ValueError(
                    "Recorded Joy-Brain execution requires finite qpos14 targets"
                )
            directory = self.root / f"step_{self.action_index:06d}"
            directory.mkdir(exist_ok=False)
            record = {
                **copy.deepcopy(self.context),
                "step": self.action_index,
                "before": self.snapshot(task, directory, "before"),
                "action": {"representation": "qpos14", "values": raw.copy()},
                "native_count_before": int(task.take_action_cnt),
                "wall_start_ns": time.time_ns(),
                "physical_duration_s": None,
                "state_unit": "radians and normalized grippers",
            }
            try:
                result = original(action, *args, **kwargs)
                record.update(
                    native_count_after=int(task.take_action_cnt),
                    wall_end_ns=time.time_ns(),
                )
                if record["native_count_after"] != record["native_count_before"] + 1:
                    raise RuntimeError(
                        "Native action count did not advance exactly once"
                    )
                record.update(
                    after=self.snapshot(task, directory, "after"),
                    execution="executed",
                    native_step_limit=int(task.step_lim),
                    reward=float(bool(task.eval_success)),
                    partial_score=None,
                    terminated=bool(task.eval_success),
                    truncated=bool(
                        task.take_action_cnt >= task.step_lim and not task.eval_success
                    ),
                )
            except BaseException as exc:
                self.failed = True
                record.update(
                    execution="unknown_or_incomplete",
                    error=f"{type(exc).__name__}: {exc}",
                    native_count_after=int(task.take_action_cnt),
                )
                write_json(directory / "transition.json", record)
                self.summary()
                raise
            write_json(directory / "transition.json", record)
            self.action_index += 1
            self.actions_by_source[record["source"]] = (
                self.actions_by_source.get(record["source"], 0) + 1
            )
            self.last_status = {
                key: record[key]
                for key in ("terminated", "truncated", "reward", "partial_score")
            }
            self.summary()
            return result

        def guarded_action(*args, **kwargs):
            try:
                return take_action(*args, **kwargs)
            except BaseException:
                self.failed = True
                try:
                    self.summary()
                except Exception:
                    pass
                raise

        task.take_action = guarded_action

    def summary(self) -> dict:
        """Persist confirmed counts; uncertainty never becomes a training label."""
        result = {
            "executed_actions": self.action_index,
            "actions_by_source": self.actions_by_source,
            "vla_inferences": self.proposals,
            "last_status": self.last_status,
            "recording_failed": self.failed,
        }
        write_json(self.root / "summary.json", result)
        return result


def validate_trace(root: Path) -> dict:
    """Check all transitions, including short episodes, without requiring a training window."""
    episode = json.loads((root / "episode.json").read_text())
    if episode["embodiment"] != "robotwin_aloha" or episode["camera_order"] != list(
        CAMERAS
    ):
        raise ValueError("Trace identity/camera order mismatch")
    count = 0
    previous = None
    for directory in sorted(root.glob("step_*")):
        record = json.loads((directory / "transition.json").read_text())
        if record["execution"] != "executed" or record["step"] != count:
            raise ValueError("Trace includes an incomplete or out-of-order action")
        if record["native_count_after"] != record["native_count_before"] + 1 or (
            previous is not None and previous != record["native_count_before"]
        ):
            raise ValueError("Native count gap")
        if record["action"]["representation"] != "qpos14":
            raise ValueError("Mixed action representations")
        for values in (
            record["action"]["values"],
            record["before"]["qpos_target14"],
            record["after"]["qpos_target14"],
        ):
            if np.asarray(values).shape != (14,) or not np.isfinite(values).all():
                raise ValueError("Invalid qpos14 sample")
        if np.any(np.asarray(record["action"]["values"])[[6, 13]] < -1e-6) or np.any(
            np.asarray(record["action"]["values"])[[6, 13]] > 1 + 1e-6
        ):
            raise ValueError("Gripper targets must be normalized to [0,1]")
        for side in ("before", "after"):
            if set(record[side]["camera_paths"]) != set(CAMERAS):
                raise ValueError("Missing camera in transition")
            for filename in record[side]["camera_paths"].values():
                if (
                    not (directory / filename)
                    .resolve()
                    .is_relative_to(directory.resolve())
                    or not (directory / filename).is_file()
                ):
                    raise ValueError("Missing camera observation")
        count += 1
        previous = record["native_count_after"]
    summary = json.loads((root / "summary.json").read_text())
    if summary["recording_failed"] or summary["executed_actions"] != count:
        raise ValueError("Trace summary disagrees with transitions")
    return {
        "passed": True,
        "executed_actions": count,
        "last_native_count": previous,
        "complete_16_action_windows": max(0, count - 15),
    }


def install_file_guards(toolkit: Any, output: Path, memory: Path) -> None:
    """Keep evaluation file tools away from credentials and controller evidence."""
    from rpent.tools.toolkit import readonly
    from rpent.utils.config import get_repo_root

    roots = [get_repo_root().resolve(), output.resolve(), memory.resolve()]

    def guard(name, handler):
        @readonly
        @wraps(handler)
        def call(*args, **kwargs):
            raw = Path(kwargs.get("path", args[0] if args else "") or output)
            path = (raw if raw.is_absolute() else roots[0] / raw).resolve()
            if not any(path.is_relative_to(root) for root in roots):
                raise PermissionError(
                    "File access is restricted to repository, episode and memory"
                )
            if any(
                part in {".git", ".codex", ".ssh"} or part.startswith((".env", ".venv"))
                for part in path.parts
            ):
                raise PermissionError("Credential/private paths are not exposed")
            if name == "write_text_file" and not path.is_relative_to(
                roots[1] / "agent_notes"
            ):
                raise PermissionError(
                    "Write notes only inside this episode's agent_notes directory"
                )
            return handler(*args, **kwargs)

        return call

    for name in ("read_text_file", "write_text_file", "list_dir"):
        spec, handler = toolkit._tools[name]
        toolkit.add_tool(name, spec, guard(name, handler))


def finalize_run(context: Any) -> Path:
    """Save the CLI's native result separately from planner declarations."""
    from rpent.evaluation import write_json_atomic

    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "platform": "robotwin",
            "task": dict(context.task_desc),
            "official_success": context.environment_success,
            "success_source": "native eval_success",
            "error": context.agent_error,
            "planner": context.planner,
            "model": context.model,
            "elapsed_s": context.elapsed_s,
            "stats": dict(context.stats),
            "finish": dict(context.finish_result) if context.finish_result else None,
        },
    )
