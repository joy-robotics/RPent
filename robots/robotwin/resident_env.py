"""Worker-owned, single-task RoboTwin runtime with explicit episode transitions."""

from __future__ import annotations

import argparse
import gc
import os
import time
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from robots.robotwin.env_server import RoboTwinEnvFacade, make_env
from robots.robotwin.recording import NativeRecorder, load_metadata
from robots.robotwin.robot_spec import ROBOTWIN_TASK_CONFIGS, env_runtime_contract


class ResidentEnvFacade(RoboTwinEnvFacade):
    """Keep the runtime alive without allowing reset within an active episode."""

    def __init__(
        self, env: Any, *, metadata: dict[str, Any], output_root: Path
    ) -> None:
        if len(env.venv.envs) != 1:
            raise ValueError("Resident recording requires one native environment")
        self.output_root = output_root.expanduser().resolve()
        self._episode_id: str | None = None
        self._poisoned = False
        self._reset_pending = False
        self._completed = 0
        self._native_args = deepcopy(env.venv.envs[0].args)
        self._initial_seed = metadata["seed"]
        super().__init__(env, metadata=metadata)

    def _register_rpc(self) -> None:
        super()._register_rpc()
        for name in ("begin", "end", "proposal", "summary"):
            self._rpc[f"trace.{name}"] = partial(self._trace, name)
        self._rpc["resident.end"] = self.end_episode
        for name, handler in list(self._rpc.items()):
            self._rpc[name] = self._scope(handler)
        self._rpc["resident.begin"] = self.begin_episode

    def _scope(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        def scoped(*args, _episode_id=None, **kwargs):
            if self._poisoned:
                raise RuntimeError(
                    "Resident environment is poisoned; retire this worker"
                )
            if not self._episode_id or _episode_id != self._episode_id:
                raise RuntimeError("Stale or missing episode identity")
            if self._reset_pending and handler not in (self.get_env_meta, self.reset):
                raise RuntimeError("Episode reset has not completed")
            return handler(*args, **kwargs)

        return scoped

    def _trace(self, method: str, *args, **kwargs) -> Any:
        return getattr(self._recorder, method)(*args, **kwargs)

    def begin_episode(self, seed: int, trace_dir: str, metadata_path: str) -> dict:
        """Prepare a new recorder and native language prewalk, never replaying a run."""
        if self._poisoned or self._episode_id is not None:
            raise RuntimeError("Previous episode is active or the worker is poisoned")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        trace_path = Path(trace_dir).resolve()
        config_path = Path(metadata_path).resolve()
        for path in (trace_path, config_path):
            if not path.is_relative_to(self.output_root) or path == self.output_root:
                raise ValueError(
                    "Resident artifacts must remain inside the worker root"
                )
        if trace_path.name != "trace" or config_path.parent != trace_path.parent:
            raise ValueError(
                "Trace and metadata must belong to the same episode directory"
            )
        if trace_path.exists():
            raise ValueError("Refusing to overwrite an attempted episode")
        metadata = load_metadata(config_path)
        cell = metadata.get("cell", {})
        if cell.get("seed") != seed or cell.get("task") != self._metadata["task_name"]:
            raise ValueError("Recorder metadata differs from the requested task/seed")
        if cell.get("job_id", trace_path.parent.name) != trace_path.parent.name:
            raise ValueError("Episode directory differs from recorder job identity")
        started = time.monotonic()
        try:
            sub_env = self._env.venv.envs[0]
            if self._completed or seed != self._initial_seed:
                import torch

                sub_env.close(clear_cache=True)
                sub_env.task = None
                gc.collect()
                torch.cuda.empty_cache()
                sub_env.args = deepcopy(self._native_args)
                sub_env.env_seed = seed
                sub_env.reset_count = 0
                sub_env.setup_task()
                gc.collect()
                torch.cuda.empty_cache()
            self._metadata = {**self._metadata, "seed": seed}
            self._recorder = NativeRecorder(trace_path, metadata)
            self._recorder.attach(sub_env.task)
            self._reset_pending = True
            self._episode_id = uuid4().hex
        except BaseException:
            self._poisoned = True
            raise
        return {
            "episode_id": self._episode_id,
            "env_meta": self.get_env_meta(),
            "env_pid": os.getpid(),
            "completed_episodes": self._completed,
            "prepare_s": time.monotonic() - started,
        }

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset exactly once and verify clean counters before exposing observations."""
        if not self._reset_pending:
            raise RuntimeError("Reset within a resident episode is forbidden")
        self._reset_pending = False
        try:
            result = super().reset()
            status = result[1]["episode_status"]
            if status["take_action_cnt"] != 0 or status["eval_success"]:
                raise RuntimeError("Native reset retained action or success state")
            if not self._recorder.is_attached_to(self._env.venv.envs[0].task):
                raise RuntimeError(
                    "Native reset replaced the recorded task or action hook"
                )
            return result
        except BaseException:
            self._poisoned = True
            raise

    def end_episode(self) -> dict:
        """Release the recorder only after the owner has drained the Agent episode."""
        if self._reset_pending:
            raise RuntimeError("Cannot finish an episode before reset")
        try:
            result = self._recorder.summary()
            self._recorder.detach()
        except BaseException:
            self._poisoned = True
            raise
        self._episode_id = None
        self._recorder = None
        self._completed += 1
        return {**result, "env_pid": os.getpid(), "completed_episodes": self._completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", choices=ROBOTWIN_TASK_CONFIGS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    env = make_env(
        args.task_name,
        args.task_config,
        args.seed,
        args.assets_path,
        args.max_episode_steps,
    )
    facade = ResidentEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name=args.task_name,
            task_config=args.task_config,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            record_actions=True,
        ),
        output_root=args.output_root,
    )
    facade.serve(
        transport="http",
        host="127.0.0.1",
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
