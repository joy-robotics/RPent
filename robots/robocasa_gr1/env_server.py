"""Native GR1 Gym owner with single-threaded rendering and recorded joint actions."""

from __future__ import annotations

import argparse
import importlib.util
import random
import time
from pathlib import Path

import numpy as np

from robots.robocasa_gr1.contracts import (
    EVAL_TYPES,
    GROUPS,
    action_array,
    action_dict,
    env_contract,
    observation,
    task_id,
)
from rpent.evaluation import write_json_atomic
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin


def native_task_ids(repo: Path, evaluation: str) -> tuple[str, ...]:
    """Load the original GR1 task split without importing its rollout script."""
    path = repo / "benchmark/Robocasa_GR1/task_lists.py"
    spec = importlib.util.spec_from_file_location("rpent_gr1_native_tasks", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Missing GR1 task list: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.EVAL_TYPE_TO_TASKS[evaluation])


def reset_native(env, seed: int):
    """Match the upstream reset of Python, NumPy and the wrapped Generator."""
    random.seed(seed)
    np.random.seed(seed)
    inner = getattr(env.unwrapped, "env", None)
    if inner is not None:
        inner.rng = np.random.default_rng(seed)
    return env.reset(seed=seed)


class GR1EnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One recorded episode; uncertain execution cannot be replayed or reset."""

    def __init__(self, env, metadata: dict, output: Path):
        self.env = env
        self.metadata = metadata
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self._obs = None
        self._reset = False
        self._failed = False
        self._uncertain = False
        self._count = 0
        self._success = False
        self._terminated = False
        self._truncated = False
        self._closed = False
        super().__init__()
        write_json_atomic(self.output / "episode.json", metadata)
        self._save_status()

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["env.observe"] = self.observe
        self._rpc["env.get_status"] = self.get_status
        self._readonly_methods.update(("env.observe", "env.get_status"))

    def get_env_meta(self) -> dict:
        return dict(self.metadata)

    def get_status(self) -> dict:
        return {
            "native_success": self._success,
            "native_terminated": self._terminated,
            "native_truncated": self._truncated,
            "step_count": self._count,
            "step_limit": self.metadata["max_episode_steps"],
            "seed": self.metadata["seed"],
            "budget_exhausted": self._count >= self.metadata["max_episode_steps"],
            "recording_failed": self._failed,
            "execution_uncertain": self._uncertain,
        }

    def _save_status(self):
        write_json_atomic(self.output / "status.json", self.get_status())

    def reset(self):
        if self._reset or self._failed:
            raise RuntimeError("A recorded GR1 episode cannot be reset or replayed")
        self._reset = True
        try:
            for _ in range(self.metadata["warmup_resets"]):
                reset_native(self.env, self.metadata["seed"])
            raw, _ = reset_native(self.env, self.metadata["seed"])
            self._obs = observation(raw)
            self._save_status()
            return self.observe(), {"episode_status": self.get_status()}
        except BaseException:
            self._failed = True
            self._save_status()
            raise

    def observe(self) -> dict:
        if self._obs is None:
            raise RuntimeError("GR1 reset has not completed")
        return {**self._obs, "episode_status": self.get_status()}

    def render_camera(self, camera_name: str, depth: bool = False):
        if camera_name != "ego" or depth:
            raise ValueError(
                "GR1 currently exposes only the native processed ego RGB view"
            )
        return self.observe()["ego_rgb"]

    def get_camera_meta(self, camera_name: str):
        image = self.render_camera(camera_name)
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "metric_depth": False,
            "image_key": self._obs["image_key"],
            "preprocessing": "native GR1 wrapper; VLA crop is server-owned",
        }

    def get_task_language(self) -> str:
        return self.observe()["task_language"]

    def _require_active(self):
        if self._failed or self._uncertain:
            raise RuntimeError(
                "GR1 recording/execution is uncertain; no further actions allowed"
            )
        if (
            self._obs is None
            or self._success
            or self._terminated
            or self._truncated
            or self._count >= self.metadata["max_episode_steps"]
        ):
            raise RuntimeError("GR1 episode is uninitialized or terminal")

    def _snapshot(self, directory: Path, prefix: str) -> dict:
        from PIL import Image

        path = directory / f"{prefix}_ego.png"
        Image.fromarray(self._obs["ego_rgb"]).save(path)
        return {
            "state44": self._obs["state44"].tolist(),
            "image": path.name,
            "task_language": self._obs["task_language"],
        }

    def step(self, action, source: str = "native_joint_targets"):
        array = action_array(action)
        if len(array) != 1:
            raise ValueError("env.step requires one 29D action")
        self._require_active()
        directory = self.output / f"step_{self._count:06d}"
        entered = False
        created = False
        record = {
            "index": self._count,
            "source": str(source),
            "action29": array[0].tolist(),
        }
        try:
            directory.mkdir(exist_ok=False)
            created = True
            record["before"] = self._snapshot(directory, "before")
            record["wall_start_ns"] = time.time_ns()
            entered = True
            raw, reward, terminated, truncated, info = self.env.step(
                action_dict(array[0])
            )
            record["wall_end_ns"] = time.time_ns()
            self._count += 1
            if not isinstance(info, dict) or "success" not in info:
                raise ValueError("Native GR1 step omitted its success verdict")
            self._success |= bool(info["success"]) or bool(terminated)
            self._terminated |= bool(terminated)
            self._truncated |= bool(truncated)
            self._obs = observation(raw)
            record.update(
                after=self._snapshot(directory, "after"),
                execution="executed",
                reward=float(reward),
                native_terminated=bool(terminated),
                native_truncated=bool(truncated),
                native_info_success=bool(info["success"]),
                episode_status=self.get_status(),
            )
            write_json_atomic(directory / "transition.json", record)
            self._save_status()
        except BaseException as error:
            self._failed = True
            self._uncertain = entered
            record.update(
                execution="unknown_or_incomplete" if entered else "not_executed",
                error=f"{type(error).__name__}: {error}",
            )
            try:
                if created:
                    write_json_atomic(directory / "transition.json", record)
                self._save_status()
            except Exception:
                pass
            raise
        status = self.get_status()
        return (
            self.observe(),
            float(reward),
            self._success or self._terminated,
            self._truncated or status["budget_exhausted"],
            {"episode_status": status, "executed_actions": 1},
        )

    def chunk_step(
        self, actions, return_all_frames: bool = True, source: str = "gr1_act"
    ):
        array = action_array(actions)
        self._require_active()
        frames = []
        executed = 0
        for action in array:
            status = self.get_status()
            if (
                status["native_success"]
                or status["native_terminated"]
                or status["native_truncated"]
                or status["budget_exhausted"]
            ):
                break
            if return_all_frames:
                frames.append(self._obs["ego_rgb"].copy())
            result = self.step(action, source=source)
            executed += 1
        status = self.get_status()
        return (
            {"final": self.observe(), "frames": frames},
            result[1],
            self._success or self._terminated,
            self._truncated or status["budget_exhausted"],
            {"episode_status": status, "executed_actions": executed},
        )

    def close(self):
        if not self._closed:
            self._closed = True
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--eval-type", choices=EVAL_TYPES, default="id")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--joybrain-repo", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.max_episode_steps < 1:
        raise ValueError("GR1 seed must be nonnegative and action budget positive")
    task = task_id(args.task_name)
    if task not in native_task_ids(args.joybrain_repo, args.eval_type):
        raise ValueError(
            f"Task {task} is not in the original {args.eval_type} GR1 split"
        )
    import gymnasium as gym
    from robocasa.utils.gym_utils import GrootRoboCasaEnv

    env = gym.make(task, enable_render=True)
    facade = None
    try:
        if not isinstance(env.unwrapped, GrootRoboCasaEnv):
            raise ValueError("Gym task did not create the native GR1 wrapper")
        for name, dimension in GROUPS:
            if env.action_space[f"action.{name}"].shape != (dimension,):
                raise ValueError(f"Unexpected GR1 native action space for {name}")
        facade = GR1EnvFacade(
            env,
            env_contract(task, args.eval_type, args.seed, args.max_episode_steps),
            args.trace_dir,
        )
        facade.serve(
            transport="http",
            host="127.0.0.1",
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        if facade is not None:
            facade.close()
        else:
            env.close()


if __name__ == "__main__":
    main()
