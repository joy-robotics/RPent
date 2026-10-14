"""GR1 observations, bounded VLA calls and native joint-target primitives."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from robots.robocasa_gr1.contracts import EFFECTIVE_INDICES, GROUPS, action_array
from rpent.evaluation import write_json_atomic
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly


def schema(name: str, description: str, properties: dict, required=()) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


class GR1Toolkit(Toolkit):
    """Keep prediction, execution and native success distinct in Agent-visible results."""

    def __init__(
        self, *, env, model, replan_steps: int, output: Path, memory, dashboard_events
    ):
        super().__init__(
            dashboard_events=dashboard_events, state=EnvState(output), memory=memory
        )
        self.env = env
        self.model = model
        self.output = output
        self.replan_steps = replan_steps
        self._frames = []
        self._predictions = 0
        self._failed = False
        self._closed = False
        if not 1 <= replan_steps <= model.chunk_size:
            raise ValueError("replan-steps must be within the checkpoint chunk length")
        self._tools.pop("finish", None)
        self.add_tool(
            "view_env_state",
            schema(
                "view_env_state",
                "Read the latest GR1 ego image, joints, native task language and episode status.",
                {},
            ),
            self.view_env_state,
        )
        self.add_tool(
            "gr1_act",
            schema(
                "gr1_act",
                "Run Joy-Brain for one to four receding-horizon windows; inspect the returned native status and image before continuing.",
                {"chunks": {"type": "integer", "minimum": 1, "maximum": 4}},
                (),
            ),
            self.gr1_act,
        )
        groups = {
            name: {
                "type": "array",
                "items": {"type": "number"},
                "minItems": dimension,
                "maxItems": dimension,
            }
            for name, dimension in GROUPS
        }
        self.add_tool(
            "set_joint_targets",
            schema(
                "set_joint_targets",
                "Execute one native GR1 joint-target step. Provide only the named joint groups to change; all other groups hold their currently observed positions. These are not Cartesian coordinates or scalar gripper commands.",
                {
                    "targets": {
                        "type": "object",
                        "properties": groups,
                        "minProperties": 1,
                        "additionalProperties": False,
                    }
                },
                ("targets",),
            ),
            self.set_joint_targets,
        )
        self.add_tool(
            "finish",
            schema(
                "finish",
                "End the episode. The reported success is determined by the native environment, not the requested status.",
                {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure", "stuck"],
                    },
                    "summary": {"type": "string"},
                },
                ("status", "summary"),
            ),
            self.finish,
        )
        self.get_env_state(
            command={"action": "reset"}, result={"initialized": True}, elapsed_s=0
        )
        self._publish_step(self.state.latest_record())

    def _active(self) -> bool:
        status = self.env.last_info["episode_status"]
        if (
            self._failed
            or self.env.execution_uncertain
            or status["recording_failed"]
            or status["execution_uncertain"]
        ):
            raise RuntimeError(
                "GR1 execution or recording failed; no more mutations allowed"
            )
        return not any(
            status[key]
            for key in (
                "native_success",
                "native_terminated",
                "native_truncated",
                "budget_exhausted",
            )
        )

    def gr1_act(self, chunks: int = 1) -> dict:
        """Execute canonical predicted windows, never an old chunk after a tool boundary."""
        if type(chunks) is not int or not 1 <= chunks <= 4:
            raise ValueError("chunks must be an integer from 1 to 4")
        executed = predictions = 0
        inference_s = execution_s = 0.0
        for _ in range(chunks):
            self.raise_if_cancelled()
            if not self._active():
                break
            observed = self.env.observe()
            started = time.monotonic()
            actions = self.model.infer(observed)
            inference_s += time.monotonic() - started
            try:
                self.state.save(
                    f"proposal_{self._predictions:06d}.npy", actions, step=None
                )
            except BaseException:
                self._failed = True
                raise
            self._predictions += 1
            predictions += 1
            self.raise_if_cancelled()
            remaining = (
                observed["episode_status"]["step_limit"]
                - observed["episode_status"]["step_count"]
            )
            count = min(self.replan_steps, remaining)
            started = time.monotonic()
            payload, _, _, _, info = self.env.chunk_step(
                actions[:count], source=f"gr1_act:{uuid4().hex}"
            )
            execution_s += time.monotonic() - started
            executed += info["executed_actions"]
            self._frames.extend(payload["frames"])
        return {
            "executed_actions": executed,
            "vla_inferences": predictions,
            "vla_inference_s": inference_s,
            "native_execution_rpc_s": execution_s,
            "replan_steps": self.replan_steps,
            "episode_status": self.env.last_info["episode_status"],
        }

    def set_joint_targets(self, targets: dict) -> dict:
        """Hold unmentioned joints and send exactly one native effective action."""
        if (
            not isinstance(targets, dict)
            or not targets
            or set(targets) - {name for name, _ in GROUPS}
        ):
            raise ValueError("Provide nonempty named GR1 joint groups")
        self.raise_if_cancelled()
        if not self._active():
            raise RuntimeError("GR1 episode is already terminal")
        observed = self.env.observe()
        command = observed["state44"][list(EFFECTIVE_INDICES)].copy()
        offset = 0
        for name, dimension in GROUPS:
            if name in targets:
                values = np.asarray(targets[name], dtype=np.float32)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(
                        f"{name} requires {dimension} finite joint targets"
                    )
                command[offset : offset + dimension] = values
            offset += dimension
        payload, _, _, _, info = self.env.chunk_step(
            action_array(command), source=f"set_joint_targets:{uuid4().hex}"
        )
        self._frames.extend(payload["frames"])
        return {
            "executed_actions": info["executed_actions"],
            "episode_status": info["episode_status"],
        }

    @readonly
    def view_env_state(self) -> dict:
        record = self.state.latest_record()
        return {
            "state": record.state,
            "_image_bytes": self.state.artifact_path(
                "ego.png", step=record.step_idx
            ).read_bytes(),
        }

    @readonly
    def finish(self, status: str, summary: str) -> dict:
        if status not in ("success", "failure", "stuck"):
            raise ValueError("Unknown finish status")
        native = self.env.observe()["episode_status"]
        return {
            "_finish": True,
            "status": "success" if native["native_success"] else "failure",
            "requested_status": status,
            "summary": summary,
            "success": native["native_success"],
            "episode_status": native,
        }

    def get_env_state(self, *, command: dict, result: dict, elapsed_s: float) -> dict:
        try:
            observed = self.env.observe()
        except BaseException:
            self._failed = True
            raise
        status = observed["episode_status"]
        visible = {
            "task_language": observed["task_language"],
            "joint_positions": {
                name: values.tolist()
                for name, values in observed["joint_positions"].items()
            },
            "state44": observed["state44"].tolist(),
            "image_key": observed["image_key"],
            "episode_status": status,
            "robot_type": self.env.server_meta["robot_type"],
        }
        try:
            with self.state.record_step(
                state=visible,
                command=command,
                result=result,
                elapsed_s=elapsed_s,
                terminated=status["native_success"] or status["native_terminated"],
                truncated=status["budget_exhausted"] or status["native_truncated"],
            ) as step:
                self.state.save("ego.png", observed["ego_rgb"], step=step)
                self.state.save("state44.npy", observed["state44"], step=step)
            image_bytes = self.state.artifact_path("ego.png", step=step).read_bytes()
        except BaseException:
            self._failed = True
            raise
        return {
            "state": visible,
            "result": result,
            "_image_bytes": image_bytes,
        }

    def solved(self) -> bool:
        return bool(self.env.last_info["episode_status"]["native_success"])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            write_json_atomic(
                self.output / "native_status.json",
                {
                    **self.env.last_info["episode_status"],
                    "client_execution_uncertain": self.env.execution_uncertain,
                    "client_recording_failed": self._failed,
                    "vla_inferences": self._predictions,
                },
            )
            if self._frames:
                self.state.save("episode.mp4", self._frames, step=None, fps=20)
        finally:
            try:
                self.model.close()
            finally:
                self.env._client.close()
                self._frames.clear()
