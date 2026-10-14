"""Typed RPent client for the independent GR1 native environment."""

from __future__ import annotations

from robots.robocasa_gr1.contracts import action_array
from rpent.robots.components.env_client_base import BaseEnvClient


class GR1EnvClient(BaseEnvClient):
    """Cache fresh observations and fail closed after uncertain mutations."""

    _TIMEOUT_S = {**BaseEnvClient._TIMEOUT_S, "default": 120.0, "env.reset": 600.0}

    def __init__(self, client, *, expected_meta: dict):
        self.execution_uncertain = False
        self.last_info = {}
        self.server_meta = expected_meta
        super().__init__(client, expected_meta=expected_meta)

    def _validate_status(self, observed: dict, status: dict, count: int) -> None:
        if (
            observed["episode_status"] != status
            or status["step_count"] != count
            or status["seed"] != self.server_meta["seed"]
            or status["step_limit"] != self.server_meta["max_episode_steps"]
            or not 0 <= count <= status["step_limit"]
        ):
            raise ValueError("GR1 observation does not match the recorded episode")

    def reset(self):
        observation, info = self._client.call("env.reset", timeout_s=600)
        self._validate_status(observation, info["episode_status"], 0)
        self.last_obs, self.last_info = observation, info
        return observation, info

    def observe(self) -> dict:
        result = self._client.call("env.observe", timeout_s=120)
        self._validate_status(
            result,
            result["episode_status"],
            self.last_info["episode_status"]["step_count"],
        )
        self.last_obs = result
        self.last_info = {"episode_status": result["episode_status"]}
        return result

    def chunk_step(self, actions, *, return_all_frames=True, source="gr1_act"):
        array = action_array(actions)
        if self.execution_uncertain:
            raise RuntimeError("GR1 action result is uncertain; do not replay")
        try:
            result = self._client.call(
                "env.chunk_step",
                args=(array,),
                kwargs={"return_all_frames": return_all_frames, "source": source},
                timeout_s=600,
            )
            payload, reward, terminated, truncated, info = result
            count = info["executed_actions"]
            if type(count) is not int or not 1 <= count <= len(array):
                raise ValueError("Invalid native GR1 execution count")
            self._validate_status(
                payload["final"],
                info["episode_status"],
                self.last_info["episode_status"]["step_count"] + count,
            )
            if return_all_frames and len(payload["frames"]) != count:
                raise ValueError("GR1 frames do not match executed actions")
            self.last_obs, self.last_info = payload["final"], info
            return payload, reward, terminated, truncated, info
        except BaseException:
            self.execution_uncertain = True
            raise

    def step(self, action, *, source="native_joint_targets"):
        array = action_array(action)
        if len(array) != 1:
            raise ValueError("GR1 step requires one action")
        payload, reward, terminated, truncated, info = self.chunk_step(
            array, source=source
        )
        return payload["final"], reward, terminated, truncated, info
