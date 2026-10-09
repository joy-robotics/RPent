from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
from websockets.sync.client import connect

from rpent.utils.msgpack_numpy import pack, unpack

EMBODIMENT = "robotwin_aloha"
CAMERAS = ("head", "left_wrist", "right_wrist")
POLICY_TYPES = {"joybrain_vla", "joybrain_vla_mot"}


def validate_metadata(metadata: dict) -> int:
    if not isinstance(metadata, dict):
        raise ValueError("JoyBrain metadata must be a mapping")
    expected = {
        "protocol_version": "2.1",
        "embodiment_id": EMBODIMENT,
        "stats_key": EMBODIMENT,
        "action_dim": 14,
        "expected_state_dim": 14,
        "expected_num_input_images": 3,
        "returns": "actions",
        "preprocessing_owner": "server_canonical",
        "strict_input_validation": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Incompatible JoyBrain {key}: {metadata.get(key)!r}; expected {value!r}"
            )
    if metadata.get("policy_type") not in POLICY_TYPES:
        raise ValueError(
            f"Unsupported JoyBrain policy_type: {metadata.get('policy_type')!r}"
        )
    for key in ("state_layout", "action_layout"):
        if key in metadata and metadata[key] != "qpos14":
            raise ValueError(f"Incompatible {key}: {metadata[key]!r}; expected qpos14")
    if "camera_order" in metadata and list(metadata["camera_order"]) != list(CAMERAS):
        raise ValueError("Incompatible camera_order")
    if "robot_type" in metadata and metadata["robot_type"] != "aloha":
        raise ValueError("Incompatible robot_type; expected aloha")
    expected_keys = [
        f"observation.images.{camera}"
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    ]
    if (
        "required_image_keys" in metadata
        and list(metadata["required_image_keys"]) != expected_keys
    ):
        raise ValueError("Incompatible required_image_keys")
    chunk = metadata.get("chunk_size")
    if type(chunk) is not int or not 1 <= chunk <= 1024:
        raise ValueError("JoyBrain chunk_size must be an integer in [1, 1024]")
    return chunk


def observation_example(observation: dict, prompt: str) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("VLA prompt must be a non-empty string")
    state = np.asarray(observation["robot_state"]["qpos_target14"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("robot_state.qpos_target14 must be finite with shape (14,)")
    images = []
    for camera in CAMERAS:
        image = np.asarray(observation["views"][camera]["rgb"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{camera} must be uint8 RGB [H,W,3]")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"{camera} image is empty")
        images.append(np.ascontiguousarray(image))
    return {
        "state": state.copy(),
        "image": images,
        "lang": prompt,
        "embodiment_id": EMBODIMENT,
        "stats_key": EMBODIMENT,
    }


def validate_actions(
    actions, chunk_size: int, *, binarize_gripper: bool = True
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (1, chunk_size, 14) or not np.isfinite(array).all():
        raise ValueError(
            f"Expected finite actions [1,{chunk_size},14], got {array.shape}"
        )
    result = array[0].copy()
    if binarize_gripper:
        result[:, [6, 13]] = (result[:, [6, 13]] >= 0.5).astype(np.float32)
    elif np.any((result[:, [6, 13]] < 0) | (result[:, [6, 13]] > 1)):
        raise ValueError("Continuous gripper values must lie in [0,1]")
    return result


class JoyBrainVLAClient:
    """One synchronous connection; errors close it instead of replaying stale requests."""

    backend = "joybrain"

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 120.0,
        binarize_gripper: bool = True,
        expected_checkpoint: str | None = None,
        close_on_toolkit_close: bool = True,
    ):
        address = urlsplit(endpoint)
        if address.scheme not in {"ws", "wss"} or not address.hostname:
            raise ValueError("--vla-endpoint must be ws://host:port or wss://host:port")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("VLA timeout must be positive and finite")
        self.endpoint = endpoint
        self.close_on_toolkit_close = close_on_toolkit_close
        self.timeout = timeout
        self.binarize_gripper = binarize_gripper
        self._lock = threading.Lock()
        self._closed = False
        self._connection = connect(
            endpoint,
            proxy=None,
            compression=None,
            max_size=128 * 1024 * 1024,
            open_timeout=min(timeout, 30),
            close_timeout=2,
            ping_interval=None,
        )
        try:
            self.metadata = unpack(self._connection.recv(timeout=timeout))
            self.chunk_size = validate_metadata(self.metadata)
            if (
                expected_checkpoint is not None
                and Path(self.metadata.get("ckpt_path", "")).resolve()
                != Path(expected_checkpoint).expanduser().resolve()
            ):
                raise ValueError(
                    "JoyBrain endpoint checkpoint differs from --vla-model-path"
                )
        except Exception:
            self.close()
            raise

    def infer(self, observation: dict) -> np.ndarray:
        example = observation_example(observation, observation["task_language"])
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "JoyBrain connection closed; start a new session, do not replay"
                )
            request_id = uuid4().hex
            try:
                self._connection.send(
                    pack(
                        {
                            "type": "infer",
                            "request_id": request_id,
                            "payload": {"examples": [example], "do_sample": False},
                        }
                    )
                )
                response = unpack(self._connection.recv(timeout=self.timeout))
                if (
                    not isinstance(response, dict)
                    or response.get("request_id") != request_id
                ):
                    raise ValueError("JoyBrain response request_id mismatch")
                if response.get("protocol_version") != "2.1":
                    raise ValueError("JoyBrain response protocol_version mismatch")
                if response.get("status") != "ok" or response.get("ok") is not True:
                    raise RuntimeError(
                        f"JoyBrain inference failed: {response.get('error')!r}"
                    )
                if response.get("type") != "inference_result":
                    raise ValueError("JoyBrain response type mismatch")
                data = response["data"]
                if (
                    data.get("action_dim") != 14
                    or data.get("chunk_size") != self.chunk_size
                ):
                    raise ValueError(
                        "JoyBrain response action contract differs from handshake"
                    )
                return validate_actions(
                    data["actions"],
                    self.chunk_size,
                    binarize_gripper=self.binarize_gripper,
                )
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
