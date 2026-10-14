"""Strict Joy-Brain GR1 WebSocket client without automatic inference replay."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from robots.robocasa_gr1.contracts import EMBODIMENT, ROBOT_TYPE
from rpent.utils.msgpack_numpy import pack, unpack


class GR1VLAClient:
    """Receive canonical unnormalized 29D actions from the original policy backend."""

    def __init__(self, endpoint: str, *, checkpoint: str, timeout: float = 600):
        from websockets.sync.client import connect

        if (
            urlsplit(endpoint).scheme not in ("ws", "wss")
            or not np.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "GR1 VLA requires a WebSocket endpoint and finite positive timeout"
            )
        self.endpoint = endpoint
        self.timeout = timeout
        self.closed = False
        self._lock = threading.Lock()
        self.connection = connect(
            endpoint,
            proxy=None,
            compression=None,
            max_size=128 * 1024 * 1024,
            open_timeout=min(timeout, 30),
            close_timeout=2,
            ping_interval=None,
        )
        try:
            self.metadata = unpack(self.connection.recv(timeout=timeout))
            expected = {
                "protocol_version": "2.1",
                "embodiment_id": EMBODIMENT,
                "stats_key": EMBODIMENT,
                "robot_type": ROBOT_TYPE,
                "expected_state_dim": 44,
                "action_dim": 29,
                "expected_num_input_images": 1,
                "preprocessing_owner": "server_canonical",
                "strict_input_validation": True,
                "returns": "actions",
            }
            if not isinstance(self.metadata, dict) or any(
                self.metadata.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(
                    "Endpoint is not a compatible canonical Joy-Brain GR1 policy"
                )
            if self.metadata.get("policy_type") not in (
                "joybrain_vla",
                "joybrain_vla_mot",
            ):
                raise ValueError("Unsupported GR1 policy type")
            self.chunk_size = self.metadata.get("chunk_size")
            if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 1024:
                raise ValueError("Invalid GR1 policy chunk size")
            if (
                Path(self.metadata.get("ckpt_path", "")).resolve()
                != Path(checkpoint).expanduser().resolve()
            ):
                raise ValueError(
                    "GR1 endpoint checkpoint differs from the requested checkpoint"
                )
        except BaseException:
            self.close()
            raise

    def infer(self, observed: dict) -> np.ndarray:
        """Predict one complete chunk, leaving crop/normalization to the server."""
        state = np.asarray(observed["state44"], dtype=np.float32)
        image = np.asarray(observed["ego_rgb"])
        if state.shape != (44,) or not np.isfinite(state).all():
            raise ValueError("GR1 VLA expects a finite raw 44D state")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("GR1 VLA expects uint8 RGB")
        language = observed["task_language"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError("GR1 VLA requires the current native task language")
        with self._lock:
            if self.closed:
                raise RuntimeError("GR1 VLA client is closed; do not replay requests")
            request_id = uuid4().hex
            try:
                self.connection.send(
                    pack(
                        {
                            "type": "infer",
                            "request_id": request_id,
                            "payload": {
                                "do_sample": False,
                                "examples": [
                                    {
                                        "state": state,
                                        "image": [image],
                                        "lang": language,
                                        "embodiment_id": EMBODIMENT,
                                        "stats_key": EMBODIMENT,
                                    }
                                ],
                            },
                        }
                    )
                )
                reply = unpack(self.connection.recv(timeout=self.timeout))
                if (
                    not isinstance(reply, dict)
                    or reply.get("request_id") != request_id
                    or reply.get("protocol_version") != "2.1"
                ):
                    raise ValueError("Invalid GR1 inference response identity")
                if (
                    reply.get("type") != "inference_result"
                    or reply.get("status") != "ok"
                    or reply.get("ok") is not True
                ):
                    raise RuntimeError(f"GR1 inference failed: {reply.get('error')}")
                data = reply["data"]
                actions = np.asarray(data["actions"], dtype=np.float32)
                if (
                    data.get("action_dim") != 29
                    or data.get("chunk_size") != self.chunk_size
                    or actions.shape != (1, self.chunk_size, 29)
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError(
                        "GR1 response is not the declared finite [1,chunk,29] action chunk"
                    )
                return actions[0].copy()
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Close this connection, never the shared VLA server."""
        if not self.closed:
            self.closed = True
            self.connection.close()
