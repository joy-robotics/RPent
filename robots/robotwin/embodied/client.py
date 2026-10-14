"""Bounded read-only VLM RPC, deliberately separate from the qpos14 VLA contract."""

from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from websockets.sync.client import connect

from rpent.utils.msgpack_numpy import pack, unpack

from .tasks import (
    DEFAULT_TOOL,
    MAX_MESSAGE_BYTES,
    PROTOCOL,
    TASKS,
    build_task_prompt,
    parse_task_response,
    prompt_fingerprint,
    validate_inputs,
)


class EmbodiedClient:
    def __init__(self, endpoint, *, checkpoint, timeout=180):
        address = urlsplit(endpoint)
        if address.scheme not in {"ws", "wss"} or not address.hostname:
            raise ValueError("VLM endpoint must be ws://host:port or wss://host:port")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("VLM timeout must be positive and finite")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._closed = False
        self._connection = connect(
            endpoint,
            proxy=None,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=min(timeout, 30),
            close_timeout=2,
            ping_interval=None,
        )
        try:
            self.metadata = unpack(self._connection.recv(timeout=timeout))
            if self.metadata.get("protocol") != PROTOCOL:
                raise ValueError("VLM protocol mismatch; do not connect a VLA endpoint")
            if (
                Path(self.metadata["checkpoint"]).resolve()
                != Path(checkpoint).resolve()
            ):
                raise ValueError("VLM endpoint serves a different checkpoint")
            for task in TASKS:
                if self.metadata.get("task_prompts", {}).get(
                    task
                ) != prompt_fingerprint(task):
                    raise ValueError(
                        "VLM task prompt does not match the local tool definition"
                    )
        except Exception:
            self.close()
            raise

    def infer(self, image, target, *, task=DEFAULT_TOOL, camera=None):
        if task not in TASKS:
            raise ValueError(f"Unknown embodied task: {task}")
        images = image if TASKS[task].get("video") else [image]
        images, target, camera = validate_inputs(task, images, target, camera)
        prompt = build_task_prompt(task, target, camera=camera, frame_count=len(images))
        with self._lock:
            if self._closed:
                raise RuntimeError("VLM connection is closed; start a new connection")
            request_id = uuid4().hex
            try:
                self._connection.send(
                    pack(
                        {
                            "protocol": PROTOCOL,
                            "request_id": request_id,
                            "task": task,
                            "images": images,
                            "target": target,
                            "camera": camera,
                        }
                    )
                )
                response = unpack(self._connection.recv(timeout=self.timeout))
                if (
                    response.get("protocol") != PROTOCOL
                    or response.get("request_id") != request_id
                ):
                    raise ValueError("VLM response protocol/request_id mismatch")
                if response.get("error"):
                    raise RuntimeError(f"VLM inference failed: {response['error']}")
                raw_text = response["raw_text"]
                parsed = parse_task_response(
                    task,
                    raw_text,
                    images[-1].shape[1],
                    images[-1].shape[0],
                    target=target,
                )
                return {
                    **parsed,
                    "request_id": request_id,
                    "task": task,
                    "target": target,
                    "checkpoint": self.metadata["checkpoint"],
                    "prompt_sha256": prompt_fingerprint(task),
                    "raw_text": raw_text,
                    "input_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "frame_count": len(images),
                    **({"camera": camera} if camera else {}),
                    "inference_seconds": response["inference_seconds"],
                }
            except Exception:
                self.close()
                raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
