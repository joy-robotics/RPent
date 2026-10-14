"""Serve prompt-defined Qwen3.5 perception without loading a simulator or robot controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from rpent.utils.msgpack_numpy import pack, unpack

from .tasks import (
    DEFAULT_CHECKPOINT,
    MAX_IMAGE_PIXELS,
    MAX_MESSAGE_BYTES,
    MAX_TOTAL_PIXELS,
    MAX_VIDEO_FRAMES,
    PROTOCOL,
    TASKS,
    build_task_prompt,
    parse_task_response,
    prompt_fingerprint,
    validate_inputs,
)


class QwenBackend:
    def __init__(self, checkpoint, *, device="cuda", max_new_tokens=1024):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        root = Path(checkpoint).resolve(strict=True)
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise ValueError("This backend requires a Qwen3.5 multimodal checkpoint")
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Use a CUDA-enabled policy process for this 9B checkpoint"
            )
        if not 32 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be in 32..4096")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            root, local_files_only=True, use_fast=False
        )
        self.model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            root,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Checkpoint weights did not load exactly: {loading}")
        self.model = self.model.to(device).eval()
        self.metadata = {
            "protocol": PROTOCOL,
            "checkpoint": str(root),
            "model_type": config["model_type"],
            "config_sha256": hashlib.sha256(
                (root / "config.json").read_bytes()
            ).hexdigest(),
            "checkpoint_files_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "config.json",
                    "generation_config.json",
                    "processor_config.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
                if (root / name).is_file()
            },
            "weights_loading": {
                name: list(loading.get(name, []))
                for name in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            },
            "processor": type(self.processor).__name__,
            "image_processor": type(self.processor.image_processor).__name__,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(device),
            "input_limits": {
                "max_frames": MAX_VIDEO_FRAMES,
                "max_image_pixels": MAX_IMAGE_PIXELS,
                "max_total_pixels": MAX_TOTAL_PIXELS,
            },
            "task_prompts": {name: prompt_fingerprint(name) for name in TASKS},
            "generation": {
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": max_new_tokens,
                "dtype": "bfloat16",
                "attention": "sdpa",
                "eos_token_id": self.model.generation_config.eos_token_id,
                "pad_token_id": self.model.generation_config.pad_token_id,
            },
            "returns": "validated_perception_json_not_actions",
        }

    def predict(self, task, images, target, camera=None):
        import torch

        prompt = build_task_prompt(task, target, camera=camera, frame_count=len(images))
        content = []
        for image_index in range(len(images)):
            if image_index:
                content.append({"type": "text", "text": " "})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=[text],
            images=[Image.fromarray(image) for image in images],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class EmbodiedService:
    def __init__(self, backend):
        self.backend = backend
        self._lock = threading.Lock()

    def handle(self, connection):
        try:
            connection.send(pack(self.backend.metadata))
            for message in connection:
                request_id = None
                try:
                    request = unpack(message)
                    request_id = request.get("request_id")
                    if request.get("protocol") != PROTOCOL or not isinstance(
                        request_id, str
                    ):
                        raise ValueError("Invalid VLM protocol or request_id")
                    task = request["task"]
                    if task not in TASKS:
                        raise ValueError("Unsupported VLM task")
                    images, target, camera = validate_inputs(
                        task,
                        request["images"],
                        request["target"],
                        request.get("camera"),
                    )
                    started = time.monotonic()
                    with self._lock:
                        raw_text = self.backend.predict(task, images, target, camera)
                    try:
                        parse_task_response(
                            task,
                            raw_text,
                            images[-1].shape[1],
                            images[-1].shape[0],
                            target=target,
                        )
                    except ValueError:
                        logging.warning(
                            "Rejected VLM response request_id=%s raw_text=%r",
                            request_id,
                            raw_text,
                        )
                        raise
                    response = {
                        "raw_text": raw_text,
                        "inference_seconds": time.monotonic() - started,
                    }
                except Exception as error:
                    response = {
                        "error": {"type": type(error).__name__, "message": str(error)}
                    }
                connection.send(
                    pack({"protocol": PROTOCOL, "request_id": request_id, **response})
                )
        except ConnectionClosed:
            return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19194)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This unauthenticated prototype must bind to loopback")
    logging.basicConfig(level=logging.INFO)
    backend = QwenBackend(
        args.checkpoint, device=args.device, max_new_tokens=args.max_new_tokens
    )
    service = EmbodiedService(backend)
    with serve(
        service.handle,
        args.host,
        args.port,
        max_size=MAX_MESSAGE_BYTES,
        compression=None,
        ping_interval=None,
        close_timeout=2,
    ) as server:
        print(
            json.dumps(
                {
                    "ready": True,
                    "endpoint": f"ws://{args.host}:{args.port}",
                    **backend.metadata,
                }
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
