"""Launch the original JoyBrain backend in its own policy Python environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path


def validate_checkpoint(path):
    root = Path(path).expanduser().resolve()
    for name in ("config.json", "stats.json", "model.safetensors"):
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise ValueError(f"Incomplete JoyBrain checkpoint: {root / name}")
    config = json.loads((root / "config.json").read_text())
    if config.get("type") not in {"joybrain_vla", "joybrain_vla_mot"}:
        raise ValueError("Expected a JoyBrain VLA/MoT checkpoint")
    if config.get("n_obs_steps", 1) != 1 or config.get("max_num_views") != 3:
        raise ValueError(
            "This RoboTwin backend requires one observation and three views"
        )
    chunk_size = config.get("chunk_size")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 1024:
        raise ValueError("Invalid JoyBrain chunk_size")
    stats = json.loads((root / "stats.json").read_text())
    for feature in ("observation.state", "action"):
        values = stats.get("robotwin_aloha", {}).get(feature, {})
        if any(len(values.get(name, [])) != 14 for name in ("mean", "std")):
            raise ValueError(
                "Checkpoint requires explicit robotwin_aloha 14D state/action statistics"
            )
        for name in ("mean", "std"):
            if any(not math.isfinite(float(value)) for value in values[name]):
                raise ValueError("Non-finite JoyBrain normalization statistics")
        if any(float(value) < 0 for value in values["std"]):
            raise ValueError("Negative JoyBrain normalization standard deviation")
    return root


def _watch_parent():
    sys.stdin.buffer.read()
    os._exit(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joybrain-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args(argv)
    if args.parent_watch:
        threading.Thread(target=_watch_parent, daemon=True).start()
    checkpoint = validate_checkpoint(args.model_path)
    source = args.joybrain_repo.expanduser().resolve() / "src"
    if not (
        source / "lerobot/serving/backends/policy_backend_joybrain_vla.py"
    ).is_file():
        raise ValueError(f"Joy-Brain-vla source not found: {source}")
    loaded = sys.modules.get("lerobot")
    if (
        loaded is not None
        and Path(getattr(loaded, "__file__", "")).resolve()
        != source / "lerobot/__init__.py"
    ):
        raise RuntimeError("A different LeRobot is loaded; use a separate --vla-python")
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source.parent))
    import torch
    from lerobot.serving.backends.policy_backend_joybrain_vla import JoyBrainVLABackend
    from lerobot.serving.transport.websocket_policy_server import WebsocketPolicyServer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("JoyBrain CUDA device is unavailable")
    backend = JoyBrainVLABackend(
        ckpt_path=str(checkpoint),
        device=args.device,
        stats_key="robotwin_aloha",
        robot_type="robotwin_aloha",
        resize_size=224,
    )
    metadata = {
        **backend.metadata(),
        "state_layout": "qpos14",
        "action_layout": "qpos14",
        "camera_order": ["head", "left_wrist", "right_wrist"],
    }
    print(f"JoyBrain policy ready: {metadata}", flush=True)
    WebsocketPolicyServer(
        backend, host=args.host, port=args.port, idle_timeout=1800, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
