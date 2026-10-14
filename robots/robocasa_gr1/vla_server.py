"""Launch the original Joy-Brain GR1 backend in its model interpreter."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path


def validate_checkpoint(path: Path) -> Path:
    """Require GR1 statistics and a complete explicit checkpoint, never a latest alias."""
    root = path.expanduser().resolve()
    for name in ("config.json", "stats.json", "model.safetensors"):
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise ValueError(f"Incomplete GR1 checkpoint: {root / name}")
    config = json.loads((root / "config.json").read_text())
    if (
        config.get("type") not in ("joybrain_vla", "joybrain_vla_mot")
        or config.get("n_obs_steps", 1) != 1
    ):
        raise ValueError("Expected a single-observation Joy-Brain GR1 checkpoint")
    chunk = config.get("chunk_size")
    if type(chunk) is not int or not 1 <= chunk <= 1024:
        raise ValueError("Invalid GR1 chunk size")
    stats = json.loads((root / "stats.json").read_text()).get("robocasa_gr1", {})
    for name in ("observation.state", "action"):
        for field in ("mean", "std"):
            values = stats.get(name, {}).get(field, [])
            if (
                not isinstance(values, list)
                or len(values) != 29
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or (field == "std" and value < 0)
                    for value in values
                )
            ):
                raise ValueError(
                    f"GR1 {name}.{field} must describe 29 effective joints"
                )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--joybrain-repo", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.parent_watch:

        def watch():
            sys.stdin.buffer.read()
            os._exit(0)

        threading.Thread(target=watch, daemon=True).start()
    checkpoint = validate_checkpoint(args.model_path)
    source = args.joybrain_repo.expanduser().resolve() / "src"
    if not (
        source / "lerobot/serving/backends/policy_backend_joybrain_vla.py"
    ).is_file():
        raise ValueError(
            "--joybrain-repo must contain the original Joy-Brain serving backend"
        )
    loaded = sys.modules.get("lerobot")
    if (
        loaded is not None
        and Path(loaded.__file__).resolve() != source / "lerobot/__init__.py"
    ):
        raise RuntimeError(
            "Another LeRobot package is loaded; use a dedicated model Python"
        )
    sys.path[:0] = [str(source), str(source.parent)]
    from lerobot.serving.backends.policy_backend_joybrain_vla import JoyBrainVLABackend
    from lerobot.serving.transport.websocket_policy_server import WebsocketPolicyServer

    backend = JoyBrainVLABackend(
        ckpt_path=str(checkpoint),
        device="cuda",
        stats_key="robocasa_gr1",
        robot_type="robocasa_gr1",
        resize_size=224,
    )
    WebsocketPolicyServer(
        backend,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=1800,
        metadata=backend.metadata(),
    ).serve_forever()


if __name__ == "__main__":
    main()
