"""Process-local simulator and policy caches without modifying installed packages."""

from __future__ import annotations

import os
from pathlib import Path


def cache_environment(root: Path | str | None, component: str) -> dict[str, str]:
    """Create isolated compiler/render cache directories for a runtime component."""
    if root is None:
        return {}
    if component not in {"env", "vla"}:
        raise ValueError(f"Unknown runtime cache component: {component}")
    directory = Path(root).expanduser().resolve() / component
    result = {}
    for variable, child in (
        ("TORCH_EXTENSIONS_DIR", "torch"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("WARP_CACHE_PATH", "warp"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("XDG_CACHE_HOME", "xdg"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        path = directory / child
        path.mkdir(parents=True, exist_ok=True)
        result[variable] = str(path)
    (directory / "xdg/torch/kernels").mkdir(parents=True, exist_ok=True)
    result["PYTHONFAULTHANDLER"] = "1"
    result["MAX_JOBS"] = os.environ.get("MAX_JOBS", "2")
    result["TORCHINDUCTOR_COMPILE_THREADS"] = os.environ.get(
        "TORCHINDUCTOR_COMPILE_THREADS", "2"
    )
    return result
