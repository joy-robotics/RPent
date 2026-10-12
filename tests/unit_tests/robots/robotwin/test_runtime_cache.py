import os

import pytest

from robots.robotwin.runtime_cache import cache_environment


def test_unconfigured_runtime_keeps_existing_environment(tmp_path):
    before = dict(os.environ)
    assert cache_environment(None, "env") == {}
    assert dict(os.environ) == before
    assert not list(tmp_path.iterdir())


def test_components_and_worker_slots_have_private_complete_caches(tmp_path):
    before = dict(os.environ)
    first = cache_environment(tmp_path / "worker0", "env")
    second = cache_environment(tmp_path / "worker1", "env")
    policy = cache_environment(tmp_path / "worker0", "vla")
    for variable in (
        "WARP_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
    ):
        assert len({first[variable], second[variable], policy[variable]}) == 3
    assert (tmp_path / "worker0/env/xdg/torch/kernels").is_dir()
    assert first["PYTHONFAULTHANDLER"] == "1"
    assert dict(os.environ) == before


def test_unknown_cache_component_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        cache_environment(tmp_path, "other")
