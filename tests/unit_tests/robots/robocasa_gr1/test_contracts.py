import json

import numpy as np
import pytest

from robots.robocasa_gr1.contracts import (
    EFFECTIVE_INDICES,
    GROUPS,
    IMAGE_KEYS,
    action_array,
    action_dict,
    observation,
    task_id,
)
from robots.robocasa_gr1.vla_server import validate_checkpoint
from tests.unit_tests.robots.robocasa_gr1.conftest import raw_observation


def test_native_mapping_roundtrips_without_panda_padding():
    raw = raw_observation()
    observed = observation(raw)
    effective = observed["state44"][list(EFFECTIVE_INDICES)]
    result = action_dict(effective)
    for name, _ in GROUPS:
        np.testing.assert_array_equal(result[f"action.{name}"], raw[f"state.{name}"])
    np.testing.assert_array_equal(observed["state44"][13:22], 0)
    np.testing.assert_array_equal(observed["state44"][35:41], 0)
    assert observed["task_language"] == "put the cup away"
    assert observed["image_key"] == IMAGE_KEYS[0]


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(12),
        np.zeros(44),
        np.zeros((0, 29)),
        np.zeros((1, 1, 29)),
        np.full(29, np.nan),
    ],
)
def test_bad_action_layouts_fail(value):
    with pytest.raises(ValueError):
        action_array(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state.left_arm", np.zeros(6)),
        (IMAGE_KEYS[0], np.zeros((4, 4, 3))),
        ("annotation.human.coarse_action", ""),
    ],
)
def test_invalid_native_observation_fails(field, value):
    raw = raw_observation()
    raw[field] = value
    with pytest.raises(ValueError):
        observation(raw)


def test_task_names_are_not_arbitrary_imports():
    canonical = task_id("PnPCupToDrawerClose")
    assert task_id(canonical) == canonical
    with pytest.raises(ValueError):
        task_id("../../other/task")


def test_checkpoint_requires_gr1_stats(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"test fixture, not real weights")
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "joybrain_vla", "chunk_size": 16})
    )
    stats = {
        name: {"mean": [0] * 29, "std": [1] * 29}
        for name in ("observation.state", "action")
    }
    (tmp_path / "stats.json").write_text(json.dumps({"robocasa_gr1": stats}))
    assert validate_checkpoint(tmp_path) == tmp_path
    stats["action"]["std"][0] = float("nan")
    (tmp_path / "stats.json").write_text(json.dumps({"robocasa_gr1": stats}))
    with pytest.raises(ValueError):
        validate_checkpoint(tmp_path)
