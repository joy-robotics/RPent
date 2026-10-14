"""Canonical GR1 raw-state and effective-joint contracts, independent of PandaOmron."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

EMBODIMENT = "robocasa_gr1"
ROBOT_TYPE = "GR1ArmsAndWaistFourierHands"
GROUPS = (
    ("left_arm", 7),
    ("left_hand", 6),
    ("right_arm", 7),
    ("right_hand", 6),
    ("waist", 3),
)
RAW_SLICES = ((0, 7), (7, 13), (22, 29), (29, 35), (41, 44))
EFFECTIVE_INDICES = tuple(range(13)) + tuple(range(22, 35)) + tuple(range(41, 44))
EVAL_TYPES = (
    "id",
    "ood_object_appearance",
    "ood_container_combination",
    "ood_object_type",
)
IMAGE_KEYS = (
    "video.ego_view_bg_crop_pad_res256_freq20",
    "video.ego_view_pad_res256_freq20",
)


def task_id(name: str) -> str:
    """Accept a GR1 task basename or its exact Gym registration name."""
    suffix = f"_{ROBOT_TYPE}_Env"
    if name.startswith("gr1_unified/") and name.endswith(suffix):
        name = name[len("gr1_unified/") : -len(suffix)]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Use a GR1 task basename or gr1_unified Gym ID")
    return f"gr1_unified/{name}{suffix}"


def action_array(value: Any) -> np.ndarray:
    """Require complete finite 29D native joint targets; never crop unknown layouts."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if (
        array.ndim != 2
        or array.shape[1] != 29
        or len(array) < 1
        or not np.isfinite(array).all()
    ):
        raise ValueError("GR1 actions must be finite [N,29] joint targets")
    return array


def action_dict(value: Any) -> dict[str, np.ndarray]:
    """Invert the trained effective order into the native GR1 action dictionary."""
    array = action_array(value)
    if len(array) != 1:
        raise ValueError("A native GR1 step requires one action")
    result = {}
    offset = 0
    for name, dimension in GROUPS:
        result[f"action.{name}"] = array[0, offset : offset + dimension].copy()
        offset += dimension
    return result


def observation(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve the training-defined 44D reserved slots and native ego image choice."""
    state = np.zeros(44, dtype=np.float32)
    groups = {}
    for (name, dimension), (start, end) in zip(GROUPS, RAW_SLICES):
        values = np.asarray(raw[f"state.{name}"], dtype=np.float32)
        if values.shape != (dimension,) or not np.isfinite(values).all():
            raise ValueError(f"GR1 state.{name} must contain {dimension} finite values")
        state[start:end] = values
        groups[name] = values.copy()
    image_key = next((key for key in IMAGE_KEYS if key in raw), None)
    if image_key is None:
        raise ValueError("GR1 ego-view observation is missing")
    image = np.asarray(raw[image_key])
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("GR1 ego view must be nonempty uint8 RGB")
    language = next(
        (
            raw[key]
            for key in (
                "annotation.human.coarse_action",
                "annotation.human.action.task_description",
            )
            if key in raw
        ),
        None,
    )
    if not isinstance(language, str) or not language.strip():
        raise ValueError("GR1 native task language is missing")
    language = language.split(": ", 1)[-1]
    return {
        "state44": state,
        "joint_positions": groups,
        "ego_rgb": image.copy(),
        "image_key": image_key,
        "task_language": language,
    }


def env_contract(task: str, evaluation: str, seed: int, limit: int) -> dict[str, Any]:
    """Identify the exact simulator, observation/action layout and episode budget."""
    return {
        "runtime": "robocasa_gr1_native_v1",
        "task_id": task_id(task),
        "eval_type": evaluation,
        "robot_type": ROBOT_TYPE,
        "seed": int(seed),
        "raw_state_dim": 44,
        "action_dim": 29,
        "action_groups": [name for name, _ in GROUPS],
        "camera": "ego",
        "max_episode_steps": int(limit),
        "success_source": "latched native info.success or native terminated",
        "warmup_resets": 2,
    }
