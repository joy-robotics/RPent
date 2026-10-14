"""Prompt-defined perception tasks with explicit, validated output contracts."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .definitions import TASKS
from .definitions.common import MAX_TARGET_LENGTH, MAX_VIDEO_FRAMES
from .definitions.common import VIEWS as VIEWS
from .definitions.grounding_2d import SPEC as TOOL_SPEC

GROUNDING_PROMPT = TASKS["grounding_2d"]["prompt"]
parse_grounding_response = TASKS["grounding_2d"]["parse"]

PROTOCOL = "rpent-embodied-vlm/2"
DEFAULT_CHECKPOINT = (
    "/mnt/workspace1/share/ckpt/"
    "qwen35_9b_midv5_sftv7_joybrain_rl_082120_sftv7_1M_v3_nothinking_n8_globalbatch1280_minibatch640_40gpu"
)
DEFAULT_TOOL = TOOL_SPEC["name"]
MAX_IMAGE_PIXELS = 4_000_000
MAX_TOTAL_PIXELS = 16_000_000
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def validate_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image [height,width,3]")
    height, width = image.shape[:2]
    if min(height, width) < 1 or height * width > MAX_IMAGE_PIXELS:
        raise ValueError("Image is empty or exceeds the 4-megapixel limit")
    return np.ascontiguousarray(image)


def validate_target(target):
    if not isinstance(target, str) or not 1 <= len(target.strip()) <= MAX_TARGET_LENGTH:
        raise ValueError(f"target must contain 1..{MAX_TARGET_LENGTH} characters")
    return target.strip()


def validate_task_target(task, target):
    return TASKS[task].get("validate_target", validate_target)(target)


def validate_camera(camera, width, height):
    if not isinstance(camera, dict) or set(camera) != {
        "intrinsic_K",
        "width",
        "height",
    }:
        raise ValueError("Camera requires intrinsic_K, width and height only")
    if any(type(camera[name]) is not int for name in ("width", "height")):
        raise ValueError("Camera dimensions must be integers")
    if (camera["width"], camera["height"]) != (width, height):
        raise ValueError("Camera calibration dimensions do not match the source image")
    matrix = camera["intrinsic_K"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in row
            )
            for row in matrix
        )
    ):
        raise ValueError("intrinsic_K must be a finite numeric 3x3 matrix")
    if (
        matrix[0][0] <= 0
        or matrix[1][1] <= 0
        or matrix[0][1] != 0
        or matrix[1][0] != 0
        or matrix[2] != [0, 0, 1]
    ):
        raise ValueError(
            "Only positive-focal-length, zero-skew pinhole intrinsics are supported"
        )
    return {"intrinsic_K": matrix, "width": width, "height": height}


def validate_inputs(task, images, target, camera=None):
    if task not in TASKS:
        raise ValueError(f"Unknown embodied task: {task}")
    if not isinstance(images, list):
        raise ValueError("images must be a list of ordered RGB frames")
    video = TASKS[task].get("video", False)
    if not (2 <= len(images) <= MAX_VIDEO_FRAMES if video else len(images) == 1):
        raise ValueError(
            "Video requires 2..64 frames; single-image tools require exactly one"
        )
    images = [validate_image(image) for image in images]
    if sum(image.shape[0] * image.shape[1] for image in images) > MAX_TOTAL_PIXELS:
        raise ValueError("Combined images exceed the 16-megapixel input limit")
    if any(image.shape != images[-1].shape for image in images):
        raise ValueError("Video frames must share image dimensions")
    if TASKS[task].get("requires_camera") or (
        TASKS[task].get("optional_camera") and camera is not None
    ):
        camera = validate_camera(camera, images[-1].shape[1], images[-1].shape[0])
    elif camera is not None:
        raise ValueError("This 2D tool does not accept camera parameters")
    return images, validate_task_target(task, target), camera


def build_task_prompt(task, target, *, camera=None, frame_count=1):
    target = validate_task_target(task, target)
    if TASKS[task].get("build_prompt"):
        if camera is not None:
            if not isinstance(camera, dict):
                raise ValueError("Camera calibration must be a dictionary")
            camera = validate_camera(camera, camera.get("width"), camera.get("height"))
        return TASKS[task]["build_prompt"](target, camera=camera)
    prompt = TASKS[task]["prompt"]
    if TASKS[task].get("requires_camera"):
        if not isinstance(camera, dict):
            raise ValueError("Saved camera calibration is required")
        camera = validate_camera(camera, camera["width"], camera["height"])
        matrix = camera["intrinsic_K"]
        replacements = {
            "fx": matrix[0][0],
            "fy": matrix[1][1],
            "cx": matrix[0][2],
            "cy": matrix[1][2],
            "width": camera["width"],
            "height": camera["height"],
            "frame_count": frame_count,
        }
        for name, value in replacements.items():
            prompt = prompt.replace("{" + name + "}", str(value))
    return prompt.replace("{target}", validate_target(target))


def prompt_fingerprint(task):
    prompt = TASKS[task]["prompt"]
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(prompt.encode()).hexdigest()


def parse_task_response(task, text, width, height, *, target):
    parsed = TASKS[task]["parse"](
        text, width, height, **TASKS[task].get("parse_options", {})
    )
    if TASKS[task].get("bind_target"):
        parsed = TASKS[task]["bind_target"](parsed, validate_task_target(task, target))
    return parsed
