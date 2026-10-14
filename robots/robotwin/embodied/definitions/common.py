"""Bounded data-only parsing and shared contracts for prompt-defined perception."""

from __future__ import annotations

import ast
import math
import re
from copy import deepcopy

VIEWS = ("head", "left_wrist", "right_wrist")
MAX_TARGET_LENGTH = 512
MAX_VIDEO_FRAMES = 64

CAMERA_PREFIX = (
    "\nHere are the detailed camera parameters for the image. Camera intrinsic parameters: "
    "Focal length f_x={fx}, f_y={fy}. Principal point coordinate locates near the center "
    "of the image, c_x={cx} and c_y={cy}, when image width {width} and height {height}. "
    "We do not consider distortion parameters here. Therefore, the intrinsic matrix K = "
    "[[{fx}, 0, {cx}], [0, {fy}, {cy}], [0, 0, 1]]. Camera coordinate: X-axis points "
    "rightward, Y-axis points downward, and Z-axis points forward. The origin point is "
    "the camera location. We take the camera coordinate system as the world coordinate "
    "system, namely the camera extrinsic matrix is [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]. "
)


def make_spec(
    name,
    *,
    purpose,
    target_description,
    returns,
    coordinates,
    usage,
    limitations,
    camera="none",
    video=False,
    target_schema=None,
):
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (
            purpose,
            target_description,
            returns,
            coordinates,
            usage,
            limitations,
        )
    ):
        raise ValueError("Tool description sections must be nonempty strings")
    camera_descriptions = {
        "none": "Uses saved RGB only; camera parameters are neither required nor accepted.",
        "required": (
            "Requires real saved intrinsic_K and image dimensions for the selected view/step. "
            "Missing or inconsistent calibration is an error; never invent camera parameters."
        ),
        "optional": (
            "Uses real saved intrinsics when available and validates them. Without calibration, "
            "uses the existing no-intrinsics training template; never invent camera parameters."
        ),
    }
    if camera not in camera_descriptions:
        raise ValueError("camera must be none, required or optional")
    target = (
        deepcopy(target_schema)
        if target_schema is not None
        else {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_TARGET_LENGTH,
        }
    )
    target["description"] = target_description
    properties = {
        "target": target,
        "view": {
            "type": "string",
            "enum": list(VIEWS),
            "default": "head",
            "description": "Saved RGB camera. Keep this view for any calibration or world-map lookup.",
        },
    }
    required = ["target"]
    if video:
        properties["steps"] = {
            "type": "array",
            "minItems": 2,
            "maxItems": MAX_VIDEO_FRAMES,
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 0},
            "description": "2..64 strictly increasing saved observation indices, not simulator action counts. "
            "Use one view with identical image dimensions and intrinsics. The last selected step "
            "defines the output camera frame. Does not capture new frames.",
        }
        required.append("steps")
    else:
        properties["step"] = {
            "type": "integer",
            "minimum": -1,
            "default": -1,
            "description": "Saved observation index, not simulator action count. -1 selects the latest saved "
            "observation; it does not capture a new image. Prefer a fresh frame after robot motion.",
        }
    description = "\n".join(
        (
            f"Purpose: {purpose}",
            f"Input: {target_description}",
            (
                f"Returns: {returns} Successful calls also include image_size, step_idx, view, step_indices, "
                "source_image_sha256 and action_executed=false."
            ),
            f"Coordinates: {coordinates}",
            f"Camera: {camera_descriptions[camera]}",
            f"Usage: {usage}",
            (
                f"Limitations: {limitations} Read-only VLM estimates may be inaccurate. Does not execute "
                "actions, advance state or capture new observations. Not sensor ground truth, a collision "
                "check or official task success. On an error or uncertain result, inspect the real image "
                "and do not treat failure as a valid prediction."
            ),
        )
    )
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def literal_data(text, *, node_limit=16384):
    if not isinstance(text, str) or len(text) > 64000:
        raise ValueError("Expected bounded literal perception data")
    fenced = re.fullmatch(
        r"```(?:json|python)?\s*(.*?)\s*```", text.strip(), flags=re.DOTALL
    )
    try:
        expression = ast.parse(
            (fenced.group(1) if fenced else text).strip(), mode="eval"
        )
        nodes = list(ast.walk(expression))
        allowed = (
            ast.Expression,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Constant,
            ast.UnaryOp,
            ast.USub,
            ast.UAdd,
            ast.Load,
        )
        if len(nodes) > node_limit or any(
            not isinstance(node, allowed) for node in nodes
        ):
            raise ValueError("Only bounded literal data is allowed")
        for node in nodes:
            if isinstance(node, ast.Dict):
                if any(
                    not isinstance(key, ast.Constant) or not isinstance(key.value, str)
                    for key in node.keys
                ):
                    raise ValueError("Dictionary keys must be strings")
                keys = [key.value for key in node.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate dictionary keys")
        return ast.literal_eval(expression)
    except (SyntaxError, TypeError, RecursionError) as error:
        raise ValueError("Invalid perception literal syntax") from error


def entries(text, field):
    result = literal_data(text)
    if not isinstance(result, list) or len(result) > 32:
        raise ValueError("Expected a list of at most 32 predictions")
    for entry in result:
        if not isinstance(entry, dict) or set(entry) != {field, "label"}:
            raise ValueError(f"Each prediction must contain only {field} and label")
        if (
            not isinstance(entry["label"], str)
            or not 1 <= len(entry["label"].strip()) <= 512
        ):
            raise ValueError("Invalid prediction label")
    return result


def coordinates(value, dimension, *, normalized=False):
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"Expected {dimension} coordinates")
    if any(
        type(number) not in (int, float) or not math.isfinite(number)
        for number in value
    ):
        raise ValueError("Coordinates must be finite numbers, not booleans")
    if normalized and any(not 0 <= number <= 1000 for number in value):
        raise ValueError("2D coordinates must be within 0..1000")
    return list(value)


def image_metadata(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Image dimensions must be positive integers")
    return {"image_size": {"width": width, "height": height}}


def pixel_point(point, width, height):
    pixels = [point[0] * width / 1000, point[1] * height / 1000]
    address = [
        min(height - 1, math.floor(pixels[1])),
        min(width - 1, math.floor(pixels[0])),
    ]
    return [round(number, 4) for number in pixels], address


def parse_boxes_3d(text, width, height, *, video=False):
    metadata = image_metadata(width, height)
    detections = []
    for entry in entries(text, "bbox_3d"):
        nested = entry["bbox_3d"]
        if isinstance(nested, list) and len(nested) == 1:
            box = coordinates(nested[0], 9)
            layout = "nested_tuple9"
        elif isinstance(nested, (list, tuple)) and len(nested) == 9:
            box = coordinates(nested, 9)
            layout = "flat_tuple9"
        else:
            raise ValueError(
                "bbox_3d must be a 9-coordinate vector or a list containing one such vector"
            )
        if any(size <= 0 for size in box[3:6]):
            raise ValueError("3D box dimensions must be positive")
        if any(not -1 <= angle <= 1 for angle in box[6:]):
            raise ValueError("Euler/pi angles must be within -1..1")
        detections.append(
            {
                "label": entry["label"].strip(),
                "bbox_3d": [box],
                "source_bbox_layout": layout,
                "center_xyz_m": box[:3],
                "size_xyz_m": box[3:6],
                "rpy_over_pi": box[6:],
                "rpy_rad": [angle * math.pi for angle in box[6:]],
            }
        )
    return {
        **metadata,
        "found": bool(detections),
        "detections": detections,
        "coordinate_space": "last_frame_camera_cv" if video else "camera_cv",
        "units": "meters",
        "axes": "x right, y down, z forward",
        "rotation_encoding": "roll,pitch,yaw divided by pi",
        "rotation_composition_order": "not specified by training template",
        "model_output_format": "bbox_3d_tuple9",
        "world_transform_applied": False,
    }
