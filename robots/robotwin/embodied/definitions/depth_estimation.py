"""Scalar camera-to-target estimates from the three 4metricdepth_*_v1 datasets."""

import math
import re
from decimal import Decimal, InvalidOperation

from .common import (
    MAX_TARGET_LENGTH,
    coordinates,
    image_metadata,
    make_spec,
    pixel_point,
)

MARKERS = {
    **{label: f"the point labeled {label}" for label in ("A", "B", "C", "D")},
    "red_dot": "the point highlighted by the red dot",
    "red_arrow": "the point marked by the red arrow",
}
PROMPT = {
    "object": "\nHow far is {name} from the camera, in meters?",
    "marker": "\nAbout how many meters away is {marker_name} from the camera?",
    "point": "\nEstimate the distance in meters from the camera to the point at ({point_x}, {point_y}), where coordinates are normalized to 0-1000.",
    "camera": "\nCamera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy} (image size {width}x{height}).",
}
NAME_SCHEMA = {"type": "string", "minLength": 1, "maxLength": MAX_TARGET_LENGTH}
TARGET_SCHEMA = {
    "oneOf": [
        {
            **NAME_SCHEMA,
            "description": "Object referring expression, e.g. the red cup; shorthand for kind=object.",
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "object", "type": "string"},
                "name": NAME_SCHEMA,
            },
            "required": ["kind", "name"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "marker", "type": "string"},
                "marker": {"type": "string", "enum": list(MARKERS)},
            },
            "required": ["kind", "marker"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "point", "type": "string"},
                "point_xy_1000": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number", "minimum": 0, "maximum": 1000},
                },
            },
            "required": ["kind", "point_xy_1000"],
            "additionalProperties": False,
        },
    ]
}

SPEC = make_spec(
    "depth_estimation",
    purpose="Estimate one object's, existing visible marker's or image point's distance from the camera.",
    target_description='Object referring string (e.g. the red cup), or {"kind":"object","name":"..."}, '
    '{"kind":"marker","marker":"A"}, or {"kind":"point","point_xy_1000":[500,400]}. '
    "Supported markers: A/B/C/D/red_dot/red_arrow; the requested marker must already be visible.",
    returns="found and one depths entry containing depth_m, reported_value, reported_unit, "
    "unit_source and query. Point queries also include query_point_xy_px/query_point_row_col. "
    "calibration_used records whether saved intrinsics were available; dense_depth_map=false.",
    coordinates="depth_m is a scalar camera-to-target estimate in meters. Training wording does "
    "not disambiguate axial Z versus distance along the viewing ray. Input point_xy_1000 uses "
    "[x,y] in 0..1000; returned query_point_row_col uses source-image [row,column].",
    usage="Use for qualitative distance reasoning with uncertainty. Do not draw or invent markers. "
    "For measured world targets use sample_world_xyz on the same returned step_idx/view instead; "
    "do not combine this ambiguous scalar with K as if it were calibrated sensor Z.",
    limitations="Not a dense depth map, measured axial Z, world position, object size or pairwise "
    "distance. A bare numeric reply takes meters from the request, not an observed unit token. "
    "No safe approach distance or successful contact is guaranteed.",
    camera="optional",
    target_schema=TARGET_SCHEMA,
)


def validate_target(target):
    if isinstance(target, str):
        target = {"kind": "object", "name": target}
    if not isinstance(target, dict):
        raise ValueError(
            "Depth target must be an object name or a structured object/marker/point query"
        )
    kind = target.get("kind")
    fields = {"object": "name", "marker": "marker", "point": "point_xy_1000"}
    if (
        not isinstance(kind, str)
        or kind not in fields
        or set(target) != {"kind", fields[kind]}
    ):
        raise ValueError(
            "Depth query requires exactly kind and its name, marker or point_xy_1000 field"
        )
    if kind == "object":
        name = target["name"]
        if (
            not isinstance(name, str)
            or not 1 <= len(name.strip()) <= MAX_TARGET_LENGTH
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError(
                "Object name must be a bounded, single-line referring expression"
            )
        return {"kind": kind, "name": name.strip()}
    if kind == "marker":
        marker = target["marker"]
        if not isinstance(marker, str) or marker not in MARKERS:
            raise ValueError(
                "Marker must already be visible: A/B/C/D, red_dot or red_arrow"
            )
        return {"kind": kind, "marker": marker}
    if not isinstance(target["point_xy_1000"], list):
        raise ValueError("point_xy_1000 must be an x,y array in 0..1000")
    return {
        "kind": kind,
        "point_xy_1000": coordinates(target["point_xy_1000"], 2, normalized=True),
    }


def build_prompt(target, *, camera=None):
    target = validate_target(target)
    values = dict(target)
    if target["kind"] == "marker":
        values["marker_name"] = MARKERS[target["marker"]]
    elif target["kind"] == "point":
        values.update(
            zip(
                ("point_x", "point_y"),
                (
                    str(int(value)) if float(value).is_integer() else str(value)
                    for value in target["point_xy_1000"]
                ),
            )
        )
    prompt = PROMPT[target["kind"]].format(**values)
    if camera:
        matrix = camera["intrinsic_K"]
        prompt = (
            PROMPT["camera"].format(
                fx=matrix[0][0],
                fy=matrix[1][1],
                cx=matrix[0][2],
                cy=matrix[1][2],
                width=camera["width"],
                height=camera["height"],
            )
            + prompt
        )
    return prompt


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
UNIT = r"centimeters?|centimetres?|meters?|metres?|cm|m|厘米|米"
AMOUNT = rf"(?P<value>{NUMBER})\s*(?P<unit>{UNIT})"
REPLIES = (
    rf"(?:The\s+)?(?P<subject>.{{1,512}}?)\s+is\s+(?:about\s+|approximately\s+|around\s+)?{AMOUNT}\s+(?:away\s+)?from\s+(?:the\s+)?camera[.!]?",
    rf"(?P<subject>.{{1,512}}?)(?:距离|距|离)(?:相机|镜头)(?:大约|大概|约为|约|为)?\s*{AMOUNT}[。.]?",
    rf"{AMOUNT}[。.]?",
)


def parse(text, width, height, *, requested_unit=None):
    metadata = image_metadata(width, height)
    if requested_unit not in (None, "meters"):
        raise ValueError("The depth templates explicitly request meters only")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4096:
        raise ValueError("Expected one bounded metric-depth answer, not a depth map")
    match = next(
        (
            matched
            for pattern in REPLIES
            if (matched := re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE))
        ),
        None,
    )
    unit_source = "model_response"
    if match is None and requested_unit == "meters":
        match = re.fullmatch(rf"(?P<value>{NUMBER})", text.strip())
        unit_source = "request_prompt"
    if match is None:
        raise ValueError(
            "Expected one camera-to-target estimate with meters or centimeters; no size, ranking, range or pairwise distance"
        )
    if (
        unit_source == "model_response"
        and len(
            re.findall(
                rf"(?:{NUMBER})\s*(?:{UNIT})(?![A-Za-z])", text, flags=re.IGNORECASE
            )
        )
        != 1
    ):
        raise ValueError("Depth answer must contain exactly one metric amount")
    unit = match.groupdict().get("unit", "meters").lower()
    centimeters = unit.startswith("cent") or unit in ("cm", "厘米")
    value = float(match["value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Depth must be finite and nonnegative")
    try:
        amount = Decimal(match["value"])
        if amount != 0 and value == 0:
            raise ValueError("Depth underflows numeric representation")
        depth = float(amount * (Decimal("0.01") if centimeters else Decimal(1)))
    except InvalidOperation as error:
        raise ValueError("Invalid numeric depth") from error
    if value != 0 and depth == 0:
        raise ValueError("Depth unit conversion underflows numeric representation")
    if not math.isfinite(value) or not math.isfinite(depth) or value < 0:
        raise ValueError(
            "Depth must be finite and nonnegative; coordinates and units are not repaired"
        )
    prediction = {
        "depth_m": depth,
        "reported_value": value,
        "reported_unit": ("centimeters" if centimeters else "meters")
        if unit_source == "model_response"
        else None,
        "unit_source": unit_source,
        "requested_unit": requested_unit,
        "reported_target": match.groupdict().get("subject"),
    }
    return {
        **metadata,
        "found": True,
        "depths": [prediction],
        "units": "meters",
        "coordinate_space": "camera_distance",
        "model_output_format": "metric_depth_text",
        "depth_semantics": "camera-to-target as worded in training; axial-Z versus ray distance is unspecified",
        "dense_depth_map": False,
        "world_transform_applied": False,
    }


def bind_target(parsed, target):
    target = validate_target(target)
    depth = parsed["depths"][0]
    depth["query"] = target
    if target["kind"] == "point":
        reported = re.search(
            rf"\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)", depth["reported_target"] or ""
        )
        if (
            reported
            and [float(value) for value in reported.groups()] != target["point_xy_1000"]
        ):
            raise ValueError(
                "Reported point differs from the requested normalized coordinates"
            )
        size = parsed["image_size"]
        pixels, address = pixel_point(
            target["point_xy_1000"], size["width"], size["height"]
        )
        depth.update(query_point_xy_px=pixels, query_point_row_col=address)
    elif target["kind"] == "marker":
        reported = re.search(
            r"point labeled ([A-D])\b",
            depth["reported_target"] or "",
            flags=re.IGNORECASE,
        )
        if reported and reported[1].upper() != target["marker"]:
            raise ValueError("Reported marker differs from the requested marker")
        depth["requires_existing_visible_marker"] = True
    return parsed


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
    "validate_target": validate_target,
    "parse_options": {"requested_unit": "meters"},
    "build_prompt": build_prompt,
    "bind_target": bind_target,
    "optional_camera": True,
}
