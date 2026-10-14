"""Object boxes, following Object_2D/finecop_posti_83k."""

import ast
import math
import re

from .common import make_spec

PROMPT = (
    "\nLocate {target}. Your answer should be a list of bounding boxes, each with a "
    '"bbox_2d" and a "label", e.g. [{"bbox_2d": [(x1,y1,x2,y2)], "label": "..."}], '
    "where coordinates are between 0 and 1000."
)

SPEC = make_spec(
    "grounding_2d",
    purpose="Localize named objects in one saved RGB observation using the shared fine-tuned VLM.",
    target_description="Object name or referring expression, e.g. the red block left of the bowl.",
    returns="found and detections with label, bbox_xyxy_1000, bbox_xyxy_px and center_row_col. "
    "An empty detections list means found=false.",
    coordinates="Boxes use [x_min,y_min,x_max,y_max]; normalized edges are in 0..1000. "
    "Pixel edges scale to the source width/height. center_row_col is [row,column], not [x,y].",
    usage="Use the chosen center_row_col with sample_world_xyz on exactly the returned step_idx/view "
    "to look up a measured world point before constructing an atomic motion target.",
    limitations="A box is not a world pose, grasp pose or verified object identity. Its center may "
    "sample background or an occluder; check the source image and validity of world-map samples.",
    camera="none",
)


def parse(text, width, height):
    if not isinstance(text, str) or len(text) > 20_000:
        raise ValueError("VLM response must be a bounded grounding literal")
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Image dimensions must be positive integers")
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
        if len(nodes) > 512 or any(not isinstance(node, allowed) for node in nodes):
            raise ValueError("Only bounded literal grounding data is allowed")
        for node in nodes:
            if isinstance(node, ast.Dict):
                if any(
                    not isinstance(key, ast.Constant) or not isinstance(key.value, str)
                    for key in node.keys
                ):
                    raise ValueError("Grounding dictionary keys must be strings")
                keys = [key.value for key in node.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate grounding dictionary keys")
        detections = ast.literal_eval(expression)
    except (SyntaxError, TypeError, RecursionError) as error:
        raise ValueError("Invalid grounding literal syntax") from error
    if not isinstance(detections, list) or len(detections) > 10:
        raise ValueError("Expected at most 10 detections")
    normalized = []
    for detection in detections:
        if not isinstance(detection, dict) or set(detection) != {"label", "bbox_2d"}:
            raise ValueError("Each detection must contain only label and bbox_2d")
        nested_box = detection["bbox_2d"]
        if not isinstance(nested_box, list) or len(nested_box) != 1:
            raise ValueError("bbox_2d must contain one coordinate tuple or list")
        label, box = detection["label"], nested_box[0]
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 200:
            raise ValueError("Invalid detection label")
        if (
            not isinstance(box, (tuple, list))
            or len(box) != 4
            or any(
                type(value) not in (int, float)
                or not 0 <= value <= 1000
                or not math.isfinite(value)
                for value in box
            )
        ):
            raise ValueError("bbox must have four finite coordinates in 0..1000")
        left, top, right, bottom = box
        if left >= right or top >= bottom:
            raise ValueError(
                "bbox must have positive width and height; coordinates are not repaired"
            )
        pixels = [
            left * width / 1000,
            top * height / 1000,
            right * width / 1000,
            bottom * height / 1000,
        ]
        normalized.append(
            {
                "label": label.strip(),
                "bbox_xyxy_1000": list(box),
                "bbox_xyxy_px": [round(value, 4) for value in pixels],
                "center_row_col": [
                    math.floor((pixels[1] + pixels[3]) / 2),
                    math.floor((pixels[0] + pixels[2]) / 2),
                ],
            }
        )
    return {
        "found": bool(normalized),
        "detections": normalized,
        "image_size": {"width": width, "height": height},
        "bbox_format": "xyxy",
        "coordinate_space": "image_pixels",
        "model_output_format": "finecop_posti_bbox_2d",
        "bbox_bounds": "continuous edges: x in [0,width], y in [0,height]",
    }


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
}
