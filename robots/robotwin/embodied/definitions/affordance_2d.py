"""Affordance region boxes, following Affordance_2D_rewrite/affordance_distill_v4_0808_box."""

import math

from .common import coordinates, entries, image_metadata, make_spec

PROMPT = (
    "\n{target} Your answer should be a list of affordance regions, each with an "
    '"afford_2d" and a "label", e.g. [{"afford_2d": [(x1,y1,x2,y2)], "label": "..."}], '
    "where coordinates are between 0 and 1000."
)

SPEC = make_spec(
    "affordance_2d",
    purpose="Locate interaction regions for a requested manipulation in one saved RGB observation.",
    target_description="Complete manipulation question naming the object and intended action, "
    "e.g. Where should the mug be held to lift it? Not just an object category.",
    returns="found and regions with label, afford_2d, bbox_xyxy_1000, bbox_xyxy_px and center_row_col. "
    "An empty regions list means found=false.",
    coordinates="Interaction boxes use [x_min,y_min,x_max,y_max] in normalized 0..1000 and source "
    "image pixels. center_row_col is the [row,column] address of the region center.",
    usage="Compare the region to the visible handle/contact surface; use center_row_col with "
    "sample_world_xyz on the returned step_idx/view for a measured world-space candidate.",
    limitations="This is an interaction-region estimate, not whole-object detection, a 6D grasp, "
    "gripper orientation, contact force or proof of graspability. The region center need not be "
    "a safe contact. Malformed output, including an empty label, is rejected rather than repaired.",
    camera="none",
)


def parse(text, width, height):
    metadata = image_metadata(width, height)
    regions = []
    for entry in entries(text, "afford_2d"):
        nested = entry["afford_2d"]
        if not isinstance(nested, list) or len(nested) != 1:
            raise ValueError("afford_2d must contain one coordinate tuple/list")
        box = coordinates(nested[0], 4, normalized=True)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Affordance bbox must have positive width and height")
        pixels = [
            box[0] * width / 1000,
            box[1] * height / 1000,
            box[2] * width / 1000,
            box[3] * height / 1000,
        ]
        regions.append(
            {
                "label": entry["label"].strip(),
                "afford_2d": [box],
                "bbox_xyxy_1000": box,
                "bbox_xyxy_px": [round(number, 4) for number in pixels],
                "center_row_col": [
                    math.floor((pixels[1] + pixels[3]) / 2),
                    math.floor((pixels[0] + pixels[2]) / 2),
                ],
            }
        )
    return {
        **metadata,
        "found": bool(regions),
        "regions": regions,
        "coordinate_space": "image_pixels",
        "model_output_format": "afford_2d_boxes",
    }


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
}
