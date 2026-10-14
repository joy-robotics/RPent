"""Spatial point localization, following Area_Spatial_Pointing_2D/refspatial."""

from .common import coordinates, entries, image_metadata, make_spec, pixel_point

PROMPT = (
    '\n{target} Your answer should be a list of points, each with a "point_2d" '
    'and a "label", e.g. [{"point_2d": [(x,y)], "label": "..."}], '
    "where coordinates are between 0 and 1000."
)

SPEC = make_spec(
    "pointing_2d",
    purpose="Locate a requested object part or spatial point in one saved RGB observation.",
    target_description="Complete spatial localization question, e.g. Where is the center of the "
    "mug handle? Specify the desired part or relation, not a robot motion command.",
    returns="found and points with label, point_2d, point_xy_1000, point_xy_px and point_row_col. "
    "An empty points list means found=false.",
    coordinates="Model points are [x,y] in 0..1000; point_xy_px uses source image pixels. "
    "point_row_col is [row,column], using floor and mapping a 1000 boundary to the last pixel.",
    usage="Pass point_row_col to sample_world_xyz with the returned step_idx/view. Check that the "
    "selected visible surface and valid measured world point correspond to the requested target.",
    limitations="A point has no box extent, orientation, grasp configuration or depth. A spatial "
    "relation can indicate free space whose image pixel instead samples a background surface; "
    "do not treat that sample as a verified contact or directly use pixels as world coordinates.",
    camera="none",
)


def parse(text, width, height):
    metadata = image_metadata(width, height)
    points = []
    for entry in entries(text, "point_2d"):
        nested = entry["point_2d"]
        if not isinstance(nested, list) or len(nested) != 1:
            raise ValueError("point_2d must contain exactly one 2D point")
        point = coordinates(nested[0], 2, normalized=True)
        pixels, address = pixel_point(point, width, height)
        points.append(
            {
                "label": entry["label"].strip(),
                "point_2d": [point],
                "point_xy_1000": point,
                "point_xy_px": pixels,
                "point_row_col": address,
            }
        )
    return {
        **metadata,
        "found": bool(points),
        "points": points,
        "coordinate_space": "image_pixels",
        "model_output_format": "point_2d",
        "pixel_address_rule": "floor continuous edge coordinates; the 1000 edge maps to the last valid pixel",
    }


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
}
