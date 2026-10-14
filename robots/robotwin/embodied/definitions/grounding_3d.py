"""Oriented 3D boxes in camera coordinates, following Object_3D/joyai_openspatial."""

from .common import CAMERA_PREFIX, make_spec, parse_boxes_3d

PROMPT = (
    CAMERA_PREFIX
    + "Locate the {target} in the provided image and output their positions and "
    "dimensions using 3D bounding boxes. The results must be in the JSON format: "
    '[{"bbox_3d":[(x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw)],'
    '"label":"category"}]. Note: (1) x_center, y_center, z_center: the center of the object '
    "in the camera coordinate, in meters. (2) x_size, y_size, z_size: The dimensions of the "
    "object along the XYZ axes, in meters, when the rotation angles are zero. (3) roll, pitch, yaw: "
    "Euler angles representing rotations around the X, Y, and Z axes respectively, "
    "normalized to (-1, 1) by dividing by pi."
)

SPEC = make_spec(
    "grounding_3d",
    purpose="Estimate oriented 3D boxes for named objects in one calibrated saved RGB image.",
    target_description="Object category or referring expression, e.g. the mug beside the plate. "
    "Requests object position, dimensions and orientation, not a grasp or motion.",
    returns="found and detections with label, bbox_3d, center_xyz_m, size_xyz_m, rpy_over_pi and "
    "rpy_rad. bbox_3d is a list containing one 9-value center/size/rotation vector per detection.",
    coordinates="Center and zero-rotation XYZ dimensions are in meters in the source camera_cv "
    "frame (x right, y down, z forward). Rotation values encode roll/pitch/yaw divided by pi, "
    "also exposed in radians; the training template does not specify composition order. "
    "world_transform_applied=false.",
    usage="Use the box as an object-extent/orientation hypothesis and verify it against the source "
    "image and measured world-map targets. A world-frame conversion requires real extrinsics "
    "from the returned step_idx/view; never treat the template's camera origin as robot-world.",
    limitations="Monocular estimates are not measured object poses, 6D grasps or clearance "
    "guarantees. Never pass center_xyz_m directly to world-frame move_to. The optional overlay "
    "projects the center only; it is not a rendered oriented 3D box.",
    camera="required",
)


def parse(text, width, height):
    return parse_boxes_3d(text, width, height)


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
    "requires_camera": True,
}
