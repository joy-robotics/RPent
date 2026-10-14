"""Multi-frame all-instance grounding, following Video_Object_3D/arkit_multi."""

from .common import make_spec, parse_boxes_3d

PROMPT = (
    "\nYou are given a video of {frame_count} frames from an indoor scene scan. "
    "Camera intrinsic parameters (shared across all frames): Focal length f_x={fx}, f_y={fy}. "
    "Principal point c_x={cx}, c_y={cy}, image width {width} and height {height}. The intrinsic "
    "matrix K = [[{fx}, 0, {cx}], [0, {fy}, {cy}], [0, 0, 1]]. All 3D coordinates are "
    "expressed in the LAST frame's camera coordinate system: X-axis points rightward, Y-axis "
    "points downward, and Z-axis points forward. The origin is the camera location of the last "
    "frame, i.e. the extrinsic matrix is [[1,0,0,0],[0,1,0,0],[0,0,1,0]]. Detect ALL instances "
    'of "{target}" visible in the video and output each one\'s position and dimensions as a 3D '
    "bounding box. The result must be a JSON list, one entry per instance: "
    '[{"bbox_3d":[(x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw)], '
    '"label":"{target}"}, ...]. Note: (1) x_center, y_center, z_center: the object center in the '
    "last frame's camera coordinate, in meters. (2) x_size, y_size, z_size: dimensions along XYZ "
    "axes in meters at zero rotation. (3) roll, pitch, yaw: Euler angles representing rotations "
    "around the X, Y, and Z axes respectively, normalized to (-1, 1) by dividing by pi. "
    "If there is exactly one instance, still return a one-element list."
)

SPEC = make_spec(
    "video_grounding_3d",
    purpose="Estimate 3D boxes for all instances of a category visible across ordered saved frames.",
    target_description="Object category to locate across the sequence, e.g. chair. Supply steps "
    "with 2..64 strictly increasing saved indices and one shared view; dimensions and intrinsics "
    "must match across the selected frames.",
    returns="found and detections with label, bbox_3d, center_xyz_m, size_xyz_m, rpy_over_pi and "
    "rpy_rad. One combined list of predicted instances, not one result per frame. Also includes "
    "frame_count and reference_step_idx for the last selected observation.",
    coordinates="All centers and zero-rotation sizes use meters in last_frame_camera_cv, with "
    "x right, y down, z forward. Rotations are Euler/pi plus radians, with composition order "
    "unspecified. world_transform_applied=false; earlier-frame camera coordinates are not used.",
    usage="Use several observations over time to resolve object presence and spatial extent. Verify "
    "predictions against the recorded images. Any world conversion must use the last selected "
    "step's real camera extrinsics, not the first frame's pose or an invented identity matrix.",
    limitations="Not tracking IDs, per-frame boxes, velocities or trajectories. An instance visible "
    "earlier can lie behind the final camera, so negative final-camera z is not automatically "
    "invalid. Estimates are not grasp poses or executable world targets; no new video is captured.",
    camera="required",
    video=True,
)


def parse(text, width, height):
    return parse_boxes_3d(text, width, height, video=True)


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
    "requires_camera": True,
    "video": True,
}
