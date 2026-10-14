"""Camera-frame object-centric trajectory, following Trajectory_3D/ca1m_trace_3d."""

from .common import CAMERA_PREFIX, coordinates, entries, image_metadata, make_spec

PROMPT = (
    CAMERA_PREFIX + 'Point the 3D object-centric visual trace for the task "{target}"\n'
    "Predict the trajectory of the target object being manipulated. Your answer should be a list "
    'of objects, each with a "trace_3d" and a "label", e.g. [{"trace_3d": '
    '[(x1,y1,z1),(x2,y2,z2),...], "label": "..."}], where x,y,z are real-world physical coordinates in meters.'
)

SPEC = make_spec(
    "trace_3d",
    purpose="Predict a proposed object-centric 3D trajectory from one calibrated saved RGB image.",
    target_description="Manipulation instruction naming the object and intended motion, e.g. "
    "Lift the mug off the table. This tool predicts the manipulated object's trajectory.",
    returns="found and trajectories with label, trace_3d and points_xyz_m, each containing "
    "2..128 ordered 3D points. trajectory_of=manipulated_object; timestamps_available=false.",
    coordinates="Points are [x,y,z] in meters in the selected source camera_cv frame: x right, "
    "y down, z forward. world_transform_applied=false; no robot-world conversion is performed.",
    usage="Use to reason about a possible object path. Verify the source image, geometry and "
    "measured world anchors before planning robot actions. Any camera-to-world conversion "
    "requires the real extrinsics of the returned step_idx/view, not an assumed identity.",
    limitations="Unlike trace_2d, this is not a gripper trajectory. No object orientation, "
    "grasp attachment, timestamps, robot kinematics or reachability is supplied. Never send "
    "these camera-frame predictions directly to world-frame move_to or execute them as a path.",
    camera="required",
)


def parse(text, width, height):
    metadata = image_metadata(width, height)
    trajectories = []
    for entry in entries(text, "trace_3d"):
        values = entry["trace_3d"]
        if not isinstance(values, list) or not 2 <= len(values) <= 128:
            raise ValueError("trace_3d must contain 2..128 ordered key points")
        points = [coordinates(point, 3) for point in values]
        trajectories.append(
            {
                "label": entry["label"].strip(),
                "trace_3d": points,
                "points_xyz_m": points,
            }
        )
    return {
        **metadata,
        "found": bool(trajectories),
        "trajectories": trajectories,
        "coordinate_space": "camera_cv",
        "units": "meters",
        "axes": "x right, y down, z forward",
        "trajectory_of": "manipulated_object",
        "model_output_format": "trace_3d",
        "world_transform_applied": False,
        "timestamps_available": False,
    }


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
    "requires_camera": True,
}
