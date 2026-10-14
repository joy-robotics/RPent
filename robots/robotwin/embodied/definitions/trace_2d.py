"""Projected end-effector trajectory, following Trajectory_2D/traj_distill."""

from .common import coordinates, entries, image_metadata, make_spec, pixel_point

PROMPT = (
    "\nYou are currently a robot performing robotic manipulation tasks. The task instruction is: "
    "{target}. Please predict up to 10 key 2D trajectory points to complete the task. "
    "This trajectory belongs to the robot arm's end effector (gripper), tracing how the end effector moves. "
    "It represents the robot arm's motion, not any object's trajectory. Your answer should be a list "
    'of trajectories, each with a "trace_2d" and a "label", e.g. [{"trace_2d": [(x1,y1),(x2,y2),...], '
    '"label": "..."}], where coordinates are between 0 and 1000.'
)

SPEC = make_spec(
    "trace_2d",
    purpose="Predict a proposed robot end-effector trajectory projected into one saved RGB image.",
    target_description="Full manipulation task instruction, e.g. Pick up the red block and place "
    "it in the bowl. The requested trajectory belongs to the gripper, not the moved object.",
    returns="found and trajectories with label, trace_2d, points_xy_1000, points_xy_px and "
    "points_row_col. Each trajectory contains 2..10 ordered key points; timestamps_available=false.",
    coordinates="Key points are [x,y] in 0..1000 and source image pixels; points_row_col supplies "
    "[row,column] addresses. Point order denotes a proposed path, not measured elapsed time.",
    usage="Use as a visual planning hypothesis. sample_world_xyz on the returned step_idx/view "
    "can ground visible current surface anchors, but cannot recover future or free-space path "
    "depth from a single image. Separately plan and verify any 3D robot motion.",
    limitations="Not an object trajectory, executed motion, joint commands, timing, IK solution "
    "or collision-checked path. Do not execute the returned pixels as a robot trajectory.",
    camera="none",
)


def parse(text, width, height):
    metadata = image_metadata(width, height)
    trajectories = []
    for entry in entries(text, "trace_2d"):
        values = entry["trace_2d"]
        if not isinstance(values, list) or not 2 <= len(values) <= 10:
            raise ValueError("trace_2d must contain 2..10 ordered key points")
        points = [coordinates(point, 2, normalized=True) for point in values]
        projected = [pixel_point(point, width, height) for point in points]
        trajectories.append(
            {
                "label": entry["label"].strip(),
                "trace_2d": points,
                "points_xy_1000": points,
                "points_xy_px": [pair[0] for pair in projected],
                "points_row_col": [pair[1] for pair in projected],
            }
        )
    return {
        **metadata,
        "found": bool(trajectories),
        "trajectories": trajectories,
        "coordinate_space": "image_pixels",
        "trajectory_of": "robot_end_effector",
        "model_output_format": "trace_2d",
        "timestamps_available": False,
    }


TASK = {
    "prompt": PROMPT,
    "spec": SPEC,
    "parse": parse,
}
