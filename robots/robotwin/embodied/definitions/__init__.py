"""Eight perception definitions; all share one VLM checkpoint/backend."""

from . import (
    affordance_2d,
    depth_estimation,
    grounding_2d,
    grounding_3d,
    pointing_2d,
    trace_2d,
    trace_3d,
    video_grounding_3d,
)

TASKS = {
    module.TASK["spec"]["name"]: module.TASK
    for module in (
        grounding_2d,
        affordance_2d,
        pointing_2d,
        trace_2d,
        trace_3d,
        grounding_3d,
        video_grounding_3d,
        depth_estimation,
    )
}
