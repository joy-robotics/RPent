"""Expose the canonical embodied tools to native RPent's optional plugin hook."""

from __future__ import annotations

import json
from pathlib import Path

from .client import EmbodiedClient
from .tasks import TASKS
from .tools import install_embodied_tools

PLANNER_GUIDANCE = """You also have eight read-only perception tools backed by one local VLM.
Use grounding_2d to localize named objects, affordance_2d for interaction regions,
and pointing_2d for a requested spatial point. For object-specific atomic targets,
prefer these dedicated localization tools over guessing image pixels by eye.
Pass a returned center_row_col or point row/column to sample_world_xyz with the
SAME view and saved step to obtain a world-frame target. Reuse reliable recent
localization when the scene has not changed; do not add redundant calls.
Use trace_2d or trace_3d
for a proposed trajectory, grounding_3d for camera-frame object boxes, and
video_grounding_3d for objects across two or more ordered recorded observations.
Use depth_estimation for a requested object or point's estimated metric depth.
Read each tool's schema for its target syntax. step=-1 uses the latest saved
observation; view selects head, left_wrist or right_wrist. Read-only calls do not
advance the robot or create a new observation. Prefer fresh frames after motion.
Tools load real camera intrinsics from that observation; never supply invented
camera parameters. 2D results contain source-image pixels and row/column lookup
coordinates. Ground motion targets with the existing world-map tools: do not
pass pixels or camera-frame 3D predictions directly to world-frame move_to.
3D outputs use x-right, y-down, z-forward camera coordinates, not the RoboTwin
world frame. Depth and trajectories are model estimates, not sensor ground truth,
collision checks, executable commands or evidence of a successful grasp.
Use these tools when they help decide an action; do not call them merely to meet
a quota. If perception fails or is uncertain, inspect the real image/world map
and retain both lingbot_act and atomic tools for recovery. The full task goal,
native success condition and existing time/action budgets remain unchanged."""


def tool_names():
    """Return the canonical registry without duplicating tool definitions."""
    return tuple(TASKS)


def planner_prompt():
    """Describe when and how the planner may use the extra tools."""
    return PLANNER_GUIDANCE


def source_files():
    """Freeze the plugin, model client and all prompt/parser implementations."""
    from rpent.utils import msgpack_numpy

    root = Path(__file__).parent
    names = (
        "__init__.py",
        "plugin.py",
        "client.py",
        "tools.py",
        "tasks.py",
        "server.py",
    )
    return (
        [root / name for name in names]
        + sorted((root / "definitions").glob("*.py"))
        + [Path(msgpack_numpy.__file__)]
    )


def install(toolkit, options, output):
    """Connect all eight tools to one borrowed VLM service and return its owner."""
    client = EmbodiedClient(
        options["endpoint"],
        checkpoint=options["checkpoint"],
        timeout=options["timeout"],
    )
    try:
        install_embodied_tools(toolkit, client, output)
        (Path(output) / "embodied_model_metadata.json").write_text(
            json.dumps(client.metadata, indent=2, ensure_ascii=False) + "\n"
        )
    except BaseException:
        client.close()
        raise
    return client
