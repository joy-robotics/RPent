"""Explicit real-planner grounding diagnostic, separate from autonomous evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIAGNOSTIC_GUIDANCE = """
INTERFACE DIAGNOSTIC (not an autonomous benchmark): before executing any robot
action, call grounding_2d on the task's target object using the latest saved head
observation. After a successful nonempty detection, pass its center_row_col to
sample_world_xyz with exactly the same view and step_idx. Read that result before
choosing an action. Then pursue the unchanged native task with lingbot_act and/or
the original atomic actions, within the existing budgets. If perception fails,
report the actual error instead of inventing coordinates. Do not infer task
success from this diagnostic: only the native environment can confirm success.
"""


def main(argv: list[str] | None = None) -> None:
    """Forward native CLI options with one explicitly disclosed diagnostic prompt."""
    from robots.robotwin.embodied import plugin
    from rpent.cli.main import main as native_main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot", default="robotwin")
    parser.add_argument("--tool-plugin", default="robots.robotwin.embodied.plugin")
    options, arguments = parser.parse_known_args(argv)
    if options.robot != "robotwin" or options.tool_plugin != plugin.__name__:
        parser.error(
            "This diagnostic requires RoboTwin and its built-in embodied plugin"
        )
    original = plugin.PLANNER_GUIDANCE
    try:
        plugin.PLANNER_GUIDANCE = original + DIAGNOSTIC_GUIDANCE
        native_main(
            ["--robot", "robotwin", "--tool-plugin", plugin.__name__, *arguments]
        )
    finally:
        plugin.PLANNER_GUIDANCE = original


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    main()
