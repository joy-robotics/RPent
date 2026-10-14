"""Independent RoboCasa GR1 integration; simulator imports remain optional."""

from __future__ import annotations

from typing import Any


def get_robot_spec():
    """Load the descriptor without importing the GR1 simulator or policy."""
    from robots.robocasa_gr1.robot_spec import get_robot_spec as factory

    return factory()


def get_toolkit(**kwargs: Any):
    """Build the GR1 toolkit only when a runtime is requested."""
    from robots.robocasa_gr1.robot_spec import get_toolkit as factory

    return factory(**kwargs)


__all__ = ["get_robot_spec", "get_toolkit"]
