"""Keep explicit diagnostics out of normal benchmark prompts."""

import pytest

from robots.robotwin.embodied import plugin
from tests.e2e_tests.robotwin import embodied_planner_smoke


@pytest.mark.parametrize("fails", [False, True])
def test_diagnostic_forwards_options_and_restores_prompt(monkeypatch, fails):
    original = plugin.PLANNER_GUIDANCE
    recorded = []

    def fake_main(arguments):
        recorded.append(arguments)
        assert original in plugin.PLANNER_GUIDANCE
        assert embodied_planner_smoke.DIAGNOSTIC_GUIDANCE in plugin.PLANNER_GUIDANCE
        if fails:
            raise RuntimeError("test failure")

    monkeypatch.setattr("rpent.cli.main.main", fake_main)
    arguments = ["--task-name", "click_bell", "--seed", "100000"]
    if fails:
        with pytest.raises(RuntimeError, match="test failure"):
            embodied_planner_smoke.main(arguments)
    else:
        embodied_planner_smoke.main(arguments)
    assert recorded == [
        ["--robot", "robotwin", "--tool-plugin", plugin.__name__, *arguments]
    ]
    assert plugin.PLANNER_GUIDANCE == original


@pytest.mark.parametrize(
    "arguments", [["--robot", "franka"], ["--tool-plugin", "other.plugin"]]
)
def test_diagnostic_rejects_unrelated_robot_or_plugin(arguments):
    with pytest.raises(SystemExit):
        embodied_planner_smoke.main(arguments)
