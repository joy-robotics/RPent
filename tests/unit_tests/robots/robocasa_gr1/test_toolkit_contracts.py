import json

import numpy as np
import pytest

from robots.robocasa_gr1.toolkit import GR1Toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager

pytestmark = pytest.mark.usefixtures("video_codec")


class Model:
    chunk_size = 16

    def __init__(self):
        self.calls = 0
        self.closed = False

    def infer(self, observed):
        self.calls += 1
        return np.zeros((16, 29), dtype=np.float32)

    def close(self):
        self.closed = True


def make_toolkit(tmp_path, stack):
    model = Model()
    toolkit = GR1Toolkit(
        env=stack[2],
        model=model,
        replan_steps=16,
        output=tmp_path / "run",
        memory=MemoryManager(root=tmp_path / "memory", memory_access="read_only"),
        dashboard_events=NullDashboardEventSink(),
    )
    return toolkit, model


def test_agent_tools_execute_and_do_not_invent_success(stack, tmp_path):
    toolkit, model = make_toolkit(tmp_path, stack)
    assert toolkit.view_env_state()["_image_bytes"]
    result = toolkit.execute_tool("gr1_act", {"chunks": 1}).result
    assert result["result"]["executed_actions"] == 3
    assert result["state"]["episode_status"]["budget_exhausted"]
    assert model.calls == 1
    finish = toolkit.execute_tool(
        "finish", {"status": "success", "summary": "done"}
    ).result
    assert finish["_finish"] and not finish["success"]
    toolkit.close()
    assert model.closed and stack[2]._client.closed
    assert (tmp_path / "run/episode.mp4").stat().st_size > 0
    status = json.loads((tmp_path / "run/native_status.json").read_text())
    assert status["step_count"] == 3 and not status["native_success"]


def test_native_correction_holds_unspecified_groups(stack, tmp_path):
    toolkit, model = make_toolkit(tmp_path, stack)
    result = toolkit.execute_tool(
        "set_joint_targets", {"targets": {"waist": [0.1, 0.2, 0.3]}}
    ).result
    assert result["result"]["executed_actions"] == 1
    native_action = stack[0].actions[0]
    np.testing.assert_allclose(native_action["action.waist"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(native_action["action.left_arm"], np.arange(7))
    assert model.calls == 0
    toolkit.close()
