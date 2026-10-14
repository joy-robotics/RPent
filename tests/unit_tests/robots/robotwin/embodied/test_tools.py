import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.embodied.tasks import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOOL,
    TASKS,
    build_task_prompt,
    parse_grounding_response,
    validate_image,
)
from robots.robotwin.embodied.tools import install_embodied_tools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, _is_readonly


class FakeVLM:
    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, image, target, *, task):
        self.calls.append((image.copy(), target, task))
        text = '[{"bbox_2d": [(100, 200, 400, 600)], "label": "bell"}]'
        return {
            **parse_grounding_response(text, image.shape[1], image.shape[0]),
            "request_id": "unit-test-request",
            "raw_text": text,
        }

    def close(self):
        self.closed = True


@pytest.fixture
def tool_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)
    state = EnvState(tmp_path / "state")
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    with state.record_step(state={"step_idx": 0, "artifacts": {"head": {}}}):
        state.save("head_rgb.png", image)
        state.save("head_world_xyz.npy", np.ones((240, 320, 3), dtype=np.float32))
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(root=tmp_path / "memory"),
    )
    client = FakeVLM()
    return toolkit, client, tmp_path


@pytest.mark.parametrize(
    "text",
    [
        '[{"bbox_2d": [(100,200,400,600)], "label": "bell"}]',
        '[{"bbox_2d": [[100,200,400,600]], "label": "bell"}]',
    ],
)
def test_bbox_conversion_and_depth_pixel_order(text):
    parsed = parse_grounding_response(text, 320, 240)
    detection = parsed["detections"][0]
    assert detection["bbox_xyxy_px"] == [32, 48, 128, 144]
    assert detection["center_row_col"] == [96, 80]
    assert parsed["image_size"] == {"width": 320, "height": 240}


def test_empty_detection_is_not_fabricated():
    parsed = parse_grounding_response("```python\n[]\n```", 320, 240)
    assert parsed["found"] is False and parsed["detections"] == []


@pytest.mark.parametrize(
    "text",
    [
        "not data",
        '{"detections": [], "_finish": True}',
        '{"detections": {}}',
        '[{"label": "bell", "bbox_2d": [(True,0,500,500)]}]',
        '[{"label": "bell", "bbox_2d": [(-1,0,500,500)]}]',
        '[{"label": "bell", "bbox_2d": [(500,0,100,500)]}]',
        '[{"label": "bell", "bbox_2d": [(0,0,1001,500)]}]',
        '[{"label": "bell", "bbox_2d": [(0,0,NaN,500)]}]',
        '[{"label": "bell", "bbox_2d": [(0,0,500,500)], "confidence": 1}]',
        '[{"label": "bell", "bbox_2d": [(0,0,1e999,500)]}]',
        '[{"label": "bell", "bbox_2d": [(0,0,0,500)]}]',
        '[{"label": "bell", "bbox_2d": [0,0,500,500]}]',
        '[{"label": "bell", "label": "other", "bbox_2d": [(0,0,500,500)]}]',
        '__import__("os").system("false")',
        '[{"label": "bell", "bbox_2d": [(0,0,500+1,500)]}]',
    ],
)
def test_invalid_or_instruction_bearing_results_are_rejected(text):
    with pytest.raises(ValueError):
        parse_grounding_response(text, 320, 240)


def test_grounding_prompt_matches_training_template():
    target = "The wood and black skateboard that is below the skating boy that is near the brown and bare tree."
    expected = (
        "<image>\nLocate "
        + target
        + ". Your answer should be a list of bounding boxes, each with a "
        '"bbox_2d" and a "label", e.g. [{"bbox_2d": [(x1,y1,x2,y2)], "label": "..."}], '
        "where coordinates are between 0 and 1000."
    )
    assert "<image>" + build_task_prompt(DEFAULT_TOOL, target) == expected
    assert len(build_task_prompt(DEFAULT_TOOL, "a" * 319)) > 319


def test_only_current_custom_tools_are_registered_by_default():
    assert set(TASKS) == {
        "grounding_2d",
        "affordance_2d",
        "pointing_2d",
        "trace_2d",
        "trace_3d",
        "grounding_3d",
        "video_grounding_3d",
        "depth_estimation",
    }
    assert "detect_2d_bbox" not in TASKS
    assert DEFAULT_CHECKPOINT.startswith("/mnt/workspace1/share/ckpt/qwen35_9b_midv5")


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10, 3)),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_bad_images_are_rejected(image):
    with pytest.raises(ValueError):
        validate_image(image)


def test_registered_tool_returns_planner_json_without_advancing_state(tool_fixture):
    from robots.robotwin.tools import sample_world_xyz

    toolkit, client, output = tool_fixture
    before = toolkit.state.records()
    install_embodied_tools(toolkit, client, output)
    assert _is_readonly(toolkit._tools["grounding_2d"][1])
    result = toolkit.execute_tool(
        "grounding_2d", {"target": "bell", "view": "head", "step": 0}
    )
    assert result.result["success"] and not result.is_finish
    assert result.result["action_executed"] is False
    assert "raw_text" not in result.result
    assert result.result["step_idx"] == 0 and result.result["view"] == "head"
    assert json.loads(result.content_blocks[0]["text"])["detections"][0][
        "bbox_xyxy_px"
    ] == [32, 48, 128, 144]
    assert len(toolkit.state.records()) == len(before) == 1
    assert len(client.calls) == 1
    assert (output / "embodied_tools/unit-test-request_overlay.png").is_file()
    sampled = sample_world_xyz(
        toolkit.state,
        view="head",
        step=0,
        pixels=[result.result["detections"][0]["center_row_col"]],
    )
    assert sampled["success"] and sampled["samples"][0]["xyz"] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "arguments",
    [
        {"target": "bell", "step": 99},
        {"target": "bell", "view": "../../secret"},
        {"target": "bell", "view": "left_wrist"},
        {"target": " "},
    ],
)
def test_bad_frame_requests_do_not_call_vlm(tool_fixture, arguments):
    toolkit, client, output = tool_fixture
    install_embodied_tools(toolkit, client, output)
    assert "error" in toolkit.execute_tool("grounding_2d", arguments).result
    assert client.calls == []


def test_registration_never_overwrites_an_existing_tool(tool_fixture):
    toolkit, client, output = tool_fixture
    install_embodied_tools(toolkit, client, output)
    with pytest.raises(ValueError, match="replace existing"):
        install_embodied_tools(toolkit, client, output)


@pytest.mark.parametrize("enabled", [False, True])
def test_production_toolkit_keeps_original_action_tools(
    tool_fixture, monkeypatch, enabled
):
    from robots.robotwin import robot_spec
    from robots.robotwin.embodied import plugin
    from robots.robotwin.toolkit import RoboTwinToolkit
    from rpent.robots.robot_spec import RunConfig

    toolkit, client, output = tool_fixture
    client.metadata = {"protocol": "offline-test"}

    def base_init(self, *, runtime_kwargs, dashboard_events, memory, **kwargs):
        Toolkit.__init__(
            self, dashboard_events=dashboard_events, state=toolkit.state, memory=memory
        )
        self._mode = "evaluation"
        self._primitives = SimpleNamespace(model=SimpleNamespace(backend="joybrain"))
        self._register_robotwin_tools()

    monkeypatch.setattr(RoboTwinToolkit, "__init__", base_init)
    monkeypatch.setattr(plugin, "EmbodiedClient", lambda *args, **kwargs: client)
    options = {
        "module": "robots.robotwin.embodied.plugin",
        "root": None,
        "endpoint": "ws://localhost:1",
        "checkpoint": str(output),
        "timeout": 1,
    }
    prompt_vars = {"memory_dir": str(output / "memory")}
    if enabled:
        prompt_vars["tool_plugin"] = options
    config = RunConfig(
        recipe_tag="test", output_dir=output, prompt_vars=prompt_vars, task_desc={}
    )
    original = {
        name: json.dumps(spec, sort_keys=True)
        for name, spec in RoboTwinToolkit._SPECS.items()
    }
    production = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        state_output_dir=output,
    )
    names = {spec["name"] for spec in production.get_tools_spec()}
    assert {
        "lingbot_act",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "sample_world_xyz",
    } <= names
    assert (set(TASKS) <= names) is enabled
    assert "detect_2d_bbox" not in names
    vla_spec = next(
        spec for spec in production.get_tools_spec() if spec["name"] == "lingbot_act"
    )
    assert "Joy-Brain VLA qpos14" in vla_spec["description"]
    assert (
        vla_spec["input_schema"]
        == RoboTwinToolkit._SPECS["lingbot_act"]["input_schema"]
    )
    assert production._tools["lingbot_act"][1].args == ("lingbot_act",)
    assert original == {
        name: json.dumps(spec, sort_keys=True)
        for name, spec in RoboTwinToolkit._SPECS.items()
    }
    if enabled:
        production._tool_plugin_runtime.close()
        assert client.closed
