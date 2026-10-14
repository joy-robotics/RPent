import importlib
import json
import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robots.robotwin.embodied import definitions as embodied_definitions
from robots.robotwin.embodied.tasks import (
    TASKS,
    build_task_prompt,
    parse_task_response,
    validate_inputs,
)
from robots.robotwin.embodied.tools import install_embodied_tools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, _is_readonly

CAMERA = {
    "intrinsic_K": [[100.0, 0, 160.0], [0, 100.0, 120.0], [0, 0, 1]],
    "width": 320,
    "height": 240,
}
RESPONSES = {
    "grounding_2d": '[{"bbox_2d": [(100,200,400,600)], "label":"box"}, '
    '{"bbox_2d": [(500,200,800,600)], "label":"second box"}]',
    "affordance_2d": '[{"afford_2d": [(100,200,400,600)], "label":"handle"}]',
    "pointing_2d": '[{"point_2d": [(500,1000)], "label":"target"}]',
    "trace_2d": '[{"trace_2d": [(100,200),(400,600)], "label":"robot_arm"}]',
    "trace_3d": '[{"trace_3d": [(0,0,1),(-0.2,0.3,1.2)], "label":"object"}]',
    "grounding_3d": '[{"bbox_3d": [(0,0,1,0.2,0.3,0.4,0,0,0.5)], "label":"box"}]',
    "video_grounding_3d": '[{"bbox_3d": [(0,0,-1,0.2,0.3,0.4,-1,0,1)], "label":"box"}]',
    "depth_estimation": "The box is about 25 centimeters from the camera.",
}


@pytest.mark.parametrize("task", RESPONSES)
def test_all_tasks_are_defined_and_exported_from_one_package(task):
    from robots.robotwin.embodied.suite import SOURCES

    module = importlib.import_module("robots.robotwin.embodied.definitions." + task)
    assert TASKS is embodied_definitions.TASKS
    assert TASKS[task] is module.TASK
    assert callable(module.TASK["parse"])
    assert module.TASK["parse"].__module__.startswith(
        "robots.robotwin.embodied.definitions."
    )
    assert module.TASK["prompt"] == module.PROMPT
    assert module.TASK["spec"] is module.SPEC
    assert module.TASK["parse"] is module.parse
    assert set(TASKS) == set(RESPONSES) == set(SOURCES)


def test_grounding_relocation_preserves_public_contract_and_legacy_imports():
    from robots.robotwin.embodied.definitions import grounding_2d
    from robots.robotwin.embodied.tasks import (
        GROUNDING_PROMPT,
        TOOL_SPEC,
        parse_grounding_response,
        prompt_fingerprint,
    )

    assert GROUNDING_PROMPT == grounding_2d.PROMPT
    assert TOOL_SPEC is grounding_2d.TASK["spec"]
    assert parse_grounding_response is grounding_2d.parse
    assert (
        prompt_fingerprint("grounding_2d")
        == "e31d346a4eab1fe604d2002b207de034773cf6fe06fa4fd444da06931b7d5766"
    )
    assert TOOL_SPEC is grounding_2d.SPEC


@pytest.mark.parametrize(
    "task", [task for task in RESPONSES if task != "depth_estimation"]
)
def test_literal_and_json_encodings_match(task):
    from robots.robotwin.embodied.definitions.common import literal_data

    parsed = TASKS[task]["parse"](RESPONSES[task], 320, 240)
    equivalent = json.dumps(literal_data(RESPONSES[task]))
    assert TASKS[task]["parse"](equivalent, 320, 240) == parsed
    assert parsed["found"]
    assert not TASKS[task]["parse"]("[]", 320, 240)["found"]


def test_coordinate_units_order_and_negative_video_depth():
    points = TASKS["pointing_2d"]["parse"](RESPONSES["pointing_2d"], 320, 240)
    assert points["points"][0]["point_xy_px"] == [160, 240]
    assert points["points"][0]["point_row_col"] == [239, 160]
    box = TASKS["grounding_3d"]["parse"](RESPONSES["grounding_3d"], 320, 240)
    assert box["detections"][0]["rpy_rad"][2] == math.pi / 2
    video = TASKS["video_grounding_3d"]["parse"](
        RESPONSES["video_grounding_3d"], 320, 240
    )
    assert video["coordinate_space"] == "last_frame_camera_cv"
    assert video["detections"][0]["center_xyz_m"][2] == -1
    assert not video["world_transform_applied"]


@pytest.mark.parametrize("task", ["grounding_3d", "video_grounding_3d"])
def test_flat_3d_vector_is_losslessly_normalized(task):
    flat = [
        {
            "bbox_3d": [
                -0.617,
                0.0414,
                1.9323,
                0.6846,
                0.707,
                1.0211,
                0.542,
                0.0001,
                -0.6421,
            ],
            "label": "chair",
        }
    ]
    result = TASKS[task]["parse"](json.dumps(flat), 320, 240)
    assert result["detections"][0]["bbox_3d"] == [flat[0]["bbox_3d"]]
    assert result["detections"][0]["source_bbox_layout"] == "flat_tuple9"
    with pytest.raises(ValueError):
        TASKS[task]["parse"](
            '[{"bbox_3d": [0,0,1,1,1,1,0,0], "label":"box"}]', 320, 240
        )


@pytest.mark.parametrize(
    "task,text",
    [
        ("grounding_2d", '[{"bbox_2d": [(500,100,200,300)], "label":"x"}]'),
        ("affordance_2d", '[{"afford_2d": [(500,100,200,300)], "label":"x"}]'),
        ("pointing_2d", '[{"point_2d": [(1001,20)], "label":"x"}]'),
        ("pointing_2d", '[{"point_2d": [(True,20)], "label":"x"}]'),
        ("pointing_2d", '[{"point_2d": [(1,2)], "label":"x", "label":"y"}]'),
        ("trace_2d", '[{"trace_2d": [(1,2)], "label":"x"}]'),
        ("trace_2d", repr([{"trace_2d": [(1, 2)] * 11, "label": "x"}])),
        ("trace_3d", '[{"trace_3d": [(0,0,1),(0,0,1e999)], "label":"x"}]'),
        ("grounding_3d", '[{"bbox_3d": [(0,0,1,0,1,1,0,0,0)], "label":"x"}]'),
        ("grounding_3d", '[{"bbox_3d": [(0,0,1,1,1,1,0,0,3.14)], "label":"x"}]'),
        (
            "video_grounding_3d",
            '[{"bbox_3d": [(__import__("os"),0,1,1,1,1,0,0,0)], "label":"x"}]',
        ),
    ],
)
def test_malformed_or_unsafe_geometry_is_rejected(task, text):
    with pytest.raises(ValueError):
        TASKS[task]["parse"](text, 320, 240)


def test_prompt_includes_real_calibration_and_frame_count():
    prompt = build_task_prompt(
        "video_grounding_3d", "chair", camera=CAMERA, frame_count=2
    )
    assert "video of 2 frames" in prompt and "f_x=100.0" in prompt
    assert "LAST frame's camera" in prompt and '"label":"chair"' in prompt
    with pytest.raises(ValueError):
        build_task_prompt("grounding_3d", "chair")


@pytest.mark.parametrize(
    "camera",
    [
        None,
        {**CAMERA, "width": 321},
        {**CAMERA, "intrinsic_K": [[0, 0, 160], [0, 100, 120], [0, 0, 1]]},
        {**CAMERA, "intrinsic_K": [[100, 1, 160], [0, 100, 120], [0, 0, 1]]},
        {**CAMERA, "intrinsic_K": [[True, 0, 160], [0, 100, 120], [0, 0, 1]]},
    ],
)
def test_calibration_is_required_and_not_guessed(camera):
    with pytest.raises(ValueError):
        validate_inputs(
            "grounding_3d", [np.zeros((240, 320, 3), dtype=np.uint8)], "box", camera
        )


@pytest.mark.parametrize("count", [0, 1, 65])
def test_video_frame_count_is_bounded(count):
    with pytest.raises(ValueError):
        validate_inputs(
            "video_grounding_3d",
            [np.zeros((240, 320, 3), dtype=np.uint8)] * count,
            "box",
            CAMERA,
        )


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)
    state = EnvState(tmp_path / "state")
    for frame_index in range(2):
        with state.record_step(state={"frame": frame_index}):
            state.save(
                "head_rgb.png", np.full((240, 320, 3), frame_index * 20, dtype=np.uint8)
            )
            state.save("head_camera_meta.json", CAMERA)
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(root=tmp_path / "memory"),
    )
    calls = []

    def infer(image, target, *, task, camera=None):
        images = image if TASKS[task].get("video") else [image]
        images, target, camera = validate_inputs(task, images, target, camera)
        calls.append((task, images, camera))
        parsed = parse_task_response(task, RESPONSES[task], 320, 240, target=target)
        return {**parsed, "request_id": "test-" + task, "raw_text": RESPONSES[task]}

    install_embodied_tools(toolkit, SimpleNamespace(infer=infer), tmp_path)
    return toolkit, calls, tmp_path


@pytest.mark.parametrize("task", RESPONSES)
def test_registered_tools_return_planner_content_without_actions(scene, task):
    toolkit, calls, output = scene
    arguments = {"target": "move or locate the box", "view": "head"}
    arguments.update({"steps": [0, 1]} if TASKS[task].get("video") else {"step": 0})
    result = toolkit.execute_tool(task, arguments)
    assert result.result["success"], result.result
    assert _is_readonly(toolkit._tools[task][1])
    assert not result.is_finish and not result.result["action_executed"]
    assert json.loads(result.content_blocks[0]["text"]) == result.result
    assert len(toolkit.state.records()) == 2 and len(calls) == 1
    assert (output / "embodied_tools" / ("test-" + task + "_overlay.png")).is_file()
    if TASKS[task].get("video"):
        assert result.result["reference_step_idx"] == 1
        assert result.result["step_indices"] == [0, 1]
        assert calls[0][1][0][0, 0, 0] == 0 and calls[0][1][1][0, 0, 0] == 20
        assert len(list((output / "embodied_tools").glob("*_frame_*_input.png"))) == 2


@pytest.mark.parametrize("steps", [[1, 0], [0, 0], [0], [-1, 0], [False, 1], [0, 99]])
def test_invalid_video_steps_never_reach_model(scene, steps):
    toolkit, calls, _output = scene
    result = toolkit.execute_tool(
        "video_grounding_3d", {"target": "box", "steps": steps}
    )
    assert "error" in result.result and not calls


def test_changed_video_intrinsics_are_rejected(scene):
    toolkit, calls, _output = scene
    camera = deepcopy(CAMERA)
    camera["intrinsic_K"][0][0] = 101
    with toolkit.state.record_step(state={}):
        toolkit.state.save("head_rgb.png", np.zeros((240, 320, 3), dtype=np.uint8))
        toolkit.state.save("head_camera_meta.json", camera)
    result = toolkit.execute_tool(
        "video_grounding_3d", {"target": "box", "steps": [0, 2]}
    )
    assert "error" in result.result and not calls


def test_missing_saved_calibration_never_reaches_model(scene):
    toolkit, calls, _output = scene
    with toolkit.state.record_step(state={}):
        toolkit.state.save("head_rgb.png", np.zeros((240, 320, 3), dtype=np.uint8))
    result = toolkit.execute_tool("grounding_3d", {"target": "box", "step": 2})
    assert "error" in result.result and not calls


def test_video_pixel_budget_and_dimensions():
    with pytest.raises(ValueError, match="16-megapixel"):
        validate_inputs(
            "video_grounding_3d",
            [np.zeros((2000, 2000, 3), dtype=np.uint8)] * 5,
            "box",
            CAMERA,
        )
    with pytest.raises(ValueError, match="share image dimensions"):
        validate_inputs(
            "video_grounding_3d",
            [
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.zeros((240, 321, 3), dtype=np.uint8),
            ],
            "box",
            CAMERA,
        )


@pytest.mark.parametrize("mode", ["audit", "vlm", "planner"])
def test_suite_defaults_include_all_tools_and_depth_kinds(tmp_path, monkeypatch, mode):
    from robots.robotwin.embodied import suite as embodied_suite

    calls = []

    def prepare(task, data_root, **options):
        kind = options.get("depth_kind")
        calls.append((task, kind))
        return {"task": task, "case": task + "_" + kind if kind else task}

    def run(sample, args):
        return {**sample, "passed": True}

    monkeypatch.setattr(embodied_suite, "prepare_sample", prepare)
    if mode != "audit":
        monkeypatch.setattr(
            embodied_suite, "run_planner_case" if mode == "planner" else "run_case", run
        )
    output = tmp_path / "suite"
    arguments = ["--output", str(output)]
    if mode != "vlm":
        arguments.append("--audit-only" if mode == "audit" else "--planner")
    assert embodied_suite.main(arguments) == 0
    expected = [
        (task, kind)
        for task in TASKS
        for kind in (
            ["object", "marker", "point"] if task == "depth_estimation" else [None]
        )
    ]
    assert calls == expected
    reports = json.loads((output / "summary.json").read_text())
    assert len(reports) == 10 and all(report["passed"] for report in reports)


@pytest.mark.parametrize(
    "task,mutation",
    [(task, "exact") for task in RESPONSES]
    + [
        ("grounding_2d", mutation)
        for mutation in (
            "partial_predictions",
            "changed_coordinate",
            "request_id",
            "coordinate_space",
            "view",
            "step_idx",
        )
    ],
)
def test_probe_uses_complete_perception_receipt_for_all_tools(
    tmp_path, monkeypatch, task, mutation
):
    from robots.robotwin.embodied import probe as embodied_probe
    from rpent.planner import api_loop, base

    image = tmp_path / "input.png"
    Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(image)
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps(CAMERA))
    config = tmp_path / "planner.json"
    config.write_text(json.dumps({"planner": {}}))
    planner = {
        "model": "openai:offline-unit-test",
        "base_url": "http://127.0.0.1:1",
        "reasoning_effort": None,
    }

    class FakeClient:
        metadata = {"backend": "offline_test_double"}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def infer(self, image, target, *, task, camera=None):
            return {
                **parse_task_response(task, RESPONSES[task], 320, 240, target=target),
                "request_id": "unit-" + task,
                "raw_text": RESPONSES[task],
            }

    class FakePlanner:
        def __init__(self, *args, **kwargs):
            pass

        def solve(self, *, system_prompt, user_message, toolkit, max_turns):
            assert "report_received_perception" in system_prompt
            assert "report_received_bbox" not in toolkit._tools
            response = toolkit.execute_tool(task, json.loads(user_message)).result
            predictions = next(
                response[name]
                for name in (
                    "detections",
                    "regions",
                    "points",
                    "trajectories",
                    "depths",
                )
                if name in response
            )
            receipt = {
                name: response[name]
                for name in ("request_id", "coordinate_space", "view", "step_idx")
            }
            receipt["predictions"] = deepcopy(predictions)
            if mutation == "partial_predictions":
                receipt["predictions"] = receipt["predictions"][:1]
            elif mutation == "changed_coordinate":
                receipt["predictions"][1]["bbox_xyxy_px"][0] += 1
            elif mutation != "exact":
                receipt[mutation] = 99 if mutation == "step_idx" else "incorrect"
            accepted = toolkit.execute_tool(
                "report_received_perception", receipt
            ).result["accepted"]
            assert accepted is (mutation == "exact")
            finish = toolkit.execute_tool("finish", {})
            return SimpleNamespace(
                messages=[],
                stats={"requests": 0},
                error=None,
                finish_result=finish.result,
            )

    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)
    monkeypatch.setattr("rpent.utils.logging._log_initialized", True)
    monkeypatch.setattr(embodied_probe, "EmbodiedClient", FakeClient)
    monkeypatch.setattr(
        embodied_probe,
        "_planner_settings",
        lambda args: (planner, {"OPENAI_API_KEY": "offline-unit-test"}),
    )
    monkeypatch.setattr(base, "build_api_model", lambda *args: object())
    monkeypatch.setattr(api_loop, "ApiAgentLoop", FakePlanner)
    output = tmp_path / "probe"
    report = embodied_probe.run_probe(
        SimpleNamespace(
            command="planner",
            tool=task,
            image=image,
            additional_image=[image] if TASKS[task].get("video") else [],
            camera_meta=camera if TASKS[task].get("requires_camera") else None,
            target="box",
            view="head",
            output=output,
            endpoint="ws://127.0.0.1:1",
            checkpoint="offline-test-double",
            timeout=1,
            planner_config=config,
        )
    )
    assert report["passed"] is (mutation == "exact")
    assert report["planner_received_exact_perception"] is (mutation == "exact")
    assert report["observation_steps_after"] == (2 if TASKS[task].get("video") else 1)
    if task == "grounding_2d":
        assert report["planner_received_exact_bbox"] is (mutation == "exact")
    events = json.loads((output / "tool_events.json").read_text())
    assert [event["tool"] for event in events] == [
        task,
        "report_received_perception",
        "finish",
    ]
