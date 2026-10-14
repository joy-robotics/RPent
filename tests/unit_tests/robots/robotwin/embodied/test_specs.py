import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from robots.robotwin.embodied.definitions.common import make_spec
from robots.robotwin.embodied.tasks import TASKS, prompt_fingerprint

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/embodied_tool_contract_v1.json").read_text()
)
SECTIONS = (
    "Purpose",
    "Input",
    "Returns",
    "Coordinates",
    "Camera",
    "Usage",
    "Limitations",
)
EXAMPLES = {
    "grounding_2d": {"target": "the red block"},
    "affordance_2d": {"target": "Where should the mug be held to lift it?"},
    "pointing_2d": {"target": "Where is the center of the mug handle?"},
    "trace_2d": {"target": "Pick up the red block and place it in the bowl."},
    "trace_3d": {"target": "Lift the mug off the table."},
    "grounding_3d": {"target": "the mug beside the plate"},
    "video_grounding_3d": {"target": "chair", "steps": [0, 2, 5]},
    "depth_estimation": {"target": "the red cup"},
}


def without_descriptions(value):
    if isinstance(value, dict):
        return {
            key: without_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


@pytest.mark.parametrize("name", TASKS)
def test_explicit_definition_template_and_complete_description(name):
    module = importlib.import_module("robots.robotwin.embodied.definitions." + name)
    assert module.TASK is TASKS[name]
    assert module.TASK["spec"] is module.SPEC
    assert module.TASK["prompt"] is module.PROMPT
    assert module.TASK["parse"] is module.parse
    assert module.SPEC["name"] == name
    sections = module.SPEC["description"].splitlines()
    assert [section.split(":", 1)[0] for section in sections] == list(SECTIONS)
    assert all(section.split(":", 1)[1].strip() for section in sections)
    for field in (
        "step_idx",
        "view",
        "image_size",
        "source_image_sha256",
        "action_executed=false",
    ):
        assert field in sections[2]
    assert "Read-only" in sections[-1]
    assert "official task success" in sections[-1]
    properties = module.SPEC["input_schema"]["properties"]
    assert all(properties[field]["description"] for field in properties)
    if module.TASK.get("requires_camera"):
        assert "Requires real saved intrinsic_K" in sections[4]
    elif module.TASK.get("optional_camera"):
        assert "Without calibration" in sections[4]
    else:
        assert "neither required nor accepted" in sections[4]


@pytest.mark.parametrize("name", TASKS)
def test_training_prompt_and_input_contract_match_evaluated_version(name):
    assert prompt_fingerprint(name) == BASELINE[name]["prompt_sha256"]
    assert without_descriptions(TASKS[name]["spec"]) == without_descriptions(
        BASELINE[name]["spec"]
    )
    assert TASKS[name]["spec"]["description"] != BASELINE[name]["spec"]["description"]


@pytest.mark.parametrize("name", TASKS)
def test_valid_examples_and_common_parameter_rejections(name):
    schema = TASKS[name]["spec"]["input_schema"]
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema)
    example = EXAMPLES[name]
    assert validator.is_valid(example)
    assert validator.is_valid({**example, "view": "right_wrist"})
    for invalid in (
        {**example, "target": ""},
        {**example, "view": "overhead"},
        {**example, "camera": {}},
        {**example, "unexpected": True},
    ):
        assert not validator.is_valid(invalid)
    if TASKS[name].get("video"):
        assert not validator.is_valid({**example, "steps": [0]})
        assert not validator.is_valid({**example, "steps": [0, 0]})
        assert not validator.is_valid({**example, "steps": [-1, 0]})
        assert not validator.is_valid({**example, "step": -1})
    else:
        assert validator.is_valid({**example, "step": -1})
        assert not validator.is_valid({**example, "step": -2})
        assert not validator.is_valid({**example, "steps": [0, 1]})


@pytest.mark.parametrize(
    "target",
    [
        "the red cup",
        {"kind": "object", "name": "the red cup"},
        {"kind": "marker", "marker": "red_dot"},
        {"kind": "point", "point_xy_1000": [500, 400]},
    ],
)
def test_depth_retains_all_target_variants(target):
    assert Draft7Validator(TASKS["depth_estimation"]["spec"]["input_schema"]).is_valid(
        {"target": target}
    )


def factory_arguments():
    return {
        "purpose": "Purpose text.",
        "target_description": "Target text.",
        "returns": "Return text.",
        "coordinates": "Coordinate text.",
        "usage": "Usage text.",
        "limitations": "Limitation text.",
    }


def test_factory_does_not_mutate_or_share_nested_schemas():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    original = deepcopy(schema)
    first = make_spec("first", target_schema=schema, **factory_arguments())
    second = make_spec("second", target_schema=schema, **factory_arguments())
    first["input_schema"]["properties"]["target"]["required"].append("extra")
    first["input_schema"]["properties"]["view"]["enum"].append("extra")
    assert schema == original
    assert second["input_schema"]["properties"]["target"]["required"] == ["name"]
    assert second["input_schema"]["properties"]["view"]["enum"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]


@pytest.mark.parametrize("field", factory_arguments())
def test_factory_requires_complete_description_fields(field):
    with pytest.raises(ValueError, match="nonempty strings"):
        make_spec("incomplete", **{**factory_arguments(), field: ""})


def test_factory_rejects_unknown_camera_contract():
    with pytest.raises(ValueError, match="camera must be"):
        make_spec("invalid_camera", camera="invented", **factory_arguments())


def test_task_specific_distinctions_are_exposed_to_planner():
    descriptions = {name: task["spec"]["description"] for name, task in TASKS.items()}
    assert (
        "end-effector" in descriptions["trace_2d"]
        and "2..10" in descriptions["trace_2d"]
    )
    assert (
        "manipulated_object" in descriptions["trace_3d"]
        and "2..128" in descriptions["trace_3d"]
    )
    assert "composition order" in descriptions["grounding_3d"]
    assert "last_frame_camera_cv" in descriptions["video_grounding_3d"]
    assert "negative final-camera z" in descriptions["video_grounding_3d"]
    assert "axial Z versus" in descriptions["depth_estimation"]
    assert "empty label" in descriptions["affordance_2d"]
