import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from robots.robotwin.embodied.definitions import depth_estimation
from robots.robotwin.embodied.tasks import (
    TASKS,
    build_task_prompt,
    parse_task_response,
    prompt_fingerprint,
    validate_inputs,
)

CAMERA = {
    "intrinsic_K": [[100, 0, 160], [0, 100, 120], [0, 0, 1]],
    "width": 320,
    "height": 240,
}
IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)
QUERIES = [
    (
        {"kind": "object", "name": "the mug"},
        "The mug is about 25 centimeters from the camera.",
    ),
    (
        {"kind": "marker", "marker": "A"},
        "The point labeled A is about 0.25 meters from the camera.",
    ),
    (
        {"kind": "point", "point_xy_1000": [500, 1000]},
        "The surface at (500, 1000) is about 0.25 meters from the camera.",
    ),
]


@pytest.mark.parametrize("target,answer", QUERIES)
def test_depth_modes_share_one_scalar_output_contract(target, answer):
    parsed = parse_task_response("depth_estimation", answer, 320, 240, target=target)
    assert parsed["depths"][0]["depth_m"] == 0.25
    assert parsed["depths"][0]["query"] == target
    assert (
        parsed["coordinate_space"] == "camera_distance" and parsed["units"] == "meters"
    )
    assert not parsed["dense_depth_map"] and not parsed["world_transform_applied"]
    if target["kind"] == "point":
        assert parsed["depths"][0]["query_point_xy_px"] == [160, 240]
        assert parsed["depths"][0]["query_point_row_col"] == [239, 160]


@pytest.mark.parametrize(
    "text,depth",
    [
        ("图中红点所在的位置距相机约 21 厘米。", 0.21),
        ("坐标 (310, 50) 处表面距相机约 0.20 米。", 0.2),
        ("25 cm", 0.25),
        ("0.25 meters", 0.25),
        ("0 centimeters", 0),
    ],
)
def test_depth_units_are_explicit_and_normalized_to_meters(text, depth):
    assert depth_estimation.parse(text, 320, 240)["depths"][0]["depth_m"] == depth


@pytest.mark.parametrize("text,expected", [("2.07", 2.07), ("0.47", 0.47), ("0", 0)])
def test_bare_model_numbers_require_explicit_request_unit_binding(text, expected):
    with pytest.raises(ValueError):
        depth_estimation.parse(text, 320, 240)
    result = parse_task_response("depth_estimation", text, 320, 240, target="the mug")
    prediction = result["depths"][0]
    assert prediction["depth_m"] == prediction["reported_value"] == expected
    assert prediction["reported_unit"] is None
    assert (
        prediction["unit_source"] == "request_prompt"
        and prediction["requested_unit"] == "meters"
    )


@pytest.mark.parametrize(
    "text", ["1-2", "1,2", "[1]", "about 2", "-0.1", "nan", "1e999", "1e-999"]
)
def test_request_unit_binding_does_not_repair_invalid_bare_answers(text):
    with pytest.raises(ValueError):
        parse_task_response("depth_estimation", text, 320, 240, target="the mug")


def test_unit_binding_cannot_claim_a_unit_not_requested_by_the_template():
    with pytest.raises(ValueError):
        depth_estimation.parse("25", 320, 240, requested_unit="centimeters")


@pytest.mark.parametrize(
    "text",
    [
        "25",
        "[]",
        '{"depth": 25}',
        "The mug is about 25 centimeters wide.",
        "The mug and the spoon are about 25 centimeters apart.",
        "From nearest to farthest: A, B, C.",
        "The point is about 20-25 centimeters from the camera.",
        "-0.1 meters",
        "nan meters",
        "inf meters",
        "1e999 meters",
        "1e-999 meters",
        "25 millimeters",
        "I cannot estimate the depth.",
        "The mug is 1 meter wide and the point is about 2 meters from the camera.",
        "The point is about 2 meters from the camera. The object is 3 meters away.",
    ],
)
def test_depth_rejects_other_tasks_ambiguous_units_and_invalid_numbers(text):
    with pytest.raises(ValueError):
        depth_estimation.parse(text, 320, 240)


@pytest.mark.parametrize(
    "target",
    [
        "",
        "a\nb",
        {"kind": "point", "point_xy_1000": [1001, 1]},
        {"kind": "point", "point_xy_1000": [True, 1]},
        {"kind": "point", "point_xy_1000": [float("nan"), 1]},
        {"kind": "point", "point_xy_1000": [1]},
        {"kind": "point", "point_xy_1000": (1, 2)},
        {"kind": "marker", "marker": "E"},
        {"kind": "object", "name": "mug", "depth_m": 1},
        {"kind": "pairwise", "name": "mug and spoon"},
        {"kind": [], "name": "mug"},
    ],
)
def test_invalid_depth_queries_are_rejected_before_inference(target):
    with pytest.raises(ValueError):
        validate_inputs("depth_estimation", [IMAGE], target)


def test_depth_uses_optional_saved_camera_without_inventing_one():
    assert validate_inputs("depth_estimation", [IMAGE], " the mug ")[1:] == (
        {"kind": "object", "name": "the mug"},
        None,
    )
    assert (
        build_task_prompt("depth_estimation", "the mug")
        == "\nHow far is the mug from the camera, in meters?"
    )
    prompt = build_task_prompt("depth_estimation", "the mug", camera=CAMERA)
    assert prompt.startswith(
        "\nCamera intrinsics: fx=100, fy=100, cx=160, cy=120 (image size 320x240).\n"
    )
    with pytest.raises(ValueError, match="dimensions"):
        validate_inputs(
            "depth_estimation", [IMAGE], "the mug", {**CAMERA, "width": 321}
        )
    with pytest.raises(ValueError):
        validate_inputs("grounding_2d", [IMAGE], "the mug", CAMERA)


def test_point_prompt_preserves_coordinate_precision_and_marker_does_not_invent_coordinates():
    target = {"kind": "point", "point_xy_1000": [123.123456789, 500.0]}
    assert "(123.123456789, 500)" in build_task_prompt("depth_estimation", target)
    assert "point highlighted by the red dot" in build_task_prompt(
        "depth_estimation", {"kind": "marker", "marker": "red_dot"}
    )
    assert "point_xy" not in build_task_prompt(
        "depth_estimation", {"kind": "marker", "marker": "A"}
    )


@pytest.mark.parametrize(
    "target,text",
    [
        (
            {"kind": "point", "point_xy_1000": [500, 1000]},
            "The surface at (500, 999) is about 1 meter from the camera.",
        ),
        (
            {"kind": "marker", "marker": "A"},
            "The point labeled B is about 1 meter from the camera.",
        ),
    ],
)
def test_explicit_wrong_query_in_model_answer_is_rejected(target, text):
    with pytest.raises(ValueError, match="differs"):
        parse_task_response("depth_estimation", text, 320, 240, target=target)


@pytest.mark.parametrize("variant", ["object", "marker", "point", "camera"])
def test_prompt_fingerprint_covers_all_depth_variants(monkeypatch, variant):
    original = prompt_fingerprint("depth_estimation")
    changed = deepcopy(depth_estimation.PROMPT)
    changed[variant] += "changed"
    monkeypatch.setitem(TASKS["depth_estimation"], "prompt", changed)
    assert prompt_fingerprint("depth_estimation") != original


@pytest.mark.parametrize("target,answer", QUERIES)
def test_depth_suite_selects_only_standalone_absolute_depth_examples(
    tmp_path, target, answer
):
    from robots.robotwin.embodied.suite import SOURCES, prepare_depth_sample

    kind = target["kind"]
    subset, task_type = SOURCES["depth_estimation"][kind]
    folder = tmp_path / subset
    folder.mkdir()
    image = tmp_path / "image.png"
    Image.fromarray(IMAGE).save(image)
    sample = {
        "id": "synthetic-template-test",
        "task_type": task_type,
        "images": [str(image)],
        "conversations": [
            {
                "from": "human",
                "value": "<image>" + build_task_prompt("depth_estimation", target),
            },
            {"from": "gpt", "value": answer},
        ],
    }
    distractor = {**sample, "task_type": "object_size_estimation"}
    source = folder / "samples.jsonl"
    lines = [json.dumps(item) + "\n" for item in (distractor, sample)]
    source.write_text("".join(lines))
    selected = prepare_depth_sample(kind, tmp_path)
    assert selected["case"] == "depth_estimation_" + kind
    assert selected["target"] == target and selected["dataset"]["line"] == 2
    assert (
        selected["dataset"]["row_sha256"]
        == hashlib.sha256(lines[1].encode()).hexdigest()
    )
    assert (
        selected["template_exact_match"]
        and selected["dataset"]["conversation_turn"] == 0
    )
    assert answer not in selected["vlm_prompt"]
