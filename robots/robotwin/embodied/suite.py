"""Reproducible training-template audit and real local VLM/toolkit smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .client import EmbodiedClient
from .definitions import depth_estimation
from .definitions.common import literal_data
from .tasks import (
    DEFAULT_CHECKPOINT,
    TASKS,
    build_task_prompt,
    parse_task_response,
    validate_camera,
)

DATA_ROOT = Path("/mnt/workspace/share/robot_data/grounding")
SPATIAL_DATA_ROOT = Path("/mnt/workspace/share/robot_data/spatial_understanding")
SOURCES = {
    "grounding_2d": (
        "Object_2D/finecop_posti_83k",
        "\nLocate ",
        ". Your answer should",
    ),
    "affordance_2d": (
        "Affordance_2D_rewrite/affordance_distill_v4_0808_box",
        "\n",
        " Your answer should",
    ),
    "pointing_2d": ("Area_Spatial_Pointing_2D/refspatial", "\n", " Your answer should"),
    "trace_2d": (
        "Trajectory_2D/traj_distill",
        "The task instruction is: ",
        ". Please predict",
    ),
    "trace_3d": (
        "Trajectory_3D/ca1m_trace_3d",
        'visual trace for the task "',
        '"\nPredict',
    ),
    "grounding_3d": (
        "Object_3D/joyai_openspatial",
        "Locate the ",
        " in the provided image",
    ),
    "video_grounding_3d": (
        "Video_Object_3D/arkit_multi",
        'Detect ALL instances of "',
        '" visible in the video',
    ),
    "depth_estimation": {
        "object": ("4metricdepth_object_v1", "object_distance_to_camera"),
        "marker": ("4metricdepth_marker_v1", "visual_point_absolute_depth"),
        "point": ("4metricdepth_point_v1", "point_absolute_depth"),
    },
}


def prepare_depth_sample(kind, data_root=SPATIAL_DATA_ROOT):
    subset, task_type = SOURCES["depth_estimation"][kind]
    source = sorted((data_root / subset).glob("*.jsonl"))[0]
    pattern = re.escape(depth_estimation.PROMPT[kind])
    for field in ("name", "marker_name", "point_x", "point_y"):
        pattern = pattern.replace(re.escape("{" + field + "}"), f"(?P<{field}>.+?)")
    with source.open() as stream:
        for line_number, line in enumerate(stream, 1):
            sample = json.loads(line)
            if sample.get("task_type") != task_type:
                continue
            human = sample["conversations"][0]["value"]
            if not human.startswith("<image>"):
                continue
            matched = re.fullmatch(pattern, human.removeprefix("<image>"))
            if matched is None:
                continue
            if kind == "object":
                target = {"kind": kind, "name": matched["name"]}
            elif kind == "marker":
                marker = next(
                    (
                        name
                        for name, description in depth_estimation.MARKERS.items()
                        if description == matched["marker_name"]
                    ),
                    None,
                )
                if marker is None:
                    continue
                target = {"kind": kind, "marker": marker}
            else:
                target = {
                    "kind": kind,
                    "point_xy_1000": [
                        float(matched[field]) for field in ("point_x", "point_y")
                    ],
                }
            target = depth_estimation.validate_target(target)
            prompt = build_task_prompt("depth_estimation", target)
            if "<image>" + prompt != human:
                raise ValueError(
                    "Depth prompt does not exactly reconstruct the selected training question"
                )
            if len(sample["images"]) != 1:
                raise ValueError("Scalar depth requires exactly one training image")
            with Image.open(sample["images"][0]) as image:
                width, height = image.size
            answer = sample["conversations"][1]["value"]
            return {
                "task": "depth_estimation",
                "case": "depth_estimation_" + kind,
                "target": target,
                "camera": None,
                "images": sample["images"],
                "image_size": [width, height],
                "vlm_prompt": prompt,
                "template_exact_match": True,
                "frame_count": 1,
                "dataset": {
                    "source": str(source),
                    "line": line_number,
                    "id": sample.get("id"),
                    "task_type": task_type,
                    "conversation_turn": 0,
                    "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    "human": human,
                    "annotation": answer,
                    "parsed_annotation": parse_task_response(
                        "depth_estimation", answer, width, height, target=target
                    ),
                    "scope": "training_example_sanity_check_not_held_out_accuracy",
                },
            }
    raise ValueError(
        f"No standalone training example matching the {kind} depth template in {source}"
    )


def prepare_sample(
    task,
    data_root=DATA_ROOT,
    *,
    depth_kind="object",
    spatial_data_root=SPATIAL_DATA_ROOT,
):
    if task == "depth_estimation":
        return prepare_depth_sample(depth_kind, spatial_data_root)
    subset, prefix, suffix = SOURCES[task]
    source = sorted((data_root / subset).glob("*.jsonl"))[0]
    with source.open() as stream:
        line = next(stream)
    sample = json.loads(line)
    human = sample["conversations"][0]["value"]
    body = re.sub(r"^(?:<image> *)+", "", human)
    if prefix not in body or suffix not in body.split(prefix, 1)[1]:
        raise ValueError("Training prompt does not match the selected contract")
    target = body.split(prefix, 1)[1].split(suffix, 1)[0]
    image_paths = [Path(filename) for filename in sample["images"]]
    sizes = []
    for filename in image_paths:
        with Image.open(filename) as image:
            sizes.append(image.size)
    if not sizes or len(set(sizes)) != 1:
        raise ValueError("Expected readable frames with matching image sizes")
    width, height = sizes[-1]
    camera = None
    if TASKS[task].get("requires_camera"):
        matrix = re.search(r"intrinsic matrix K = (\[\[.*?\]\])\.", body)
        dimensions = re.search(r"image width (\d+) and height (\d+)", body)
        if not matrix or not dimensions:
            raise ValueError("Training example is missing explicit camera calibration")
        camera = validate_camera(
            {
                "intrinsic_K": literal_data(matrix.group(1)),
                "width": int(dimensions.group(1)),
                "height": int(dimensions.group(2)),
            },
            width,
            height,
        )
    prompt = build_task_prompt(
        task, target, camera=camera, frame_count=len(image_paths)
    )
    reconstructed = " ".join(["<image>"] * len(image_paths)) + prompt
    if reconstructed != human:
        raise ValueError(
            f"Generated prompt is not identical to the selected training example: {task}"
        )
    parsed_annotation = TASKS[task]["parse"](
        sample["conversations"][1]["value"], width, height
    )
    return {
        "task": task,
        "target": target,
        "camera": camera,
        "images": [str(path) for path in image_paths],
        "image_size": [width, height],
        "vlm_prompt": prompt,
        "template_exact_match": True,
        "frame_count": len(image_paths),
        "dataset": {
            "source": str(source),
            "line": 1,
            "id": sample.get("id"),
            "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "human": human,
            "annotation": sample["conversations"][1]["value"],
            "parsed_annotation": parsed_annotation,
            "scope": "training_example_sanity_check_not_held_out_accuracy",
        },
    }


def run_case(sample, args):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.session import EnvState
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.logging import init_output_dir

    from .tools import install_embodied_tools

    task = sample["task"]
    case = sample.get("case", task)
    output = args.output / case
    output.mkdir(parents=True, exist_ok=False)
    init_output_dir(output)
    (output / "dataset_reference.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False) + "\n"
    )
    report = {
        "task": task,
        "case": case,
        "checkpoint": args.checkpoint,
        "endpoint": args.endpoint,
        "started_at": datetime.now().astimezone().isoformat(),
        "passed": False,
        "real_vlm": True,
        "real_api_planner": False,
        "live_simulator": False,
        "frame_count": sample["frame_count"],
        "template_exact_match": sample["template_exact_match"],
        "vlm_prompt": sample["vlm_prompt"],
    }
    state = EnvState(output / "state")
    for frame_index, filename in enumerate(sample["images"]):
        with Image.open(filename) as opened:
            image = np.asarray(opened.convert("RGB"))
        with state.record_step(state={"source": filename, "frame_index": frame_index}):
            state.save("head_rgb.png", image)
            if sample["camera"]:
                state.save("head_camera_meta.json", sample["camera"])
    arguments = {"target": sample["target"], "view": "head"}
    arguments.update(
        {"steps": list(range(sample["frame_count"]))}
        if TASKS[task].get("video")
        else {"step": 0}
    )
    report["tool_arguments"] = arguments
    try:
        with EmbodiedClient(
            args.endpoint, checkpoint=args.checkpoint, timeout=args.timeout
        ) as client:
            report["vlm_metadata"] = client.metadata
            toolkit = Toolkit(
                dashboard_events=NullDashboardEventSink(),
                state=state,
                memory=MemoryManager(root=output / "memory"),
            )
            install_embodied_tools(toolkit, client, output)
            report["tool_schemas"] = [
                spec for spec in toolkit.get_tools_spec() if spec["name"] in TASKS
            ]
            result = toolkit.execute_tool(task, arguments)
            report["tool_result"] = result.result
            report["observation_steps_after"] = len(state.records())
            report["planner_content_blocks"] = result.content_blocks
            receipt = json.loads(result.content_blocks[0]["text"]) == result.result
            report["planner_content_roundtrip_exact"] = receipt
            report["passed"] = (
                result.result.get("success") is True
                and receipt
                and not result.is_finish
                and result.result.get("action_executed") is False
                and len(state.records()) == sample["frame_count"]
            )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["finished_at"] = datetime.now().astimezone().isoformat()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def run_planner_case(sample, args):
    from .probe import run_probe

    camera_path = None
    case = sample.get("case", sample["task"])
    if sample["camera"]:
        camera_path = args.output / (case + "_camera.json")
        camera_path.write_text(json.dumps(sample["camera"], indent=2) + "\n")
    output = args.output / case
    report = run_probe(
        SimpleNamespace(
            command="planner",
            tool=sample["task"],
            image=Path(sample["images"][0]),
            additional_image=[Path(path) for path in sample["images"][1:]],
            camera_meta=camera_path,
            target=sample["target"],
            view="head",
            output=output,
            endpoint=args.endpoint,
            checkpoint=args.checkpoint,
            timeout=args.timeout,
            planner_config=args.planner_config,
            model=getattr(args, "model", None),
            base_url=getattr(args, "base_url", None),
            reasoning_effort=getattr(args, "reasoning_effort", None),
            api_key_env=getattr(args, "api_key_env", None),
        )
    )
    (output / "dataset_reference.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False) + "\n"
    )
    return {**report, "task": sample["task"], "case": case}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19194")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--spatial-data-root", type=Path, default=SPATIAL_DATA_ROOT)
    parser.add_argument(
        "--depth-kinds",
        nargs="+",
        choices=list(SOURCES["depth_estimation"]),
        default=list(SOURCES["depth_estimation"]),
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--planner",
        action="store_true",
        help="Also call the configured external planner API to verify exact receipt",
    )
    from .probe import add_planner_args

    add_planner_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.audit_only and args.planner:
        parser.error("--audit-only and --planner cannot be combined")
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    cases = [
        (task, kind)
        for task in args.tasks
        for kind in (args.depth_kinds if task == "depth_estimation" else [None])
    ]
    for task, kind in cases:
        try:
            options = (
                {"depth_kind": kind, "spatial_data_root": args.spatial_data_root}
                if kind
                else {}
            )
            sample = prepare_sample(task, args.data_root, **options)
            report = (
                sample
                if args.audit_only
                else (
                    run_planner_case(sample, args)
                    if args.planner
                    else run_case(sample, args)
                )
            )
            report["passed"] = True if args.audit_only else report["passed"]
        except Exception as error:
            report = {
                "task": task,
                "case": task + "_" + kind if kind else task,
                "passed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        reports.append(report)
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in ("task", "case", "passed", "frame_count", "error")
                }
            ),
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
    )
    return 0 if all(report["passed"] for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
