"""Real VLM / optional real planner on a saved RGB observation; not a robot baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .client import EmbodiedClient
from .tasks import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOOL,
    TASKS,
    VIEWS,
    build_task_prompt,
    validate_inputs,
)


def add_planner_args(parser: argparse.ArgumentParser) -> None:
    """Add native API planner settings without a second repository's config."""
    parser.add_argument("--planner-config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--api-key-env", default=None)


def _planner_settings(args: argparse.Namespace) -> tuple[dict, dict]:
    """Resolve optional JSON settings and provider credentials from environment."""
    path = getattr(args, "planner_config", None)
    spec = json.loads(path.read_text())["planner"] if path else {}
    if any(name in spec for name in ("api_key", "token", "secret", "authorization")):
        raise ValueError(
            "Inline credentials are not allowed; export the provider API key"
        )
    model = getattr(args, "model", None) or os.environ.get(
        spec.get("model_env", "RPENT_MODEL"), spec.get("model", "")
    )
    if not model or not model.startswith(("openai:", "openai-chat:", "anthropic:")):
        raise ValueError(
            "Pass an openai:, openai-chat: or anthropic: --model for the planner probe"
        )
    base_url = getattr(args, "base_url", None) or os.environ.get(
        spec.get("base_url_env", "RPENT_BASE_URL"), spec.get("base_url", "")
    )
    if base_url:
        address = urlsplit(base_url)
        if (
            address.scheme not in {"http", "https"}
            or not address.hostname
            or any(
                (address.username, address.password, address.query, address.fragment)
            )
        ):
            raise ValueError(
                "Planner base URL must not contain credentials, query or fragment"
            )
    auth_name = (
        "ANTHROPIC_API_KEY" if model.startswith("anthropic:") else "OPENAI_API_KEY"
    )
    key_name = getattr(args, "api_key_env", None) or spec.get("api_key_env", auth_name)
    secret = os.environ.get(key_name, os.environ.get(auth_name, "")).strip()
    if not secret:
        raise ValueError(
            f"Export {key_name}; credential files from other applications are not loaded"
        )
    return {
        "model": model,
        "base_url": base_url or None,
        "reasoning_effort": getattr(args, "reasoning_effort", None)
        or spec.get("reasoning_effort", "none"),
    }, {auth_name: secret}


def run_probe(args):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.session import EnvState
    from rpent.tools.toolkit import Toolkit, readonly
    from rpent.utils.logging import init_output_dir

    from .tools import install_embodied_tools

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    init_output_dir(output)
    filenames = [args.image, *args.additional_image]
    images = []
    for filename in filenames:
        with Image.open(filename) as opened:
            images.append(np.asarray(opened.convert("RGB")))
    camera = None
    if args.camera_meta:
        metadata = json.loads(args.camera_meta.read_text())
        camera = {name: metadata[name] for name in ("intrinsic_K", "width", "height")}
    images, target, camera = validate_inputs(args.tool, images, args.target, camera)
    state = EnvState(output / "state")
    for frame_index, image in enumerate(images):
        with state.record_step(
            state={
                "source": "saved_real_observation",
                "view": args.view,
                "frame": frame_index,
            }
        ):
            state.save(f"{args.view}_rgb.png", image)
            if camera:
                state.save(f"{args.view}_camera_meta.json", camera)
    tool_input = {
        "target": target,
        "view": args.view,
        **(
            {"steps": list(range(len(images)))}
            if TASKS[args.tool].get("video")
            else {"step": 0}
        ),
    }
    report = {
        "kind": "embodied_tool_probe_not_robot_baseline",
        "real_vlm": True,
        "tool": args.tool,
        "tool_spec": TASKS[args.tool]["spec"],
        "vlm_prompt": build_task_prompt(
            args.tool, target, camera=camera, frame_count=len(images)
        ),
        "frame_count": len(images),
        "image_sources": [str(path.resolve()) for path in filenames],
        "real_planner": args.command == "planner",
        "live_simulator": False,
        "action_execution": False,
        "image_source": str(args.image.resolve()),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "passed": False,
    }
    transcript = []
    secret = ""

    class ProbeToolkit(Toolkit):
        def _register_common_tools(self):
            return None

        def __init__(self, client):
            super().__init__(
                dashboard_events=NullDashboardEventSink(),
                state=state,
                memory=MemoryManager(root=output / "memory"),
            )
            self.events = []
            self.last_detection = None
            self.received = False
            install_embodied_tools(self, client, output)
            self.add_tool(
                "report_received_perception",
                {
                    "name": "report_received_perception",
                    "description": "Copy the complete returned detections/regions/points/trajectories/depths list and provenance exactly.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "request_id": {"type": "string"},
                            "coordinate_space": {"type": "string"},
                            "predictions": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                            "view": {"type": "string"},
                            "step_idx": {"type": "integer"},
                        },
                        "required": [
                            "request_id",
                            "coordinate_space",
                            "predictions",
                            "view",
                            "step_idx",
                        ],
                        "additionalProperties": False,
                    },
                },
                self.receive_perception,
            )
            self.add_tool(
                "finish",
                {
                    "name": "finish",
                    "description": "Finish after the receipt was accepted.",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                self.finish,
            )

        def execute_tool(self, name, input_dict):
            result = super().execute_tool(name, input_dict)
            self.events.append(
                {
                    "tool": name,
                    "input": input_dict,
                    "result": result.result,
                    "planner_content_blocks": result.content_blocks,
                }
            )
            if name == args.tool and result.result.get("success"):
                self.last_detection = result.result
            return result

        @readonly
        def receive_perception(
            self, request_id, coordinate_space, predictions, view, step_idx
        ):
            response = self.last_detection
            if response:
                expected = next(
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
                self.received = (
                    bool(expected)
                    and predictions == expected
                    and request_id == response["request_id"]
                    and coordinate_space == response["coordinate_space"]
                    and view == response["view"]
                    and step_idx == response["step_idx"]
                )
            return {
                "accepted": self.received,
                "meaning": "Exact perception receipt, not robot task success",
            }

        @readonly
        def finish(self):
            return {
                "_finish": True,
                "success": self.received,
                "status": "success" if self.received else "failure",
            }

    with EmbodiedClient(
        args.endpoint, checkpoint=args.checkpoint, timeout=args.timeout
    ) as client:
        toolkit = ProbeToolkit(client)
        report["vlm_metadata"] = client.metadata
        try:
            if args.command == "detect":
                result = toolkit.execute_tool(args.tool, tool_input)
                report["passed"] = result.result.get("success") is True
                report["tool_result"] = result.result
            else:
                from rpent.planner.api_loop import ApiAgentLoop
                from rpent.planner.base import build_api_model

                planner, environment = _planner_settings(args)
                auth_name = (
                    "ANTHROPIC_API_KEY"
                    if planner["model"].startswith("anthropic:")
                    else "OPENAI_API_KEY"
                )
                secret = environment[auth_name]
                changed = {
                    name: environment[name]
                    for name in (auth_name, "NO_PROXY", "no_proxy")
                    if name in environment
                }
                previous = {name: os.environ.get(name) for name in changed}
                previous_logging = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                os.environ.update(changed)
                try:
                    model = build_api_model(planner["model"], planner["base_url"])
                    loop = ApiAgentLoop(
                        model=model,
                        max_tokens=4096,
                        reasoning_effort=planner["reasoning_effort"],
                        dashboard_events=NullDashboardEventSink(),
                        timeout_s=600,
                    )
                    receipt_instruction = (
                        "Copy the complete returned detections, regions, points, trajectories or depths list into "
                        "report_received_perception.predictions, plus request_id, coordinate_space, view and step_idx. "
                        "Keep every numeric value and field unchanged."
                    )
                    result = loop.solve(
                        system_prompt=(
                            "This is a perception tool integration test on a recorded robot image, not a robot task. "
                            f"First call {args.tool} with exactly the requested arguments. Read its returned JSON. "
                            + receipt_instruction
                            + " "
                            "Do not guess coordinates or invent a detection. If receipt is accepted, call finish."
                        ),
                        user_message=json.dumps(tool_input),
                        toolkit=toolkit,
                        max_turns=4,
                    )
                    transcript = result.messages
                    report.update(
                        planner=planner,
                        stats=result.stats,
                        error=result.error,
                        probe_limits={
                            "max_turns": 4,
                            "max_tokens": 4096,
                            "timeout_s": 600,
                        },
                        finish_result=result.finish_result,
                        planner_received_exact_perception=toolkit.received,
                        passed=toolkit.received
                        and bool(result.finish_result)
                        and not result.error,
                    )
                    if args.tool == DEFAULT_TOOL:
                        report["planner_received_exact_bbox"] = toolkit.received
                finally:
                    logging.disable(previous_logging)
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
            report["observation_steps_after"] = len(state.records())
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            for name, content in (
                ("report.json", report),
                ("tool_events.json", toolkit.events),
                ("planner_transcript.json", transcript),
            ):
                text = json.dumps(content, ensure_ascii=False, indent=2)
                if secret:
                    text = text.replace(secret, "<redacted>")
                (output / name).write_text(text + "\n")
    return json.loads((output / "report.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["detect", "planner"])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--additional-image",
        type=Path,
        action="append",
        default=[],
        help="Additional ordered video frames; repeated up to 63 times",
    )
    parser.add_argument(
        "--camera-meta",
        type=Path,
        help="Saved intrinsic_K/width/height JSON; required for 3D tools",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Referring expression; depth_estimation also accepts a JSON object/marker/point query",
    )
    parser.add_argument("--tool", choices=list(TASKS), default=DEFAULT_TOOL)
    parser.add_argument("--view", choices=VIEWS, default="head")
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19194")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--timeout", type=float, default=300)
    add_planner_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.tool == "depth_estimation" and args.target.lstrip().startswith("{"):
        try:
            args.target = json.loads(args.target)
        except json.JSONDecodeError as error:
            parser.error(f"Invalid depth query JSON: {error}")
    report = run_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
