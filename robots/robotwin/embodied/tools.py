"""Register perception as ordinary read-only RPent tools on synchronized EnvState frames."""

from __future__ import annotations

import hashlib
import io
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rpent.tools.toolkit import readonly

from .tasks import MAX_VIDEO_FRAMES, TASKS, VIEWS, validate_camera, validate_task_target


def project_point(point, camera):
    if point[2] <= 0:
        return None
    matrix = camera["intrinsic_K"]
    pixel = (
        matrix[0][0] * point[0] / point[2] + matrix[0][2],
        matrix[1][1] * point[1] / point[2] + matrix[1][2],
    )
    if (
        all(math.isfinite(number) for number in pixel)
        and 0 <= pixel[0] < camera["width"]
        and 0 <= pixel[1] < camera["height"]
    ):
        return pixel
    return None


def write_overlay(image, result, path):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    def mark(point, label=""):
        if point is not None:
            horizontal, vertical = point
            draw.ellipse(
                (horizontal - 3, vertical - 3, horizontal + 3, vertical + 3), fill="red"
            )
            draw.text((horizontal, max(0, vertical - 12)), label, fill="red")

    for detection in result.get("detections", []) + result.get("regions", []):
        if "bbox_xyxy_px" in detection:
            box = detection["bbox_xyxy_px"]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], max(0, box[1] - 12)), detection["label"], fill="red")
        elif "center_xyz_m" in detection:
            mark(
                project_point(detection["center_xyz_m"], result["camera"]),
                detection["label"],
            )
    for point in result.get("points", []):
        mark(point["point_xy_px"], point["label"])
    for trajectory in result.get("trajectories", []):
        points = trajectory.get("points_xy_px")
        if points is None:
            points = [
                project_point(point, result["camera"])
                for point in trajectory["points_xyz_m"]
            ]
        for previous, current in zip(points, points[1:]):
            if previous is not None and current is not None:
                draw.line([tuple(previous), tuple(current)], fill="red", width=2)
        for point in points:
            mark(point)
    for depth in result.get("depths", []):
        label = f"VLM estimate: {depth['depth_m']:g} m (not sensor depth)"
        draw.text((8, 8), label, fill="red")
        if "query_point_xy_px" in depth:
            mark(depth["query_point_xy_px"])
    overlay.save(path)


class EmbodiedTools:
    def __init__(self, toolkit, client, output_root):
        self.toolkit = toolkit
        self.client = client
        self.output_root = Path(output_root) / "embodied_tools"

    @readonly
    def grounding_2d(self, target, view="head", step=-1):
        return self._infer("grounding_2d", target, view, [step])

    @readonly
    def affordance_2d(self, target, view="head", step=-1):
        return self._infer("affordance_2d", target, view, [step])

    @readonly
    def pointing_2d(self, target, view="head", step=-1):
        return self._infer("pointing_2d", target, view, [step])

    @readonly
    def trace_2d(self, target, view="head", step=-1):
        return self._infer("trace_2d", target, view, [step])

    @readonly
    def trace_3d(self, target, view="head", step=-1):
        return self._infer("trace_3d", target, view, [step])

    @readonly
    def grounding_3d(self, target, view="head", step=-1):
        return self._infer("grounding_3d", target, view, [step])

    @readonly
    def depth_estimation(self, target, view="head", step=-1):
        return self._infer("depth_estimation", target, view, [step])

    @readonly
    def video_grounding_3d(self, target, steps, view="head"):
        if (
            not isinstance(steps, list)
            or not 2 <= len(steps) <= MAX_VIDEO_FRAMES
            or any(type(step) is not int or step < 0 for step in steps)
            or any(previous >= current for previous, current in zip(steps, steps[1:]))
        ):
            raise ValueError(
                "steps must contain 2..64 strictly increasing nonnegative recorded indices"
            )
        return self._infer("video_grounding_3d", target, view, steps)

    def _infer(self, task, target, view, steps):
        target = validate_task_target(task, target)
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        if any(type(step) is not int or step < -1 for step in steps):
            raise ValueError("step must be -1 or a nonnegative integer")
        self.toolkit.raise_if_cancelled()
        artifact = f"{view}_rgb.png"
        records, sources, images = [], [], []
        camera = None
        for step in steps:
            record = self.toolkit.state.get(step)
            if artifact not in record.artifacts:
                raise ValueError(f"No {view} RGB image in step {record.step_idx}")
            image_bytes = self.toolkit.state.load_bytes(artifact, step=record.step_idx)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
            camera_artifact = f"{view}_camera_meta.json"
            if TASKS[task].get("requires_camera") or (
                TASKS[task].get("optional_camera")
                and camera_artifact in record.artifacts
            ):
                if camera_artifact not in record.artifacts:
                    raise ValueError(
                        f"No saved {view} camera calibration in step {record.step_idx}"
                    )
                metadata = json.loads(
                    self.toolkit.state.load_bytes(camera_artifact, step=record.step_idx)
                )
                selected = {
                    name: metadata[name] for name in ("intrinsic_K", "width", "height")
                }
                selected = validate_camera(selected, image.width, image.height)
                if camera is not None and selected != camera:
                    raise ValueError(
                        "Video requires shared intrinsics/dimensions across all selected frames"
                    )
                camera = selected
            records.append(record)
            sources.append(image_bytes)
            images.append(np.asarray(image))
        arguments = {"camera": camera} if camera else {}
        response = self.client.infer(
            images if TASKS[task].get("video") else images[-1],
            target,
            task=task,
            **arguments,
        )
        self.toolkit.raise_if_cancelled()
        self.output_root.mkdir(parents=True, exist_ok=True)
        stem = response["request_id"]
        result = {key: value for key, value in response.items() if key != "raw_text"}
        result.update(
            success=True,
            step_idx=records[-1].step_idx,
            view=view,
            step_indices=[record.step_idx for record in records],
            source_image_sha256=hashlib.sha256(sources[-1]).hexdigest(),
            source_image_sha256s=[
                hashlib.sha256(source).hexdigest() for source in sources
            ],
            action_executed=False,
            warning="Perception estimate, not ground truth, an executable action, a collision check, or official task success.",
        )
        if camera:
            result.update(
                camera=camera,
                reference_step_idx=records[-1].step_idx,
                overlay_kind="3D center/trace projection only; not an oriented 3D box rendering",
            )
        else:
            result["overlay_kind"] = "2D geometry in source image pixels"
        if "depths" in result:
            result["overlay_kind"] = (
                "Estimated scalar depth label and optional query point; not a dense depth map"
            )
            result["calibration_used"] = camera is not None
        (self.output_root / f"{stem}_input.png").write_bytes(sources[-1])
        if len(sources) > 1:
            for frame_index, source in enumerate(sources):
                (
                    self.output_root / f"{stem}_frame_{frame_index:03d}_input.png"
                ).write_bytes(source)
        write_overlay(image, result, self.output_root / f"{stem}_overlay.png")
        (self.output_root / f"{stem}.json").write_text(
            json.dumps(
                {**result, "raw_text": response["raw_text"]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return result


def install_embodied_tools(toolkit, client, output_root):
    tools = EmbodiedTools(toolkit, client, output_root)
    for name, task in TASKS.items():
        if any(spec["name"] == name for spec in toolkit.get_tools_spec()):
            raise ValueError(f"Refusing to replace existing robot tool: {name}")
        toolkit.add_tool(name, deepcopy(task["spec"]), getattr(tools, name))
    return tools
