import threading
from contextlib import contextmanager

import numpy as np
import pytest
from websockets.sync.server import serve

from robots.robotwin.embodied.client import EmbodiedClient
from robots.robotwin.embodied.server import EmbodiedService
from robots.robotwin.embodied.tasks import PROTOCOL, TASKS, prompt_fingerprint


@pytest.fixture
def backend(tmp_path):
    class FakeBackend:
        metadata = {
            "protocol": PROTOCOL,
            "checkpoint": str(tmp_path),
            "task_prompts": {name: prompt_fingerprint(name) for name in TASKS},
        }
        response = [{"label": "unit-test box", "bbox_2d": [(100, 200, 400, 600)]}]
        calls = []

        def predict(self, task, images, target, camera=None):
            self.calls.append(
                (task, [image.copy() for image in images], target, camera)
            )
            return (
                self.response if isinstance(self.response, str) else repr(self.response)
            )

    return FakeBackend()


@contextmanager
def running_server(backend):
    service = EmbodiedService(backend)
    with serve(
        service.handle, "127.0.0.1", 0, compression=None, ping_interval=None
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_real_wire_roundtrip_keeps_perception_separate_from_actions(backend):
    with running_server(backend) as endpoint:
        with EmbodiedClient(
            endpoint, checkpoint=backend.metadata["checkpoint"], timeout=5
        ) as client:
            result = client.infer(np.zeros((240, 320, 3), dtype=np.uint8), "bell")
    assert result["detections"][0]["bbox_xyxy_px"] == [32, 48, 128, 144]
    assert len(backend.calls) == 1
    assert "actions" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "2.1"),
        ("checkpoint", "/another/model"),
        ("task_prompts", {"grounding_2d": "unreviewed-prompt"}),
    ],
)
def test_model_protocol_and_prompt_binding(backend, field, value):
    checkpoint = backend.metadata["checkpoint"]
    backend.metadata[field] = value
    with running_server(backend) as endpoint:
        with pytest.raises(ValueError):
            EmbodiedClient(endpoint, checkpoint=checkpoint, timeout=5)


def test_invalid_model_response_fails_without_fabricating_a_box(backend):
    backend.response = [{"label": "bad", "bbox_2d": [(900, 0, 100, 500)]}]
    with running_server(backend) as endpoint:
        with EmbodiedClient(
            endpoint, checkpoint=backend.metadata["checkpoint"], timeout=5
        ) as client:
            with pytest.raises(RuntimeError, match="positive width"):
                client.infer(np.zeros((240, 320, 3), dtype=np.uint8), "bell")
            with pytest.raises(RuntimeError, match="closed"):
                client.infer(np.zeros((240, 320, 3), dtype=np.uint8), "bell")
    assert len(backend.calls) == 1


def test_native_embodied_tools_are_opt_in_and_fingerprinted(
    backend, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from robots.robotwin import tool_plugin
    from robots.robotwin.embodied import plugin
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.tools.toolkit import Toolkit

    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)
    assert tool_plugin.configuration(SimpleNamespace()) is None
    with running_server(backend) as endpoint:
        options = tool_plugin.configuration(
            SimpleNamespace(
                tool_plugin="robots.robotwin.embodied.plugin",
                tool_plugin_root=None,
                tool_plugin_endpoint=endpoint,
                tool_plugin_checkpoint=backend.metadata["checkpoint"],
                tool_plugin_timeout=5,
            )
        )
        toolkit = Toolkit(
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
        )
        resource = plugin.install(toolkit, options, tmp_path)
        try:
            assert resource.metadata == backend.metadata
            assert set(plugin.tool_names()) <= {
                spec["name"] for spec in toolkit.get_tools_spec()
            }
            names = {path.name for path in plugin.source_files()}
            assert {
                "tasks.py",
                "grounding_2d.py",
                "depth_estimation.py",
                "video_grounding_3d.py",
            } <= names
            assert all(path.is_file() for path in plugin.source_files())
        finally:
            resource.close()
    assert not backend.calls


@pytest.mark.parametrize(
    "task,field,geometry",
    [
        ("grounding_2d", "bbox_2d", [(100, 200, 400, 600)]),
        ("affordance_2d", "afford_2d", [(100, 200, 400, 600)]),
        ("pointing_2d", "point_2d", [(100, 200)]),
        ("trace_2d", "trace_2d", [(100, 200), (200, 300)]),
        ("trace_3d", "trace_3d", [(0, 0, 1), (0.1, 0.2, 1)]),
        ("grounding_3d", "bbox_3d", [(0, 0, 1, 0.1, 0.2, 0.3, 0, 0, 0)]),
        ("video_grounding_3d", "bbox_3d", [(0, 0, -1, 0.1, 0.2, 0.3, 0, 0, 0)]),
    ],
)
def test_all_tasks_over_real_websocket(backend, task, field, geometry):
    backend.response = [{field: geometry, "label": "target"}]
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    camera = (
        {
            "intrinsic_K": [[100, 0, 160], [0, 100, 120], [0, 0, 1]],
            "width": 320,
            "height": 240,
        }
        if TASKS[task].get("requires_camera")
        else None
    )
    images = [image, image + 1] if TASKS[task].get("video") else image
    with running_server(backend) as endpoint:
        with EmbodiedClient(
            endpoint, checkpoint=backend.metadata["checkpoint"], timeout=5
        ) as client:
            result = client.infer(images, "target", task=task, camera=camera)
    assert result["found"] and "actions" not in result
    assert result["frame_count"] == (2 if TASKS[task].get("video") else 1)
    assert backend.calls[0][3] == camera
    if TASKS[task].get("video"):
        assert backend.calls[0][1][1][0, 0, 0] == 1


@pytest.mark.parametrize(
    "target,answer",
    [
        ("the mug", "The mug is about 25 centimeters from the camera."),
        ("the mug", "0.25"),
        (
            {"kind": "marker", "marker": "A"},
            "The point labeled A is about 0.25 meters from the camera.",
        ),
        (
            {"kind": "point", "point_xy_1000": [500, 400]},
            "The surface at (500, 400) is about 25 centimeters from the camera.",
        ),
    ],
)
@pytest.mark.parametrize("calibrated", [False, True])
def test_depth_queries_over_real_websocket(backend, target, answer, calibrated):
    backend.response = answer
    camera = (
        {
            "intrinsic_K": [[100, 0, 160], [0, 100, 120], [0, 0, 1]],
            "width": 320,
            "height": 240,
        }
        if calibrated
        else None
    )
    with running_server(backend) as endpoint:
        with EmbodiedClient(
            endpoint, checkpoint=backend.metadata["checkpoint"], timeout=5
        ) as client:
            result = client.infer(
                np.zeros((240, 320, 3), dtype=np.uint8),
                target,
                task="depth_estimation",
                camera=camera,
            )
    assert result["depths"][0]["depth_m"] == 0.25
    assert result["depths"][0]["query"] == backend.calls[0][2]
    assert result["target"] == backend.calls[0][2] and backend.calls[0][3] == camera
    assert (
        result["coordinate_space"] == "camera_distance"
        and not result["dense_depth_map"]
    )
    if answer == "0.25":
        assert result["depths"][0]["unit_source"] == "request_prompt"
        assert result["depths"][0]["reported_unit"] is None


def test_wrong_depth_point_response_is_rejected_over_wire(backend):
    backend.response = "The surface at (501, 400) is about 0.25 meters from the camera."
    with running_server(backend) as endpoint:
        with EmbodiedClient(
            endpoint, checkpoint=backend.metadata["checkpoint"], timeout=5
        ) as client:
            with pytest.raises(RuntimeError, match="Reported point differs"):
                client.infer(
                    np.zeros((240, 320, 3), dtype=np.uint8),
                    {"kind": "point", "point_xy_1000": [500, 400]},
                    task="depth_estimation",
                )
