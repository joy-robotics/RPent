import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robotwin import evaluate, tool_plugin
from robots.robotwin.prompt_bundle import system_prompt
from rpent.tools.toolkit import _is_readonly, readonly


class FakeToolkit:
    def __init__(self):
        self._tools = {
            "lingbot_act": ({"name": "lingbot_act"}, lambda: {"success": True})
        }

    def add_tool(self, name, spec, handler):
        self._tools[name] = (spec, handler)


@pytest.fixture
def plugin_fixture(tmp_path, monkeypatch):
    closed = []

    @readonly
    def perceive(target):
        if target == "bad":
            raise ValueError("invalid target")
        return {"success": True, "request_id": "prediction-1", "step_idx": 0}

    def install(toolkit, options, output):
        toolkit.add_tool("perceive", {"name": "perceive"}, perceive)
        return SimpleNamespace(close=lambda: closed.append(True))

    module = SimpleNamespace(
        __file__=str(tmp_path / "fake_plugin.py"),
        install=install,
        tool_names=lambda: ("perceive",),
        planner_prompt=lambda: (
            "Use perceive only when useful; it never moves the robot."
        ),
        source_files=lambda: [],
    )
    monkeypatch.setitem(sys.modules, "fake_perception_plugin", module)
    options = {
        "module": "fake_perception_plugin",
        "root": str(tmp_path),
        "endpoint": "ws://127.0.0.1:19194",
        "checkpoint": str(tmp_path),
        "timeout": 180,
    }
    return module, options, closed


def test_disabled_plugin_does_not_load_optional_packages(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled plugin must not import a module")

    monkeypatch.setattr(tool_plugin.importlib, "import_module", forbidden)
    assert tool_plugin.configuration(SimpleNamespace()) is None
    assert "READ-ONLY PERCEPTION TOOLS" not in system_prompt({})


def test_plugin_options_require_explicit_identity_and_checkpoint(tmp_path):
    parser = argparse.ArgumentParser()
    tool_plugin.add_cli_args(parser)
    with pytest.raises(ValueError, match="require --tool-plugin"):
        tool_plugin.configuration(
            parser.parse_args(["--tool-plugin-endpoint", "ws://localhost:1"])
        )
    with pytest.raises(ValueError, match="endpoint and checkpoint"):
        tool_plugin.configuration(parser.parse_args(["--tool-plugin", "test_plugin"]))
    arguments = [
        "--tool-plugin",
        "test_plugin",
        "--tool-plugin-root",
        str(tmp_path),
        "--tool-plugin-endpoint",
        "ws://localhost:19194",
        "--tool-plugin-checkpoint",
        str(tmp_path),
    ]
    options = tool_plugin.configuration(parser.parse_args(arguments))
    forwarded = parser.parse_args(tool_plugin.worker_arguments(options))
    assert tool_plugin.configuration(forwarded) == options
    with pytest.raises(ValueError, match="positive and finite"):
        tool_plugin.configuration(
            parser.parse_args(arguments + ["--tool-plugin-timeout", "nan"])
        )


def test_readonly_invocations_preserve_native_binding_and_record_errors(
    tmp_path, plugin_fixture
):
    module, options, closed = plugin_fixture
    toolkit = FakeToolkit()
    native = toolkit._tools["lingbot_act"]
    runtime = tool_plugin.ToolPluginRuntime(toolkit, options, tmp_path)
    handler = toolkit._tools["perceive"][1]
    assert _is_readonly(handler)
    assert toolkit._tools["lingbot_act"] is native
    assert handler(target="bell")["request_id"] == "prediction-1"
    with pytest.raises(ValueError, match="invalid target"):
        handler(target="bad")
    with pytest.raises(TypeError):
        handler(unexpected="wrong")
    usage = tool_plugin.read_usage(tmp_path)
    assert usage["calls"] == {"perceive": 3}
    assert usage["successes"] == {"perceive": 1}
    assert usage["failures"] == {"perceive": 2}
    assert usage["incomplete"] == 0
    runtime.close()
    runtime.close()
    assert closed == [True]


def test_registered_tools_are_exposed_in_the_native_prompt(plugin_fixture):
    module, options, closed = plugin_fixture
    sections = system_prompt({"strategy": "atomic-preferred", "tool_plugin": options})
    assert "ATOMIC PREFERENCE" in sections
    assert sections["READ-ONLY PERCEPTION TOOLS"] == module.planner_prompt()


def test_missing_invocation_end_is_visible_not_discarded(tmp_path):
    root = tmp_path / "tool_plugin"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"tools": ["perceive"]}))
    begin = {"event": "tool_begin", "call_id": "unfinished", "tool": "perceive"}
    (root / "events.jsonl").write_text(json.dumps(begin) + "\n")
    usage = tool_plugin.read_usage(tmp_path)
    assert usage["calls"] == {"perceive": 1}
    assert usage["incomplete"] == 1
    assert usage["successes"] == {"perceive": 0}


def test_plugin_cannot_replace_action_tools(tmp_path, plugin_fixture):
    module, options, closed = plugin_fixture

    def install(toolkit, options, output):
        toolkit.add_tool("lingbot_act", {"name": "lingbot_act"}, lambda: {})
        toolkit.add_tool("perceive", {"name": "perceive"}, readonly(lambda: {}))
        return SimpleNamespace(close=lambda: closed.append(True))

    module.install = install
    with pytest.raises(ValueError, match="must not replace"):
        tool_plugin.ToolPluginRuntime(FakeToolkit(), options, tmp_path)
    assert closed == [True]


def test_plugin_cannot_register_mutating_tools(tmp_path, plugin_fixture):
    module, options, closed = plugin_fixture

    def install(toolkit, options, output):
        toolkit.add_tool("perceive", {"name": "perceive"}, lambda: {})
        return SimpleNamespace(close=lambda: closed.append(True))

    module.install = install
    with pytest.raises(ValueError, match="read-only"):
        tool_plugin.ToolPluginRuntime(FakeToolkit(), options, tmp_path)
    assert closed == [True]


def test_task_report_separates_perception_from_motion_and_includes_timings(tmp_path):
    cell = {"task": "click_bell"}
    episode = {
        "cell": cell,
        "official_success": True,
        "loop_verified": True,
        "stop_reason": "native_success",
        "tool_calls": {"lingbot_act": 2, "move_to": 1},
        "trace_validation": {"executed_actions": 116},
        "elapsed_s": 250.0,
        "scheduler": {"wall_s": 335.0},
        "perception_tool_usage": {"calls": {"perceive": 3}},
    }
    summary = {"episodes": [episode], "active": {}, "perception_tools": ["perceive"]}
    evaluate.save_results(tmp_path, summary, [cell])
    report = json.loads((tmp_path / "task_summary.json").read_text())["tasks"][0]
    assert report["vla_calls"] == 2
    assert report["atomic_calls"] == 1
    assert report["perception_calls"] == report["perceive_calls"] == 3
    assert report["executed_actions"] == 116
    assert report["loop_seconds"] == 250.0
    assert report["worker_wall_seconds"] == 335.0


def test_resident_controller_freezes_builtin_tools_and_summarizes_usage(
    tmp_path, monkeypatch
):
    from robots.robotwin import resident_evaluate
    from robots.robotwin.embodied import plugin
    from robots.robotwin.rollout_pool import GPU

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)
    monkeypatch.setattr("rpent.utils.logging._log_initialized", True)
    gpu = GPU(0, "GPU-resident-plugin-test", 140000, 138000)
    monkeypatch.setattr(resident_evaluate, "discover_gpus", lambda: [gpu])
    monkeypatch.setattr(resident_evaluate.tempfile, "gettempdir", lambda: str(tmp_path))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"offline-test-checkpoint")
    monkeypatch.setattr(
        resident_evaluate.robot_spec,
        "_resolve_vla_runtime_path",
        lambda args: checkpoint,
    )
    monkeypatch.setattr(
        resident_evaluate.robot_spec, "_resolve_env_runtime_path", lambda args: tmp_path
    )

    class FakePool:
        def __init__(self, *args):
            self.endpoints = {gpu.uuid: "ws://localhost:1"}

        def start(self):
            pass

        def check(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(resident_evaluate, "PolicyPool", FakePool)
    monkeypatch.setattr(
        resident_evaluate.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(wait=lambda timeout: 0),
    )
    monkeypatch.setattr(resident_evaluate, "stop_process_group", lambda process: None)

    def forbidden_client(*args, **kwargs):
        raise AssertionError("The controller must not create a perception client")

    monkeypatch.setattr(plugin, "EmbodiedClient", forbidden_client)
    output = tmp_path / "evaluation"

    def run_pool(shards, slots, launch, **callbacks):
        assert len(shards) == len(slots) == 1
        shard, slot = shards[0], slots[0]
        job = launch(shard, slot)
        assert "robots.robotwin.resident_worker" in job.command
        callbacks["on_start"](shard, slot, 12345)
        directory = output / shard["job_id"]
        directory.mkdir()
        rows = [
            {
                "cell": cell,
                "official_success": False,
                "loop_verified": False,
                "perception_tool_usage": {"calls": {"grounding_2d": 1}},
            }
            for cell in shard["episodes"]
        ]
        (directory / "worker-result.json").write_text(json.dumps({"episodes": rows}))
        callbacks["on_finish"](
            {"job_id": shard["job_id"], "returncode": 0, "runner_error": None}
        )

    monkeypatch.setattr(resident_evaluate, "run_pool", run_pool)
    assert (
        resident_evaluate.main(
            [
                "--gpu-ids",
                "0",
                "--episodes-per-task",
                "2",
                "--model",
                "openai:offline-test",
                "--output-dir",
                str(output),
                "--tool-plugin",
                "robots.robotwin.embodied.plugin",
                "--tool-plugin-endpoint",
                "ws://localhost:19194",
                "--tool-plugin-checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["perception_tools"] == list(plugin.tool_names())
    assert {str(path.resolve()) for path in plugin.source_files()} <= set(
        plan["source_sha256"]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["completed_attempts"] == 2
    assert summary["official_successes"] == 0
    assert summary["perception_tool_calls"] == {
        name: 2 if name == "grounding_2d" else 0 for name in plugin.tool_names()
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_resident_worker_forwards_tools_and_closes_each_episode_client(
    tmp_path, monkeypatch, plugin_fixture, enabled
):
    from robots.robotwin import resident_worker, robot_spec
    from rpent.cli import main as cli_module

    _, options, closed = plugin_fixture
    calls, finished, runtime_closed = [], [], []
    configuration = {
        "runtime_cache_dir": str(tmp_path / "cache"),
        "reset_timeout_s": 1,
        "joybrain_timeout": 1,
        "vla_model_path": str(tmp_path),
        "task_config": "demo_randomized",
        "robotwin_assets_path": str(tmp_path),
        "max_episode_steps": 100,
        "strategy": "upstream",
        "model": "openai:offline-test",
        "max_turns": 2,
        "max_tokens": 1024,
        "planner_timeout_s": 1,
    }
    if enabled:
        configuration.update(
            tool_plugin=options["module"],
            tool_plugin_root=options["root"],
            tool_plugin_endpoint=options["endpoint"],
            tool_plugin_checkpoint=options["checkpoint"],
            tool_plugin_timeout=options["timeout"],
        )
    runtime = SimpleNamespace(
        initialize=lambda *args, **kwargs: None,
        finish=lambda directory: finished.append(directory),
        close=lambda: runtime_closed.append(True),
    )
    monkeypatch.setattr(resident_worker, "ResidentRuntime", lambda *args: runtime)
    monkeypatch.setattr(resident_worker, "reset_output_logging", lambda: None)

    def cli_main(arguments, *, runtime_initializer):
        calls.append(arguments)
        assert runtime_initializer == runtime.initialize
        parser = argparse.ArgumentParser()
        robot_spec._add_cli_args(parser, False)
        parsed, _ = parser.parse_known_args(arguments)
        actual = tool_plugin.configuration(parsed)
        assert actual == (options if enabled else None)
        directory = Path(arguments[arguments.index("--output-dir") + 1])
        if actual:
            toolkit = FakeToolkit()
            resource = tool_plugin.ToolPluginRuntime(toolkit, actual, directory)
            try:
                assert toolkit._tools["perceive"][1](target="object")["success"]
            finally:
                resource.close()
        return 0

    monkeypatch.setattr(cli_module, "main", cli_main)
    monkeypatch.setattr(
        evaluate,
        "inspect_episode",
        lambda directory, process: {
            "official_success": False,
            "loop_verified": False,
            "perception_tool_usage": tool_plugin.read_usage(directory),
        },
    )
    shard = {
        "gpu": {"uuid": "GPU-resident-plugin-test"},
        "episodes": [
            {"job_id": f"episode_{index}", "task": "click_bell", "seed": 100000 + index}
            for index in range(2)
        ],
    }
    plan = {
        "configuration": configuration,
        "output_dir": str(tmp_path),
        "source_sha256": {},
    }
    root = tmp_path / "worker"
    root.mkdir()
    assert resident_worker.run_worker(plan, shard, root, "ws://localhost:1") == 0
    assert len(calls) == len(finished) == 2
    assert closed == ([True, True] if enabled else [])
    assert runtime_closed == [True]
    result = json.loads((root / "worker-result.json").read_text())
    assert result["status"] == "complete"
    assert [row["perception_tool_usage"]["calls"] for row in result["episodes"]] == [
        {"perceive": 1} if enabled else {},
        {"perceive": 1} if enabled else {},
    ]
