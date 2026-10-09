import ast
import json
from pathlib import Path

import pytest

from robots.robotwin import evaluate
from robots.robotwin.rollout_pool import GPU


def test_500_jobs_have_ten_unique_seeds_per_task():
    manifest = json.loads(
        (Path(evaluate.__file__).parent / "eval/demo_randomized.json").read_text()
    )
    cells = evaluate.make_cells(manifest, "all", 10)
    assert len(cells) == len({(cell["task"], cell["seed"]) for cell in cells}) == 500
    for task in manifest["tasks"]:
        selected = [cell for cell in cells if cell["task"] == task]
        assert len({cell["seed"] for cell in selected}) == 10
    assert len({cell["task"] for cell in cells[:50]}) == 50


def test_explicit_seed_range_is_used_without_replacement():
    manifest = {"tasks": {"task": [{"seed": 100001}]}}
    cells = evaluate.make_cells(manifest, "task", 3, 200000)
    assert [cell["seed"] for cell in cells] == [200000, 200001, 200002]
    assert all(cell["seed_source"] == "fixed_not_expert_verified" for cell in cells)
    with pytest.raises(ValueError):
        evaluate.make_cells(manifest, "task,task", 1)


def test_dry_run_is_self_contained_and_does_not_launch_processes(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        evaluate, "discover_gpus", lambda: [GPU(0, "GPU-cpu-fixture", 160000, 150000)]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not launch processes")

    monkeypatch.setattr(evaluate.subprocess, "Popen", forbidden)
    assert (
        evaluate.main(
            [
                "--tasks",
                "all",
                "--episodes-per-task",
                "10",
                "--workers-per-gpu",
                "4",
                "--output-dir",
                str(tmp_path / "plan"),
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads((tmp_path / "plan/plan.json").read_text())
    assert len(plan["cells"]) == 500 and plan["gpu_count"] == 1
    assert plan["configuration"]["record_actions"] is True


def test_missing_gpu_never_aliases_existing_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        evaluate, "discover_gpus", lambda: [GPU(0, "GPU-cpu-fixture", 160000, 150000)]
    )
    with pytest.raises(ValueError, match="unavailable"):
        evaluate.main(
            ["--gpu-ids", "0,1", "--output-dir", str(tmp_path / "plan"), "--dry-run"]
        )
    assert not (tmp_path / "plan").exists()


def test_unknown_worker_result_is_not_reported_as_normal_failure(tmp_path):
    row = evaluate.inspect_episode(tmp_path, {"returncode": 1, "runner_error": None})
    assert row["official_success"] is None and row["error"]
    assert row["stop_reason"] == "initialization_or_worker_error"


def test_no_production_import_of_legacy_adapter():
    root = Path(evaluate.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(name.startswith("rpent_joybrain") for name in modules), path


def test_strategy_is_explicit_and_does_not_change_default_prompt():
    from robots.robotwin.prompt_bundle import system_prompt

    assert "ATOMIC PREFERENCE" not in system_prompt({"strategy": "upstream"})
    assert "ATOMIC PREFERENCE" in system_prompt({"strategy": "atomic-preferred"})


def test_recording_requires_owned_joybrain_evaluation():
    from robots.robotwin import robot_spec
    from rpent.cli.main import _build_argparser

    parser = _build_argparser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--robot",
            "robotwin",
            "--task-name",
            "click_bell",
            "--seed",
            "0",
            "--record-actions",
        ]
    )
    with pytest.raises(ValueError, match="Joy-Brain"):
        robot_spec._parse_config(args)


def test_controller_builds_native_cli_workers_without_legacy_wrapper(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from robots.robotwin import robot_spec
    from rpent.cli.main import _build_argparser

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    gpu = GPU(0, f"CPU-fixture-{tmp_path.name}", 160000, 150000)
    monkeypatch.setattr(evaluate, "discover_gpus", lambda: [gpu])
    monkeypatch.setattr(evaluate, "init_output_dir", lambda path: None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"unit-test-only")
    monkeypatch.setattr(
        robot_spec, "_resolve_vla_runtime_path", lambda args: checkpoint
    )
    monkeypatch.setattr(robot_spec, "_resolve_env_runtime_path", lambda args: tmp_path)
    closes = []
    model = SimpleNamespace(
        endpoint="ws://127.0.0.1:12345",
        metadata={"ckpt_path": str(checkpoint)},
        close=lambda: closes.append(True),
    )
    monkeypatch.setattr(
        robot_spec, "_init_runtime", lambda *args: ([], {"model": model})
    )
    monkeypatch.setattr(
        evaluate.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(wait=lambda timeout: 0),
    )
    monkeypatch.setattr(evaluate, "stop_process_group", lambda process: None)
    monkeypatch.setattr(
        evaluate,
        "inspect_episode",
        lambda directory, process: {
            "official_success": True,
            "loop_verified": True,
            "stop_reason": "native_success",
            "tool_calls": {"lingbot_act": 1},
        },
    )

    def run(jobs, slots, build_launch, **callbacks):
        launch = build_launch(jobs[0], slots[0])
        assert launch.command[1:3] == ["-m", "rpent.cli.main"]
        assert not any("rpent_joybrain" in arg for arg in launch.command)
        parser = _build_argparser()
        robot_spec._add_cli_args(parser, False)
        native_args = parser.parse_args(launch.command[3:])
        config = robot_spec._parse_config(native_args)
        assert config.task_desc["record_actions"] is True
        metadata = json.loads(native_args.recording_config.read_text())
        assert metadata["policy_metadata"]["ckpt_path"] == str(checkpoint)
        callbacks["on_start"](jobs[0], slots[0], 1)
        callbacks["on_finish"](
            {"job_id": jobs[0]["job_id"], "returncode": 0, "runner_error": None}
        )
        return []

    monkeypatch.setattr(evaluate, "run_pool", run)
    output = tmp_path / "run"
    assert (
        evaluate.main(["--model", "openai:unit-test-only", "--output-dir", str(output)])
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["completed_attempts"] == 1
    assert closes
