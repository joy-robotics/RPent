import ast
import json
from pathlib import Path

import pytest

from robots.robotwin import evaluate
from robots.robotwin.rollout_pool import GPU


@pytest.mark.parametrize(
    "row,kind",
    [
        (
            {"trace_error": "missing transition", "stop_reason": "planner_stop"},
            "recording_error",
        ),
        (
            {
                "error": "invalid scene seed",
                "stop_reason": "initialization_or_worker_error",
            },
            "initialization_or_worker_error",
        ),
        ({"error": "model HTTP 500", "stop_reason": "planner_error"}, "planner_error"),
        ({"error": "timed out", "stop_reason": "planner_timeout"}, "planner_timeout"),
    ],
)
def test_episode_errors_are_explicitly_skipped_without_replay(row, kind):
    disposition = evaluate.error_disposition(row, "continue")
    assert disposition == {
        "kind": kind,
        "action": "skip_and_continue",
        "replay": False,
        "artifacts_preserved": True,
    }
    assert evaluate.error_disposition(row, "stop")["action"] == "drain_and_stop"


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "OutOfMemoryError",
        "No space left on device",
        "Disk quota exceeded",
    ],
)
def test_resource_failures_cannot_be_skipped(message):
    assert (
        evaluate.error_disposition({"error": message}, "continue")["action"]
        == "drain_and_stop"
    )


def test_native_task_failure_is_not_an_exception():
    assert (
        evaluate.error_disposition(
            {"official_success": False, "stop_reason": "planner_stop", "error": None},
            "continue",
        )
        is None
    )


def test_clean_requires_its_own_manifest_before_gpu_work(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main(
            [
                "--task-config",
                "demo_clean",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("track", ["demo_clean", "demo_randomized"])
def test_explicit_track_manifest_preserves_seeds_and_provenance(
    tmp_path, monkeypatch, track
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        evaluate, "discover_gpus", lambda: [GPU(0, "GPU-cpu", 160000, 150000)]
    )
    monkeypatch.setattr(
        evaluate.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry-run launched a service"),
    )
    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "task_config": track,
                "seed_source": "upstream_expert_validated_run",
                "tasks": {"click_bell": [{"seed": 4300001}, {"seed": 4300004}]},
            }
        )
    )
    output = tmp_path / "run"
    assert (
        evaluate.main(
            [
                "--task-config",
                track,
                "--eval-manifest",
                str(manifest),
                "--episodes-per-task",
                "2",
                "--dry-run",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert [cell["seed"] for cell in plan["cells"]] == [4300001, 4300004]
    assert {cell["seed_source"] for cell in plan["cells"]} == {
        "upstream_expert_validated_run"
    }
    assert plan["seed_manifest"]["task_config"] == track
    assert plan["configuration"]["eval_manifest"] == str(manifest)
    assert len(plan["seed_manifest"]["sha256"]) == 64


@pytest.mark.parametrize("seeds", [[1, 1], [-1], [True], ["100000"]])
def test_seed_manifest_rejects_invalid_entries(tmp_path, seeds):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "task_config": "demo_clean",
                "tasks": {"task": [{"seed": value} for value in seeds]},
            }
        )
    )
    with pytest.raises(ValueError):
        evaluate.load_eval_manifest(path, "demo_clean")


def test_seed_manifest_cannot_be_used_for_another_track(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {"task_config": "demo_randomized", "tasks": {"task": [{"seed": 100000}]}}
        )
    )
    with pytest.raises(ValueError, match="task_config"):
        evaluate.load_eval_manifest(path, "demo_clean")


@pytest.mark.parametrize("value", [[], "not a manifest", None])
def test_seed_manifest_requires_an_object(tmp_path, value):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="JSON object"):
        evaluate.load_eval_manifest(path, "demo_clean")


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


def test_measured_gpu_limits_allow_small_card_without_disabling_protection(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    gpu = GPU(0, "GPU-small-fixture", 32607, 31000)
    monkeypatch.setattr(evaluate, "discover_gpus", lambda: [gpu])
    arguments = ["--output-dir", str(tmp_path / "plan"), "--dry-run"]
    with pytest.raises(RuntimeError, match="required=38000"):
        evaluate.main(arguments)
    assert (
        evaluate.main(
            arguments
            + [
                "--gpu-start-min-free-mib",
                "30000",
                "--gpu-runtime-min-free-mib",
                "2048",
            ]
        )
        == 0
    )
    plan = json.loads((tmp_path / "plan/plan.json").read_text())
    assert plan["configuration"]["gpu_start_min_free_mib"] == 30000
    assert plan["configuration"]["gpu_runtime_min_free_mib"] == 2048
    evaluate.require_gpu_headroom([gpu], [GPU(0, gpu.uuid, 32607, 2048)], 2048)
    with pytest.raises(RuntimeError, match="required=2048"):
        evaluate.require_gpu_headroom([gpu], [GPU(0, gpu.uuid, 32607, 2047)], 2048)
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluate.require_gpu_headroom([gpu], [], 2048)


@pytest.mark.parametrize("startup,runtime", [(30000, 0), (30000, 1023), (1024, 2048)])
def test_gpu_memory_limits_cannot_disable_safety(tmp_path, startup, runtime):
    with pytest.raises(SystemExit):
        evaluate.main(
            [
                "--output-dir",
                str(tmp_path / "plan"),
                "--dry-run",
                "--gpu-start-min-free-mib",
                str(startup),
                "--gpu-runtime-min-free-mib",
                str(runtime),
            ]
        )
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize(
    "layout,workers,models,capacity", [("colocated", 1, 8, 8), ("split", 5, 4, 20)]
)
def test_eight_card_dry_run_records_exact_layout(
    tmp_path, monkeypatch, layout, workers, models, capacity
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    inventory = [GPU(index, f"GPU-{index}", 32607, 32110) for index in range(8)]
    monkeypatch.setattr(evaluate, "discover_gpus", lambda: inventory)
    monkeypatch.setattr(
        evaluate.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry run started a process"),
    )
    output = tmp_path / "plan"
    assert (
        evaluate.main(
            [
                "--gpu-ids",
                "0,1,2,3,4,5,6,7",
                "--gpu-layout",
                layout,
                "--workers-per-gpu",
                str(workers),
                "--gpu-start-min-free-mib",
                "30052",
                "--gpu-runtime-min-free-mib",
                "2048",
                "--dry-run",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["gpu_count"] == 8
    assert plan["model_server_count"] == models
    assert plan["worker_capacity"] == plan["max_active_planners"] == capacity
    assert plan["gpu_layout"] == layout
    pairs = [
        (group["model_gpu"]["index"], group["env_gpu"]["index"])
        for group in plan["gpu_groups"]
    ]
    assert pairs == (
        [(index, index) for index in range(8)]
        if layout == "colocated"
        else [(0, 1), (2, 3), (4, 5), (6, 7)]
    )


@pytest.mark.parametrize(
    "extra,error",
    [
        (["--gpu-ids", "0,1,2"], ValueError),
        (["--gpu-ids", "0,1", "--max-active-planners", "6"], SystemExit),
    ],
)
def test_split_layout_rejects_odd_cards_and_excess_planners(
    tmp_path, monkeypatch, extra, error
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        evaluate,
        "discover_gpus",
        lambda: [GPU(index, f"GPU-{index}", 160000, 150000) for index in range(3)],
    )
    with pytest.raises(error):
        evaluate.main(
            [
                "--gpu-layout",
                "split",
                "--workers-per-gpu",
                "5",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "plan"),
                *extra,
            ]
        )
    assert not (tmp_path / "plan").exists()


def test_split_layout_respects_visibility_and_borrowed_service_scope(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        evaluate,
        "discover_gpus",
        lambda: [GPU(index, f"GPU-{index}", 160000, 150000) for index in range(4)],
    )
    arguments = [
        "--gpu-layout",
        "split",
        "--gpu-ids",
        "0,1",
        "--dry-run",
        "--output-dir",
        str(tmp_path / "one"),
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="visibility|outside"):
        evaluate.main(arguments)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert evaluate.main([*arguments, "--vla-endpoint", "ws://127.0.0.1:12345"]) == 0
    with pytest.raises(SystemExit):
        evaluate.main(
            [
                *arguments,
                "--gpu-ids",
                "0,1,2,3",
                "--output-dir",
                str(tmp_path / "two"),
                "--vla-endpoint",
                "ws://127.0.0.1:12345",
            ]
        )


def test_startup_timeout_is_not_reported_as_planner_timeout(tmp_path):
    row = evaluate.inspect_episode(
        tmp_path, {"returncode": -15, "runner_error": "startup_timeout"}
    )
    assert row["stop_reason"] == "worker_startup_timeout"
    assert row["official_success"] is None


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


@pytest.mark.parametrize("image_limit", [None, 48])
def test_controller_builds_native_cli_workers_without_legacy_wrapper(
    tmp_path, monkeypatch, image_limit
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

    class Pool:
        def __init__(self, groups, args, output, cache_root, check_resources):
            self.endpoints = {gpu.uuid: "ws://127.0.0.1:12345"}
            self.metadata = {gpu.uuid: {"ckpt_path": str(checkpoint)}}
            self.check = check_resources

        def start(self):
            pass

        def close(self):
            closes.append(True)

    monkeypatch.setattr(evaluate, "PolicyPool", Pool)
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
        clock = SimpleNamespace(monotonic=lambda: 100.0, time_ns=evaluate.time.time_ns)
        monkeypatch.setattr(evaluate, "time", clock)
        monkeypatch.setattr(
            evaluate,
            "discover_gpus",
            lambda: [GPU(gpu.index, gpu.uuid, gpu.total_mib, 3000)],
        )
        assert callbacks["should_stop"]() is False
        clock.monotonic = lambda: 200.0
        monkeypatch.setattr(
            evaluate,
            "discover_gpus",
            lambda: [GPU(gpu.index, gpu.uuid, gpu.total_mib, 2047)],
        )
        with pytest.raises(RuntimeError, match="required=2048"):
            callbacks["should_stop"]()
        launch = build_launch(jobs[0], slots[0])
        assert launch.command[1:3] == ["-m", "rpent.cli.main"]
        assert not any("rpent_joybrain" in arg for arg in launch.command)
        parser = _build_argparser()
        robot_spec._add_cli_args(parser, False)
        native_args = parser.parse_args(launch.command[3:])
        assert native_args.api_image_limit == image_limit
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
        evaluate.main(
            [
                "--model",
                "openai:unit-test-only",
                "--gpu-runtime-min-free-mib",
                "2048",
                "--output-dir",
                str(output),
                *(
                    ["--api-image-limit", str(image_limit)]
                    if image_limit is not None
                    else []
                ),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["completed_attempts"] == 1
    assert closes


@pytest.mark.parametrize(
    "layout,workers,capacity", [("colocated", 1, 8), ("split", 5, 20)]
)
def test_native_workers_receive_environment_gpu_and_group_policy(
    tmp_path, monkeypatch, layout, workers, capacity
):
    from types import SimpleNamespace

    from robots.robotwin import robot_spec
    from rpent.cli.main import _build_argparser

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    inventory = [
        GPU(index, f"{tmp_path.name}-GPU-{index}", 32607, 32110) for index in range(8)
    ]
    monkeypatch.setattr(evaluate, "discover_gpus", lambda: inventory)
    monkeypatch.setattr(evaluate, "init_output_dir", lambda path: None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"unit-test-only")
    monkeypatch.setattr(
        robot_spec, "_resolve_vla_runtime_path", lambda args: checkpoint
    )
    monkeypatch.setattr(robot_spec, "_resolve_env_runtime_path", lambda args: tmp_path)
    groups_seen = []

    class Pool:
        def __init__(self, groups, args, output, cache_root, check_resources):
            groups_seen.extend(groups)
            self.endpoints = {
                group.model_gpu.uuid: f"ws://127.0.0.1:{12000 + group.model_gpu.index}"
                for group in groups
            }
            self.metadata = {
                group.model_gpu.uuid: {
                    "ckpt_path": str(checkpoint),
                    "owner_gpu": group.model_gpu.uuid,
                }
                for group in groups
            }
            self.check = check_resources

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(evaluate, "PolicyPool", Pool)
    monkeypatch.setattr(
        evaluate.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(wait=lambda timeout: 0),
    )
    monkeypatch.setattr(evaluate, "stop_process_group", lambda process: None)
    monkeypatch.setattr(
        evaluate,
        "inspect_episode",
        lambda *args: {
            "official_success": False,
            "loop_verified": True,
            "stop_reason": "planner_stop",
            "tool_calls": {"lingbot_act": 1},
        },
    )

    def run(jobs, slots, build_launch, **callbacks):
        assert len(slots) == capacity
        assignments = {
            group.env_gpu.uuid: group.model_gpu.uuid for group in groups_seen
        }
        caches = set()
        for index, job in enumerate(jobs):
            slot = slots[index % len(slots)]
            launch = build_launch(job, slot)
            parser = _build_argparser()
            robot_spec._add_cli_args(parser, False)
            worker_args = parser.parse_args(launch.command[3:])
            metadata = json.loads(worker_args.recording_config.read_text())
            assert (
                worker_args.env_cuda_device
                == launch.environment["CUDA_VISIBLE_DEVICES"]
                == slot.gpu_uuid
            )
            assert slot.policy_gpu_uuid == assignments[slot.gpu_uuid]
            assert metadata["policy_metadata"]["owner_gpu"] == slot.policy_gpu_uuid
            assert metadata["assignment"]["model_gpu_uuid"] == slot.policy_gpu_uuid
            assert worker_args.vla_endpoint == slot.endpoint
            assert worker_args.joybrain_timeout == 600
            assert (
                launch.ready_path == worker_args.recording_config.parent / "states.json"
            )
            assert launch.startup_timeout_s == 1800
            caches.add(str(worker_args.runtime_cache_dir))
            callbacks["on_start"](job, slot, index + 1)
            callbacks["on_finish"](
                {"job_id": job["job_id"], "returncode": 0, "runner_error": None}
            )
        assert len(caches) == capacity
        return []

    monkeypatch.setattr(evaluate, "run_pool", run)
    output = tmp_path / "run"
    assert (
        evaluate.main(
            [
                "--model",
                "openai:unit-test-only",
                "--gpu-layout",
                layout,
                "--gpu-ids",
                "0,1,2,3,4,5,6,7",
                "--workers-per-gpu",
                str(workers),
                "--joybrain-timeout",
                "600",
                "--gpu-start-min-free-mib",
                "30052",
                "--gpu-runtime-min-free-mib",
                "2048",
                "--episodes-per-task",
                str(capacity),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["completed_attempts"] == capacity
    assert summary["model_server_count"] == len(groups_seen)
