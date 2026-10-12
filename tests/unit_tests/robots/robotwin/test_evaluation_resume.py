import hashlib
import json
from argparse import Namespace
from copy import deepcopy

import pytest

from robots.robotwin.evaluation_resume import (
    inherited_defaults,
    load_parent,
    validate_configuration,
    verify_worker_sources,
)


def parent_run(root):
    root.mkdir()
    cells = [
        {"job_id": f"episode_{index}", "task": "task", "seed": index}
        for index in range(4)
    ]
    source = root / "worker.py"
    source.write_text("unchanged worker\n")
    plan = {
        "cells": cells,
        "configuration": {
            "model": "openai:fixture",
            "max_turns": 24,
            "planner_timeout_s": 900,
            "max_episode_steps": 1000,
            "strategy": "upstream",
            "gpu_ids": "0",
        },
        "base_url": None,
        "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    rows = []
    for index in range(2):
        directory = root / cells[index]["job_id"]
        directory.mkdir()
        row = {
            "cell": cells[index],
            "official_success": index == 0,
            "loop_verified": index == 0,
            "stop_reason": "native_success" if index == 0 else "planner_timeout",
            "scheduler": {"returncode": index},
        }
        (directory / "evaluation.json").write_text(json.dumps(row))
        rows.append(row)
    summary = {
        "status": "stopped_by_request",
        "active": {},
        "episodes": rows,
        "completed_attempts": 2,
        "workers_per_gpu": 4,
        "gpu_count": 1,
    }
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "summary.json").write_text(json.dumps(summary))
    return plan, summary


def test_resume_excludes_successes_and_failed_attempts(tmp_path):
    root = tmp_path / "parent"
    parent_run(root)
    state = load_parent(root)
    assert [cell["seed"] for cell in state.pending] == [2, 3]
    assert len(state.completed) == 2
    assert all(row["phase"]["workers_per_gpu"] == 4 for row in state.completed)


def test_active_parent_is_not_resumed(tmp_path):
    root = tmp_path / "parent"
    _, summary = parent_run(root)
    summary["status"] = "running"
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="drain"):
        load_parent(root)


def test_partial_attempt_is_not_replayed(tmp_path):
    root = tmp_path / "parent"
    parent_run(root)
    (root / "episode_2").mkdir()
    with pytest.raises(ValueError, match="Unfinalized"):
        load_parent(root)


def test_result_mismatch_is_rejected(tmp_path):
    root = tmp_path / "parent"
    _, summary = parent_run(root)
    summary["episodes"][0]["official_success"] = False
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="disagrees"):
        load_parent(root)


def test_policy_and_budget_cannot_change_on_resume(tmp_path):
    root = tmp_path / "parent"
    parent_run(root)
    state = load_parent(root)
    arguments = Namespace(**inherited_defaults(state), workers_per_gpu=8)
    validate_configuration(state, arguments)
    arguments.max_turns = 25
    with pytest.raises(ValueError, match="max_turns"):
        validate_configuration(state, arguments)


def test_only_controller_source_change_is_allowed(tmp_path):
    root = tmp_path / "parent"
    parent_run(root)
    state = load_parent(root)
    source = root / "worker.py"
    assert verify_worker_sources(state, tmp_path / "controller.py") == []
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        verify_worker_sources(state, tmp_path / "controller.py")
    assert verify_worker_sources(state, source) == [str(source)]


def test_multiple_resume_phases_keep_original_provenance(tmp_path):
    root = tmp_path / "parent"
    plan, _ = parent_run(root)
    first = load_parent(root)
    second = tmp_path / "second"
    second.mkdir()
    (second / "plan.json").write_text(json.dumps(plan))
    rows = deepcopy(first.completed)
    summary = {
        "status": "stopped_by_request",
        "active": {},
        "episodes": rows,
        "completed_attempts": 2,
        "workers_per_gpu": 8,
        "gpu_count": 1,
    }
    (second / "summary.json").write_text(json.dumps(summary))
    state = load_parent(second)
    assert len(state.pending) == 2
    assert all(row["phase"]["workers_per_gpu"] == 4 for row in state.completed)


def test_layout_defaults_are_preserved_on_resume_and_old_runs_are_colocated(tmp_path):
    root = tmp_path / "parent"
    plan, _ = parent_run(root)
    assert inherited_defaults(load_parent(root))["gpu_layout"] == "colocated"
    plan["configuration"]["gpu_layout"] = "split"
    (root / "plan.json").write_text(json.dumps(plan))
    assert inherited_defaults(load_parent(root))["gpu_layout"] == "split"
