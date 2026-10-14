"""Resume a drained evaluation without replaying any attempted task/seed."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEMANTIC_ARGUMENTS = (
    "task_name",
    "seed",
    "task_config",
    "tasks",
    "episodes_per_task",
    "record_actions",
    "vla_backend",
    "vla_model_path",
    "vla_python",
    "joybrain_repo",
    "joybrain_timeout",
    "robotwin_assets_path",
    "vla_endpoint",
    "env_endpoint",
    "strategy",
    "model",
    "base_url",
    "allow_insecure_http",
    "max_turns",
    "max_tokens",
    "planner_timeout_s",
    "max_episode_steps",
    "tool_plugin",
    "tool_plugin_root",
    "tool_plugin_endpoint",
    "tool_plugin_checkpoint",
    "tool_plugin_timeout",
)


@dataclass
class ResumeState:
    """Validated parent results, including failed but finished attempts."""

    root: Path
    plan: dict[str, Any]
    summary: dict[str, Any]
    completed: list[dict[str, Any]]
    pending: list[dict[str, Any]]


def load_parent(root: Path) -> ResumeState:
    """Require a fully drained parent and a persisted result for every attempt."""
    root = root.expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary.get("status") not in {"complete", "stopped_by_request"}
        or summary.get("active")
        or summary.get("interrupted_workers")
    ):
        raise ValueError("Parent evaluation must drain normally before resuming")
    cells = plan["cells"]
    by_id = {cell["job_id"]: cell for cell in cells}
    if len(by_id) != len(cells) or len(
        {(cell["task"], cell["seed"]) for cell in cells}
    ) != len(cells):
        raise ValueError("Parent plan contains duplicate task/seed pairs")
    completed = []
    seen = set()
    for entry in summary["episodes"]:
        cell = entry["cell"]
        job_id = cell["job_id"]
        if job_id in seen or by_id.get(job_id) != cell:
            raise ValueError("Completed attempt does not match the frozen plan")
        directory = Path(entry.get("directory", root / job_id)).resolve()
        if directory.name != job_id:
            raise ValueError("Episode directory does not match its job ID")
        saved = json.loads((directory / "evaluation.json").read_text())
        for name in (
            "cell",
            "official_success",
            "loop_verified",
            "stop_reason",
            "scheduler",
        ):
            if saved.get(name) != entry.get(name):
                raise ValueError("Summary disagrees with the persisted episode result")
        saved = deepcopy(saved)
        saved["directory"] = str(directory)
        saved["phase"] = entry.get(
            "phase",
            {
                "directory": str(root),
                "workers_per_gpu": summary["workers_per_gpu"],
                "gpu_count": summary["gpu_count"],
            },
        )
        completed.append(saved)
        seen.add(job_id)
    if summary.get("completed_attempts") != len(completed):
        raise ValueError("Parent completed-attempt count is inconsistent")
    for cell in cells:
        if (root / cell["job_id"]).exists() and cell["job_id"] not in seen:
            raise ValueError(
                "Unfinalized attempt exists; do not automatically replay it"
            )
    return ResumeState(
        root,
        plan,
        summary,
        completed,
        [cell for cell in cells if cell["job_id"] not in seen],
    )


def inherited_defaults(state: ResumeState) -> dict[str, Any]:
    """Reuse policy, planner, seed and budget settings rather than new CLI defaults."""
    configuration = state.plan["configuration"]
    defaults = {
        name: configuration[name]
        for name in SEMANTIC_ARGUMENTS
        if name in configuration
    }
    defaults["base_url"] = state.plan.get("base_url", defaults.get("base_url"))
    defaults["gpu_ids"] = configuration.get("gpu_ids", "0")
    defaults["gpu_layout"] = configuration.get("gpu_layout", "colocated")
    defaults["_vla_model_path_explicit"] = defaults.get("vla_model_path") is not None
    return defaults


def validate_configuration(state: ResumeState, arguments: Any) -> None:
    """Allow scaling/output changes, but reject silent policy or budget changes."""
    expected = inherited_defaults(state)
    for name in SEMANTIC_ARGUMENTS:
        if name in expected:
            actual = getattr(arguments, name)
            actual = str(actual) if isinstance(actual, Path) else actual
            if actual != expected[name]:
                raise ValueError(f"Resume cannot change frozen setting {name}")


def verify_worker_sources(state: ResumeState, controller: Path) -> list[str]:
    """Only the scheduling controller may change during this resume migration."""
    hashes = state.plan.get("source_sha256")
    if not hashes:
        raise ValueError("Parent evaluation has no frozen source hashes")
    changed = []
    for filename, expected in hashes.items():
        path = Path(filename)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            if path.resolve() != controller.resolve():
                raise ValueError(f"Worker/model/recorder source changed: {filename}")
            changed.append(filename)
    return changed
