"""Recorded RoboTwin evaluations with colocated or paired model/environment GPUs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from robots.robotwin import robot_spec, tool_plugin
from robots.robotwin.policy_pool import PolicyPool
from robots.robotwin.recording import validate_trace, write_json
from robots.robotwin.rollout_pool import (
    GPU,
    JobLaunch,
    WorkerSlot,
    discover_gpus,
    make_gpu_groups,
    run_pool,
    select_gpus,
    stop_process_group,
)
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("robotwin.eval")


def require_gpu_headroom(
    selected: list[GPU], inventory: list[GPU], minimum_free_mib: int
) -> None:
    """Reject unavailable or undersized devices without stopping existing jobs."""
    available = {gpu.uuid: gpu for gpu in inventory}
    for gpu in selected:
        current = available.get(gpu.uuid)
        if current is None or current.free_mib < minimum_free_mib:
            free = current.free_mib if current is not None else "unavailable"
            raise RuntimeError(
                f"Insufficient GPU headroom on {gpu.uuid}: free={free} MiB, "
                f"required={minimum_free_mib} MiB; no unrelated jobs will be stopped"
            )


def make_cells(
    manifest: dict, tasks: str, count: int, seed_start: int | None = None
) -> list[dict]:
    """Freeze distinct task/seed pairs; never reuse or silently replace a seed."""
    names = list(manifest["tasks"]) if tasks == "all" else tasks.split(",")
    if (
        not 1 <= count <= 100
        or len(set(names)) != len(names)
        or any(name not in manifest["tasks"] for name in names)
    ):
        raise ValueError("Choose unique listed tasks and 1..100 seeds per task")
    selected = {}
    for name in names:
        known = manifest["tasks"][name]
        seeds = [int(entry["seed"]) for entry in known]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Official seed list must be nonempty and unique")
        if seed_start is not None:
            if seed_start < 0:
                raise ValueError("seed must be nonnegative")
            chosen = list(range(seed_start, seed_start + count))
        else:
            chosen = seeds[:count]
            chosen += list(range(max(seeds) + 1, max(seeds) + 1 + count - len(chosen)))
        selected[name] = [
            (
                value,
                "official_manifest" if value in seeds else "fixed_not_expert_verified",
            )
            for value in chosen
        ]
    return [
        {
            "job_id": f"episode_{trial * len(names) + index:04d}_{name}_s{selected[name][trial][0]}",
            "task": name,
            "seed": selected[name][trial][0],
            "seed_source": selected[name][trial][1],
            "trial": trial + 1,
        }
        for trial in range(count)
        for index, name in enumerate(names)
    ]


def inspect_episode(directory: Path, process: dict) -> dict:
    """Keep native outcomes, infrastructure failures and trace validity separate."""
    path = directory / "result.json"
    final = json.loads(path.read_text()) if path.exists() else {}
    status_path = directory / "native_status.json"
    native = json.loads(status_path.read_text()) if status_path.exists() else {}
    error = process["runner_error"] or final.get("error")
    if not final:
        error = error or f"worker exited {process['returncode']} before finalization"
    trace_error = None
    try:
        trace = validate_trace(directory / "trace")
        if (
            trace["executed_actions"]
            and trace["last_native_count"]
            != native["episode_status"]["take_action_cnt"]
        ):
            raise ValueError("Trace and environment action counts differ")
    except Exception as exc:
        trace, trace_error = {}, f"{type(exc).__name__}: {exc}"
    events_path = directory / "trace/events.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    sequence = [
        {key: event[key] for key in ("tool_id", "tool", "arguments", "source")}
        for event in events
        if event["event"] == "tool_begin"
    ]
    success = final.get("official_success")
    verified = (
        process["returncode"] == 0
        and not error
        and not trace_error
        and not native.get("recording_failed", True)
        and not native.get("execution_uncertain", True)
        and trace.get("executed_actions", 0) > 0
        and isinstance(success, bool)
    )
    reason = (
        "worker_startup_timeout"
        if process["runner_error"] == "startup_timeout"
        else "worker_timeout"
        if process["runner_error"]
        else "initialization_or_worker_error"
        if not final
        else "planner_timeout"
        if error and "timed out" in error
        else "planner_error"
        if error
        else "native_success"
        if success
        else "action_budget"
        if native.get("truncated")
        else "planner_stop"
    )
    perception_error = None
    try:
        perception = tool_plugin.read_usage(directory)
    except Exception as exc:
        perception = {
            "calls": {},
            "successes": {},
            "failures": {},
            "incomplete": 0,
            "elapsed_s": 0.0,
        }
        perception_error = f"{type(exc).__name__}: {exc}"
    return {
        "official_success": success,
        "loop_verified": verified and not perception_error,
        "error": error,
        "trace_error": trace_error,
        "trace_validation": trace,
        "stop_reason": reason,
        "tool_calls": dict(Counter(item["tool"] for item in sequence)),
        "perception_tool_usage": perception,
        "perception_tool_error": perception_error,
        "tool_sequence": sequence,
        "planner_stats": final.get("stats", {}),
        "elapsed_s": final.get("elapsed_s"),
        "video": str(directory / "episode.mp4"),
        "directory": str(directory),
        "training_eligible": False,
    }


def save_results(output: Path, summary: dict, cells: list[dict]) -> None:
    """Publish controller-owned progress and per-task attempt statistics."""
    episodes = summary["episodes"]
    summary["completed_attempts"] = len(episodes)
    summary["phase_completed_attempts"] = len(episodes) - summary.get(
        "resumed_attempts", 0
    )
    summary["remaining_not_started"] = (
        len(cells) - len(episodes) - len(summary.get("active", {}))
    )
    summary["official_successes"] = sum(
        row["official_success"] is True for row in episodes
    )
    summary["valid_loops"] = sum(row["loop_verified"] for row in episodes)
    write_json(output / "summary.json", summary)
    rows = []
    for task, planned in Counter(cell["task"] for cell in cells).items():
        results = [row for row in episodes if row["cell"]["task"] == task]
        successes = sum(row["official_success"] is True for row in results)
        rows.append(
            {
                "task": task,
                "planned": planned,
                "completed": len(results),
                "successes": successes,
                "success_rate": successes / len(results) if results else None,
                "timeouts": sum("timeout" in row["stop_reason"] for row in results),
                "unknown": sum(row["official_success"] is None for row in results),
                "vla_calls": sum(
                    row["tool_calls"].get("lingbot_act", 0) for row in results
                ),
                "atomic_calls": sum(
                    sum(
                        value
                        for name, value in row["tool_calls"].items()
                        if name in {"move_to", "release", "set_gripper", "rotate_wrist"}
                    )
                    for row in results
                ),
                "executed_actions": sum(
                    row.get("trace_validation", {}).get("executed_actions", 0)
                    for row in results
                ),
                "episodes_with_action_counts": sum(
                    "executed_actions" in row.get("trace_validation", {})
                    for row in results
                ),
                "loop_seconds": sum(row.get("elapsed_s") or 0.0 for row in results),
                "episodes_with_loop_time": sum(
                    row.get("elapsed_s") is not None for row in results
                ),
                "worker_wall_seconds": sum(
                    row.get("scheduler", {}).get("wall_s") or 0.0 for row in results
                ),
                "perception_calls": sum(
                    sum(row.get("perception_tool_usage", {}).get("calls", {}).values())
                    for row in results
                ),
                **{
                    f"{name}_calls": sum(
                        row.get("perception_tool_usage", {})
                        .get("calls", {})
                        .get(name, 0)
                        for row in results
                    )
                    for name in summary.get("perception_tools", [])
                },
            }
        )
    write_json(output / "task_summary.json", {"tasks": rows})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = output / ".task_summary.csv.tmp"
    temporary.write_text(buffer.getvalue(), encoding="utf-8-sig")
    temporary.replace(output / "task_summary.csv")


def main(argv: list[str] | None = None) -> int:
    """Run isolated native CLI workers; credentials come only from their environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    robot_spec._add_cli_args(parser, True)
    parser.set_defaults(
        vla_backend="joybrain",
        vla_model_path=os.environ.get("JOYBRAIN_MODEL_PATH"),
        record_actions=True,
    )
    parser.add_argument("--tasks", default="click_bell")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument(
        "--gpu-layout",
        choices=["colocated", "split"],
        default="colocated",
        help="Colocated: one VLA per GPU; split: consecutive GPU IDs form model/env pairs.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Worker slots per environment GPU (not per model GPU in split mode).",
    )
    parser.add_argument(
        "--worker-startup-timeout-s",
        type=int,
        default=1800,
        help="Maximum initialization time before the first complete observation.",
    )
    parser.add_argument(
        "--gpu-start-min-free-mib",
        type=int,
        help=(
            "Measured model/worker peak plus safety margin; "
            "defaults to 30000 + 8000 * workers-per-gpu (colocated), "
            "or max(30000, 8000 * workers-per-gpu) (split)."
        ),
    )
    parser.add_argument(
        "--gpu-runtime-min-free-mib",
        type=int,
        default=12000,
        help="Minimum remaining GPU memory during execution (MiB, at least 1024).",
    )
    parser.add_argument("--max-active-planners", type=int)
    parser.add_argument("--model", default=os.environ.get("RPENT_MODEL"))
    parser.add_argument("--base-url")
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--planner-timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume only unattempted cells from a normally drained run.",
    )
    preview, _ = parser.parse_known_args(argv)
    resume = None
    controller_changes = []
    if preview.resume_from:
        from robots.robotwin.evaluation_resume import (
            inherited_defaults,
            load_parent,
            validate_configuration,
            verify_worker_sources,
        )

        resume = load_parent(preview.resume_from)
        parser.set_defaults(**inherited_defaults(resume))
    args = parser.parse_args(argv)
    plugin_options = tool_plugin.configuration(args)
    plugin = tool_plugin.load_plugin(plugin_options) if plugin_options else None
    if resume is not None:
        validate_configuration(resume, args)
        controller_changes = verify_worker_sources(resume, Path(__file__))
        if not resume.pending:
            logger.info("All parent attempts are finalized; nothing to resume")
            return 0
    if args.vla_backend != "joybrain" or args.env_endpoint or args.recording_config:
        parser.error("This runner owns recorded Joy-Brain evaluation workers")
    if args.task_config != "demo_randomized":
        parser.error("Batch evaluation uses the demo_randomized task manifest")
    if (
        not 1 <= args.workers_per_gpu <= 8
        or args.max_turns < 1
        or args.planner_timeout_s <= 0
        or args.max_tokens < 1
        or args.max_episode_steps < 1
        or args.worker_startup_timeout_s <= 0
    ):
        parser.error("Invalid worker or planner budget")
    if args.gpu_start_min_free_mib is None:
        args.gpu_start_min_free_mib = (
            30000 + 8000 * args.workers_per_gpu
            if args.gpu_layout == "colocated"
            else max(30000, 8000 * args.workers_per_gpu)
        )
    if (
        args.gpu_runtime_min_free_mib < 1024
        or args.gpu_start_min_free_mib < args.gpu_runtime_min_free_mib
    ):
        parser.error("GPU memory limits require startup >= runtime >= 1024 MiB")
    if any(
        value is not None
        for value in (args.cuda_device, args.env_cuda_device, args.vla_cuda_device)
    ):
        parser.error("Use --gpu-ids for pool GPU assignment")
    gpus = select_gpus(
        args.gpu_ids.split(","), discover_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    groups = make_gpu_groups(gpus, args.gpu_layout)
    capacity = len(groups) * args.workers_per_gpu
    maximum = (
        args.max_active_planners if args.max_active_planners is not None else capacity
    )
    if not 1 <= maximum <= capacity:
        parser.error("Invalid max-active-planners")
    if args.vla_endpoint and len(groups) != 1:
        parser.error("A borrowed VLA endpoint supports one GPU group only")
    require_gpu_headroom(gpus, gpus, args.gpu_start_min_free_mib)
    base_url = args.base_url or os.environ.get(
        "ANTHROPIC_BASE_URL"
        if (args.model or "").startswith("anthropic:")
        else "OPENAI_BASE_URL"
    )
    if base_url:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            parser.error(
                "Use a provider base URL without embedded credentials or query parameters"
            )
        if parsed.scheme == "http" and not args.allow_insecure_http:
            parser.error("Plain HTTP requires --allow-insecure-http")
    manifest_path = Path(__file__).parent / "eval/demo_randomized.json"
    cells = make_cells(
        json.loads(manifest_path.read_text()),
        args.task_name or args.tasks,
        args.episodes_per_task,
        args.seed,
    )
    if resume is not None and cells != resume.plan["cells"]:
        raise ValueError("Generated task/seed plan differs from the frozen parent")
    pending = resume.pending if resume is not None else cells
    output = args.output_dir.expanduser().resolve()
    args.runtime_cache_dir = (
        (args.runtime_cache_dir or output / "runtime-cache").expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "kind": "robotwin_evaluation",
        "resume_from": str(resume.root) if resume is not None else None,
        "controller_source_changes": controller_changes,
        "rng_continuation": "new service RNG; previous actions are not replayed"
        if resume is not None
        else None,
        "gpu_count": len(gpus),
        "gpu_layout": args.gpu_layout,
        "gpu_groups": [asdict(group) for group in groups],
        "model_server_count": len(groups),
        "worker_capacity": capacity,
        "worker_lifecycle": "new process per episode; model services remain shared",
        "worker_startup": "one initializing worker per environment GPU until states.json",
        "workers_per_gpu": args.workers_per_gpu,
        "max_active_planners": maximum,
        "cells": cells,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "base_url": base_url,
        "physical_gpus": [asdict(gpu) for gpu in gpus],
        "training_eligible": False,
        "policy_execution": "one serial VLA service per model GPU; no dynamic batching",
        "policy_randomness": "server RNG; scene seeds do not fix VLA action noise",
    }
    write_json(output / "plan.json", plan)
    if args.dry_run:
        logger.info(
            "Plan contains %d distinct task/seed pairs; no services started", len(cells)
        )
        return 0
    if not args.model:
        parser.error("--model or RPENT_MODEL is required for real evaluations")
    if not args.vla_endpoint:
        args.vla_model_path = str(robot_spec._resolve_vla_runtime_path(args))
        plan["configuration"]["vla_model_path"] = args.vla_model_path
    assets_path = robot_spec._resolve_env_runtime_path(args)
    if assets_path is None:
        parser.error("--robotwin-assets-path or ROBOTWIN_ASSETS_PATH is required")
    args.robotwin_assets_path = str(assets_path)
    plan["configuration"]["robotwin_assets_path"] = args.robotwin_assets_path
    init_output_dir(output)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sources = list(Path(__file__).parent.glob("*.py")) + list(
        (Path(__file__).parent / "prompts").glob("*.py")
    )
    if plugin is not None:
        sources.extend(Path(path).resolve() for path in plugin.source_files())
        plan["perception_tools"] = list(plugin.tool_names())
    plan["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    if args.vla_model_path and not args.vla_endpoint:
        with (Path(args.vla_model_path) / "model.safetensors").open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(block)
            plan["weight_sha256"] = checksum.hexdigest()
            if (
                resume is not None
                and resume.plan.get("weight_sha256") != plan["weight_sha256"]
            ):
                raise ValueError("Checkpoint bytes changed since the parent evaluation")
    write_json(output / "plan.json", plan)
    summary = {
        "kind": "robotwin_evaluation",
        "status": "preflight",
        "episodes": resume.completed if resume is not None else [],
        "resumed_attempts": len(resume.completed) if resume is not None else 0,
        "resume_from": str(resume.root) if resume is not None else None,
        "active": {},
        "planned_episodes": len(cells),
        "phase_planned_episodes": len(pending),
        "started_ns": resume.summary["started_ns"]
        if resume is not None
        else time.time_ns(),
        "phase_started_ns": time.time_ns(),
        "mixed_concurrency": bool(
            resume is not None
            and (
                resume.summary.get("mixed_concurrency")
                or resume.summary["workers_per_gpu"] != args.workers_per_gpu
                or resume.summary["gpu_count"] != len(gpus)
                or resume.plan.get("gpu_layout", "colocated") != args.gpu_layout
            )
        ),
        "gpu_count": len(gpus),
        "gpu_layout": args.gpu_layout,
        "model_server_count": len(groups),
        "worker_capacity": capacity,
        "workers_per_gpu": args.workers_per_gpu,
        "training_eligible": False,
        "perception_tools": list(plugin.tool_names()) if plugin is not None else [],
    }
    save_results(output, summary, cells)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(get_repo_root())
    last_check = 0.0

    def check_resources():
        nonlocal last_check
        if time.monotonic() - last_check > 1:
            last_check = time.monotonic()
            require_gpu_headroom(gpus, discover_gpus(), args.gpu_runtime_min_free_mib)

    pool = PolicyPool(groups, args, output, args.runtime_cache_dir, check_resources)

    def stop_services():
        pool.close()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {
        name: signal.signal(name, interrupted)
        for name in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with ExitStack() as leases:
            lock_root = Path(tempfile.gettempdir()) / f"rpent-gpu-leases-{os.getuid()}"
            lock_root.mkdir(mode=0o700, exist_ok=True)
            for gpu in gpus:
                descriptor = os.open(
                    lock_root / gpu.uuid, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
                )
                lease = leases.enter_context(os.fdopen(descriptor, "a+"))
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            leases.callback(stop_services)
            require_gpu_headroom(gpus, discover_gpus(), args.gpu_start_min_free_mib)
            check_command = [
                sys.executable,
                "-m",
                "rpent.cli.check_llm",
                "--planner",
                "api",
                "--model",
                args.model,
            ]
            if base_url:
                check_command += ["--base-url", base_url]
            with (output / "planner-check.log").open("x") as log:
                process = subprocess.Popen(
                    check_command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=150)
                finally:
                    stop_process_group(process)
                if code:
                    raise RuntimeError(
                        "Planner preflight failed; see planner-check.log"
                    )
            pool.start()
            slots = [
                WorkerSlot(
                    f"gpu{group.env_gpu.index}_worker{worker}",
                    group.env_gpu.uuid,
                    pool.endpoints[group.model_gpu.uuid],
                    group.model_gpu.uuid,
                )
                for worker in range(args.workers_per_gpu)
                for group in groups
            ]
            by_id = {cell["job_id"]: cell for cell in cells}

            def launch(cell, slot):
                for filename, checksum in plan["source_sha256"].items():
                    if (
                        hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                        != checksum
                    ):
                        raise RuntimeError("Source files changed during evaluation")
                directory = output / cell["job_id"]
                directory.mkdir()
                metadata = directory / "run_config.json"
                write_json(
                    metadata,
                    {
                        **plan,
                        "cell": cell,
                        "assignment": asdict(slot),
                        "policy_metadata": pool.metadata[slot.policy_gpu_uuid],
                    },
                )
                command = [
                    sys.executable,
                    "-m",
                    "rpent.cli.main",
                    "--robot",
                    "robotwin",
                    "--vla-backend",
                    "joybrain",
                    "--vla-endpoint",
                    slot.endpoint,
                    "--joybrain-timeout",
                    str(args.joybrain_timeout),
                    "--record-actions",
                    "--recording-config",
                    str(metadata),
                    "--task-name",
                    cell["task"],
                    "--task-config",
                    args.task_config,
                    "--seed",
                    str(cell["seed"]),
                    "--env-cuda-device",
                    slot.gpu_uuid,
                    "--runtime-cache-dir",
                    str(
                        args.runtime_cache_dir
                        / "workers"
                        / slot.gpu_uuid
                        / slot.slot_id
                    ),
                    "--robotwin-assets-path",
                    str(robot_spec._resolve_env_runtime_path(args)),
                    "--max-episode-steps",
                    str(args.max_episode_steps),
                    "--output-dir",
                    str(directory),
                    "--memory-profile",
                    "local",
                    "--memory-dir",
                    str(directory / "memory"),
                    "--strategy",
                    args.strategy,
                    "--planner",
                    "api",
                    "--model",
                    args.model,
                    "--max-turns",
                    str(args.max_turns),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--planner-timeout-s",
                    str(args.planner_timeout_s),
                ]
                if args.vla_model_path:
                    command += ["--vla-model-path", args.vla_model_path]
                if base_url:
                    command += ["--base-url", base_url]
                if plugin_options:
                    command += tool_plugin.worker_arguments(plugin_options)
                write_json(directory / "command.json", {"argv": command})
                return JobLaunch(
                    command,
                    {**environment, "CUDA_VISIBLE_DEVICES": slot.gpu_uuid},
                    directory / "console.log",
                    args.worker_startup_timeout_s + args.planner_timeout_s + 60,
                    ready_path=directory / "states.json",
                    startup_timeout_s=args.worker_startup_timeout_s,
                )

            def started(cell, slot, pid):
                summary["active"][cell["job_id"]] = {**asdict(slot), "pid": pid}
                summary["peak_active_workers"] = max(
                    summary.get("peak_active_workers", 0), len(summary["active"])
                )
                save_results(output, summary, cells)
                logger.info("START %s on %s", cell["job_id"], slot.slot_id)

            def finished(process):
                cell = by_id[process["job_id"]]
                directory = output / cell["job_id"]
                try:
                    row = inspect_episode(directory, process)
                except Exception as exc:
                    row = {
                        "official_success": None,
                        "loop_verified": False,
                        "tool_calls": {},
                        "stop_reason": "analysis_error",
                        "error": str(exc),
                    }
                row.update(
                    cell=cell,
                    scheduler=process,
                    phase={
                        "directory": str(output),
                        "workers_per_gpu": args.workers_per_gpu,
                        "gpu_count": len(gpus),
                        "gpu_layout": args.gpu_layout,
                        "model_server_count": len(groups),
                    },
                )
                write_json(directory / "evaluation.json", row)
                summary["episodes"].append(row)
                summary["active"].pop(cell["job_id"], None)
                save_results(output, summary, cells)
                logger.info(
                    "DONE %s success=%s", cell["job_id"], row["official_success"]
                )

            def stop_requested():
                pool.check()
                return (output / "STOP_AFTER_CURRENT").exists()

            summary["status"] = "running"
            save_results(output, summary, cells)
            run_pool(
                pending,
                slots,
                launch,
                max_active=maximum,
                on_start=started,
                on_finish=finished,
                should_stop=stop_requested,
            )
            summary["status"] = (
                "complete"
                if len(summary["episodes"]) == len(cells)
                else "stopped_by_request"
            )
    except BaseException as exc:
        summary.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_services()
        for name, handler in handlers.items():
            signal.signal(name, handler)
        if summary["active"]:
            summary["interrupted_workers"] = summary["active"]
            summary["active"] = {}
        summary["finished_ns"] = time.time_ns()
        summary["phase_finished_ns"] = summary["finished_ns"]
        save_results(output, summary, cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
