"""Experimental single-task evaluation with one resident Agent/Env worker per GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from robots.robotwin import robot_spec, tool_plugin
from robots.robotwin.evaluate import make_cells, require_gpu_headroom
from robots.robotwin.policy_pool import PolicyPool
from robots.robotwin.recording import write_json
from robots.robotwin.rollout_pool import (
    JobLaunch,
    WorkerSlot,
    discover_gpus,
    make_gpu_groups,
    run_pool,
    select_gpus,
    stop_process_group,
)
from rpent.utils.config import get_repo_root
from rpent.utils.logging import init_output_dir


def main(argv: list[str] | None = None) -> int:
    """Run a fixed task/seed plan without replacing failed resident workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    robot_spec._add_cli_args(parser, True)
    parser.set_defaults(
        vla_backend="joybrain",
        vla_model_path=os.environ.get("JOYBRAIN_MODEL_PATH"),
        record_actions=True,
        joybrain_timeout=600,
    )
    parser.add_argument("--tasks", default="click_bell")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--model", default=os.environ.get("RPENT_MODEL"))
    parser.add_argument("--base-url")
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--planner-timeout-s", type=int, default=600)
    parser.add_argument("--reset-timeout-s", type=int, default=600)
    parser.add_argument("--worker-startup-timeout-s", type=int, default=1800)
    parser.add_argument("--gpu-start-min-free-mib", type=int, default=31000)
    parser.add_argument("--gpu-runtime-min-free-mib", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plugin_options = tool_plugin.configuration(args)
    plugin = tool_plugin.load_plugin(plugin_options) if plugin_options else None
    task = args.task_name or args.tasks
    if task == "all" or "," in task or args.task_config != "demo_randomized":
        parser.error("Resident evaluation currently supports one demo_randomized task")
    if (
        args.vla_backend != "joybrain"
        or args.env_endpoint
        or args.vla_endpoint
        or args.recording_config
    ):
        parser.error(
            "This runner owns both the resident environment and Joy-Brain services"
        )
    if any(
        value is not None
        for value in (args.cuda_device, args.env_cuda_device, args.vla_cuda_device)
    ):
        parser.error("Use --gpu-ids for physical GPU placement")
    if (
        min(
            args.max_turns,
            args.max_tokens,
            args.planner_timeout_s,
            args.reset_timeout_s,
            args.worker_startup_timeout_s,
            args.max_episode_steps,
        )
        < 1
    ):
        parser.error("Resident evaluation budgets must be positive")
    if not args.gpu_start_min_free_mib >= args.gpu_runtime_min_free_mib >= 1024:
        parser.error("Memory guards require startup >= runtime >= 1024 MiB")
    args.base_url = args.base_url or os.environ.get(
        "ANTHROPIC_BASE_URL"
        if (args.model or "").startswith("anthropic:")
        else "OPENAI_BASE_URL"
    )
    if args.base_url:
        url = urlsplit(args.base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            parser.error(
                "Provider URL must not contain embedded credentials or query parameters"
            )
        if url.scheme == "http" and not args.allow_insecure_http:
            parser.error("HTTP requires --allow-insecure-http")
    gpus = select_gpus(
        args.gpu_ids.split(","), discover_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    groups = make_gpu_groups(gpus, "colocated")
    require_gpu_headroom(gpus, gpus, args.gpu_start_min_free_mib)
    manifest = json.loads(
        (Path(__file__).parent / "eval/demo_randomized.json").read_text()
    )
    cells = make_cells(manifest, task, args.episodes_per_task, args.seed)
    if len(cells) < 2 * len(gpus):
        parser.error("Use at least two episodes per GPU to exercise resident reuse")
    output = args.output_dir.expanduser().resolve()
    args.runtime_cache_dir = (
        (args.runtime_cache_dir or output / "runtime-cache").expanduser().resolve()
    )
    shards = [
        {
            "job_id": f"resident_gpu_{gpu.index}",
            "gpu": asdict(gpu),
            "episodes": cells[index :: len(gpus)],
        }
        for index, gpu in enumerate(gpus)
    ]
    plan = {
        "kind": "robotwin_resident_evaluation",
        "output_dir": str(output),
        "cells": cells,
        "gpu_layout": "colocated",
        "gpu_count": len(gpus),
        "model_server_count": len(gpus),
        "worker_capacity": len(gpus),
        "shards": shards,
        "training_eligible": False,
        "worker_lifecycle": "resident Agent and Env processes; fresh CLI conversation/recorder per episode",
        "scene_lifecycle": "native task/language prewalk refreshed for each seed; no task switching",
        "perception_tools": list(plugin.tool_names()) if plugin is not None else [],
        "configuration": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    if args.dry_run:
        return 0
    if not args.model:
        parser.error("--model or RPENT_MODEL is required for actual Agent requests")
    args.vla_model_path = str(robot_spec._resolve_vla_runtime_path(args))
    args.robotwin_assets_path = str(robot_spec._resolve_env_runtime_path(args))
    plan["configuration"].update(
        vla_model_path=args.vla_model_path,
        robotwin_assets_path=args.robotwin_assets_path,
    )
    sources = list(Path(__file__).parent.glob("*.py")) + list(
        (Path(__file__).parent / "prompts").glob("*.py")
    )
    sources += list((Path(__file__).parent / "guides").glob("*.md"))
    if plugin is not None:
        sources.extend(Path(path).resolve() for path in plugin.source_files())
    sources += [
        get_repo_root() / name
        for name in (
            "rpent/cli/main.py",
            "rpent/utils/logging.py",
            "rpent/planner/api_loop.py",
        )
    ]
    plan["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    checksum = hashlib.sha256()
    with (Path(args.vla_model_path) / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    plan["checkpoint_sha256"] = checksum.hexdigest()
    write_json(output / "plan.json", plan)
    init_output_dir(output)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    environment = {
        **os.environ,
        "PYTHONPATH": str(get_repo_root()),
        "TOKENIZERS_PARALLELISM": "false",
    }
    summary = {
        "status": "preflight",
        "planned_episodes": len(cells),
        "episodes": [],
        "active_workers": {},
        "started_ns": time.time_ns(),
        "worker_capacity": len(gpus),
        "perception_tools": plan["perception_tools"],
    }
    last_check = 0.0

    def save_progress():
        episodes = []
        for shard in shards:
            result_path = output / shard["job_id"] / "worker-result.json"
            if result_path.exists():
                episodes.extend(json.loads(result_path.read_text()).get("episodes", []))
        if len({row["cell"]["job_id"] for row in episodes}) != len(episodes):
            raise RuntimeError("Duplicate finalized episode in resident results")
        summary.update(
            episodes=episodes,
            completed_attempts=len(episodes),
            official_successes=sum(row["official_success"] is True for row in episodes),
            valid_loops=sum(row["loop_verified"] for row in episodes),
            perception_tool_calls={
                name: sum(
                    row.get("perception_tool_usage", {}).get("calls", {}).get(name, 0)
                    for row in episodes
                )
                for name in plan["perception_tools"]
            },
        )
        write_json(output / "summary.json", summary)

    def check_resources():
        nonlocal last_check
        if time.monotonic() - last_check >= 1:
            last_check = time.monotonic()
            require_gpu_headroom(gpus, discover_gpus(), args.gpu_runtime_min_free_mib)
            save_progress()

    pool = PolicyPool(groups, args, output, args.runtime_cache_dir, check_resources)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {
        name: signal.signal(name, interrupt) for name in (signal.SIGTERM, signal.SIGINT)
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
            leases.callback(pool.close)
            require_gpu_headroom(gpus, discover_gpus(), args.gpu_start_min_free_mib)
            command = [
                sys.executable,
                "-B",
                "-m",
                "rpent.cli.check_llm",
                "--planner",
                "api",
                "--model",
                args.model,
            ]
            if args.base_url:
                command += ["--base-url", args.base_url]
            with (output / "planner-check.log").open("x") as console:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    if process.wait(timeout=180) != 0:
                        raise RuntimeError(
                            "Planner preflight failed; see planner-check.log"
                        )
                finally:
                    stop_process_group(process)
            pool.start()
            slots = [
                WorkerSlot(
                    shard["job_id"],
                    shard["gpu"]["uuid"],
                    pool.endpoints[shard["gpu"]["uuid"]],
                )
                for shard in shards
            ]

            def launch(shard, slot):
                index = next(
                    index
                    for index, candidate in enumerate(shards)
                    if candidate["job_id"] == shard["job_id"]
                )
                if slot.slot_id != shard["job_id"]:
                    raise RuntimeError("Resident shard/slot assignment changed")
                return JobLaunch(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "robots.robotwin.resident_worker",
                        "--plan",
                        str(output / "plan.json"),
                        "--shard",
                        str(index),
                        "--endpoint",
                        slot.endpoint,
                    ],
                    {**environment, "CUDA_VISIBLE_DEVICES": slot.gpu_uuid},
                    output / f"{slot.slot_id}.log",
                    args.worker_startup_timeout_s
                    + len(shard["episodes"])
                    * (args.reset_timeout_s + args.planner_timeout_s + 120),
                    ready_path=output / slot.slot_id / "ready.json",
                    startup_timeout_s=args.worker_startup_timeout_s,
                )

            def started(shard, slot, pid):
                summary["active_workers"][shard["job_id"]] = {
                    "pid": pid,
                    "gpu_uuid": slot.gpu_uuid,
                }
                save_progress()

            def finished(process):
                summary["active_workers"].pop(process["job_id"], None)
                save_progress()
                if process["returncode"] != 0 or process["runner_error"]:
                    raise RuntimeError(
                        f"Resident worker failed without replacement: {process['job_id']}"
                    )

            def should_stop():
                pool.check()
                return (output / "STOP_AFTER_CURRENT").exists()

            summary["status"] = "running"
            save_progress()
            run_pool(
                shards,
                slots,
                launch,
                max_active=len(gpus),
                on_start=started,
                on_finish=finished,
                should_stop=should_stop,
            )
            save_progress()
            summary["status"] = (
                "complete"
                if len(summary["episodes"]) == len(cells)
                else "stopped_by_request"
            )
    except BaseException as error:
        summary.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            pool.close()
        finally:
            for name, handler in handlers.items():
                signal.signal(name, handler)
            if summary["active_workers"]:
                summary["interrupted_workers"] = summary["active_workers"]
                summary["active_workers"] = {}
            summary["finished_ns"] = time.time_ns()
            save_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
