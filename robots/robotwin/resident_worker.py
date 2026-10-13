"""Run independent native CLI episodes inside one resident Agent worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from robots.robotwin import robot_spec
from robots.robotwin.recording import write_json
from robots.robotwin.resident_protocol import EpisodeRpcClient
from robots.robotwin.runtime_cache import cache_environment
from rpent.dashboard.events import DashboardEventSink
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import reset_output_logging
from rpent.utils.rpc.http_rpc import HttpRpcClient


class ResidentRuntime:
    """Own a persistent EnvServer; each CLI call receives fresh clients and state."""

    def __init__(self, root: Path, cache_root: Path, reset_timeout_s: float) -> None:
        self.root = root
        self.cache_root = cache_root
        self.reset_timeout_s = reset_timeout_s
        self.daemon = None
        self.transport = None
        self.signature = None
        self.episode = None
        self.client = None
        self.model = None
        self.retired = False

    def initialize(
        self,
        args: argparse.Namespace,
        output_dir: Path,
        events: DashboardEventSink,
        components: set[str] | None,
    ) -> tuple[list[ProcessDaemon], dict[str, Any]]:
        if (
            components is not None
            or args.env_endpoint
            or args.vla_backend != "joybrain"
        ):
            raise ValueError(
                "Resident worker owns a complete Joy-Brain evaluation runtime"
            )
        if self.retired or self.episode is not None:
            raise RuntimeError("Previous resident episode was not cleanly finalized")
        signature = (
            args.task_name,
            args.task_config,
            args.max_episode_steps,
            str(robot_spec._resolve_env_runtime_path(args)),
        )
        if self.signature is not None and signature != self.signature:
            raise ValueError(
                "A resident worker cannot change task, assets, or action budget"
            )
        from robots.robotwin.env_client import RoboTwinEnvClient
        from robots.robotwin.joybrain_client import JoyBrainVLAClient

        started = time.monotonic()
        started_ns = time.time_ns()
        try:
            if self.daemon is None:
                self.signature = signature
                port = pick_free_port()
                self.daemon = ProcessDaemon(
                    "resident_robotwin_env",
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "robots.robotwin.resident_env",
                        "--task-name",
                        args.task_name,
                        "--task-config",
                        args.task_config,
                        "--seed",
                        str(args.seed),
                        "--assets-path",
                        signature[3],
                        "--max-episode-steps",
                        str(args.max_episode_steps),
                        "--output-root",
                        str(self.root),
                        "--port",
                        str(port),
                        "--parent-watch",
                    ],
                    env_overrides={
                        **cache_environment(self.cache_root, "env"),
                        "CUDA_VISIBLE_DEVICES": args.env_cuda_device,
                    },
                    log_path=str(self.root / "env-server.log"),
                )
                for name in (
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "RPENT_BASELINE_API_KEY",
                ):
                    self.daemon.subprocess_env.pop(name, None)
                self.daemon.start()
                robot_spec._wait_for_tcp("127.0.0.1", port, self.daemon, timeout_s=1800)
                self.transport = HttpRpcClient(f"http://127.0.0.1:{port}")
            if self.daemon.poll() is not None:
                raise RuntimeError("Resident EnvServer exited")
            prepared = self.transport.call(
                "resident.begin",
                args=(
                    int(args.seed),
                    str(output_dir / "trace"),
                    str(args.recording_config),
                ),
                timeout_s=self.reset_timeout_s,
            )
            self.episode = EpisodeRpcClient(self.transport, prepared["episode_id"])
            expected = robot_spec.env_runtime_contract(
                task_name=args.task_name,
                task_config=args.task_config,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
                record_actions=True,
            )
            self.client = RoboTwinEnvClient(self.episode, expected_meta=expected)
            self.model = JoyBrainVLAClient(
                args.vla_endpoint,
                timeout=args.joybrain_timeout,
                expected_checkpoint=robot_spec._configured_model_path(args),
                close_on_toolkit_close=True,
            )
            write_json(
                output_dir / "resident-runtime.json",
                {
                    **prepared,
                    "worker_pid": os.getpid(),
                    "initialize_started_ns": started_ns,
                    "initialized_ns": time.time_ns(),
                    "runtime_initialize_s": time.monotonic() - started,
                    "initial_status": self.client.last_info["episode_status"],
                    "initial_robot_state": self.client.last_info.get("robot_state"),
                    "initial_instruction": self.client.last_reset_info["instruction"],
                },
            )
            ready = self.root / "ready.json"
            if not ready.exists():
                write_json(
                    ready, {"worker_pid": os.getpid(), "env_pid": prepared["env_pid"]}
                )
            return [], {
                "env": self.client,
                "model": self.model,
                "seed": int(args.seed),
                "seed_mode": "exact",
            }
        except BaseException:
            self.retired = True
            raise

    def finish(self, output_dir: Path) -> None:
        """Retire uncertain runs; otherwise detach the recorder before the next seed."""
        native = json.loads((output_dir / "native_status.json").read_text())
        if native["execution_uncertain"] or native["recording_failed"]:
            self.retired = True
            raise RuntimeError(
                "An uncertain episode cannot reuse its resident environment"
            )
        result = self.episode.call("resident.end", timeout_s=self.reset_timeout_s)
        result["ended_ns"] = time.time_ns()
        write_json(output_dir / "resident-end.json", result)
        self.episode.close()
        self.episode = None
        self.client = None
        if self.model is not None:
            self.model.close()
            self.model = None

    def close(self) -> None:
        """Stop only this worker's owned environment and client connections."""
        try:
            if self.model is not None:
                self.model.close()
            if self.episode is not None:
                self.episode.close()
        finally:
            if self.daemon is not None:
                self.daemon.stop()


def run_worker(plan: dict, shard: dict, root: Path, endpoint: str) -> int:
    """Keep processes, not planner histories, alive between separate episodes."""
    from robots.robotwin.evaluate import inspect_episode
    from rpent.cli.main import main as cli_main

    configuration = plan["configuration"]
    runtime = ResidentRuntime(
        root,
        Path(configuration["runtime_cache_dir"]) / shard["gpu"]["uuid"],
        configuration["reset_timeout_s"],
    )
    result = {
        "worker_pid": os.getpid(),
        "gpu": shard["gpu"],
        "episodes": [],
        "status": "running",
    }
    try:
        for cell in shard["episodes"]:
            if (Path(plan["output_dir"]) / "STOP_AFTER_CURRENT").exists():
                result["status"] = "stopped_by_request"
                break
            for filename, digest in plan["source_sha256"].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(
                        f"Source changed during resident evaluation: {filename}"
                    )
            directory = root / cell["job_id"]
            directory.mkdir(exist_ok=False)
            metadata = directory / "run_config.json"
            write_json(metadata, {**plan, "cell": cell, "assignment": shard["gpu"]})
            arguments = [
                "--robot",
                "robotwin",
                "--vla-backend",
                "joybrain",
                "--vla-endpoint",
                endpoint,
                "--joybrain-timeout",
                str(configuration["joybrain_timeout"]),
                "--vla-model-path",
                configuration["vla_model_path"],
                "--task-name",
                cell["task"],
                "--task-config",
                configuration["task_config"],
                "--seed",
                str(cell["seed"]),
                "--env-cuda-device",
                shard["gpu"]["uuid"],
                "--robotwin-assets-path",
                configuration["robotwin_assets_path"],
                "--max-episode-steps",
                str(configuration["max_episode_steps"]),
                "--output-dir",
                str(directory),
                "--memory-profile",
                "local",
                "--memory-dir",
                str(directory / "memory"),
                "--strategy",
                configuration["strategy"],
                "--planner",
                "api",
                "--model",
                configuration["model"],
                "--max-turns",
                str(configuration["max_turns"]),
                "--max-tokens",
                str(configuration["max_tokens"]),
                "--planner-timeout-s",
                str(configuration["planner_timeout_s"]),
                "--record-actions",
                "--recording-config",
                str(metadata),
            ]
            if configuration.get("base_url"):
                arguments += ["--base-url", configuration["base_url"]]
            write_json(directory / "command.json", {"argv": arguments})
            started = time.time_ns()
            reset_output_logging()
            try:
                with (
                    (directory / "console.log").open("x") as console,
                    redirect_stdout(console),
                    redirect_stderr(console),
                ):
                    code = cli_main(arguments, runtime_initializer=runtime.initialize)
            finally:
                reset_output_logging()
            process = {
                "job_id": cell["job_id"],
                "pid": os.getpid(),
                "returncode": code,
                "runner_error": None,
                "started_ns": started,
                "finished_ns": time.time_ns(),
                "gpu_uuid": shard["gpu"]["uuid"],
                "model_gpu_uuid": shard["gpu"]["uuid"],
            }
            row = {
                **inspect_episode(directory, process),
                "cell": cell,
                "scheduler": process,
            }
            write_json(directory / "evaluation.json", row)
            result["episodes"].append(row)
            write_json(root / "worker-result.json", result)
            if code != 0:
                raise RuntimeError(
                    f"Resident episode failed: {cell['job_id']}; no replay"
                )
            runtime.finish(directory)
        else:
            result["status"] = "complete"
    except BaseException as error:
        result.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            runtime.close()
        finally:
            write_json(root / "worker-result.json", result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    shard = plan["shards"][args.shard]
    root = Path(plan["output_dir"]) / shard["job_id"]
    root.mkdir(exist_ok=False)
    return run_worker(plan, shard, root, args.endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
