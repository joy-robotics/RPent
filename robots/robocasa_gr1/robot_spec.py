"""GR1 registration and runtime wiring, separate from PandaOmron/RLDX."""

from __future__ import annotations

import math
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from robots.robocasa_gr1.contracts import EVAL_TYPES, env_contract, task_id
from robots.robocasa_gr1.prompt_bundle import system_prompt, user_prompt
from rpent.evaluation import write_json_atomic
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="robocasa_gr1",
        prompts=PromptBundle(system_prompt, user_prompt),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
        validate_args=validate_args,
        prepare_memory=prepare_memory,
        finalize_run=finalize_run,
        dashboard={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <task_name> <eval_type> <seed>",
                "fields": (
                    {"name": "task_name"},
                    {"name": "eval_type", "choices": EVAL_TYPES},
                    {"name": "seed", "kind": "integer", "minimum": 0},
                ),
                "display": "{task_name} / {eval_type} / seed {seed}",
                "output_slug": "{task_name}_{eval_type}_s{seed}",
            },
            "runtime_components": (
                {"name": "env", "label": "ENV", "scope": "unique"},
                {"name": "vla", "label": "VLA", "scope": "shared"},
            ),
            "primitives": ("gr1_act", "set_joint_targets"),
        },
    )


def add_cli_args(parser, use_dashboard: bool) -> None:
    parser.add_argument("--task-name", required=not use_dashboard)
    parser.add_argument("--eval-type", choices=EVAL_TYPES, default="id")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--joybrain-repo", default=os.environ.get("JOYBRAIN_REPO_ROOT"))
    parser.add_argument(
        "--vla-model-path", default=os.environ.get("JOYBRAIN_MODEL_PATH")
    )
    parser.add_argument("--vla-python", default=os.environ.get("JOYBRAIN_PYTHON"))
    parser.add_argument("--gr1-repo", default=os.environ.get("ROBOCASA_GR1_REPO"))
    parser.add_argument("--gr1-python", default=os.environ.get("ROBOCASA_GR1_PYTHON"))
    parser.add_argument(
        "--env-ld-library-path", default=os.environ.get("ROBOCASA_GR1_LD_LIBRARY_PATH")
    )
    parser.add_argument("--vla-endpoint")
    parser.add_argument("--cuda-device")
    parser.add_argument("--env-cuda-device")
    parser.add_argument("--vla-cuda-device")
    parser.add_argument("--egl-device-id", type=int)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--joybrain-timeout", type=float, default=600)


def validate_args(args) -> None:
    if args.task_name:
        task_id(args.task_name)
    if (
        args.seed < 0
        or (args.egl_device_id is not None and args.egl_device_id < 0)
        or min(args.max_episode_steps, args.replan_steps) < 1
    ):
        raise ValueError("GR1 seeds/devices must be nonnegative and budgets positive")
    if args.cuda_device is not None and (
        args.env_cuda_device is not None or args.vla_cuda_device is not None
    ):
        raise ValueError("Use either --cuda-device or separate Env/VLA devices")
    if args.memory_profile != "local":
        raise ValueError(
            "GR1 has no shared PandaOmron memory; use --memory-profile local"
        )
    if not math.isfinite(args.joybrain_timeout) or args.joybrain_timeout <= 0:
        raise ValueError("joybrain-timeout must be finite and positive")


def prepare_memory(args, config) -> None:
    """Keep GR1 memory local; do not download or reuse PandaOmron recipes."""
    validate_args(args)


def parse_config(args) -> RunConfig:
    task = task_id(args.task_name)
    slug = task.split("/")[1].removesuffix("_GR1ArmsAndWaistFourierHands_Env")
    tag = f"robocasa_gr1_{slug}_{args.eval_type}_s{args.seed}"
    output = (
        Path(args.output_dir or Path("logs") / f"{datetime.now():%Y%m%d-%H%M%S}_{tag}")
        .expanduser()
        .resolve()
    )
    memory = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else output / "memory"
    )
    return RunConfig(
        recipe_tag=tag,
        output_dir=output,
        prompt_vars={
            "task_name": task,
            "seed": args.seed,
            "memory_dir": str(memory),
            "output_dir": output,
        },
        task_desc={
            "platform": "robocasa_gr1",
            "task_name": task,
            "eval_type": args.eval_type,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "max_episode_steps": args.max_episode_steps,
            "vla_model_path": args.vla_model_path,
        },
    )


def get_toolkit(*, runtime_kwargs, dashboard_events, config):
    from robots.robocasa_gr1.toolkit import GR1Toolkit
    from rpent.memory import MemoryManager

    return GR1Toolkit(
        **runtime_kwargs,
        output=config.output_dir,
        dashboard_events=dashboard_events,
        memory=MemoryManager(
            root=config.prompt_vars["memory_dir"], memory_access="read_only"
        ),
    )


def _path(value, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} must be explicitly configured")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def init_runtime(args, output_dir, dashboard_events, components):
    from robots.robocasa_gr1.env_client import GR1EnvClient
    from robots.robocasa_gr1.vla_client import GR1VLAClient
    from rpent.dashboard.events import RuntimeStatusEvent
    from rpent.robots.runtime import (
        stop_owned_daemons,
        try_spawn_server,
        try_wait_server,
    )
    from rpent.utils.config import get_repo_root
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    selected = {"env", "vla"} if components is None else set(components)
    if selected - {"env", "vla"}:
        raise ValueError("Unknown GR1 runtime component")
    root = get_repo_root()
    joybrain = _path(args.joybrain_repo, "--joybrain-repo")
    env_device = (
        args.cuda_device if args.cuda_device is not None else args.env_cuda_device
    )
    vla_device = (
        args.cuda_device if args.cuda_device is not None else args.vla_cuda_device
    )
    egl_device = args.egl_device_id
    if "env" in selected and egl_device is None:
        effective_device = env_device or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if not str(effective_device).isdecimal():
            raise ValueError(
                "Set --egl-device-id explicitly for UUID or multiple CUDA devices"
            )
        egl_device = int(effective_device)
    owned = {}
    clients = []
    result = {"replan_steps": args.replan_steps}
    try:
        if "env" in selected:
            python = _path(args.gr1_python, "--gr1-python")
            gr1_repo = _path(args.gr1_repo, "--gr1-repo")
            if not (gr1_repo / "robocasa/utils/gym_utils").is_dir():
                raise ValueError("--gr1-repo is not the GR1 tabletop fork")
            port = pick_free_port()
            cache = Path(output_dir) / "env-cache"
            cache.mkdir(parents=True, exist_ok=True)
            environment = {
                "PYTHONPATH": f"{gr1_repo}{os.pathsep}{root}",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": str(egl_device),
                "NUMBA_CACHE_DIR": str(cache / "numba"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
            }
            if env_device is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(env_device)
            if args.env_ld_library_path:
                environment["LD_LIBRARY_PATH"] = args.env_ld_library_path
            daemon = ProcessDaemon(
                "robocasa_gr1_env",
                [
                    str(python),
                    str(root / "robots/robocasa_gr1/env_server.py"),
                    "--task-name",
                    task_id(args.task_name),
                    "--eval-type",
                    args.eval_type,
                    "--seed",
                    str(args.seed),
                    "--max-episode-steps",
                    str(args.max_episode_steps),
                    "--joybrain-repo",
                    str(joybrain),
                    "--trace-dir",
                    str(Path(output_dir) / "trace"),
                    "--port",
                    str(port),
                    "--parent-watch",
                ],
                env_overrides=environment,
                log_path=str(Path(output_dir) / "env-server.log"),
            )
            for key in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "RPENT_BASELINE_API_KEY",
            ):
                daemon.subprocess_env.pop(key, None)
            rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
            clients.append(rpc)

            def spawn_env():
                daemon.start()
                return daemon, rpc

            try_spawn_server(owned, dashboard_events, "env", spawn_env)
            result["env"] = try_wait_server(
                owned,
                dashboard_events,
                "env",
                rpc,
                daemon,
                1800,
                post_fn=lambda: GR1EnvClient(
                    rpc,
                    expected_meta=env_contract(
                        args.task_name,
                        args.eval_type,
                        args.seed,
                        args.max_episode_steps,
                    ),
                ),
            )
        if "vla" in selected:
            checkpoint = _path(args.vla_model_path, "--vla-model-path")
            dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
            uri = args.vla_endpoint
            if uri is None:
                port = pick_free_port()
                model_python = _path(args.vla_python or sys.executable, "--vla-python")
                cache = Path(output_dir) / "vla-cache"
                environment = {
                    "PYTHONPATH": str(root),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                for variable, subdir in (
                    ("TORCH_EXTENSIONS_DIR", "torch"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                    ("TRITON_CACHE_DIR", "triton"),
                    ("CUDA_CACHE_PATH", "cuda"),
                    ("XDG_CACHE_HOME", "xdg"),
                ):
                    path = cache / subdir
                    path.mkdir(parents=True, exist_ok=True)
                    environment[variable] = str(path)
                (cache / "xdg/torch/kernels").mkdir(parents=True, exist_ok=True)
                if vla_device is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = str(vla_device)
                policy = ProcessDaemon(
                    "joybrain_gr1_vla",
                    [
                        str(model_python),
                        str(root / "robots/robocasa_gr1/vla_server.py"),
                        "--model-path",
                        str(checkpoint),
                        "--joybrain-repo",
                        str(joybrain),
                        "--port",
                        str(port),
                        "--parent-watch",
                    ],
                    env_overrides=environment,
                    log_path=str(Path(output_dir) / "vla-server.log"),
                )
                for key in (
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "RPENT_BASELINE_API_KEY",
                ):
                    policy.subprocess_env.pop(key, None)
                owned["vla"] = policy
                policy.start()
                deadline = time.monotonic() + 1800
                while True:
                    if policy.poll() is not None:
                        raise RuntimeError(
                            "GR1 VLA server exited; inspect vla-server.log"
                        )
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("GR1 VLA startup timed out")
                        time.sleep(0.5)
                uri = f"ws://127.0.0.1:{port}"
            if urlsplit(uri).scheme not in ("ws", "wss"):
                raise ValueError("--vla-endpoint must be a WebSocket URL")
            model = GR1VLAClient(
                uri, checkpoint=str(checkpoint), timeout=args.joybrain_timeout
            )
            clients.append(model)
            if args.replan_steps > model.chunk_size:
                raise ValueError("replan-steps exceeds the checkpoint chunk size")
            result["model"] = model
            write_json_atomic(Path(output_dir) / "policy-metadata.json", model.metadata)
            dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
        return list(owned.values()), result
    except BaseException:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        stop_owned_daemons(owned, dashboard_events)
        raise


def finalize_run(context):
    import json

    native_path = context.output_dir / "native_status.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else None
    measurement_valid = native is not None and not any(
        native.get(key, False)
        for key in (
            "execution_uncertain",
            "recording_failed",
            "client_execution_uncertain",
            "client_recording_failed",
        )
    )
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "platform": "robocasa_gr1",
            "task": dict(context.task_desc),
            "official_success": native["native_success"] if measurement_valid else None,
            "measurement_valid": measurement_valid,
            "success_source": "native GR1 info.success or native terminated, latched",
            "native_status": native,
            "error": context.agent_error,
            "elapsed_s": context.elapsed_s,
            "model": context.model,
            "planner": context.planner,
            "stats": dict(context.stats),
            "finish": dict(context.finish_result) if context.finish_result else None,
            "training_eligible": False,
        },
    )
