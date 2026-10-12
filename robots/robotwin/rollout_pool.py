"""GPU selection and isolated subprocess scheduling for recorded rollouts."""

from __future__ import annotations

import csv
import math
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class GPUGroup:
    """One policy GPU and the GPU assigned to its environment workers."""

    model_gpu: GPU
    env_gpu: GPU


def make_gpu_groups(gpus: Sequence[GPU], layout: str) -> list[GPUGroup]:
    """Build explicit, disjoint groups from an already validated GPU selection."""
    if not gpus or len({gpu.uuid for gpu in gpus}) != len(gpus):
        raise ValueError("GPU groups require distinct selected physical GPUs")
    if layout == "colocated":
        return [GPUGroup(gpu, gpu) for gpu in gpus]
    if layout != "split":
        raise ValueError(f"Unknown GPU layout: {layout}")
    if len(gpus) % 2:
        raise ValueError("Split layout requires model/env pairs: an even GPU count")
    return [GPUGroup(gpus[index], gpus[index + 1]) for index in range(0, len(gpus), 2)]


def discover_gpus() -> list[GPU]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        GPU(int(row[0]), row[1].strip(), int(row[2]), int(row[3]))
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]


def select_gpus(
    requested: Sequence[str], inventory: Sequence[GPU], visibility: str | None = None
) -> list[GPU]:
    by_name = {name: gpu for gpu in inventory for name in (str(gpu.index), gpu.uuid)}
    if not requested:
        raise ValueError("At least one physical GPU index or full UUID is required")
    selected = []
    for name in requested:
        if name not in by_name:
            raise ValueError(
                f"GPU {name!r} is unavailable; no substitute GPU will be used"
            )
        selected.append(by_name[name])
    if len({gpu.uuid for gpu in selected}) != len(selected):
        raise ValueError(
            "Duplicate physical GPU; use workers-per-gpu for shared-card workers"
        )
    if visibility is not None:
        allowed = set()
        for name in visibility.split(","):
            name = name.strip()
            if name in by_name:
                allowed.add(by_name[name].uuid)
            elif name not in ("", "-1"):
                raise ValueError("Cannot safely resolve inherited CUDA_VISIBLE_DEVICES")
        if any(gpu.uuid not in allowed for gpu in selected):
            raise ValueError("Requested GPU is outside inherited CUDA_VISIBLE_DEVICES")
    return selected


@dataclass(frozen=True)
class WorkerSlot:
    slot_id: str
    gpu_uuid: str
    endpoint: str
    model_gpu_uuid: str | None = None

    @property
    def policy_gpu_uuid(self) -> str:
        """Return the policy GPU, retaining compatibility with colocated slots."""
        return self.model_gpu_uuid or self.gpu_uuid


@dataclass(frozen=True)
class JobLaunch:
    command: Sequence[str]
    environment: Mapping[str, str]
    log_path: Path
    timeout_s: float
    ready_path: Path | None = None
    startup_timeout_s: float | None = None


@dataclass
class ActiveJob:
    job: dict
    slot: WorkerSlot
    process: subprocess.Popen
    log: object
    started: float
    started_ns: int
    timeout_s: float
    ready_path: Path | None
    startup_timeout_s: float | None
    ready: bool
    ready_ns: int | None = None


def stop_process_group(process: subprocess.Popen, grace_s: float = 3.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_pool(
    jobs: Sequence[dict],
    slots: Sequence[WorkerSlot],
    build_launch: Callable[[dict, WorkerSlot], JobLaunch],
    *,
    max_active: int,
    on_start: Callable[[dict, WorkerSlot, int], None] | None = None,
    on_finish: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    poll_s: float = 0.1,
    max_initializing_per_gpu: int = 1,
) -> list[dict]:
    if not slots or not 1 <= max_active <= len(slots):
        raise ValueError("max_active must be within the number of worker slots")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("Worker slot IDs must be unique")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs; automatic repeats are not permitted")
    if max_initializing_per_gpu < 1:
        raise ValueError("max_initializing_per_gpu must be positive")
    pending = deque(jobs)
    available = deque(slots)
    active: dict[str, ActiveJob] = {}
    results = []
    try:
        while pending or active:
            stop_requested = should_stop()
            for running in active.values():
                if (
                    not running.ready
                    and running.ready_path is not None
                    and running.ready_path.is_file()
                ):
                    running.ready = True
                    running.ready_ns = time.time_ns()
            while (
                pending
                and available
                and len(active) < max_active
                and not stop_requested
            ):
                slot = None
                for _ in range(len(available)):
                    candidate = available.popleft()
                    initializing = sum(
                        not running.ready
                        and running.slot.gpu_uuid == candidate.gpu_uuid
                        for running in active.values()
                    )
                    if initializing < max_initializing_per_gpu:
                        slot = candidate
                        break
                    available.append(candidate)
                if slot is None:
                    break
                job = pending.popleft()
                launch = build_launch(job, slot)
                if not math.isfinite(launch.timeout_s) or launch.timeout_s <= 0:
                    raise ValueError("Job timeout must be positive and finite")
                if launch.startup_timeout_s is not None and (
                    not math.isfinite(launch.startup_timeout_s)
                    or launch.startup_timeout_s <= 0
                    or launch.ready_path is None
                ):
                    raise ValueError(
                        "Startup timeout requires a readiness file and a positive finite value"
                    )
                if launch.ready_path is not None and launch.ready_path.exists():
                    raise ValueError("Worker readiness file already exists")
                launch.log_path.parent.mkdir(parents=True, exist_ok=True)
                log = launch.log_path.open("x")
                started, started_ns = time.monotonic(), time.time_ns()
                try:
                    process = subprocess.Popen(
                        list(launch.command),
                        env=dict(launch.environment),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[slot.slot_id] = ActiveJob(
                    job,
                    slot,
                    process,
                    log,
                    started,
                    started_ns,
                    launch.timeout_s,
                    launch.ready_path,
                    launch.startup_timeout_s,
                    launch.ready_path is None,
                )
                if on_start:
                    on_start(job, slot, process.pid)
                stop_requested = should_stop()
            for slot_id, running in list(active.items()):
                returncode = running.process.poll()
                timeout = (
                    returncode is None
                    and time.monotonic() - running.started >= running.timeout_s
                )
                startup_timeout = (
                    returncode is None
                    and not running.ready
                    and running.startup_timeout_s is not None
                    and time.monotonic() - running.started >= running.startup_timeout_s
                )
                if returncode is None and not timeout and not startup_timeout:
                    continue
                stop_process_group(running.process)
                running.log.close()
                result = {
                    "job_id": running.job["job_id"],
                    "slot_id": slot_id,
                    "gpu_uuid": running.slot.gpu_uuid,
                    "model_gpu_uuid": running.slot.policy_gpu_uuid,
                    "endpoint": running.slot.endpoint,
                    "pid": running.process.pid,
                    "returncode": running.process.returncode,
                    "runner_error": "startup_timeout"
                    if startup_timeout
                    else "wall_timeout"
                    if timeout
                    else None,
                    "ready_ns": running.ready_ns,
                    "started_ns": running.started_ns,
                    "finished_ns": time.time_ns(),
                    "wall_s": time.monotonic() - running.started,
                }
                del active[slot_id]
                available.append(running.slot)
                results.append(result)
                if on_finish:
                    on_finish(result)
            if not active and (not pending or stop_requested):
                break
            if active:
                time.sleep(poll_s)
    finally:
        for running in active.values():
            try:
                stop_process_group(running.process)
            finally:
                running.log.close()
    return results
