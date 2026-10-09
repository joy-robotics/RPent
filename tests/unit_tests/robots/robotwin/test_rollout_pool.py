import json
import os
import subprocess
import sys

import pytest

from robots.robotwin.rollout_pool import (
    GPU,
    JobLaunch,
    WorkerSlot,
    run_pool,
    select_gpus,
)

INVENTORY = [
    GPU(0, "GPU-cpu-fixture-0", 80000, 70000),
    GPU(1, "GPU-cpu-fixture-1", 80000, 70000),
]


def test_select_two_distinct_physical_gpus_cpu_only():
    selected = select_gpus(["1", "GPU-cpu-fixture-0"], INVENTORY)
    assert [gpu.index for gpu in selected] == [1, 0]


@pytest.mark.parametrize(
    "requested", [["1"], ["0", "0"], ["0", "GPU-cpu-fixture-0"], []]
)
def test_single_gpu_cannot_masquerade_as_two(requested):
    with pytest.raises(ValueError):
        select_gpus(requested, INVENTORY[:1])


@pytest.mark.parametrize("visibility", ["0", "GPU-cpu-fixture-0", "", "-1"])
def test_inherited_visibility_is_not_broadened(visibility):
    with pytest.raises(ValueError):
        select_gpus(["1"], INVENTORY, visibility)


def launch_factory(root):
    def build(job, slot):
        code = (
            "import json,os,time,sys; "
            "print(json.dumps({'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'endpoint':os.environ['TEST_ENDPOINT']}),flush=True); "
            f"time.sleep({job.get('sleep', 0.05)}); sys.exit({job.get('exit', 0)})"
        )
        return JobLaunch(
            [sys.executable, "-c", code],
            {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": slot.gpu_uuid,
                "TEST_ENDPOINT": slot.endpoint,
            },
            root / f"{job['job_id']}.log",
            job.get("timeout", 5),
        )

    return build


def test_two_gpu_slots_dispatch_independent_processes_cpu_only(tmp_path):
    slots = [
        WorkerSlot("slot0", INVENTORY[0].uuid, "fixture://policy0"),
        WorkerSlot("slot1", INVENTORY[1].uuid, "fixture://policy1"),
    ]
    jobs = [
        {"job_id": "slow", "sleep": 0.6},
        {"job_id": "fast", "sleep": 0.05},
        {"job_id": "next", "sleep": 0.05},
        {"job_id": "last", "sleep": 0.05},
    ]
    rows = run_pool(jobs, slots, launch_factory(tmp_path), max_active=2, poll_s=0.005)
    by_id = {row["job_id"]: row for row in rows}
    assert len(rows) == len({row["job_id"] for row in rows}) == 4
    assert by_id["next"]["slot_id"] == "slot1"
    assert by_id["fast"]["started_ns"] < by_id["slow"]["finished_ns"]
    assert by_id["slow"]["started_ns"] < by_id["fast"]["finished_ns"]
    assert len({row["pid"] for row in rows}) == 4
    for row in rows:
        actual = json.loads((tmp_path / f"{row['job_id']}.log").read_text())
        assert actual == {"gpu": row["gpu_uuid"], "endpoint": row["endpoint"]}


def test_two_workers_can_share_one_gpu_without_claiming_two_cards(tmp_path):
    slots = [
        WorkerSlot(f"slot{index}", INVENTORY[0].uuid, "fixture://shared-policy")
        for index in range(2)
    ]
    rows = run_pool(
        [{"job_id": "first"}, {"job_id": "second"}],
        slots,
        launch_factory(tmp_path),
        max_active=2,
    )
    assert len({row["gpu_uuid"] for row in rows}) == 1
    assert len({row["endpoint"] for row in rows}) == 1
    assert len({row["slot_id"] for row in rows}) == 2


def test_failure_and_timeout_do_not_stop_other_jobs(tmp_path):
    outsider = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    slots = [
        WorkerSlot(f"slot{index}", INVENTORY[0].uuid, "fixture://policy")
        for index in range(2)
    ]
    jobs = [
        {"job_id": "timeout", "sleep": 10, "timeout": 0.1},
        {"job_id": "failed", "exit": 7},
        {"job_id": "healthy"},
    ]
    try:
        rows = {
            row["job_id"]: row
            for row in run_pool(jobs, slots, launch_factory(tmp_path), max_active=2)
        }
        assert rows["timeout"]["runner_error"] == "wall_timeout"
        assert rows["failed"]["returncode"] == 7
        assert rows["healthy"]["returncode"] == 0
        assert outsider.poll() is None
    finally:
        outsider.terminate()
        outsider.wait()


def test_active_session_cap_and_stop_after_current(tmp_path):
    slots = [
        WorkerSlot(f"slot{index}", INVENTORY[0].uuid, "fixture://policy")
        for index in range(2)
    ]
    stop = []
    rows = run_pool(
        [{"job_id": "first"}, {"job_id": "second"}],
        slots,
        launch_factory(tmp_path),
        max_active=1,
        on_start=lambda *args: stop.append(True),
        should_stop=lambda: bool(stop),
    )
    assert [row["job_id"] for row in rows] == ["first"]
    assert not (tmp_path / "second.log").exists()


def test_duplicate_jobs_rejected_before_launch(tmp_path):
    with pytest.raises(ValueError, match="Duplicate job"):
        run_pool(
            [{"job_id": "same"}, {"job_id": "same"}],
            [WorkerSlot("slot", INVENTORY[0].uuid, "fixture://policy")],
            launch_factory(tmp_path),
            max_active=1,
        )
    assert not list(tmp_path.iterdir())


def test_callback_failure_cleans_up_owned_process(tmp_path):
    spawned = []

    def abort(job, slot, pid):
        spawned.append(pid)
        raise RuntimeError("controller stopped")

    with pytest.raises(RuntimeError, match="controller stopped"):
        run_pool(
            [{"job_id": "child", "sleep": 30}],
            [WorkerSlot("slot", INVENTORY[0].uuid, "fixture://policy")],
            launch_factory(tmp_path),
            max_active=1,
            on_start=abort,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(spawned[0], 0)
