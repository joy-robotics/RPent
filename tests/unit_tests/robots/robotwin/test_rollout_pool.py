import json
import os
import subprocess
import sys

import pytest

from robots.robotwin.rollout_pool import (
    GPU,
    JobLaunch,
    WorkerSlot,
    make_gpu_groups,
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


def test_layouts_use_explicit_gpu_order():
    gpus = [GPU(index, f"GPU-{index}", 32607, 32110) for index in range(8)]
    colocated = make_gpu_groups(gpus, "colocated")
    assert [(group.model_gpu.index, group.env_gpu.index) for group in colocated] == [
        (index, index) for index in range(8)
    ]
    split = make_gpu_groups(gpus, "split")
    assert [(group.model_gpu.index, group.env_gpu.index) for group in split] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    reversed_pair = make_gpu_groups(gpus[1::-1], "split")[0]
    assert reversed_pair.model_gpu.index == 1 and reversed_pair.env_gpu.index == 0


@pytest.mark.parametrize(
    "gpus,layout",
    [
        ([], "split"),
        (INVENTORY[:1], "split"),
        (INVENTORY * 2, "split"),
        (INVENTORY, "unknown"),
    ],
)
def test_invalid_groups_fail_before_starting(gpus, layout):
    with pytest.raises(ValueError):
        make_gpu_groups(gpus, layout)


def test_staggered_startup_does_not_serialize_execution_or_other_gpus(tmp_path):
    slots = [
        WorkerSlot("gpu0-first", "GPU-0", "fixture://model0", "GPU-model0"),
        WorkerSlot("gpu0-second", "GPU-0", "fixture://model0", "GPU-model0"),
        WorkerSlot("gpu1-first", "GPU-1", "fixture://model1", "GPU-model1"),
    ]
    starts = []
    gate = tmp_path / "gate"
    release = tmp_path / "release"

    def build(job, slot):
        ready = tmp_path / f"{job['job_id']}.ready"
        code = (
            "from pathlib import Path\nimport time,sys\n"
            "gate,ready,release=map(Path,sys.argv[1:])\n"
            "while not gate.exists(): time.sleep(0.01)\n"
            "ready.write_text('ready')\n"
            "while not release.exists(): time.sleep(0.01)\n"
        )
        return JobLaunch(
            [sys.executable, "-c", code, str(gate), str(ready), str(release)],
            dict(os.environ),
            tmp_path / f"{job['job_id']}.log",
            15,
            ready_path=ready,
            startup_timeout_s=10,
        )

    def on_start(job, slot, pid):
        starts.append((job["job_id"], slot.slot_id, pid))
        if len(starts) == 2:
            assert [item[1] for item in starts] == ["gpu0-first", "gpu1-first"]
            gate.touch()
        if len(starts) == 3:
            assert slot.slot_id == "gpu0-second"
            assert (tmp_path / "first.ready").exists()
            os.kill(starts[0][2], 0)
            release.touch()

    rows = run_pool(
        [{"job_id": name} for name in ("first", "second", "third")],
        slots,
        build,
        max_active=3,
        on_start=on_start,
        poll_s=0.01,
    )
    assert len(rows) == 3 and all(row["returncode"] == 0 for row in rows)
    assert {row["model_gpu_uuid"] for row in rows} == {"GPU-model0", "GPU-model1"}


def test_failed_startup_releases_slot_without_replaying_episode(tmp_path):
    launches = []

    def build(job, slot):
        launches.append(job["job_id"])
        return JobLaunch(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            dict(os.environ),
            tmp_path / f"{job['job_id']}.log",
            10,
            ready_path=tmp_path / f"{job['job_id']}.ready",
            startup_timeout_s=0.1,
        )

    rows = run_pool(
        [{"job_id": "first"}, {"job_id": "second"}],
        [WorkerSlot("slot", "GPU-0", "fixture://model")],
        build,
        max_active=1,
        poll_s=0.01,
    )
    assert launches == ["first", "second"]
    assert all(row["runner_error"] == "startup_timeout" for row in rows)


def test_stale_readiness_file_is_rejected(tmp_path):
    ready = tmp_path / "ready"
    ready.touch()
    with pytest.raises(ValueError, match="already exists"):
        run_pool(
            [{"job_id": "only"}],
            [WorkerSlot("slot", "GPU", "fixture://model")],
            lambda job, slot: JobLaunch(
                [sys.executable, "-c", "pass"],
                dict(os.environ),
                tmp_path / "log",
                10,
                ready_path=ready,
            ),
            max_active=1,
        )
    assert not (tmp_path / "log").exists()
