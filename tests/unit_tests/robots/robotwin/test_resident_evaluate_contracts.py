import json

import pytest

from robots.robotwin import resident_evaluate
from robots.robotwin.rollout_pool import GPU


def test_eight_resident_workers_get_disjoint_multi_episode_shards(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        resident_evaluate,
        "discover_gpus",
        lambda: [GPU(index, f"GPU-{index}", 32607, 32110) for index in range(8)],
    )
    output = tmp_path / "plan"
    assert (
        resident_evaluate.main(
            [
                "--gpu-ids",
                "0,1,2,3,4,5,6,7",
                "--episodes-per-task",
                "20",
                "--dry-run",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["worker_capacity"] == plan["model_server_count"] == 8
    assert sorted(len(shard["episodes"]) for shard in plan["shards"]) == [
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]
    identifiers = [
        cell["job_id"] for shard in plan["shards"] for cell in shard["episodes"]
    ]
    assert len(identifiers) == len(set(identifiers)) == 20
    assert {shard["gpu"]["uuid"] for shard in plan["shards"]} == {
        f"GPU-{index}" for index in range(8)
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tasks", "all"],
        ["--tasks", "click_bell,lift_block"],
        ["--gpu-runtime-min-free-mib", "0"],
    ],
)
def test_resident_mode_rejects_unsupported_or_unprotected_runs(tmp_path, arguments):
    with pytest.raises(SystemExit):
        resident_evaluate.main(
            ["--dry-run", "--output-dir", str(tmp_path / "plan"), *arguments]
        )
