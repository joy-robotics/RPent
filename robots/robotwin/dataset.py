"""Export confirmed qpos14 transitions to the existing JoyBrain LeRobot/transform pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robots.robotwin.recording import validate_trace
from rpent.utils.logging import get_logger

logger = get_logger("robotwin.dataset")


def activate_joybrain(repo: Path) -> None:
    """Load dataset/transform code only from the explicitly selected model repository."""
    import sys

    source = repo.expanduser().resolve() / "src"
    expected = source / "lerobot/__init__.py"
    if not expected.is_file():
        raise ValueError("--joybrain-repo must contain src/lerobot")
    loaded = sys.modules.get("lerobot")
    if loaded is not None and Path(loaded.__file__).resolve() != expected:
        raise RuntimeError(
            "A different LeRobot is already imported; use an isolated Python environment"
        )
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source.parent))


IMAGE_KEYS = {
    "head": "observation.images.cam_high",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}


def read_transitions(trace, *, allow_validation=False):
    """Read confirmed native actions; never turn incomplete windows into labels."""
    trace = Path(trace)
    validate_trace(trace)
    status_path = trace.parent / "native_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("recording_failed") or status.get("execution_uncertain"):
            raise ValueError("Episode has uncertain execution or recording failure")
    episode = json.loads((trace / "episode.json").read_text())
    result_path = trace.parent / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if not allow_validation and (
        not isinstance(result.get("official_success"), bool) or result.get("error")
    ):
        raise ValueError(
            "A finalized native result without infrastructure errors is required"
        )
    records = [
        (path.parent, json.loads(path.read_text()))
        for path in sorted(trace.glob("step_*/transition.json"))
    ]
    if len(records) < 16:
        raise ValueError(
            "Need one complete 16-action training window; raw short episodes are retained"
        )
    return episode, records, False


def export(trace, output, checkpoint, joybrain_repo, *, allow_validation=False):
    """Export a validated candidate dataset, retaining success/failure labels."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    if config["chunk_size"] != 16:
        raise ValueError(
            "This exporter currently validates 16-action Joy-Brain windows only"
        )
    episode, records, eligible = read_transitions(
        trace, allow_validation=allow_validation
    )
    configuration = episode["run_config"]
    expected_checkpoint = configuration.get("checkpoint") or configuration.get(
        "platform_config", {}
    ).get("checkpoint")
    if (
        expected_checkpoint
        and Path(expected_checkpoint).resolve() != Path(checkpoint).resolve()
    ):
        raise ValueError("Export checkpoint differs from episode checkpoint")
    activate_joybrain(Path(joybrain_repo))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    first_directory, first = records[0]
    shape = np.asarray(
        Image.open(first_directory / first["before"]["camera_paths"]["head"])
    ).shape
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"qpos_{index}" for index in range(14)],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"target_{index}" for index in range(14)],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.terminated": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
        "action.source_id": {"dtype": "int64", "shape": (1,), "names": None},
        "action.tool_call_index": {"dtype": "int64", "shape": (1,), "names": None},
        "observation.wall_time_ns": {"dtype": "int64", "shape": (1,), "names": None},
        "execution.wall_duration_ns": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for key in IMAGE_KEYS.values():
        features[key] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        "rpent/robotwin_rollout",
        fps=1,
        root=output,
        robot_type="aloha",
        features=features,
        use_videos=False,
    )
    sources = sorted({record["source"] for _, record in records})
    tool_ids = list(dict.fromkeys(record["tool_id"] for _, record in records))
    try:
        for directory, record in records:
            instruction = (
                record["effective_vla_instruction"]
                if record["source"] == "vla"
                else record["before"]["instruction"]
            )
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Missing actual instruction")
            frame = {
                "observation.state": np.asarray(
                    record["before"]["qpos_target14"], np.float32
                ),
                "action": np.asarray(record["action"]["values"], np.float32),
                "task": instruction,
                "next.reward": np.asarray([record["reward"]], np.float32),
                "next.terminated": np.asarray([record["terminated"]], bool),
                "next.truncated": np.asarray([record["truncated"]], bool),
                "action.source_id": np.asarray(
                    [sources.index(record["source"])], np.int64
                ),
                "action.tool_call_index": np.asarray(
                    [tool_ids.index(record["tool_id"])], np.int64
                ),
                "observation.wall_time_ns": np.asarray(
                    [record["wall_start_ns"]], np.int64
                ),
                "execution.wall_duration_ns": np.asarray(
                    [record["wall_end_ns"] - record["wall_start_ns"]], np.int64
                ),
            }
            for camera, key in IMAGE_KEYS.items():
                image = np.asarray(
                    Image.open(
                        directory / record["before"]["camera_paths"][camera]
                    ).convert("RGB")
                )
                if image.shape != shape:
                    raise ValueError("Camera resolution changed within episode")
                frame[key] = image
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    events = Path(trace) / "events.jsonl"
    if events.is_file():
        (output / "meta/rpent_tool_calls.jsonl").write_text(events.read_text())
    final_path = Path(trace).parent / "result.json"
    final_result = json.loads(final_path.read_text()) if final_path.exists() else None
    provenance = {
        "source_trace": str(Path(trace).resolve()),
        "episode": episode,
        "training_eligible": eligible,
        "requires_curation": True,
        "validation_only": allow_validation,
        "target": "discrete native-decision-step absolute qpos14 imitation",
        "time_axis": "decision index; fps=1 indexes decisions, NOT physical Hz; wall clocks retained",
        "physical_resampling": "not performed; not suitable for physical-time/action-dt training as-is",
        "action_units": "radians for 6 joints each arm, normalized grippers at 6 and 13",
        "sources": dict(enumerate(sources)),
        "tool_ids": dict(enumerate(tool_ids)),
        "camera_order": list(IMAGE_KEYS.values()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "complete_window_anchors": [0, len(records) - 16 + 1],
        "tail_policy": "retain raw tail, exclude incomplete 16-action windows from training",
        "success": bool(records[-1][1]["terminated"]),
        "rpent_final_result": final_result,
        "stop_reason": "native_success"
        if records[-1][1]["terminated"]
        else "native_budget"
        if records[-1][1]["truncated"]
        else "planner_stop",
        "failure_stop_reason": "see source result.json and evaluation.json; never infer task success from tool success",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def validate_batch(root, checkpoint, joybrain_repo):
    """Read a real Joy-Brain training batch and compare labels to raw actions."""
    activate_joybrain(Path(joybrain_repo))
    import torch
    from lerobot.datasets.factory import _multimodal_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.transformed_dataset import TransformedLeRobotDataset
    from lerobot.policies.joybrain_vla.transform_joybrain_vla import (
        JoyBrainVLATransform,
    )
    from torch.utils.data import DataLoader, Subset

    root = Path(root)
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    base = LeRobotDataset(
        "rpent/robotwin_rollout",
        root=root,
        delta_timestamps={"action": list(range(16))},
        video_backend="pyav",
    )
    raw = base[0]
    trace = Path(provenance["source_trace"])
    expected = [
        json.loads(path.read_text())
        for path in sorted(trace.glob("step_*/transition.json"))[:16]
    ]
    np.testing.assert_allclose(
        raw["observation.state"].numpy(),
        expected[0]["before"]["qpos_target14"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        raw["action"].numpy(),
        np.asarray([record["action"]["values"] for record in expected]),
        rtol=1e-6,
    )
    transform = JoyBrainVLATransform(
        vlm_model_name_or_path=config["vlm_model_name_or_path"],
        chunk_size=16,
        max_state_dim=config["max_state_dim"],
        max_action_dim=config["max_action_dim"],
        max_num_views=3,
        use_qwen_chat_template=config["use_qwen_chat_template"],
        prompt_format=config["prompt_format"],
        normalization_stats_path=str(Path(checkpoint) / "stats.json"),
        state_discretization_bins=config.get("state_discretization_bins", 256),
        physical_prompt_max_length=config.get("physical_prompt_max_length", 256),
        physical_prompt_encoding=config.get("physical_prompt_encoding", "legacy"),
    )
    dataset = TransformedLeRobotDataset.from_base(base, transforms=[transform])
    anchors = range(*provenance["complete_window_anchors"])
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, anchors),
                batch_size=min(2, len(anchors)),
                num_workers=0,
                collate_fn=_multimodal_collate,
            )
        )
    )
    assert batch["action"].shape[1:] == (16, config["max_action_dim"])
    assert batch["observation.state"].shape[1:] == (config["max_state_dim"],)
    assert torch.isfinite(batch["action"]).all() and batch["camera_mask"].all()
    stats = json.loads((Path(checkpoint) / "stats.json").read_text())["robotwin_aloha"]
    recovered = batch["action"][0, :, :14] * torch.tensor(
        stats["action"]["std"]
    ).clamp_min(1e-6) + torch.tensor(stats["action"]["mean"])
    np.testing.assert_allclose(
        recovered.numpy(),
        np.asarray([record["action"]["values"] for record in expected]),
        atol=2e-6,
    )
    report = {
        "passed": True,
        "loader": "original LeRobotDataset + TransformedLeRobotDataset + JoyBrainVLATransform + DataLoader",
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
        },
        "alignment_verified": "s_t before a_t; action targets a_t..a_(t+15); inverse normalization matches actual targets",
        "training_eligible": provenance["training_eligible"],
        "model_weights_loaded": False,
    }
    (root / "batch_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "validate"])
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--joybrain-repo", type=Path, required=True)
    parser.add_argument("--allow-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "export":
        if args.trace is None:
            parser.error("--trace is required")
        logger.info(
            "Exported %s",
            export(
                args.trace,
                args.dataset,
                args.checkpoint,
                args.joybrain_repo,
                allow_validation=args.allow_validation,
            ),
        )
    else:
        logger.info(
            "%s",
            json.dumps(
                validate_batch(args.dataset, args.checkpoint, args.joybrain_repo),
                indent=2,
            ),
        )


if __name__ == "__main__":
    main()
