import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.recording import (
    CAMERAS,
    NativeRecorder,
    install_file_guards,
    load_metadata,
    validate_trace,
    write_json,
)


class NativeTask:
    def __init__(self):
        self.state = np.zeros(14, dtype=np.float32)
        self.take_action_cnt = 0
        self.step_lim = 100
        self.eval_success = False

    def get_instruction(self):
        return "native task goal"

    def get_obs(self):
        return {
            "joint_action": {"vector": self.state.copy()},
            "observation": {
                name: {"rgb": np.full((4, 4, 3), index, np.uint8)}
                for index, name in enumerate(CAMERAS.values())
            },
        }

    def take_action(self, action, action_type="qpos"):
        self.take_action_cnt += 1
        self.state = np.array(action, copy=True)


def test_atomic_and_vla_actions_have_aligned_observations_and_distinct_sources(
    tmp_path,
):
    task = NativeTask()
    recorder = NativeRecorder(tmp_path / "trace", {"task": "fixture"})
    recorder.attach(task)
    with pytest.raises(RuntimeError):
        task.take_action(np.zeros(14))
    recorder = NativeRecorder(tmp_path / "valid", {"task": "fixture"})
    task = NativeTask()
    recorder.attach(task)
    recorder.begin("lingbot_act", {"chunks": 1})
    proposal = np.full((16, 14), 0.25)
    recorder.proposal(task.get_instruction(), proposal)
    task.take_action(proposal[0])
    recorder.end()
    recorder.begin("move_to", {"arm": "left", "xyz": [0, 0, 1]})
    task.take_action(np.full(14, 0.5))
    recorder.end()
    report = validate_trace(tmp_path / "valid")
    assert report == {
        "passed": True,
        "executed_actions": 2,
        "last_native_count": 2,
        "complete_16_action_windows": 0,
    }
    first = json.loads((tmp_path / "valid/step_000000/transition.json").read_text())
    second = json.loads((tmp_path / "valid/step_000001/transition.json").read_text())
    np.testing.assert_array_equal(first["before"]["qpos_target14"], np.zeros(14))
    np.testing.assert_array_equal(second["before"]["qpos_target14"], proposal[0])
    assert first["source"] == "vla" and second["source"] == "move_to"
    assert first["tool_id"] != second["tool_id"]
    assert first["effective_vla_instruction"] == "native task goal"
    assert recorder.summary()["actions_by_source"] == {"vla": 1, "move_to": 1}


def test_unknown_native_execution_is_never_replayed_or_exported(tmp_path):
    task = NativeTask()

    def fail(action, **kwargs):
        task.take_action_cnt += 1
        raise RuntimeError("native failure after mutation")

    task.take_action = fail
    recorder = NativeRecorder(tmp_path / "trace", {})
    recorder.attach(task)
    recorder.begin("set_gripper", {})
    with pytest.raises(RuntimeError, match="native failure"):
        task.take_action(np.zeros(14))
    with pytest.raises(RuntimeError):
        task.take_action(np.zeros(14))
    assert task.take_action_cnt == 1 and recorder.failed
    with pytest.raises(ValueError):
        validate_trace(tmp_path / "trace")


def test_observation_io_failure_prevents_native_action(tmp_path, monkeypatch):
    task = NativeTask()
    recorder = NativeRecorder(tmp_path / "trace", {})
    recorder.attach(task)
    recorder.begin("move_to", {})

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(recorder, "snapshot", fail)
    with pytest.raises(OSError):
        task.take_action(np.zeros(14))
    assert recorder.failed and task.take_action_cnt == 0


def test_metadata_does_not_persist_inline_keys(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"nested": [{"api_key": "do-not-persist"}]}))
    with pytest.raises(ValueError) as error:
        load_metadata(path)
    assert "do-not-persist" not in str(error.value)


def test_file_guard_blocks_credentials_escape_and_evidence_writes(
    tmp_path, monkeypatch
):
    from rpent.utils import config

    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(config, "get_repo_root", lambda: repo)
    handlers = {
        name: ({"name": name}, lambda **kwargs: kwargs)
        for name in ("read_text_file", "write_text_file", "list_dir")
    }
    toolkit = SimpleNamespace(_tools=handlers)
    toolkit.add_tool = lambda name, spec, handler: handlers.update(
        {name: (spec, handler)}
    )
    install_file_guards(toolkit, output, tmp_path / "memory")
    read = handlers["read_text_file"][1]
    write = handlers["write_text_file"][1]
    assert read(path="README.md")["path"] == "README.md"
    with pytest.raises(PermissionError):
        read(path=str(repo / ".env"))
    secret = tmp_path / "secret"
    secret.write_text("private")
    (repo / "escaped").symlink_to(secret)
    with pytest.raises(PermissionError):
        read(path=str(repo / "escaped"))
    with pytest.raises(PermissionError):
        write(path=str(output / "result.json"), content="overwrite")
    assert write(path=str(output / "agent_notes/notes.md"), content="note")


def test_recording_rpc_failure_latches_client(tmp_path):
    from robots.robotwin.env_client import RoboTwinEnvClient

    env = object.__new__(RoboTwinEnvClient)
    env.recording_failed = False
    env.execution_capabilities = {"native_action_recording": True}
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise TimeoutError("recorder unavailable")

    env._client = SimpleNamespace(call=fail)
    with pytest.raises(TimeoutError):
        env.trace("begin", "move_to", {})
    with pytest.raises(RuntimeError):
        env.trace("begin", "move_to", {})
    assert len(calls) == 1 and env.recording_failed


def test_recorded_short_success_is_valid_even_without_training_window(tmp_path):
    from robots.robotwin.evaluate import inspect_episode

    task = NativeTask()
    recorder = NativeRecorder(tmp_path / "trace", {})
    recorder.attach(task)
    recorder.begin("set_gripper", {})
    task.eval_success = True
    task.take_action(np.zeros(14))
    recorder.end()
    write_json(tmp_path / "result.json", {"official_success": True, "error": None})
    write_json(
        tmp_path / "native_status.json",
        {
            "episode_status": {"take_action_cnt": 1},
            "recording_failed": False,
            "execution_uncertain": False,
        },
    )
    result = inspect_episode(tmp_path, {"returncode": 0, "runner_error": None})
    assert result["loop_verified"] and result["official_success"]
    assert result["trace_validation"]["complete_16_action_windows"] == 0
