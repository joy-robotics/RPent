import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rlinf.envs.sim.robotwin.robotwin_env")
pytest.importorskip("robotwin.config")

from robots.robotwin import env_server
from robots.robotwin.recording import (
    CAMERAS,
    NativeRecorder,
    validate_trace,
    write_json,
)
from robots.robotwin.resident_env import ResidentEnvFacade
from robots.robotwin.resident_protocol import EpisodeRpcClient
from robots.robotwin.robot_spec import env_runtime_contract


class Task:
    def __init__(self):
        self.take_action_cnt = 0
        self.eval_success = False
        self.step_lim = 256
        self.state = np.zeros(14, np.float32)
        self.instruction = ""

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        return {
            "joint_action": {"vector": self.state.copy()},
            "observation": {
                name: {"rgb": np.zeros((4, 4, 3), np.uint8)}
                for name in CAMERAS.values()
            },
        }

    def take_action(self, action, action_type="qpos"):
        self.take_action_cnt += 1
        self.state = np.asarray(action, dtype=np.float32)


class SubEnv:
    def __init__(self, seed):
        self.task = Task()
        self.args = {"step_lim": 256}
        self.env_seed = seed
        self.info_seed = seed
        self.reset_count = 0
        self.prewalks = []

    def setup_task(self):
        self.prewalks.append(self.env_seed)
        self.info_seed = self.env_seed
        self.task = Task()

    def close(self, clear_cache=True):
        pass


class Env:
    def __init__(self, seed=100000):
        self.venv = SimpleNamespace(envs=[SubEnv(seed)])
        self.bad_reset = False

    def reset(self, *, env_idx, env_seeds):
        sub = self.venv.envs[0]
        sub.task.take_action_cnt = 7 if self.bad_reset else 0
        sub.task.eval_success = False
        sub.task.state = np.zeros(14, np.float32)
        sub.task.instruction = f"seed {env_seeds[0]} info {sub.info_seed}"
        observation = {"main_images": np.zeros((1, 4, 4, 3), np.uint8)}
        return observation, {
            "episode_status": {
                "actual_seed": env_seeds[0],
                "take_action_cnt": sub.task.take_action_cnt,
                "eval_success": sub.task.eval_success,
                "step_lim": 256,
            }
        }

    def get_task_language(self, env_id):
        return self.venv.envs[env_id].task.instruction


class Transport:
    def __init__(self, facade):
        self.facade = facade

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        return self.facade._dispatch(method, args, kwargs or {})


@pytest.fixture
def resident(tmp_path, monkeypatch):
    monkeypatch.setattr(env_server, "_evaluation_language", lambda *args: None)
    env = Env()
    facade = ResidentEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name="click_bell",
            task_config="demo_randomized",
            seed=100000,
            max_episode_steps=256,
            record_actions=True,
        ),
        output_root=tmp_path,
    )
    return env, facade, Transport(facade)


def begin(root, transport, seed):
    directory = root / str(seed)
    directory.mkdir()
    metadata = directory / "config.json"
    write_json(metadata, {"cell": {"task": "click_bell", "seed": seed}})
    prepared = transport.call(
        "resident.begin", (seed, str(directory / "trace"), str(metadata))
    )
    return directory, prepared, EpisodeRpcClient(transport, prepared["episode_id"])


def test_two_episodes_reset_language_counters_and_recorders_without_stale_access(
    tmp_path, resident
):
    env, facade, transport = resident
    first, prepared, client = begin(tmp_path, transport, 100000)
    _, info = client.call("env.reset")
    assert info["episode_status"]["take_action_cnt"] == 0
    client.call("trace.begin", ("lingbot_act", {}))
    env.venv.envs[0].task.take_action(np.ones(14))
    client.call("trace.end")
    client.call("resident.end")
    saved = (first / "trace/summary.json").read_bytes()
    old_id = prepared["episode_id"]
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.call("env.reset")
    second, prepared, client = begin(tmp_path, transport, 100001)
    assert prepared["episode_id"] != old_id and prepared["completed_episodes"] == 1
    with pytest.raises(RuntimeError, match="Stale"):
        transport.call("trace.begin", ("move_to", {}), {"_episode_id": old_id})
    _, info = client.call("env.reset")
    assert info["instruction"] == "seed 100001 info 100001"
    assert info["episode_status"]["take_action_cnt"] == 0
    assert not info["episode_status"]["eval_success"]
    assert env.venv.envs[0].prewalks == [100001]
    with pytest.raises(RuntimeError, match="Reset within"):
        client.call("env.reset")
    client.call("trace.begin", ("move_to", {}))
    env.venv.envs[0].task.take_action(np.full(14, 0.5))
    client.call("trace.end")
    client.call("resident.end")
    assert (first / "trace/summary.json").read_bytes() == saved
    assert validate_trace(first / "trace")["executed_actions"] == 1
    assert validate_trace(second / "trace")["executed_actions"] == 1
    assert "take_action" not in vars(env.venv.envs[0].task)
    assert facade._completed == 2


def test_active_or_uncertain_recording_cannot_be_reassigned(tmp_path, resident):
    env, facade, transport = resident
    directory, prepared, client = begin(tmp_path, transport, 100000)
    client.call("env.reset")
    with pytest.raises(RuntimeError, match="Previous episode"):
        transport.call(
            "resident.begin",
            (100001, str(tmp_path / "other"), str(directory / "config.json")),
        )
    client.call("trace.begin", ("lingbot_act", {}))
    with pytest.raises(RuntimeError, match="active or failed"):
        client.call("resident.end")
    with pytest.raises(RuntimeError, match="poisoned"):
        client.call("env.get_env_meta")


def test_failed_reset_poisoning_prevents_a_second_attempt(tmp_path, resident):
    env, facade, transport = resident
    env.bad_reset = True
    directory, prepared, client = begin(tmp_path, transport, 100000)
    with pytest.raises(RuntimeError, match="retained"):
        client.call("env.reset")
    with pytest.raises(RuntimeError, match="poisoned"):
        client.call("env.reset")


def test_recorder_detach_preserves_methods_and_refuses_double_wrapping(tmp_path):
    task = Task()
    task.instruction = "goal"
    first = NativeRecorder(tmp_path / "first", {})
    first.attach(task)
    second = NativeRecorder(tmp_path / "second", {})
    with pytest.raises(RuntimeError, match="already attached"):
        second.attach(task)
    first.detach()
    assert "take_action" not in vars(task)
    second.attach(task)
    second.begin("move_to", {})
    task.take_action(np.zeros(14))
    second.end()
    second.detach()
    assert (
        json.loads((tmp_path / "first/summary.json").read_text())["executed_actions"]
        == 0
    )
    assert (
        json.loads((tmp_path / "second/summary.json").read_text())["executed_actions"]
        == 1
    )


def test_unscoped_or_pre_reset_requests_cannot_mutate(tmp_path, resident):
    env, facade, transport = resident
    _, _, client = begin(tmp_path, transport, 100000)
    with pytest.raises(RuntimeError, match="missing episode"):
        transport.call("env.reset")
    with pytest.raises(RuntimeError, match="reset has not completed"):
        client.call("trace.begin", ("lingbot_act", {}))
