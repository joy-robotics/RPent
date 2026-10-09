"""Offline contracts for the in-tree JoyBrain integration."""

import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("msgpack")
pytest.importorskip("websockets", minversion="15")

from robots.robotwin import robot_spec
from robots.robotwin.joybrain_client import (
    JoyBrainVLAClient,
    observation_example,
    validate_actions,
    validate_metadata,
)
from robots.robotwin.joybrain_server import validate_checkpoint
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.utils.msgpack_numpy import pack, unpack


@pytest.fixture
def metadata():
    return {
        "protocol_version": "2.1",
        "policy_type": "joybrain_vla_mot",
        "embodiment_id": "robotwin_aloha",
        "stats_key": "robotwin_aloha",
        "action_dim": 14,
        "expected_state_dim": 14,
        "expected_num_input_images": 3,
        "chunk_size": 16,
        "returns": "actions",
        "preprocessing_owner": "server_canonical",
        "strict_input_validation": True,
        "ckpt_path": "/test/checkpoint",
    }


@pytest.fixture
def observation():
    return {
        "robot_state": {
            "qpos_target14": np.arange(14, dtype=np.float32),
            "left_eef_pose": np.ones(7) * 99,
            "right_eef_pose": np.ones(7) * 88,
        },
        "views": {
            camera: {"rgb": np.full((4, 4, 3), index, np.uint8)}
            for index, camera in enumerate(("head", "left_wrist", "right_wrist"))
        },
        "task_language": "complete native task",
    }


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "pretrained_model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "joybrain_vla_mot",
                "n_obs_steps": 1,
                "max_num_views": 3,
                "chunk_size": 16,
            }
        )
    )
    (root / "stats.json").write_text(
        json.dumps(
            {
                "robotwin_aloha": {
                    key: {"mean": [0] * 14, "std": [1] * 14}
                    for key in ("observation.state", "action")
                }
            }
        )
    )
    (root / "model.safetensors").write_bytes(b"not-real-weights-unit-test-only")
    return root


def arguments(extra=()):
    from rpent.cli.main import _build_argparser

    parser = _build_argparser()
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(
        ["--robot", "robotwin", "--task-name", "click_bell", "--seed", "0", *extra]
    )


def test_default_is_lingbot_and_model_environment_is_backend_specific(monkeypatch):
    monkeypatch.setenv("LINGBOT_MODEL_PATH", "/lingbot-model")
    monkeypatch.setenv("JOYBRAIN_MODEL_PATH", "/joybrain-model")
    original = arguments()
    assert original.vla_backend == "lingbot"
    assert robot_spec._configured_model_path(original) == "/lingbot-model"
    assert robot_spec._parse_config(original).task_desc["action_layout"] == "eef16"
    joy = arguments(["--vla-backend", "joybrain"])
    assert robot_spec._configured_model_path(joy) == "/joybrain-model"
    assert robot_spec._parse_config(joy).task_desc["action_layout"] == "qpos14"


def test_identity_text_only_changes_for_joybrain():
    original = system_prompt({})
    joy = system_prompt({"vla_backend": "joybrain"})
    assert joy == {
        name: text.replace("LingBot-VLA", "Joy-Brain VLA")
        for name, text in original.items()
    }
    assert "RLinf/LingBot-VLA" in user_prompt({})["CELL"]
    assert "{{vla_checkpoint}}" in user_prompt({"vla_backend": "joybrain"})["CELL"]
    assert user_prompt({})["BEGIN"] == user_prompt({"vla_backend": "joybrain"})["BEGIN"]


def test_checkpoint_validation_and_spawn_are_in_tree(checkpoint, tmp_path, monkeypatch):
    from rpent.utils.daemon import ProcessDaemon

    source = tmp_path / "Joy-Brain-vla"
    server = source / "src/lerobot/serving/server_policy.py"
    server.parent.mkdir(parents=True)
    server.write_text("")
    monkeypatch.setattr(ProcessDaemon, "start", lambda self: None)
    args = arguments(
        [
            "--vla-backend",
            "joybrain",
            "--vla-model-path",
            str(checkpoint),
            "--joybrain-repo",
            str(source),
            "--vla-python",
            "/policy/python",
            "--vla-cuda-device",
            "2",
        ]
    )
    assert validate_checkpoint(checkpoint) == checkpoint
    daemon, endpoint = robot_spec._spawn_vla_server(args, tmp_path)
    assert daemon.cmd[0] == "/policy/python"
    assert Path(daemon.cmd[1]) == Path(robot_spec.__file__).resolve().with_name(
        "joybrain_server.py"
    )
    assert "--parent-watch" in daemon.cmd
    assert daemon.subprocess_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert not any("rpent_joybrain" in part for part in daemon.cmd)
    assert endpoint[0] == "127.0.0.1"


def test_wrong_embodiment_not_accepted_even_with_same_dimensions(checkpoint):
    path = checkpoint / "stats.json"
    stats = json.loads(path.read_text())
    path.write_text(json.dumps({"robodojo_arx_x5": stats["robotwin_aloha"]}))
    with pytest.raises(ValueError, match="robotwin_aloha"):
        validate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embodiment_id", "robodojo_arx_x5"),
        ("action_dim", 16),
        ("expected_state_dim", 16),
        ("expected_num_input_images", 1),
        ("policy_type", "mock_random"),
        ("chunk_size", 0),
        ("chunk_size", True),
        ("protocol_version", "2.0"),
        ("action_layout", "eef16"),
    ],
)
def test_bad_metadata_is_rejected(metadata, field, value):
    metadata[field] = value
    with pytest.raises(ValueError):
        validate_metadata(metadata)


def test_observation_layout_and_wire_format(observation):
    result = observation_example(observation, observation["task_language"])
    np.testing.assert_array_equal(
        result["state"], observation["robot_state"]["qpos_target14"]
    )
    assert [image[0, 0, 0] for image in result["image"]] == [0, 1, 2]
    decoded = unpack(pack(result))
    np.testing.assert_array_equal(decoded["state"], result["state"])
    np.testing.assert_array_equal(decoded["image"][2], result["image"][2])


def test_action_postprocess_does_not_denormalize_twice():
    values = np.full((1, 16, 14), 0.25, np.float32)
    values[..., 6] = 0.49
    values[..., 13] = 0.5
    result = validate_actions(values, 16)
    np.testing.assert_array_equal(result[:, :6], values[0, :, :6])
    assert np.all(result[:, 6] == 0) and np.all(result[:, 13] == 1)
    values[0, -1, 0] = np.nan
    with pytest.raises(ValueError):
        validate_actions(values, 16)


@pytest.fixture
def policy_server(metadata):
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    state = SimpleNamespace(requests=[], response_change=None)

    def handler(connection):
        try:
            connection.send(pack(metadata))
            for data in connection:
                query = unpack(data)
                state.requests.append(query)
                response = {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "protocol_version": "2.1",
                    "request_id": query["request_id"],
                    "data": {
                        "actions": np.zeros((1, 16, 14), np.float32),
                        "action_dim": 14,
                        "chunk_size": 16,
                    },
                }
                if state.response_change:
                    response = state.response_change(response)
                connection.send(pack(response))
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0, compression=None, ping_interval=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        state.endpoint = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        try:
            yield state
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_client_roundtrip_preserves_proxy_env(policy_server, observation, monkeypatch):
    import os

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    before = dict(os.environ)
    with JoyBrainVLAClient(
        policy_server.endpoint, expected_checkpoint="/test/checkpoint"
    ) as model:
        result = model.infer(observation)
        assert result.shape == (16, 14)
    assert dict(os.environ) == before
    assert (
        policy_server.requests[0]["payload"]["examples"][0]["lang"]
        == "complete native task"
    )


def test_wrong_checkpoint_rejected(policy_server):
    with pytest.raises(ValueError, match="checkpoint"):
        JoyBrainVLAClient(
            policy_server.endpoint, expected_checkpoint="/other/checkpoint"
        )


def test_stale_response_closes_without_replay(policy_server, observation):
    policy_server.response_change = lambda response: {**response, "request_id": "stale"}
    with JoyBrainVLAClient(policy_server.endpoint) as model:
        with pytest.raises(ValueError, match="request_id"):
            model.infer(observation)
        with pytest.raises(RuntimeError, match="closed"):
            model.infer(observation)
    assert len(policy_server.requests) == 1


class Environment:
    def __init__(self, observation):
        self.observation = observation
        self.last_info = {
            "robot_state": deepcopy(observation["robot_state"]),
            "episode_status": {
                "eval_success": False,
                "take_action_cnt": 0,
                "step_lim": 500,
                "actual_seed": 0,
            },
        }
        self.terminated = self.truncated = False
        self.execution_capabilities = {"chunk_step_all_frames": True}
        self.calls = []
        self.error = None
        self.success_after = None

    def render_camera(self, name):
        return self.observation["views"][name]["rgb"]

    def get_task_language(self):
        return self.observation["task_language"]

    def chunk_step(self, actions, *, action_type, return_all_frames):
        self.calls.append((actions.copy(), action_type))
        if self.error:
            raise self.error
        status = self.last_info["episode_status"]
        count = len(actions)
        if self.success_after is not None:
            count = min(count, self.success_after - status["take_action_cnt"])
        status["take_action_cnt"] += count
        if (
            self.success_after is not None
            and status["take_action_cnt"] >= self.success_after
        ):
            status["eval_success"] = self.terminated = True
        self.truncated = status["take_action_cnt"] >= status["step_lim"]
        if action_type == "qpos":
            self.last_info["robot_state"]["qpos_target14"] = actions[count - 1].copy()
        return (
            {},
            0,
            self.terminated,
            self.truncated,
            {"executed_actions": count, "episode_status": dict(status)},
        )


class Model:
    backend = "joybrain"
    chunk_size = 16

    def __init__(self):
        self.calls = []

    def infer(self, observation):
        self.calls.append(deepcopy(observation))
        return np.ones((self.chunk_size, 14), np.float32) * len(self.calls)


def primitives(env, model, check_cancelled=lambda: None):
    return RoboTwinPrimitives(
        env=env, model=model, seed=0, check_cancelled=check_cancelled
    )


def test_joybrain_fixed_window_native_instruction_and_fresh_state(observation):
    env, model = Environment(observation), Model()
    result = primitives(env, model).lingbot_act(
        chunks=1, prompt="ignored local instruction"
    )
    assert [len(actions) for actions, _ in env.calls] == [16, 16, 16, 2]
    assert all(kind == "qpos" for _, kind in env.calls)
    assert result["executed_steps"] == 50 and result["vla_inferences"] == 4
    assert result["prompt"] == observation["task_language"]
    assert result["agent_prompt_ignored"]
    np.testing.assert_array_equal(
        model.calls[1]["robot_state"]["qpos_target14"], np.ones(14)
    )


@pytest.mark.parametrize("stop", ["success", "budget", "terminal"])
def test_joybrain_stop_conditions(observation, stop):
    env, model = Environment(observation), Model()
    if stop == "success":
        env.success_after = 5
    elif stop == "budget":
        env.last_info["episode_status"]["step_lim"] = 5
    else:
        env.truncated = True
    result = primitives(env, model).lingbot_act(chunks=4)
    assert result["executed_steps"] == (0 if stop == "terminal" else 5)
    assert len(model.calls) <= 1


def test_cancel_after_infer_never_executes(observation):
    from rpent.tools.toolkit import ToolCancelled

    env, model = Environment(observation), Model()

    def cancel():
        if model.calls:
            raise ToolCancelled()

    with pytest.raises(ToolCancelled):
        primitives(env, model, cancel).lingbot_act(chunks=1)
    assert not env.calls


def test_unknown_execution_cannot_be_replayed(observation):
    env, model = Environment(observation), Model()
    env.error = TimeoutError("unknown")
    primitive = primitives(env, model)
    with pytest.raises(TimeoutError):
        primitive.lingbot_act(chunks=1)
    with pytest.raises(RuntimeError, match="unknown"):
        primitive.lingbot_act(chunks=1)
    assert len(env.calls) == len(model.calls) == 1


def test_default_legacy_model_still_uses_one_eef_chunk(observation):
    env = Environment(observation)
    model = SimpleNamespace(infer=lambda obs: np.zeros((50, 16)))
    result = primitives(env, model).lingbot_act(chunks=1)
    assert result["executed_steps"] == 50
    assert len(env.calls) == 1 and env.calls[0][1] == "ee"


@pytest.mark.parametrize("owned", [False, True])
def test_toolkit_only_closes_its_owned_model(owned):
    from robots.robotwin.toolkit import RoboTwinToolkit

    calls = []
    toolkit = object.__new__(RoboTwinToolkit)
    toolkit._primitives = SimpleNamespace(
        stop_recording=lambda: [],
        model=SimpleNamespace(
            close_on_toolkit_close=owned, close=lambda: calls.append("closed")
        ),
    )
    toolkit.close()
    assert len(calls) == int(owned)


@pytest.mark.parametrize(
    "options",
    [
        ["--vla-model-path", "/explicit", "--vla-backend", "joybrain"],
        ["--vla-backend", "joybrain", "--vla-model-path", "/explicit"],
    ],
)
def test_explicit_model_path_wins_regardless_of_argument_order(monkeypatch, options):
    monkeypatch.setenv("LINGBOT_MODEL_PATH", "/lingbot")
    monkeypatch.setenv("JOYBRAIN_MODEL_PATH", "/joybrain")
    assert robot_spec._configured_model_path(arguments(options)) == "/explicit"


@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_joybrain_keeps_upstream_prompt_mode(mode, monkeypatch):
    from robots.robotwin import prompt_bundle

    monkeypatch.setattr(
        prompt_bundle.evaluate_parts,
        "system_prompt",
        lambda variables: {"ROLE": "LingBot-VLA eval"},
    )
    monkeypatch.setattr(
        prompt_bundle.explore_parts,
        "system_prompt",
        lambda: {"ROLE": "LingBot-VLA explore"},
    )
    assert prompt_bundle.system_prompt({"mode": mode, "vla_backend": "joybrain"}) == {
        "ROLE": f"Joy-Brain VLA {mode}"
    }
    assert prompt_bundle.system_prompt({"mode": mode, "vla_backend": "lingbot"}) == {
        "ROLE": f"LingBot-VLA {mode}"
    }


@pytest.mark.parametrize(
    ("components", "explore", "owned"),
    [
        (None, False, True),
        (None, True, False),
        ({"vla"}, False, False),
        ({"vla"}, True, False),
    ],
)
def test_exploration_sessions_keep_shared_policy_connection(
    components, explore, owned, monkeypatch, tmp_path
):
    from robots.robotwin import joybrain_client
    from rpent.dashboard.events import NullDashboardEventSink

    args = arguments(["--vla-backend", "joybrain"])
    args.explore = explore
    monkeypatch.setattr(robot_spec, "_spawn_env_server", lambda *args: (None, object()))
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        robot_spec, "_spawn_vla_server", lambda *args: (None, ("127.0.0.1", 12345))
    )
    monkeypatch.setattr(robot_spec, "_wait_for_tcp", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        joybrain_client,
        "JoyBrainVLAClient",
        lambda endpoint, **kwargs: SimpleNamespace(**kwargs),
    )
    daemons, runtime = robot_spec._init_runtime(
        args, tmp_path, NullDashboardEventSink(), components
    )
    assert daemons == []
    assert runtime["model"].close_on_toolkit_close is owned
