import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa_gr1 import robot_spec
from rpent.dashboard.events import NullDashboardEventSink


def arguments(tmp_path):
    parser = argparse.ArgumentParser()
    robot_spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "PnPCupToDrawerClose"])
    args.memory_profile = "local"
    args.joybrain_repo = str(tmp_path)
    args.vla_model_path = str(tmp_path)
    args.vla_endpoint = "ws://127.0.0.1:19001"
    args.gr1_repo = str(tmp_path / "gr1")
    (Path(args.gr1_repo) / "robocasa/utils/gym_utils").mkdir(parents=True)
    args.gr1_python = str(tmp_path / "python")
    Path(args.gr1_python).touch()
    return args


def test_memory_hook_uses_public_two_argument_contract(tmp_path):
    args = arguments(tmp_path)
    args.output_dir = str(tmp_path / "run")
    args.memory_dir = None
    spec = robot_spec.get_robot_spec()
    config = spec.parse_config(args)
    spec.prepare_memory(args, config)
    assert config.prompt_vars["memory_dir"] == str(tmp_path / "run/memory")


@pytest.mark.parametrize("native_success", [None, False, True])
def test_finalizer_uses_only_recorded_native_success(tmp_path, native_success):
    if native_success is not None:
        (tmp_path / "native_status.json").write_text(
            json.dumps({"native_success": native_success})
        )
    context = SimpleNamespace(
        output_dir=tmp_path,
        task_desc={"seed": 7},
        agent_error=None,
        elapsed_s=1,
        model="offline-test",
        planner="api",
        stats={},
        finish_result={"status": "success"},
    )
    result = json.loads(robot_spec.finalize_run(context).read_text())
    assert result["official_success"] is native_success
    assert not result["training_eligible"]


def test_uncertain_execution_is_not_a_native_failure_verdict(tmp_path):
    (tmp_path / "native_status.json").write_text(
        json.dumps({"native_success": False, "client_execution_uncertain": True})
    )
    context = SimpleNamespace(
        output_dir=tmp_path,
        task_desc={},
        agent_error="timeout",
        elapsed_s=1,
        model="offline-test",
        planner="api",
        stats={},
        finish_result=None,
    )
    result = json.loads(robot_spec.finalize_run(context).read_text())
    assert result["official_success"] is None
    assert not result["measurement_valid"]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_config_rejects_invalid_timeout(tmp_path, timeout):
    args = arguments(tmp_path)
    args.joybrain_timeout = timeout
    with pytest.raises(ValueError):
        robot_spec.validate_args(args)


@pytest.mark.usefixtures("joybrain_dependencies")
def test_borrowed_model_starts_no_daemon(tmp_path, monkeypatch):
    from robots.robocasa_gr1 import vla_client

    calls = []

    class Model:
        chunk_size = 16
        metadata = {"stats_key": "robocasa_gr1"}

        def __init__(self, uri, **kwargs):
            calls.append((uri, kwargs))

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(vla_client, "GR1VLAClient", Model)
    daemons, runtime = robot_spec.init_runtime(
        arguments(tmp_path), tmp_path, NullDashboardEventSink(), {"vla"}
    )
    assert daemons == [] and set(runtime) == {"model", "replan_steps"}
    assert calls[0][1]["checkpoint"] == str(tmp_path)
    runtime["model"].close()


@pytest.mark.usefixtures("joybrain_dependencies")
def test_borrowed_model_closed_on_chunk_mismatch(tmp_path, monkeypatch):
    from robots.robocasa_gr1 import vla_client

    closed = []

    class Model:
        chunk_size = 8

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(vla_client, "GR1VLAClient", Model)
    with pytest.raises(ValueError, match="chunk size"):
        robot_spec.init_runtime(
            arguments(tmp_path), tmp_path, NullDashboardEventSink(), {"vla"}
        )
    assert closed == [True]


@pytest.mark.usefixtures("joybrain_dependencies")
def test_failed_env_start_cleans_owned_process_and_keeps_secrets_out(
    tmp_path, monkeypatch
):
    from rpent.robots import runtime
    from rpent.utils import daemon
    from rpent.utils.rpc import http_rpc

    args = arguments(tmp_path)
    args.env_cuda_device = "3"
    args.env_ld_library_path = "/verified/native/lib"
    args.egl_device_id = 3
    created = []
    closed = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-secret")

    class Daemon:
        def __init__(self, name, cmd, **kwargs):
            self.cmd = cmd
            self.subprocess_env = {
                "OPENAI_API_KEY": "offline-test-secret",
                **kwargs["env_overrides"],
            }
            self.stopped = False
            created.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class Rpc:
        def __init__(self, uri):
            pass

        def close(self):
            closed.append(True)
            raise RuntimeError("close failed")

    def fail(*args, **kwargs):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(daemon, "ProcessDaemon", Daemon)
    monkeypatch.setattr(http_rpc, "HttpRpcClient", Rpc)
    monkeypatch.setattr(runtime, "try_wait_server", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        robot_spec.init_runtime(args, tmp_path, NullDashboardEventSink(), {"env"})
    assert closed and created[0].stopped
    environment = created[0].subprocess_env
    assert "OPENAI_API_KEY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert environment["LD_LIBRARY_PATH"] == "/verified/native/lib"
    assert created[0].cmd[0] == args.gr1_python
    assert "--parent-watch" in created[0].cmd
