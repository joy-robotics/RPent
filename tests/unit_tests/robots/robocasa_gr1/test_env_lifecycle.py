import json

import numpy as np
import pytest


def test_reset_warmups_budget_and_close(stack):
    native, facade, client = stack
    assert native.seeds == [7, 7, 7]
    payload, _, terminated, truncated, info = client.chunk_step(np.zeros((16, 29)))
    assert (
        len(native.actions) == info["executed_actions"] == len(payload["frames"]) == 3
    )
    assert not terminated and truncated
    assert not info["episode_status"]["native_success"]
    assert info["episode_status"]["budget_exhausted"]
    assert not info["episode_status"]["native_truncated"]
    with pytest.raises(RuntimeError):
        facade.reset()
    facade.close()
    facade.close()
    assert native.closes == 1


@pytest.mark.parametrize("flag", ["success", "terminated"])
def test_native_success_stops_chunk(stack, flag):
    native, facade, client = stack
    setattr(native, flag, True)
    _, _, terminated, _, info = client.chunk_step(np.zeros((3, 29)))
    assert terminated and info["episode_status"]["native_success"]
    assert len(native.actions) == 1
    trace = json.loads((facade.output / "step_000000/transition.json").read_text())
    assert trace["execution"] == "executed"
    assert len(trace["action29"]) == 29


def test_failed_recording_does_not_execute_or_overwrite(stack):
    native, facade, client = stack
    directory = facade.output / "step_000000"
    directory.mkdir()
    record = directory / "transition.json"
    record.write_text("preserve previous evidence")
    with pytest.raises(FileExistsError):
        client.chunk_step(np.zeros((1, 29)))
    assert not native.actions
    assert record.read_text() == "preserve previous evidence"
    assert client.execution_uncertain
    with pytest.raises(RuntimeError):
        client.chunk_step(np.zeros((1, 29)))


def test_uncertain_step_never_replayed(stack, monkeypatch):
    native, facade, client = stack

    def fail(action):
        native.actions.append(action)
        raise RuntimeError("lost response")

    monkeypatch.setattr(native, "step", fail)
    with pytest.raises(RuntimeError):
        client.step(np.zeros(29))
    with pytest.raises(RuntimeError):
        client.step(np.zeros(29))
    assert len(native.actions) == 1
    assert facade.get_status()["execution_uncertain"]


def test_client_rejects_counter_drift(stack, monkeypatch):
    _, facade, client = stack
    original = facade.chunk_step

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result[-1]["episode_status"]["step_count"] = 0
        return result

    monkeypatch.setattr(facade, "chunk_step", corrupt)
    with pytest.raises(ValueError):
        client.step(np.zeros(29))
    assert client.execution_uncertain
