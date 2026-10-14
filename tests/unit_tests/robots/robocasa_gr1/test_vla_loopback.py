import threading
from contextlib import contextmanager

import numpy as np
import pytest

from robots.robocasa_gr1.contracts import EMBODIMENT, ROBOT_TYPE, observation
from tests.unit_tests.robots.robocasa_gr1.conftest import raw_observation

pytestmark = pytest.mark.usefixtures("joybrain_dependencies")


@contextmanager
def endpoint(checkpoint, *, bad_meta=False, bad_actions=False, timeout=False):
    from websockets.sync.server import serve

    from rpent.utils.msgpack_numpy import pack, unpack

    calls = []
    release = threading.Event()

    def handler(connection):
        metadata = {
            "protocol_version": "2.1",
            "embodiment_id": EMBODIMENT,
            "stats_key": EMBODIMENT,
            "robot_type": ROBOT_TYPE,
            "expected_state_dim": 44,
            "action_dim": 12 if bad_meta else 29,
            "expected_num_input_images": 1,
            "preprocessing_owner": "server_canonical",
            "strict_input_validation": True,
            "returns": "actions",
            "policy_type": "joybrain_vla",
            "chunk_size": 16,
            "ckpt_path": str(checkpoint),
        }
        connection.send(pack(metadata))
        for payload in connection:
            request = unpack(payload)
            calls.append(request)
            if timeout:
                release.wait(3)
                return
            response = {
                "protocol_version": "2.1",
                "request_id": request["request_id"],
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "data": {
                    "action_dim": 29,
                    "chunk_size": 16,
                    "actions": np.zeros(
                        (1, 16, 12 if bad_actions else 29), dtype=np.float32
                    ),
                },
            }
            connection.send(pack(response))

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", calls
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=5)


def test_real_transport_keeps_canonical_state_and_image(tmp_path):
    from robots.robocasa_gr1.vla_client import GR1VLAClient

    with endpoint(tmp_path) as (uri, calls):
        client = GR1VLAClient(uri, checkpoint=str(tmp_path))
        observed = observation(raw_observation())
        assert client.infer(observed).shape == (16, 29)
        client.close()
        example = calls[0]["payload"]["examples"][0]
        np.testing.assert_array_equal(example["state"], observed["state44"])
        np.testing.assert_array_equal(example["image"][0], observed["ego_rgb"])


def test_incompatible_metadata_rejected_before_inference(tmp_path):
    from robots.robocasa_gr1.vla_client import GR1VLAClient

    with endpoint(tmp_path, bad_meta=True) as (uri, calls):
        with pytest.raises(ValueError):
            GR1VLAClient(uri, checkpoint=str(tmp_path))
        assert not calls


@pytest.mark.parametrize("mode", ["bad_actions", "timeout"])
def test_failed_inference_is_not_replayed(tmp_path, mode):
    from robots.robocasa_gr1.vla_client import GR1VLAClient

    with endpoint(tmp_path, **{mode: True}) as (uri, calls):
        client = GR1VLAClient(uri, checkpoint=str(tmp_path), timeout=0.2)
        with pytest.raises((ValueError, TimeoutError)):
            client.infer(observation(raw_observation()))
        assert client.closed
        with pytest.raises(RuntimeError):
            client.infer(observation(raw_observation()))
        assert len(calls) == 1
