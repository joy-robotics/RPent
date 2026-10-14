from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa_gr1.contracts import GROUPS, IMAGE_KEYS, env_contract
from robots.robocasa_gr1.env_client import GR1EnvClient
from robots.robocasa_gr1.env_server import GR1EnvFacade


@pytest.fixture
def joybrain_dependencies():
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets", minversion="15")


@pytest.fixture
def video_codec():
    pytest.importorskip("imageio_ffmpeg")


def raw_observation():
    raw = {f"state.{name}": np.arange(size, dtype=np.float32) for name, size in GROUPS}
    raw[IMAGE_KEYS[0]] = np.full((16, 16, 3), 127, dtype=np.uint8)
    raw["annotation.human.coarse_action"] = "task: put the cup away"
    return raw


class NativeEnv:
    def __init__(self):
        self.unwrapped = SimpleNamespace(env=SimpleNamespace())
        self.seeds = []
        self.actions = []
        self.closes = 0
        self.success = False
        self.terminated = False
        self.truncated = False

    def reset(self, *, seed):
        self.seeds.append(seed)
        return raw_observation(), {}

    def step(self, action):
        self.actions.append(action)
        return (
            raw_observation(),
            float(self.success),
            self.terminated,
            self.truncated,
            {"success": self.success},
        )

    def close(self):
        self.closes += 1


class LocalRpc:
    def __init__(self, facade):
        self.facade = facade
        self.closed = False

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        return getattr(self.facade, method.removeprefix("env."))(
            *args, **(kwargs or {})
        )

    def close(self):
        self.closed = True


@pytest.fixture
def stack(tmp_path):
    native = NativeEnv()
    meta = env_contract("PnPCupToDrawerClose", "id", 7, 3)
    facade = GR1EnvFacade(native, meta, tmp_path / "trace")
    rpc = LocalRpc(facade)
    client = GR1EnvClient(rpc, expected_meta=meta)
    yield native, facade, client
    facade.close()
