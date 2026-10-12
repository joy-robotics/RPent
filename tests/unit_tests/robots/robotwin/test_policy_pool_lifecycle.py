import argparse
import threading
import time

import pytest

pytest.importorskip("msgpack")
pytest.importorskip("websockets", minversion="15")

from robots.robotwin import joybrain_client, robot_spec
from robots.robotwin.policy_pool import PolicyPool
from robots.robotwin.rollout_pool import GPU, make_gpu_groups


@pytest.fixture
def pool_setup(tmp_path, monkeypatch):
    events = []
    clients = []
    daemons = []
    pinged = threading.Event()

    class Daemon:
        def __init__(self, name):
            self.name = name
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def stop(self):
            self.stopped = True
            events.append(("stop", self.name))

    class Client:
        def __init__(self, endpoint, **options):
            self.endpoint = endpoint
            self.metadata = {"ckpt_path": "/checkpoint"}
            self.closed = False
            self.failure = None
            clients.append(self)

        def ping(self):
            pinged.set()
            if self.failure:
                raise self.failure

        def close(self):
            self.closed = True

    def spawn(options, output):
        events.append(("spawn", options.vla_cuda_device))
        assert options.cuda_device is None
        assert str(options.runtime_cache_dir).endswith(options.vla_cuda_device)
        daemon = None
        if not options.vla_endpoint:
            daemon = Daemon(options.vla_cuda_device)
            daemons.append(daemon)
        return daemon, ("127.0.0.1", 12000 + len(events))

    def wait(host, port, daemon, **options):
        events.append(("wait", port))
        options["check_status"]()

    monkeypatch.setattr(robot_spec, "_spawn_vla_server", spawn)
    monkeypatch.setattr(robot_spec, "_wait_for_tcp", wait)
    monkeypatch.setattr(joybrain_client, "JoyBrainVLAClient", Client)
    args = argparse.Namespace(
        vla_endpoint=None,
        vla_backend="joybrain",
        vla_model_path="/checkpoint",
        worker_startup_timeout_s=5,
        joybrain_timeout=1,
    )
    gpus = [GPU(index, f"GPU-{index}", 32607, 32110) for index in range(4)]
    pool = PolicyPool(
        make_gpu_groups(gpus, "split"),
        args,
        tmp_path,
        tmp_path / "cache",
        lambda: None,
        keepalive_s=0.01,
    )
    return pool, events, clients, daemons, pinged


def test_services_spawn_before_wait_and_keepalive_does_not_load_extra_models(
    pool_setup,
):
    pool, events, clients, daemons, pinged = pool_setup
    try:
        pool.start()
        assert events[:2] == [("spawn", "GPU-0"), ("spawn", "GPU-2")]
        assert len(pool.endpoints) == len(clients) == len(daemons) == 2
        assert pinged.wait(2)
        pool.check()
    finally:
        pool.close()
    assert all(client.closed for client in clients)
    assert all(daemon.stopped for daemon in daemons)
    pool.close()


def test_partial_startup_failure_closes_every_owned_service(pool_setup, monkeypatch):
    pool, events, clients, daemons, _ = pool_setup
    original = robot_spec._wait_for_tcp
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("second service failed")
        original(*args, **kwargs)

    monkeypatch.setattr(robot_spec, "_wait_for_tcp", fail_second)
    with pytest.raises(RuntimeError, match="second service"):
        pool.start()
    assert len(daemons) == 2 and all(daemon.stopped for daemon in daemons)
    assert clients and all(client.closed for client in clients)


def test_resource_failure_during_startup_cleans_up_pending_models(
    pool_setup, monkeypatch
):
    pool, events, clients, daemons, _ = pool_setup

    def fail_wait(*args, **kwargs):
        pool.check_resources = lambda: (_ for _ in ()).throw(RuntimeError("low VRAM"))
        kwargs["check_status"]()

    monkeypatch.setattr(robot_spec, "_wait_for_tcp", fail_wait)
    with pytest.raises(RuntimeError, match="low VRAM"):
        pool.start()
    assert len(daemons) == 2 and all(daemon.stopped for daemon in daemons)


def test_borrowed_service_is_not_stopped(pool_setup):
    pool, events, clients, daemons, _ = pool_setup
    pool.groups = pool.groups[:1]
    pool.args.vla_endpoint = "ws://127.0.0.1:12345"
    pool.start()
    pool.close()
    assert not daemons
    assert clients[0].closed
    assert not any(event[0] == "stop" for event in events)


def test_keepalive_error_blocks_dispatch(pool_setup):
    pool, events, clients, daemons, _ = pool_setup
    try:
        pool.start()
        clients[0].failure = RuntimeError("connection lost")
        deadline = time.monotonic() + 2
        while pool._errors.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="keepalive"):
            pool.check()
        with pytest.raises(RuntimeError, match="keepalive"):
            pool.check()
    finally:
        pool.close()


def test_pool_cannot_accidentally_load_models_twice(pool_setup):
    pool, events, clients, daemons, _ = pool_setup
    try:
        pool.start()
        with pytest.raises(RuntimeError, match="new policy pool"):
            pool.start()
        assert len(daemons) == 2
    finally:
        pool.close()
    with pytest.raises(RuntimeError, match="new policy pool"):
        pool.start()


def test_cleanup_attempts_all_services_and_can_retry_failure(pool_setup):
    pool, events, clients, daemons, _ = pool_setup
    pool.start()
    original = daemons[0].stop

    def failed_stop():
        raise RuntimeError("temporary cleanup failure")

    daemons[0].stop = failed_stop
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pool.close()
    assert daemons[1].stopped and all(client.closed for client in clients)
    daemons[0].stop = original
    pool.close()
    assert daemons[0].stopped


def test_exited_service_blocks_dispatch(pool_setup):
    pool, events, clients, daemons, _ = pool_setup
    try:
        pool.start()
        daemons[0].stopped = True
        with pytest.raises(RuntimeError, match="exited"):
            pool.check()
    finally:
        pool.close()
