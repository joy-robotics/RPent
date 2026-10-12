"""Owned Joy-Brain services shared by episode workers, with idle keepalives."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable, Sequence
from copy import copy
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

from robots.robotwin import robot_spec
from robots.robotwin.rollout_pool import GPUGroup


class PolicyPool:
    """Start one service per model GPU and close only owned services/connections."""

    def __init__(
        self,
        groups: Sequence[GPUGroup],
        args: argparse.Namespace,
        output: Path,
        cache_root: Path,
        check_resources: Callable[[], None],
        *,
        keepalive_s: float = 30.0,
    ) -> None:
        if keepalive_s <= 0:
            raise ValueError("Keepalive interval must be positive")
        self.groups = list(groups)
        self.args = args
        self.output = output
        self.cache_root = cache_root
        self.check_resources = check_resources
        self.keepalive_s = keepalive_s
        self.endpoints: dict[str, str] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self._daemons: list[Any] = []
        self._clients: list[Any] = []
        self._threads: list[threading.Thread] = []
        self._errors: SimpleQueue[Exception] = SimpleQueue()
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._started = False

    def start(self) -> None:
        """Spawn all services before waiting, checking resources while they load."""
        if self._started or self._stop.is_set():
            raise RuntimeError(
                "Create a new policy pool instead of restarting services"
            )
        self._started = True
        from robots.robotwin.joybrain_client import JoyBrainVLAClient
        from robots.robotwin.recording import write_json

        pending = []
        try:
            for group in self.groups:
                self.check()
                gpu = group.model_gpu
                directory = self.output / f"policy_gpu_{gpu.index}"
                directory.mkdir()
                options = copy(self.args)
                options.cuda_device = None
                options.vla_cuda_device = gpu.uuid
                options.runtime_cache_dir = self.cache_root / "models" / gpu.uuid
                daemon, endpoint = robot_spec._spawn_vla_server(options, directory)
                if daemon is not None:
                    self._daemons.append(daemon)
                pending.append((gpu, directory, daemon, endpoint))
            for gpu, directory, daemon, endpoint in pending:
                robot_spec._wait_for_tcp(
                    *endpoint,
                    daemon,
                    timeout_s=self.args.worker_startup_timeout_s,
                    check_status=self.check,
                )
                uri = self.args.vla_endpoint or f"ws://{endpoint[0]}:{endpoint[1]}"
                if "://" not in uri:
                    uri = f"ws://{uri}"
                client = JoyBrainVLAClient(
                    uri,
                    timeout=self.args.joybrain_timeout,
                    expected_checkpoint=robot_spec._configured_model_path(self.args),
                    close_on_toolkit_close=False,
                )
                self._clients.append(client)
                self.endpoints[gpu.uuid] = client.endpoint
                self.metadata[gpu.uuid] = dict(client.metadata)
                write_json(
                    directory / "metadata.json",
                    {
                        "model_gpu_uuid": gpu.uuid,
                        "endpoint": client.endpoint,
                        "policy": client.metadata,
                    },
                )
                thread = threading.Thread(
                    target=self._keepalive,
                    args=(client,),
                    name=f"policy-keepalive-{gpu.index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _keepalive(self, client: Any) -> None:
        while not self._stop.wait(self.keepalive_s):
            try:
                client.ping()
            except Exception as error:
                if not self._stop.is_set():
                    self._errors.put(error)
                return

    def check(self) -> None:
        """Fail dispatch on lost services, failed keepalives, or resource pressure."""
        self.check_resources()
        for daemon in self._daemons:
            if daemon.poll() is not None:
                raise RuntimeError(f"Owned VLA service exited: {daemon.name}")
        if self._failure is None:
            try:
                self._failure = self._errors.get_nowait()
            except Empty:
                return
        raise RuntimeError(f"VLA keepalive failed: {self._failure}") from self._failure

    def close(self) -> None:
        """Release every owned service even if another cleanup operation fails."""
        self._stop.set()
        errors = []
        pending_clients = []
        for client in self._clients:
            try:
                client.close()
            except Exception as error:
                errors.append(error)
                pending_clients.append(client)
        pending_daemons = []
        for daemon in reversed(self._daemons):
            try:
                daemon.stop()
            except Exception as error:
                errors.append(error)
                pending_daemons.append(daemon)
        pending_threads = []
        for thread in self._threads:
            thread.join(timeout=3)
            if thread.is_alive():
                errors.append(
                    RuntimeError(f"Keepalive thread did not exit: {thread.name}")
                )
                pending_threads.append(thread)
        self._clients = pending_clients
        self._daemons = list(reversed(pending_daemons))
        self._threads = pending_threads
        if errors:
            raise RuntimeError(f"Policy pool cleanup failed: {errors}")
