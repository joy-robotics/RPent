"""Episode-scoped access to a worker-owned resident RoboTwin server."""

from __future__ import annotations

from typing import Any

from rpent.utils.rpc.rpc_client import RpcClient


class EpisodeRpcClient(RpcClient):
    """Reject calls after close and tag every request with its episode identity."""

    def __init__(self, transport: RpcClient, episode_id: str) -> None:
        super().__init__()
        self.transport = transport
        self.episode_id = episode_id

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Episode RPC client is closed")
        values = dict(kwargs or {})
        if "_episode_id" in values:
            raise ValueError("Episode identity is owned by the client")
        values["_episode_id"] = self.episode_id
        return self.transport.call(
            method, args=args, kwargs=values, timeout_s=timeout_s
        )
