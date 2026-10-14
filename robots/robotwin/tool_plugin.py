"""Opt-in read-only tool extensions with independent invocation accounting."""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from rpent.tools.toolkit import _is_readonly, readonly


def add_cli_args(parser: Any) -> None:
    """Add explicit plugin options without importing an optional extension."""
    parser.add_argument("--tool-plugin", default=None)
    parser.add_argument("--tool-plugin-root", type=Path, default=None)
    parser.add_argument("--tool-plugin-endpoint", default=None)
    parser.add_argument("--tool-plugin-checkpoint", default=None)
    parser.add_argument("--tool-plugin-timeout", type=float, default=180)


def configuration(args: Any) -> dict[str, Any] | None:
    """Validate an enabled extension and reject partially supplied options."""
    module = getattr(args, "tool_plugin", None)
    root = getattr(args, "tool_plugin_root", None)
    endpoint = getattr(args, "tool_plugin_endpoint", None)
    checkpoint = getattr(args, "tool_plugin_checkpoint", None)
    if not module:
        if root or endpoint or checkpoint:
            raise ValueError("Tool plugin options require --tool-plugin")
        return None
    if not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("Tool plugin must be a Python module name")
    if not endpoint or not checkpoint:
        raise ValueError("Tool plugin requires an endpoint and checkpoint")
    address = urlsplit(endpoint)
    if (
        address.scheme not in {"ws", "wss"}
        or not address.hostname
        or address.username
        or address.password
        or address.query
        or address.fragment
    ):
        raise ValueError("Tool endpoint must be a WebSocket URL without credentials")
    timeout = getattr(args, "tool_plugin_timeout", 180)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Tool plugin timeout must be positive and finite")
    if root is not None:
        root = Path(root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Tool plugin root must be a directory")
    return {
        "module": module,
        "root": str(root) if root is not None else None,
        "endpoint": endpoint,
        "checkpoint": str(Path(checkpoint).expanduser().resolve(strict=True)),
        "timeout": timeout,
    }


def load_plugin(options: dict[str, Any]) -> Any:
    """Load only the explicitly selected plugin from its declared source root."""
    root = options.get("root")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module(options["module"])
    if root and not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("Loaded tool plugin does not belong to the selected root")
    for name in ("install", "tool_names", "planner_prompt", "source_files"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Tool plugin is missing {name}()")
    return module


def worker_arguments(options: dict[str, Any]) -> list[str]:
    """Preserve the selected plugin across native CLI worker boundaries."""
    result = [
        "--tool-plugin",
        options["module"],
        "--tool-plugin-endpoint",
        options["endpoint"],
        "--tool-plugin-checkpoint",
        options["checkpoint"],
        "--tool-plugin-timeout",
        str(options["timeout"]),
    ]
    if options.get("root"):
        result += ["--tool-plugin-root", options["root"]]
    return result


class ToolPluginRuntime:
    """Own extension resources and record attempts, including failed handlers."""

    def __init__(self, toolkit: Any, options: dict[str, Any], output: Path):
        self.output = Path(output) / "tool_plugin"
        self.output.mkdir(parents=True, exist_ok=True)
        self.resource = None
        module = load_plugin(options)
        original = dict(toolkit._tools)
        try:
            self.resource = module.install(toolkit, options, Path(output))
            names = tuple(module.tool_names())
            if len(set(names)) != len(names) or set(toolkit._tools) - set(
                original
            ) != set(names):
                raise ValueError(
                    "Plugin registration disagrees with its declared tools"
                )
            if any(
                toolkit._tools.get(name) is not binding
                for name, binding in original.items()
            ):
                raise ValueError("Perception plugins must not replace native tools")
            for name in names:
                spec, handler = toolkit._tools[name]
                if not _is_readonly(handler):
                    raise ValueError(
                        "Perception plugins may only register read-only tools"
                    )
                toolkit.add_tool(name, spec, self._recorded(name, handler))
            metadata = {**options, "tools": list(names), "action_executed": False}
            (self.output / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        except BaseException:
            self.close()
            raise

    def _event(self, event: dict[str, Any]) -> None:
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"time_ns": time.time_ns(), **event}, ensure_ascii=False)
                + "\n"
            )

    def _recorded(self, name: str, handler: Any) -> Any:
        @readonly
        @wraps(handler)
        def invoke(**arguments: Any) -> Any:
            call_id = uuid4().hex
            started = time.monotonic()
            self._event(
                {
                    "event": "tool_begin",
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                }
            )
            try:
                result = handler(**arguments)
                if not isinstance(result, dict):
                    raise ValueError(
                        "Perception handlers must return a result dictionary"
                    )
            except BaseException as error:
                self._event(
                    {
                        "event": "tool_end",
                        "call_id": call_id,
                        "tool": name,
                        "success": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "elapsed_s": time.monotonic() - started,
                        "action_executed": False,
                    }
                )
                raise
            self._event(
                {
                    "event": "tool_end",
                    "call_id": call_id,
                    "tool": name,
                    "success": not result.get("error")
                    and result.get("success", True) is not False,
                    "request_id": result.get("request_id"),
                    "step_idx": result.get("step_idx"),
                    "calibration_used": result.get(
                        "calibration_used", "camera" in result
                    ),
                    "elapsed_s": time.monotonic() - started,
                    "action_executed": False,
                }
            )
            return result

        return invoke

    def close(self) -> None:
        """Close the borrowed model connection, never its shared server."""
        resource, self.resource = self.resource, None
        if resource is not None:
            resource.close()


def read_usage(directory: Path) -> dict[str, Any]:
    """Recompute separate perception counters from recorded handler events."""
    root = Path(directory) / "tool_plugin"
    metadata = root / "metadata.json"
    if not metadata.exists():
        return {
            "calls": {},
            "successes": {},
            "failures": {},
            "incomplete": 0,
            "elapsed_s": 0.0,
        }
    names = json.loads(metadata.read_text())["tools"]
    calls = Counter(dict.fromkeys(names, 0))
    successes = Counter(dict.fromkeys(names, 0))
    failures = Counter(dict.fromkeys(names, 0))
    starts, ends = {}, set()
    elapsed = 0.0
    path = root / "events.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        event = json.loads(line)
        call_id, name = event["call_id"], event["tool"]
        if name not in names:
            raise ValueError("Unknown tool in perception invocation trace")
        if event["event"] == "tool_begin":
            if call_id in starts:
                raise ValueError("Duplicate perception invocation ID")
            starts[call_id] = name
            calls[name] += 1
        elif event["event"] == "tool_end":
            if starts.get(call_id) != name or call_id in ends:
                raise ValueError("Unmatched perception invocation result")
            ends.add(call_id)
            (successes if event["success"] else failures)[name] += 1
            elapsed += event["elapsed_s"]
        else:
            raise ValueError("Unknown perception invocation event")
    return {
        "calls": dict(calls),
        "successes": dict(successes),
        "failures": dict(failures),
        "incomplete": len(starts) - len(ends),
        "elapsed_s": elapsed,
    }
