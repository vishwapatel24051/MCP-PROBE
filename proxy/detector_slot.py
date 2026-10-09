"""The one place a detector plugs into the proxy.

A detector is any object with a ``name`` and a ``filter_tools(tools) -> tools``
method. ``filter_tools`` may be sync or async; sync detectors run in a worker
thread so a slow model does not block the proxy's event loop.

Contract: the returned list must be a subset of the input, with every kept
tool unchanged. Detectors block tools; they never rewrite them. The proxy
enforces this (see ``check_subset``) so no condition in Experiment 3 can
quietly sanitize descriptions.

Detectors are chosen in config by dotted path, e.g.
``detectors.mcp_guard:MCPGuard``, so the proxy never imports one by name.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from typing import Any, Protocol, runtime_checkable

import anyio.to_thread
from mcp.types import Tool


@runtime_checkable
class Detector(Protocol):
    name: str

    def filter_tools(self, tools: list[Tool]) -> Any:
        """Return the tools the agent may see (``list[Tool]``, or an awaitable of one)."""
        ...


class NoDetector:
    """The no-detector condition: every tool is allowed."""

    name = "none"

    async def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        return list(tools)


class DetectorContractError(Exception):
    """A detector returned something other than an unchanged subset of its input."""


def load_detector(spec: dict[str, Any] | None) -> Detector:
    """Build a detector from config: ``{"class": "pkg.module:Class", "params": {...}}``.

    A missing spec means the no-detector condition.
    """
    if not spec:
        return NoDetector()
    target = spec["class"]
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(f"detector class must look like 'pkg.module:Class', got {target!r}")
    cls = getattr(importlib.import_module(module_name), attr)
    detector = cls(**(spec.get("params") or {}))
    if not hasattr(detector, "filter_tools"):
        raise TypeError(f"{target} has no filter_tools method")
    if not getattr(detector, "name", None):
        detector.name = cls.__name__
    return detector


async def run_detector(detector: Detector, tools: list[Tool]) -> list[Tool]:
    """Call ``filter_tools`` whether it is sync or async."""
    if inspect.iscoroutinefunction(detector.filter_tools):
        result = await detector.filter_tools(tools)
    else:
        result = await anyio.to_thread.run_sync(detector.filter_tools, tools)
        if inspect.isawaitable(result):
            result = await result
    return list(result)


def tool_sha256(tool: Tool) -> str:
    """Stable hash of everything the agent would see for this tool."""
    dump = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    return hashlib.sha256(json.dumps(dump, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_subset(given: list[Tool], kept: list[Tool]) -> None:
    """Raise unless ``kept`` is an unchanged subset of ``given``."""
    given_hashes = {tool_sha256(t) for t in given}
    seen: set[str] = set()
    for tool in kept:
        if not isinstance(tool, Tool):
            raise DetectorContractError(f"detector returned a {type(tool).__name__}, not a Tool")
        if tool_sha256(tool) not in given_hashes:
            raise DetectorContractError(f"detector returned a new or modified tool: {tool.name!r}")
        if tool.name in seen:
            raise DetectorContractError(f"detector returned {tool.name!r} twice")
        seen.add(tool.name)
