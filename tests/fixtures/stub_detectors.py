"""Stub detectors for proxy tests. Loaded by dotted path, like real ones."""

import anyio
from mcp.types import Tool

from tests.fixtures.toy_server import POISON_MARKER


class BlockMarker:
    """Sync detector: blocks tools whose description contains a marker."""

    name = "block-marker"

    def __init__(self, marker: str = POISON_MARKER) -> None:
        self.marker = marker
        self.calls = 0

    def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        self.calls += 1
        return [t for t in tools if self.marker not in (t.description or "")]


class Crashes:
    name = "crashes"

    async def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        raise RuntimeError("boom")


class Slow:
    name = "slow"

    async def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        await anyio.sleep(5)
        return tools


class Rewrites:
    """Breaks the contract by sanitizing a description instead of blocking."""

    name = "rewrites"

    async def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        return [t.model_copy(update={"description": "cleaned"}) for t in tools]


class Unnamed:
    async def filter_tools(self, tools: list[Tool]) -> list[Tool]:
        return tools
