"""The proxy: an MCP server to the agent, an MCP client to one upstream server.

Everything is relayed unchanged except:

- ``tools/list``: the full upstream list goes through the detector, and only
  the tools it keeps are returned (as a single page).
- ``tools/call``: calls to a blocked tool are refused with the same error an
  unknown tool gets, so blocking cannot be bypassed by calling a hidden name.

Only capabilities the upstream advertises are offered to the agent.
Not relayed (out of scope for our test servers): server-initiated requests
(sampling, elicitation), change notifications and resource subscriptions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import anyio
from mcp import Client
from mcp.server import Server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    CompleteRequestParams,
    CompleteResult,
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    Tool,
)

from proxy.decision_log import DecisionLog
from proxy.detector_slot import Detector, check_subset, run_detector

logger = logging.getLogger(__name__)

OnError = Literal["raise", "fail_open", "fail_closed"]
MAX_UPSTREAM_PAGES = 100


def _unknown_tool(name: str) -> MCPError:
    return MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {name}")


class ToolGate:
    """Runs the detector over upstream tool lists and remembers what it blocked."""

    def __init__(
        self,
        upstream: Client,
        detector: Detector,
        log: DecisionLog,
        *,
        timeout_s: float | None = None,
        on_error: OnError = "raise",
    ) -> None:
        if on_error not in ("raise", "fail_open", "fail_closed"):
            raise ValueError(f"on_error must be raise, fail_open or fail_closed; got {on_error!r}")
        self.upstream = upstream
        self.detector = detector
        self.log = log
        self.timeout_s = timeout_s
        self.on_error = on_error
        # list_sha256 -> names kept. Decisions are cached per whole list, not per
        # tool, so a detector that looks at the full tool set behaves the same.
        self._cache: dict[str, set[str]] = {}
        self.allowed: set[str] = set()
        self.blocked: set[str] = set()

    async def upstream_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        cursor: str | None = None
        for _ in range(MAX_UPSTREAM_PAGES):
            page = await self.upstream.list_tools(cursor=cursor, cache_mode="refresh")
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools
        raise MCPError(code=INTERNAL_ERROR, message="upstream tools/list did not finish paginating")

    async def filtered_tools(self) -> list[Tool]:
        given = await self.upstream_tools()
        digest = self.log.snapshot(given)

        if digest in self._cache:
            kept_names = self._cache[digest]
            kept = [t for t in given if t.name in kept_names]
            self._record(given, kept, digest, latency_ms=0.0, cached=True)
            return kept

        start = time.perf_counter()
        try:
            with anyio.fail_after(self.timeout_s):
                kept = await run_detector(self.detector, list(given))
            check_subset(given, kept)
        except Exception as exc:
            return self._handle_error(exc, given, digest, (time.perf_counter() - start) * 1000)
        latency_ms = (time.perf_counter() - start) * 1000

        self._cache[digest] = {t.name for t in kept}
        self._record(given, kept, digest, latency_ms=latency_ms, cached=False)
        return kept

    def _record(self, given: list[Tool], kept: list[Tool], digest: str, *, latency_ms: float, cached: bool, fallback: str | None = None) -> None:
        self.allowed = {t.name for t in kept}
        self.blocked = {t.name for t in given} - self.allowed
        self.log.decisions(given, kept, list_digest=digest, latency_ms=latency_ms, cached=cached, fallback=fallback)

    def _handle_error(self, exc: Exception, given: list[Tool], digest: str, latency_ms: float) -> list[Tool]:
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        self.log.event("detector_error", error=error, on_error=self.on_error, list_sha256=digest)
        logger.error("detector %s failed (%s): %s", self.detector.name, self.on_error, error)
        if self.on_error == "fail_open":
            kept = list(given)
        elif self.on_error == "fail_closed":
            kept = []
        else:
            raise MCPError(code=INTERNAL_ERROR, message=f"detector {self.detector.name!r} failed: {error}") from exc
        # Fallback results are not cached: the detector gets another try next time.
        self._record(given, kept, digest, latency_ms=latency_ms, cached=False, fallback=self.on_error)
        return kept

    async def check_call(self, name: str) -> None:
        """Refuse calls to tools the detector blocked."""
        if name not in self.allowed and name not in self.blocked:
            # The agent called a name before (or without) listing it; decide now.
            await self.filtered_tools()
        if name in self.blocked:
            self.log.event("call_blocked", tool=name)
            raise _unknown_tool(name)


def build_proxy_server(upstream: Client, gate: ToolGate) -> Server[Any]:
    """Build the agent-facing server. ``upstream`` must already be connected."""
    caps = upstream.server_capabilities
    info = upstream.server_info
    handlers: dict[str, Any] = {}

    if caps.tools is not None:

        async def list_tools(ctx: Any, params: PaginatedRequestParams | None) -> ListToolsResult:
            return ListToolsResult(tools=await gate.filtered_tools())

        async def call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:
            await gate.check_call(params.name)
            gate.log.event("call_forwarded", tool=params.name)
            return await upstream.call_tool(params.name, params.arguments)

        handlers.update(on_list_tools=list_tools, on_call_tool=call_tool)

    if caps.resources is not None:

        async def list_resources(ctx: Any, params: PaginatedRequestParams | None) -> ListResourcesResult:
            return await upstream.list_resources(cursor=params.cursor if params else None)

        async def list_resource_templates(ctx: Any, params: PaginatedRequestParams | None) -> ListResourceTemplatesResult:
            return await upstream.list_resource_templates(cursor=params.cursor if params else None)

        async def read_resource(ctx: Any, params: ReadResourceRequestParams) -> ReadResourceResult:
            return await upstream.read_resource(str(params.uri))

        handlers.update(
            on_list_resources=list_resources,
            on_list_resource_templates=list_resource_templates,
            on_read_resource=read_resource,
        )

    if caps.prompts is not None:

        async def list_prompts(ctx: Any, params: PaginatedRequestParams | None) -> ListPromptsResult:
            return await upstream.list_prompts(cursor=params.cursor if params else None)

        async def get_prompt(ctx: Any, params: GetPromptRequestParams) -> GetPromptResult:
            return await upstream.get_prompt(params.name, params.arguments)

        handlers.update(on_list_prompts=list_prompts, on_get_prompt=get_prompt)

    if caps.completions is not None:

        async def complete(ctx: Any, params: CompleteRequestParams) -> CompleteResult:
            context = params.context.arguments if params.context else None
            return await upstream.complete(params.ref, params.argument.model_dump(), context)

        handlers["on_completion"] = complete

    return Server(
        info.name if info else "mcp-probe-proxy",
        version=info.version if info else "",
        instructions=upstream.instructions,
        **handlers,
    )
