"""Proxy tests. A scripted MCP client stands in for the agent; the upstream is
the toy server in tests/fixtures (synthetic text, no MCPTox data)."""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import yaml
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from proxy.decision_log import DecisionLog
from proxy.detector_slot import NoDetector, load_detector
from proxy.server import ToolGate, build_proxy_server
from tests.fixtures.stub_detectors import BlockMarker, Crashes, Rewrites, Slow
from tests.fixtures.toy_server import server as toy

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def anyio_backend():
    return "asyncio"


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@asynccontextmanager
async def proxied(detector, log_dir, **gate_kw):
    async with Client(toy) as upstream:
        log = DecisionLog(log_dir, "test", "toy", detector.name)
        gate = ToolGate(upstream, detector, log, **gate_kw)
        async with Client(build_proxy_server(upstream, gate)) as agent:
            yield agent, upstream, log


@pytest.mark.anyio
async def test_no_detector_is_transparent(tmp_path):
    async with proxied(NoDetector(), tmp_path) as (agent, upstream, log):
        seen = (await agent.list_tools()).tools
        real = (await upstream.list_tools()).tools
        assert [t.model_dump() for t in seen] == [t.model_dump() for t in real]

        assert (await agent.call_tool("add", {"a": 2, "b": 3})).structured_content == {"result": 5}
        assert (await agent.read_resource("toy://readme")).contents[0].text == "toy readme"
        assert (await agent.get_prompt("greet", {"name": "A"})).messages[0].content.text == "Hello A"
        assert agent.instructions == upstream.instructions

    decisions = [e for e in read_events(log.decisions_path) if e["event"] == "tool_decision"]
    assert {(e["tool"], e["decision"]) for e in decisions} == {("add", "allowed"), ("echo", "allowed")}


@pytest.mark.anyio
async def test_blocked_tool_is_hidden_and_uncallable(tmp_path):
    async with proxied(BlockMarker(), tmp_path) as (agent, _, log):
        assert [t.name for t in (await agent.list_tools()).tools] == ["add"]
        with pytest.raises(MCPError, match="Unknown tool: echo"):
            await agent.call_tool("echo", {"message": "hi"})
        assert (await agent.call_tool("add", {"a": 1, "b": 1})).structured_content == {"result": 2}

    events = read_events(log.decisions_path)
    decision = {e["tool"]: e for e in events if e["event"] == "tool_decision"}
    assert decision["echo"]["decision"] == "blocked"
    assert decision["add"]["decision"] == "allowed"
    for e in decision.values():
        assert e["detector"] == "block-marker" and e["run_id"] == "test" and e["ts"]
        assert e["description_sha256"] and e["list_sha256"] and e["cached"] is False
    assert [e["tool"] for e in events if e["event"] == "call_blocked"] == ["echo"]
    assert [e["tool"] for e in events if e["event"] == "call_forwarded"] == ["add"]

    snapshot = read_events(log.tools_path)
    assert len(snapshot) == 1
    assert {t["name"] for t in snapshot[0]["tools"]} == {"add", "echo"}


@pytest.mark.anyio
async def test_call_without_listing_still_goes_through_detector(tmp_path):
    async with proxied(BlockMarker(), tmp_path) as (agent, _, log):
        with pytest.raises(MCPError, match="Unknown tool: echo"):
            await agent.call_tool("echo", {"message": "hi"})
    assert any(e["event"] == "call_blocked" for e in read_events(log.decisions_path))


@pytest.mark.anyio
async def test_decisions_are_cached_per_tool_list(tmp_path):
    detector = BlockMarker()
    async with proxied(detector, tmp_path) as (agent, _, log):
        await agent.list_tools()
        await agent.list_tools()
    assert detector.calls == 1
    decisions = [e for e in read_events(log.decisions_path) if e["event"] == "tool_decision"]
    assert [e["cached"] for e in decisions] == [False, False, True, True]
    assert len(read_events(log.tools_path)) == 1


@pytest.mark.anyio
async def test_detector_error_raises_by_default(tmp_path):
    async with proxied(Crashes(), tmp_path) as (agent, _, log):
        with pytest.raises(MCPError, match="boom"):
            await agent.list_tools()
    events = read_events(log.decisions_path)
    assert [e["event"] for e in events] == ["detector_error"]
    assert events[0]["on_error"] == "raise"


@pytest.mark.anyio
@pytest.mark.parametrize("on_error,expected", [("fail_open", ["add", "echo"]), ("fail_closed", [])])
async def test_detector_error_fallbacks(tmp_path, on_error, expected):
    async with proxied(Crashes(), tmp_path, on_error=on_error) as (agent, _, log):
        assert [t.name for t in (await agent.list_tools()).tools] == expected
    decisions = [e for e in read_events(log.decisions_path) if e["event"] == "tool_decision"]
    assert all(e["fallback"] == on_error for e in decisions)


@pytest.mark.anyio
async def test_detector_timeout_is_an_error(tmp_path):
    async with proxied(Slow(), tmp_path, timeout_s=0.2) as (agent, _, log):
        with pytest.raises(MCPError, match="TimeoutError"):
            await agent.list_tools()
    assert read_events(log.decisions_path)[0]["event"] == "detector_error"


@pytest.mark.anyio
async def test_detector_may_not_rewrite_tools(tmp_path):
    async with proxied(Rewrites(), tmp_path) as (agent, _, _log):
        with pytest.raises(MCPError, match="new or modified tool"):
            await agent.list_tools()


def test_load_detector_by_dotted_path():
    assert isinstance(load_detector(None), NoDetector)
    det = load_detector({"class": "tests.fixtures.stub_detectors:BlockMarker", "params": {"marker": "X"}})
    assert isinstance(det, BlockMarker) and det.marker == "X"
    assert load_detector({"class": "tests.fixtures.stub_detectors:Unnamed"}).name == "Unnamed"
    with pytest.raises(ValueError):
        load_detector({"class": "tests.fixtures.stub_detectors.BlockMarker"})


@pytest.mark.anyio
async def test_stdio_end_to_end(tmp_path):
    """Launch the proxy the way the agent will: as a stdio subprocess from config."""
    config = {
        "upstream": {"name": "toy", "command": sys.executable, "args": [str(REPO / "tests/fixtures/toy_server.py")]},
        "detector": {"class": "tests.fixtures.stub_detectors:BlockMarker"},
        "log_dir": str(tmp_path),
    }
    config_path = tmp_path / "proxy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "proxy", "--config", str(config_path), "--run-id", "e2e"],
        cwd=str(REPO),
    )
    async with Client(params, read_timeout_seconds=60) as agent:
        assert [t.name for t in (await agent.list_tools()).tools] == ["add"]
        assert (await agent.call_tool("add", {"a": 2, "b": 2})).structured_content == {"result": 4}
        with pytest.raises(MCPError, match="Unknown tool: echo"):
            await agent.call_tool("echo", {"message": "hi"})

    events = read_events(tmp_path / "e2e.toy.decisions.jsonl")
    assert events[0]["event"] == "proxy_start"
    assert events[0]["detector"] == "block-marker"
    assert {e["event"] for e in events} >= {"tool_decision", "call_forwarded", "call_blocked"}
