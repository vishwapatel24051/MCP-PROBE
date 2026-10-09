"""Run the proxy as a stdio MCP server.

    uv run python -m proxy --config configs/proxy/example.yaml [--run-id RUN]

The agent launches this command as its MCP server; the proxy launches the
upstream server from the config. Diagnostics go to stderr, decisions to the
JSONL logs in ``log_dir``. Nothing may write to stdout: it carries the protocol.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import yaml
from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.server.stdio import stdio_server

from proxy.decision_log import DecisionLog
from proxy.detector_slot import load_detector
from proxy.server import ToolGate, build_proxy_server


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def upstream_params(up: dict[str, Any]) -> StdioServerParameters:
    env = up.get("env")
    return StdioServerParameters(
        command=up["command"],
        args=[str(a) for a in up.get("args") or []],
        env={**get_default_environment(), **{k: str(v) for k, v in env.items()}} if env else None,
        cwd=up.get("cwd"),
    )


async def serve(cfg: dict[str, Any], run_id: str) -> None:
    async with stdio_server() as (read_stream, write_stream):
        # Set up inside stdio_server: while it is open, stray prints from
        # detector code go to stderr instead of corrupting the protocol.
        det_cfg = cfg.get("detector") or {}
        detector = load_detector(det_cfg)
        up = cfg["upstream"]
        log = DecisionLog(cfg.get("log_dir", "results/proxy_logs"), run_id, up["name"], detector.name)
        log.event(
            "proxy_start",
            detector_class=det_cfg.get("class", "proxy.detector_slot:NoDetector"),
            detector_params=det_cfg.get("params") or {},
            on_error=det_cfg.get("on_error", "raise"),
            timeout_s=det_cfg.get("timeout_s"),
            upstream_command=[up["command"], *map(str, up.get("args") or [])],
        )
        async with Client(upstream_params(up), cache=None) as upstream:
            gate = ToolGate(
                upstream,
                detector,
                log,
                timeout_s=det_cfg.get("timeout_s"),
                on_error=det_cfg.get("on_error", "raise"),
            )
            server = build_proxy_server(upstream, gate)
            await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m proxy", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="proxy YAML config")
    parser.add_argument("--run-id", help="overrides run_id in the config")
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s proxy %(levelname)s %(message)s")
    cfg = load_config(args.config)
    run_id = args.run_id or cfg.get("run_id") or default_run_id()
    anyio.run(serve, cfg, run_id)


if __name__ == "__main__":
    main()
