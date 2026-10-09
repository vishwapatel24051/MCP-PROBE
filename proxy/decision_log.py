"""Append-only JSONL logs of every proxy decision.

Two files per (run, upstream) under ``log_dir``:

- ``<run_id>.<upstream>.decisions.jsonl``: one line per event. Events:
  ``proxy_start``, ``tool_decision`` (one per tool per tools/list),
  ``call_forwarded``, ``call_blocked``, ``detector_error``. ``fallback`` on a
  ``tool_decision`` is set when the detector failed and ``on_error`` decided.
- ``<run_id>.<upstream>.tools.jsonl``: the full upstream tool list, written
  once per distinct list (keyed by ``list_sha256``), so every decision can be
  traced back to the exact text the detector saw.

Lines are flushed as they are written, so a crashed run still leaves a usable log.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import Tool

from proxy.detector_slot import tool_sha256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def list_sha256(tools: list[Tool]) -> str:
    return hashlib.sha256("\n".join(tool_sha256(t) for t in tools).encode()).hexdigest()


def text_sha256(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


class DecisionLog:
    def __init__(self, log_dir: str | Path, run_id: str, upstream: str, detector: str) -> None:
        self.run_id = run_id
        self.upstream = upstream
        self.detector = detector
        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe(run_id)}.{_safe(upstream)}"
        self.decisions_path = base / f"{stem}.decisions.jsonl"
        self.tools_path = base / f"{stem}.tools.jsonl"
        self._snapshotted: set[str] = set()

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": _now(),
            "run_id": self.run_id,
            "event": event,
            "detector": self.detector,
            "upstream": self.upstream,
            **fields,
        }
        self._append(self.decisions_path, record)

    def snapshot(self, tools: list[Tool]) -> str:
        """Write the full tool list once per distinct list; return its hash."""
        digest = list_sha256(tools)
        if digest not in self._snapshotted:
            self._snapshotted.add(digest)
            self._append(
                self.tools_path,
                {
                    "ts": _now(),
                    "run_id": self.run_id,
                    "upstream": self.upstream,
                    "list_sha256": digest,
                    "tools": [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools],
                },
            )
        return digest

    def decisions(
        self,
        given: list[Tool],
        kept: list[Tool],
        *,
        list_digest: str,
        latency_ms: float,
        cached: bool,
        fallback: str | None = None,
    ) -> None:
        kept_names = {t.name for t in kept}
        for tool in given:
            self.event(
                "tool_decision",
                tool=tool.name,
                decision="allowed" if tool.name in kept_names else "blocked",
                description_sha256=text_sha256(tool.description),
                tool_sha256=tool_sha256(tool),
                list_sha256=list_digest,
                latency_ms=round(latency_ms, 2),
                cached=cached,
                fallback=fallback,
            )
