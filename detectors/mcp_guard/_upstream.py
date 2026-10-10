"""Make the vendored GenTelLab/MCP-Guard code importable.

The upstream code imports itself as ``src.mcp_guard...`` relative to its repo
root, so we append that root to sys.path (at the end, so nothing of ours is
shadowed) instead of editing the vendored files.
"""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
UPSTREAM_COMMIT = "4bc97790256280480cf827c0a0fdb85e86754dfb"  # github.com/GenTelLab/MCP-Guard

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.append(str(UPSTREAM_ROOT))
