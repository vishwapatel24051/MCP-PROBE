"""MCP-Guard (arXiv 2508.10991) wrapped for MCP-PROBE. See README.md in this folder.

    from detectors.mcp_guard import MCPGuard
    guard = MCPGuard.from_config()          # configs/mcp_guard.yaml
    verdict = guard.scan(tool_name, description)
    verdict.blocked, verdict.stage, verdict.s2_score
"""

from .pipeline import MCPGuard, Verdict

__all__ = ["MCPGuard", "Verdict"]
