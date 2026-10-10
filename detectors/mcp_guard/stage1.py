"""Stage I: the upstream pattern detectors, run exactly as upstream's router runs them.

Upstream (src/mcp_guard/api/routers/guardrail.py) calls seven detectors on the
tool description only, in this order, and blocks if any returns an issue.
"""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass

from . import _upstream  # noqa: F401  (sets sys.path)
from src.mcp_guard.detectors.cross_origin_detector import CrossOriginViolationDetector
from src.mcp_guard.detectors.important_tag_detector import ImportantTagDetector
from src.mcp_guard.detectors.prompt_injection_detector import PromptInjectionDetector
from src.mcp_guard.detectors.sensitive_file_detector import SensitiveFileAccessDetector
from src.mcp_guard.detectors.shadow_detector import ShadowHijackDetector
from src.mcp_guard.detectors.shell_injection_detector import ShellInjectionDetector
from src.mcp_guard.detectors.sql_injection_detector import SQLInjectionDetector

log = logging.getLogger(__name__)

DETECTORS = {
    "prompt_injection": PromptInjectionDetector,
    "sensitive_file": SensitiveFileAccessDetector,
    "shell_injection": ShellInjectionDetector,
    "sql_injection": SQLInjectionDetector,
    "shadow_hijack": ShadowHijackDetector,
    "cross_origin": CrossOriginViolationDetector,
    "important_tag": ImportantTagDetector,
}


@dataclass
class PatternHit:
    detector: str
    issue_type: str
    message: str


class PatternStage:
    def __init__(self, enabled: list[str] | None = None):
        names = enabled if enabled is not None else list(DETECTORS)
        unknown = set(names) - set(DETECTORS)
        if unknown:
            raise ValueError(f"unknown Stage I detectors: {sorted(unknown)}")
        self.detectors = {n: DETECTORS[n]() for n in names}

    def scan(self, text: str) -> list[PatternHit]:
        hits = []
        for name, det in self.detectors.items():
            try:
                # cross_origin prints DEBUG lines on every call; keep stdout clean.
                with contextlib.redirect_stdout(io.StringIO()):
                    issues = det.detect(text)
            except Exception as e:  # upstream marks the detector degraded and carries on
                log.warning("Stage I detector %s failed: %s", name, e)
                continue
            hits.extend(PatternHit(name, str(i.type.value), i.message) for i in issues)
        return hits
