import json
import re
from typing import List, Dict, Any
import os
from src.mcp_guard.config.loader import detectors_path
from src.mcp_guard.schemas import DetectorResult, IssueType

CONFIG_PATH = detectors_path('shell_rules.json')

class Rule:
    def __init__(self, id, pattern, score, flags="g"):
        self.id = id
        self.pattern = pattern
        self.score = score
        self.flags = flags

class DetectorConfig:
    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.threshold = cfg["threshold"]
        self.shell_rules = [Rule(**r) for r in cfg["shell_rules"]]

class DetectResult:
    def __init__(self, total_score, matches, is_malicious):
        self.total_score = total_score
        self.matches = matches
        self.is_malicious = is_malicious

class ShellInjectionDetector:
    def __init__(self, config_path=CONFIG_PATH):
        self.cfg = DetectorConfig(config_path)

    def detect_with_rules(self, text: str, rules: List[Rule]) -> DetectResult:
        total_score = 0
        matches = []
        for r in rules:
            regex = re.compile(r.pattern, flags=self._parse_flags(r.flags))
            found = []
            m = regex.search(text)
            while m:
                found.append(m.group(0))
                if 'g' not in (r.flags or ''):
                    break
                m = regex.search(text, m.end())
            if found:
                total_score += r.score
                matches.append({"ruleId": r.id, "score": r.score, "matches": found})
        return DetectResult(total_score, matches, total_score >= self.cfg.threshold)

    def detect_shell(self, text: str) -> DetectResult:
        return self.detect_with_rules(text, self.cfg.shell_rules)

    def detect(self, text: str) -> List[DetectorResult]:
        shell_result = self.detect_shell(text)
        results = []
        if shell_result.is_malicious:
            results.append(DetectorResult(
                type=IssueType.shell_injection if hasattr(IssueType, "shell_injection") else IssueType.prompt_injection,
                message=f"评分式Shell检测命中，总分: {shell_result.total_score}, 详情: {shell_result.matches}",
                location="description",
                details={"matches": shell_result.matches, "total_score": shell_result.total_score}
            ))
        return results

    def _parse_flags(self, flagstr):
        flags = 0
        if not flagstr:
            return re.IGNORECASE
        if "i" in flagstr: flags |= re.IGNORECASE
        if "s" in flagstr: flags |= re.DOTALL
        if "m" in flagstr: flags |= re.MULTILINE
        return flags 