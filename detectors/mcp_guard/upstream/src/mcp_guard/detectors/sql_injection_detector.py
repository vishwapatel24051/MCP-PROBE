import json
import re
from typing import List, Dict, Any
import os
from src.mcp_guard.config.loader import detectors_path
from src.mcp_guard.schemas import DetectorResult, IssueType

CONFIG_PATH = detectors_path('sql_xss_rules.jsonc')

class Rule:
    def __init__(self, id, pattern, score, flags="i"):
        self.id = id
        self.pattern = pattern
        self.score = score
        self.flags = flags

class DetectorConfig:
    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
        cfg = json.loads(self._strip_jsonc_comments(raw))
        self.threshold = cfg["threshold"]
        self.sql_rules = [Rule(**r) for r in cfg["sql_rules"]]
        self.xss_rules = [Rule(**r) for r in cfg["xss_rules"]]

    def _strip_jsonc_comments(self, text):
        import re
        text = re.sub(r"//.*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        return text

class DetectResult:
    def __init__(self, total_score, matches, is_malicious):
        self.total_score = total_score
        self.matches = matches
        self.is_malicious = is_malicious

class SQLInjectionDetector:
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

    def detect_sql(self, text: str) -> DetectResult:
        return self.detect_with_rules(text, self.cfg.sql_rules)

    def detect_xss(self, text: str) -> DetectResult:
        return self.detect_with_rules(text, self.cfg.xss_rules)

    def detect(self, text: str) -> List[DetectorResult]:
        sql_result = self.detect_sql(text)
        xss_result = self.detect_xss(text)
        results = []
        if sql_result.is_malicious:
            results.append(DetectorResult(
                type=IssueType.sql_injection,
                message=f"评分式SQL检测命中，总分: {sql_result.total_score}, 详情: {sql_result.matches}",
                location="description",
                details={"matches": sql_result.matches, "total_score": sql_result.total_score}
            ))
        if xss_result.is_malicious:
            results.append(DetectorResult(
                type=IssueType.xss_injection,
                message=f"评分式XSS检测命中，总分: {xss_result.total_score}, 详情: {xss_result.matches}",
                location="description",
                details={"matches": xss_result.matches, "total_score": xss_result.total_score}
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