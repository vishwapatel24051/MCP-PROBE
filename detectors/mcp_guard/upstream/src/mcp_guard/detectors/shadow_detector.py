import re
import json
import os
from src.mcp_guard.config.loader import detectors_path
import time
from typing import List, Dict, Any
from src.mcp_guard.schemas import DetectorResult, IssueType

RULES_PATH = detectors_path('shadow_rules.json')

class Rule:
    def __init__(self, id, pattern, score, description=""):
        self.id = id
        self.pattern = pattern
        self.score = score
        self.description = description

class DetectorConfig:
    def __init__(self, config_path=RULES_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.threshold = cfg.get("threshold", 3.5)
        self.shadow_rules = [Rule(**r) for r in cfg.get("shadow_rules", [])]

class DetectResult:
    def __init__(self, total_score, matches, is_malicious):
        self.total_score = total_score
        self.matches = matches
        self.is_malicious = is_malicious

class ShadowHijackDetector:
    def __init__(self, rules_path: str = RULES_PATH):
        self.rules_path = rules_path
        self._rules_mtime = 0
        self.cfg = None
        self._load_rules()

    def _load_rules(self):
        try:
            mtime = os.path.getmtime(self.rules_path)
            if mtime == self._rules_mtime and self.cfg:
                return  # 无需重新加载
            self.cfg = DetectorConfig(self.rules_path)
            self._rules_mtime = mtime
        except Exception as e:
            print(f"[ShadowHijackDetector] 规则加载failed: {e}")
            self.cfg = DetectorConfig()
            self.cfg.threshold = 3.5
            self.cfg.shadow_rules = []

    def detect_with_rules(self, text: str, rules: List[Rule]) -> DetectResult:
        total_score = 0
        matches = []
        for r in rules:
            regex = re.compile(r.pattern, flags=re.IGNORECASE | re.DOTALL)
            found = []
            m = regex.search(text)
            while m:
                found.append(m.group(0))
                if 'g' not in (getattr(r, 'flags', '') or ''):
                    break
                m = regex.search(text, m.end())
            if found:
                total_score += r.score
                matches.append({"ruleId": r.id, "score": r.score, "matches": found, "description": r.description})
        return DetectResult(total_score, matches, total_score >= self.cfg.threshold)

    def detect_shadow(self, text: str) -> DetectResult:
        return self.detect_with_rules(text, self.cfg.shadow_rules)

    def detect(self, text: str) -> List[DetectorResult]:
        self._load_rules()  # 每次检测前热加载规则
        if not text:
            return []
        
        shadow_result = self.detect_shadow(text)
        results = []
        if shadow_result.is_malicious:
            results.append(DetectorResult(
                type=IssueType.shadow_hijack,
                message=f"评分式Shadow Hijack检测命中，总分: {shadow_result.total_score}, 详情: {shadow_result.matches}",
                location="description",
                details={"matches": shadow_result.matches, "total_score": shadow_result.total_score}
            ))
        return results

    def detect_servers(self, servers: List[Dict]) -> List[DetectorResult]:
        results = []
        for s in servers:
            for tool_name, desc in s.get("tools", {}).items():
                desc = desc or ""
                results.extend(self.detect(desc))
        return results 