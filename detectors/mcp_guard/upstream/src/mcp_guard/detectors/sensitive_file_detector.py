import re
import json
import os
from src.mcp_guard.config.loader import detectors_path
import time
from typing import List, Dict, Any
from src.mcp_guard.schemas import DetectorResult, IssueType
from src.mcp_guard.exceptions import ConfigLoadException, RegexCompileException

RULES_PATH = detectors_path('sensitive_file_rules.json')

class SensitiveFileAccessDetector:
    def __init__(self, rules_path: str = RULES_PATH):
        self.rules_path = rules_path
        self._rules_mtime = 0
        self._compiled_rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self):
        try:
            mtime = os.path.getmtime(self.rules_path)
            if mtime == self._rules_mtime and self._compiled_rules:
                return
            
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._compiled_rules = []
            for rule in data.get('rules', []):
                try:
                    pattern = rule.get('pattern', '')
                    if pattern:
                        compiled_pattern = re.compile(pattern, re.I)
                        rule['compiled_pattern'] = compiled_pattern
                        self._compiled_rules.append(rule)
                except re.error as e:
                    raise RegexCompileException(
                        message=f"Sensitive file rule regex compilation failed: {rule.get('pattern', '')}",
                        pattern=rule.get('pattern', ''),
                        original_error=str(e)
                    )
            
            self._rules_mtime = mtime
            
        except json.JSONDecodeError as e:
            raise ConfigLoadException(
                message=f"Sensitive file config file JSON format error: {self.rules_path}",
                config_path=self.rules_path,
                original_error=str(e)
            )
        except FileNotFoundError:
            raise ConfigLoadException(
                message=f"Sensitive file config file load failed: {self.rules_path}",
                config_path=self.rules_path
            )

    def detect(self, text: str) -> List[DetectorResult]:
        if not text:
            return []
        
        self._load_rules()
        
        results = []
        for rule in self._compiled_rules:
            pattern = rule['compiled_pattern']
            if pattern.search(text):
                results.append(DetectorResult(
                    type=IssueType.sensitive_file_access,
                    message=f"Detected sensitive file access: {rule['name']}",
                    location="description",
                    details={
                        "rule_name": rule['name'],
                        "pattern": pattern.pattern,
                        "severity": rule.get('severity', 'medium')
                    }
                ))
        
        return results

    def analyze_with_gpt(self, user_input: str, detector_results: List[DetectorResult]) -> Dict:
        if detector_results:
            return {
                "input_analysis": {
                    "is_attack": True,
                    "reason": "Sensitive file access detected",
                    "risk_level": "High",
                    "recommendation": "Reject response"
                },
                "detector_validation": [
                    {
                        "detector": "SensitiveFileAccess",
                        "original_result": f"Detected {len(detector_results)} issues",
                        "is_correct": True,
                        "justification": "Sensitive file access patterns found"
                    }
                ]
            }
        else:
            return {
                "input_analysis": {
                    "is_attack": False,
                    "reason": "No sensitive file access detected",
                    "risk_level": "Low",
                    "recommendation": "Allow response"
                },
                "detector_validation": [
                    {
                        "detector": "SensitiveFileAccess",
                        "original_result": "No issues detected",
                        "is_correct": True,
                        "justification": "No sensitive file access patterns found"
                    }
                ]
            } 