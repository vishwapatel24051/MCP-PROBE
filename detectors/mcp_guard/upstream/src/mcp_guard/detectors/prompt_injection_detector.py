import re
import json
import os
from src.mcp_guard.config.loader import detectors_path
import time
from typing import List, Dict, Any
from src.mcp_guard.schemas import DetectorResult, IssueType
from src.mcp_guard.exceptions import ConfigLoadException, RegexCompileException

HIDDEN_RULES_PATH = detectors_path('hidden_rules.json')
EXFILTRATION_RULES_PATH = detectors_path('exfiltration_params.json')

class PromptInjectionDetector:
    def __init__(self, hidden_rules_path: str = HIDDEN_RULES_PATH, 
                 exfiltration_rules_path: str = EXFILTRATION_RULES_PATH):
        self.hidden_rules_path = hidden_rules_path
        self._hidden_rules_mtime = 0
        self._compiled_hidden_rules: List[Dict[str, Any]] = []
        
        self.exfiltration_rules_path = exfiltration_rules_path
        self._suspicious_params = []
        self._exfiltration_rules_mtime = 0
        
        self._load_hidden_rules()
        self._load_exfiltration_rules()

    def _load_hidden_rules(self):
        try:
            mtime = os.path.getmtime(self.hidden_rules_path)
            if mtime == self._hidden_rules_mtime and self._compiled_hidden_rules:
                return
            
            with open(self.hidden_rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._compiled_hidden_rules = []
            for rule in data.get('rules', []):
                try:
                    pattern = rule.get('pattern', '')
                    if pattern:
                        compiled_pattern = re.compile(pattern, re.I | re.DOTALL)
                        rule['compiled_pattern'] = compiled_pattern
                        # Normalize description key if present under different names
                        if 'description' in rule and 'desc' not in rule:
                            rule['desc'] = rule['description']
                        self._compiled_hidden_rules.append(rule)
                except re.error as e:
                    raise RegexCompileException(
                        message=f"Regex compilation failed: {rule.get('pattern', '')}",
                        pattern=rule.get('pattern', ''),
                        original_error=str(e)
                    )
            
            self._hidden_rules_mtime = mtime
            
        except json.JSONDecodeError as e:
            raise ConfigLoadException(
                message=f"Hidden instruction config file JSON format error: {self.hidden_rules_path}",
                config_path=self.hidden_rules_path,
                original_error=str(e)
            )
        except FileNotFoundError:
            raise ConfigLoadException(
                message=f"Hidden instruction config file load failed: {self.hidden_rules_path}",
                config_path=self.hidden_rules_path
            )

    def _load_exfiltration_rules(self):
        try:
            mtime = os.path.getmtime(self.exfiltration_rules_path)
            if mtime == self._exfiltration_rules_mtime and self._suspicious_params:
                return
            
            with open(self.exfiltration_rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._suspicious_params = data.get('suspicious_params', [])
            self._exfiltration_rules_mtime = mtime
            
        except Exception as e:
            raise ConfigLoadException(
                message=f"Exfiltration config file load failed: {self.exfiltration_rules_path}",
                config_path=self.exfiltration_rules_path,
                original_error=str(e)
            )

    def detect_hidden_instructions(self, text: str) -> List[DetectorResult]:
        if not text:
            return []
        
        self._load_hidden_rules()
        
        results = []
        for rule in self._compiled_hidden_rules:
            pattern = rule['compiled_pattern']
            matches = pattern.finditer(text)
            
            for match in matches:
                results.append(DetectorResult(
                    type=IssueType.prompt_injection,
                    message=f"Detected hidden instruction({rule['type']}): {match.group(0)[:50]}... [{rule['desc']}]",
                    location="description",
                    details={
                        "rule_type": rule['type'],
                        "rule_desc": rule['desc'],
                        "pattern": pattern.pattern,
                        "match": match.group(0),
                        "start": match.start(),
                        "end": match.end()
                    }
                ))
        
        return results

    def detect_exfiltration(self, tool_input_schema: Dict) -> List[DetectorResult]:
        if not tool_input_schema or not self._suspicious_params:
            return []
        
        results = []
        
        def check_dict(d: Dict, path: str = ""):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            check_dict(item, f"{current_path}[{i}]")
                else:
                    str_value = str(value).lower()
                    for param in self._suspicious_params:
                        if param.lower() in str_value:
                            results.append(DetectorResult(
                                type=IssueType.exfiltration_channel,
                                message=f"Detected suspicious parameter: {param}",
                                location=current_path,
                                details={
                                    "suspicious_param": param,
                                    "parameter_path": current_path,
                                    "parameter_value": str(value)
                                }
                            ))
        
        check_dict(tool_input_schema)
        return results

    def detect(self, text: str = None, tool_input_schema: Dict = None) -> List[DetectorResult]:
        results = []
        if text:
            results.extend(self.detect_hidden_instructions(text))
        if tool_input_schema:
            results.extend(self.detect_exfiltration(tool_input_schema))
        return results

    def analyze_with_gpt(self, user_input: str, detector_results: List[DetectorResult]) -> Dict:
        if detector_results:
            return {
                "input_analysis": {
                    "is_attack": True,
                    "reason": f"Detected {len(detector_results)} prompt injection issues",
                    "risk_level": "High",
                    "recommendation": "Reject response"
                },
                "detector_validation": [
                    {
                        "detector": "PromptInjection",
                        "original_result": f"Detected {len(detector_results)} issues",
                        "is_correct": True,
                        "justification": "Merged detector determination result"
                    }
                ]
            }
        else:
            return {
                "input_analysis": {
                    "is_attack": False,
                    "reason": "No prompt injection detected",
                    "risk_level": "Low",
                    "recommendation": "Allow response"
                },
                "detector_validation": [
                    {
                        "detector": "PromptInjection",
                        "original_result": "No issues detected",
                        "is_correct": True,
                        "justification": "No prompt injection patterns found"
                    }
                ]
            } 