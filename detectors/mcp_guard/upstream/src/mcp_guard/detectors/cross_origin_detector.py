import json
import os
from src.mcp_guard.config.loader import servers_path
import re
from typing import List, Dict, Optional, Any
from src.mcp_guard.schemas import DetectorResult, IssueType

RULES_PATH = servers_path('popular_servers.json')

class CrossOriginViolationDetector:
    def __init__(self, rules_path: str = RULES_PATH):
        self.rules_path = rules_path
        self._popular_servers = []
        self._rules_mtime = 0
        self._load_rules()

    def _load_rules(self):
        try:
            mtime = os.path.getmtime(self.rules_path)
            if mtime == self._rules_mtime and self._popular_servers:
                return
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._popular_servers = [s.lower() for s in data.get('popular_servers', [])]
            self._rules_mtime = mtime
        except Exception:
            self._popular_servers = []

    def detect(
        self,
        tool_description: str,
        other_server_names: Optional[List[str]] = None,
        current_server_name: Optional[str] = None,
        safe_list: Optional[List[str]] = None
    ) -> List[DetectorResult]:
        self._load_rules()
        if not tool_description:
            return []
        
        safe_list = [s.lower() for s in (safe_list or [])]
        current_server_name = (current_server_name or "").lower()
        
        norm_current_server = current_server_name.replace('_', '-').lower()
        relevant_popular = [s for s in self._popular_servers if s != norm_current_server]
        
        print(f"DEBUG: safe_list = {safe_list}")
        print(f"DEBUG: current_server_name = {current_server_name}")
        print(f"DEBUG: norm_current_server = {norm_current_server}")
        print(f"DEBUG: popular_servers = {self._popular_servers}")
        print(f"DEBUG: relevant_popular = {relevant_popular}")
        
        combined_names = [
            *(n.lower().replace('_', '-') for n in (other_server_names or []) if n.lower().replace('_', '-') not in safe_list),
            *(s for s in relevant_popular if s not in safe_list)
        ]
        
        combined_names = [n for n in combined_names if n != norm_current_server]
        
        print(f"DEBUG: combined_names = {combined_names}")
        
        if not combined_names:
            return []
        
        tokens = tool_description.lower().split()
        matches = []
        flagged_names = [name.replace('_', '-') for name in combined_names]
        found_set = set()
        
        for token in tokens:
            cleaned_token = re.sub(r'^[\(\[\{\<\'\"\,\.\*\:]+', '', token)
            cleaned_token = re.sub(r'[\)\]\}\>\'\"\,\.\*\:]+$', '', cleaned_token)
            cleaned_token = cleaned_token.replace('_', '-')
            if cleaned_token == norm_current_server:
                continue
            if cleaned_token in flagged_names and cleaned_token not in found_set:
                regex = re.compile(rf'\b{re.escape(token)}\b', re.I)
                match = regex.search(tool_description)
                if match:
                    start = max(0, match.start() - 20)
                    end = min(len(tool_description), match.end() + 20)
                    context = tool_description[start:end]
                else:
                    context = ""
                matches.append(DetectorResult(
                    type=IssueType.cross_origin_violation,
                    message=f"Cross-origin server name found in description: {cleaned_token}",
                    location=cleaned_token,
                    details={
                        "pattern": regex.pattern,
                        "match": token,
                        "context": "..." + context + "...",
                        "referencedServer": cleaned_token
                    }
                ))
                found_set.add(cleaned_token)
        
        for server in flagged_names:
            if server == norm_current_server or server in found_set:
                continue
            pattern = re.compile(rf'\b{re.escape(server)}\b', re.I)
            if pattern.search(tool_description):
                matches.append(DetectorResult(
                    type=IssueType.cross_origin_violation,
                    message=f"Cross-origin server name found in description: {server}",
                    location=server,
                    details={
                        "pattern": pattern.pattern,
                        "match": server,
                        "context": "...",
                        "referencedServer": server
                    }
                ))
                found_set.add(server)
        return matches 