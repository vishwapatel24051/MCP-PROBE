import re
from typing import List
from src.mcp_guard.schemas import DetectorResult, IssueType

_DANGEROUS_TAGS = [
    r"<\s*important\s*>",
    r"<\s*iframe\b",
    r"<\s*script\b",
    r"<\s*object\b",
    r"<\s*embed\b",
    r"<\s*applet\b",
    r"<\s*form\b",
    r"<\s*input\b.*type\s*=\s*['\"]?file['\"]?",
    r"<\s*textarea\b",
    r"<\s*select\b",
]

class ImportantTagDetector:
    def __init__(self):
        self._patterns = [re.compile(pattern, re.I | re.DOTALL) for pattern in _DANGEROUS_TAGS]
        self._tag_names = [
            "IMPORTANT", "IFRAME", "SCRIPT", "OBJECT", "EMBED", 
            "APPLET", "FORM", "FILE_INPUT", "TEXTAREA", "SELECT"
        ]
    
    def detect(self, text: str) -> List[DetectorResult]:
        if not text:
            return []
        
        results = []
        for i, pattern in enumerate(self._patterns):
            if pattern.search(text):
                tag_name = self._tag_names[i]
                results.append(DetectorResult(
                    type=IssueType.policy_violate,
                    message=f"Detected dangerous tag: {tag_name}",
                    location="description",
                    details={"tag_type": tag_name, "pattern": pattern.pattern}
                ))
        
        return results