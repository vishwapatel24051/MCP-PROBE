"""
Data schemas for MCP Guardrail API

This module contains Pydantic models for API request/response validation.
"""

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Types of security issues that can be detected"""
    sql_injection = "sql_injection"
    xss_injection = "xss_injection"
    prompt_injection = "prompt_injection"
    shadow_hijack = "shadow_hijack"
    policy_violate = "policy_violate"
    sensitive_file_access = "sensitive_file_access"
    exfiltration_channel = "exfiltration_channel"
    cross_origin_violation = "cross_origin_violation"
    shell_injection = "shell_injection"
    SIGNATURE = "signature"
    safe = "safe"


class DetectorResult(BaseModel):
    """Result from a security detector"""
    type: IssueType
    message: str
    location: Optional[str] = None
    details: Optional[Dict] = None
    analysis: str = ""
    end_time: Optional[float] = None

class GuardrailRequest(BaseModel):
    """Request model for guardrail scanning"""
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    servers: List[Dict] = Field(default_factory=list)
    detector_name: Optional[str] = None
    tool_input_schema: Optional[Dict] = None
    current_server_name: Optional[str] = None
    other_server_names: Optional[List[str]] = None
    safe_list: Optional[List[str]] = None


class GuardrailResponse(BaseModel):
    """Response model for guardrail scanning"""
    allowed: bool
    issues: List[DetectorResult]
    detector_used: Optional[str] = None
    degraded_detectors: Optional[List[str]] = None
    degradation_note: Optional[str] = None
    final_reason: Optional[str] = None
    gpt_analysis: Optional[Dict] = None
    detection_stage: Optional[str] = None
    end_time: Optional[float] = None

