from typing import Any, Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel


class DetectorException(Exception):
    def __init__(self, message: str, error_code: str = "DETECTOR_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class PolicyLoadException(DetectorException):
    def __init__(self, message: str, file_path: str):
        super().__init__(message, "POLICY_LOAD_ERROR", {"file_path": file_path})


class GPTGuardException(DetectorException):
    def __init__(self, message: str, api_error: Optional[str] = None):
        super().__init__(message, "GPT_GUARD_ERROR", {"api_error": api_error})


class RemoteVerifyException(DetectorException):
    def __init__(self, message: str, url: str, status_code: Optional[int] = None):
        super().__init__(message, "REMOTE_VERIFY_ERROR", {"url": url, "status_code": status_code})


class ValidationException(DetectorException):
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class ConfigLoadException(DetectorException):
    def __init__(self, message: str, config_path: str, original_error: Optional[str] = None):
        super().__init__(message, "CONFIG_LOAD_ERROR", {
            "config_path": config_path,
            "original_error": original_error
        })


class RegexCompileException(DetectorException):
    def __init__(self, message: str, pattern: str, original_error: Optional[str] = None):
        super().__init__(message, "REGEX_COMPILE_ERROR", {
            "pattern": pattern,
            "original_error": original_error
        })


class RuleParseException(DetectorException):
    def __init__(self, message: str, rule_id: Optional[str] = None, rule_type: Optional[str] = None, original_error: Optional[str] = None):
        super().__init__(message, "RULE_PARSE_ERROR", {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "original_error": original_error
        })


class ErrorResponse(BaseModel):
    error: bool = True
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


def detector_exception_to_http(exc: DetectorException) -> HTTPException:
    error_mapping = {
        "POLICY_LOAD_ERROR": 500,
        "GPT_GUARD_ERROR": 503,
        "REMOTE_VERIFY_ERROR": 502,
        "VALIDATION_ERROR": 400,
        "CONFIG_LOAD_ERROR": 500,
        "REGEX_COMPILE_ERROR": 500,
        "RULE_PARSE_ERROR": 500,
        "DETECTOR_ERROR": 500
    }
    
    status_code = error_mapping.get(exc.error_code, 500)
    return HTTPException(
        status_code=status_code,
        detail={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    ) 