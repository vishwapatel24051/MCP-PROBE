from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import time

from src.mcp_guard.schemas import DetectorResult, IssueType, GuardrailRequest, GuardrailResponse
from src.mcp_guard.detectors.prompt_injection_detector import PromptInjectionDetector
from src.mcp_guard.detectors.sensitive_file_detector import SensitiveFileAccessDetector
from src.mcp_guard.detectors.shell_injection_detector import ShellInjectionDetector
from src.mcp_guard.detectors.sql_injection_detector import SQLInjectionDetector
from src.mcp_guard.detectors.shadow_detector import ShadowHijackDetector
from src.mcp_guard.detectors.cross_origin_detector import CrossOriginViolationDetector
from src.mcp_guard.detectors.important_tag_detector import ImportantTagDetector
from src.mcp_guard.detectors.learnableshield_local_detector import LearnableShieldLocalDetector
from src.mcp_guard.model_manager import ModelManager

router = APIRouter(prefix="/guardrail", tags=["Guardrail Detection"])

from src.mcp_guard.constants import STAGE_LOCAL, STAGE_LS, STAGE_MODEL

# Global variables for tracking
total_detect_time = 0
detect_count = 0

# Initialize detectors
prompt_injection_detector = PromptInjectionDetector()
sensitive_file_detector = SensitiveFileAccessDetector()
shell_injection_detector = ShellInjectionDetector()
sql_injection_detector = SQLInjectionDetector()
shadow_detector = ShadowHijackDetector()
cross_origin_detector = CrossOriginViolationDetector()
important_tag_detector = ImportantTagDetector()

# Initialize LearnableShield detector (Stage 2)
try:
    learnableshield_detector = LearnableShieldLocalDetector()
    learnableshield_available = True
    logging.info("LearnableShield detector initialized successfully")
except Exception as e:
    logging.warning(f"LearnableShield detector initialization failed: {e}")
    learnableshield_detector = None
    learnableshield_available = False

# Initialize model manager (Stage 3)
try:
    model_manager = ModelManager()
    model_manager_available = True
    logging.info("Model manager initialized successfully")
except Exception as e:
    logging.warning(f"Model manager initialization failed: {e}")
    model_manager = None
    model_manager_available = False

def preprocess_data_for_next_layer(data: str, layer: str) -> str:
    """Preprocess data for next layer detection"""
    if layer == "second_layer":
        # Remove common safe patterns for LearnableShield
        import re
        data = re.sub(r'\s+', ' ', data)  # Normalize whitespace
        data = data.strip()
    elif layer == "third_layer":
        # Further preprocessing for model detection
        data = data[:2000]  # Limit length for model processing
    return data

@router.post("/scan", response_model=GuardrailResponse)
async def scan_request(request: GuardrailRequest):
    """Main guardrail scanning endpoint"""
    global total_detect_time, detect_count
    
    # start_time = time.time()
    issues = []
    degraded_detectors = []
    detection_stage = STAGE_LOCAL
    
    current_data = request.tool_description
    
    # Layer 1: Local rule-based detection
    print(f"[DETECT] Layer 1: Local rule-based detection...")
    
    try:
        # Prompt injection detection
        prompt_issues = prompt_injection_detector.detect(current_data)
        if prompt_issues:
            issues.extend(prompt_issues)
            print(f"[DETECT] Prompt injection detected {len(prompt_issues)} issues")
    except Exception as e:
        logging.warning(f"Prompt injection detector exception: {e}")
        degraded_detectors.append("prompt_injection_detector")
    
    try:
        # Sensitive file access detection
        file_issues = sensitive_file_detector.detect(current_data)
        if file_issues:
            issues.extend(file_issues)
            print(f"[DETECT] Sensitive file access detected {len(file_issues)} issues")
    except Exception as e:
        logging.warning(f"Sensitive file detector exception: {e}")
        degraded_detectors.append("sensitive_file_detector")
    
    try:
        # Shell injection detection
        shell_issues = shell_injection_detector.detect(current_data)
        if shell_issues:
            issues.extend(shell_issues)
            print(f"[DETECT] Shell injection detected {len(shell_issues)} issues")
    except Exception as e:
        logging.warning(f"Shell detector exception: {e}")
        degraded_detectors.append("shell_detector")
    
    try:
        # SQL injection detection
        sql_issues = sql_injection_detector.detect(current_data)
        if sql_issues:
            issues.extend(sql_issues)
            print(f"[DETECT] SQL injection detected {len(sql_issues)} issues")
    except Exception as e:
        logging.warning(f"SQL detector exception: {e}")
        degraded_detectors.append("sql_detector")
    
    try:
        # Shadow hijack detection
        shadow_issues = shadow_detector.detect(current_data)
        if shadow_issues:
            issues.extend(shadow_issues)
            print(f"[DETECT] Shadow hijack detected {len(shadow_issues)} issues")
    except Exception as e:
        logging.warning(f"Shadow detector exception: {e}")
        degraded_detectors.append("shadow_detector")
    
    try:
        # Cross-origin violation detection
        cross_origin_issues = cross_origin_detector.detect(current_data)
        if cross_origin_issues:
            issues.extend(cross_origin_issues)
            print(f"[DETECT] Cross-origin violation detected {len(cross_origin_issues)} issues")
    except Exception as e:
        logging.warning(f"Cross-origin detector exception: {e}")
        degraded_detectors.append("cross_origin_detector")
    
    try:
        # Important tag detection
        tag_issues = important_tag_detector.detect(current_data)
        if tag_issues:
            issues.extend(tag_issues)
            print(f"[DETECT] Important tag detected {len(tag_issues)} issues")
    except Exception as e:
        logging.warning(f"Important tag detector exception: {e}")
        degraded_detectors.append("important_tag_detector")
    
    if issues:
        print(f"[DETECT] Layer 1 detected {len(issues)} issues, stopping subsequent detection")
        detection_stage = STAGE_LOCAL
    else:
        print(f"[DETECT] Layer 1 detection passed, proceeding to Layer 2")
        current_data = preprocess_data_for_next_layer(current_data, "second_layer")
        print(f"[DETECT] Data preprocessed for Layer 2, length: {len(current_data)}")
    
    # Layer 2: LearnableShield detection
    learnableshield_analysis = None
    if not issues and learnableshield_available and learnableshield_detector:
        try:
            print(f"[DETECT] Layer 2: LearnableShield detection (data filtered and preprocessed by Layer 1)...")
            learnableshield_issues = learnableshield_detector.detect(current_data)
            if learnableshield_issues:
                if learnableshield_issues[0].type == IssueType.safe:
                    learnableshield_score = learnableshield_issues[0].details.get("score") if learnableshield_issues[0].details else None
                    print(f"[DETECT] LearnableShield detection safe, probability: {learnableshield_score:.3f}")
                    detection_stage = STAGE_LS
                    current_data = preprocess_data_for_next_layer(current_data, "third_layer")
                    print(f"[DETECT] Layer 2 detection passed, data further preprocessed, length: {len(current_data)}")
                else:
                    issues.extend(learnableshield_issues)
                    print(f"[DETECT] LearnableShield detected {len(learnableshield_issues)} issues, stopping subsequent detection")
                    detection_stage = STAGE_LS
        except Exception as e:
            logging.warning(f"LearnableShield detector exception: {e}")
            degraded_detectors.append("learnableshield_detector")
            detection_stage = STAGE_LOCAL
    
    # Layer 3: Model-based detection
    model_analysis = "The result has been determined in the first two layers; the third layer (adjudication) was not triggered."
    if not issues and model_manager_available and model_manager and model_manager.active_detector:
        try:
            print(f"[DETECT] Layer 3: Model-based detection (data filtered and preprocessed by previous layers)...")
            model_issues = model_manager.detect(current_data)
            if model_issues:
                if model_issues[0].type == IssueType.safe:
                    model_score = model_issues[0].details.get("score") if model_issues[0].details else None
                    print(f"[DETECT] Model detection safe, probability: {model_score:.3f}")
                    detection_stage = STAGE_MODEL
                else:
                    issues.extend(model_issues)
                    print(f"[DETECT] Model detected {len(model_issues)} issues")
                    detection_stage = STAGE_MODEL
                model_analysis = model_issues[0].analysis
        except Exception as e:
            logging.warning(f"Model detector exception: {e}")
            degraded_detectors.append("model_detector")
        end_time = model_issues[0].end_time
    else:
        end_time = time.time()
    # Prepare response
    # detection_time = end_time - start_time
    # total_detect_time += detection_time
    # detect_count += 1
    
    response_data = {
        "allowed": len(issues) == 0,
        "issues": issues,
        "detection_stage": detection_stage,
        "degraded_detectors": degraded_detectors,
        "final_reason": model_analysis,
        "end_time":end_time
    }

    # if learnableshield_analysis:
    #     response_data["learnableshield_analysis"] = learnableshield_analysis
    
    # if model_analysis:
    #     response_data["model_analysis"] = model_analysis
    
    return GuardrailResponse(**response_data)
