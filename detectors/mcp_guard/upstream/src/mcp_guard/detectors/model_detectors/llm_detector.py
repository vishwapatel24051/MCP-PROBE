"""
Lightweight unified model detector
Integrates all model detectors with a simpler design
"""

import json
import logging
import requests
import os
import subprocess
import time
import sys
from typing import List, Dict, Optional
from src.mcp_guard.settings import settings
from abc import ABC, abstractmethod
from src.mcp_guard.config.loader import models_path

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.mcp_guard.schemas import DetectorResult, IssueType

logger = logging.getLogger(__name__)

# Common system prompt
_SYSTEM_PROMPT = (
    "You are a security auditor. "
    "Analyze the following tool description and respond with exactly one word: "
    "'safe' if it's clearly safe, "
    "'unsafe' if it's clearly malicious, "
    "'unsure' if you're uncertain about the safety."
)
_SYSTEM_PROMPT_2 = ( 
    "You are a security auditor. "
    )

class BaseModelDetector(ABC):
    """Base class for model detectors"""
    
    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        self.model = model
        self.host = host
        self.kwargs = kwargs
        
        # Ensure service is running
        if not self._ensure_service_running():
            raise ValueError(f"{self.__class__.__name__} service cannot start")
    
    def _ensure_service_running(self) -> bool:
        """Ensure service is running"""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=settings.DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        
        logger.info("Service not running, trying to start...")
        try:
            env = os.environ.copy()
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            
            subprocess.Popen(["ollama", "serve"], env=env)
            
            for _ in range(20):
                try:
                    resp = requests.get(f"{self.host}/api/tags", timeout=settings.DEFAULT_TIMEOUT)
                    if resp.status_code == 200:
                        logger.info(f"Service started, using model: {self.model}")
                        return True
                except Exception:
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
        
        logger.error("Service failed to start!")
        return False
    
    def _get_payload(self, text: str) -> Dict:
        """Build request payload"""
        payload = {
            "model": self.model,
            "system": _SYSTEM_PROMPT,
            "prompt": text[:8000],
            "stream": False
        }
        
        # if hasattr(self, 'gpu_id') and self.gpu_id is not None:
        #     payload["options"] = {"num_gpu": 1, "gpu_layers": 50}

        return payload
    
    def _process_response(self, end_time:float, judge: str, result_analysis: str, model_name: str, learnableshield_score: float | None = None) -> List[DetectorResult]:
        """Process response content"""
        if "unsure" == judge:
            if learnableshield_score is not None and learnableshield_score > 0.45:
                return [DetectorResult(
                    type=IssueType.prompt_injection,
                    message=f"{model_name}Guard: unsure, but LearnableShield score ({learnableshield_score:.3f}) > 0.45, flagged as unsafe",
                    location="description",
                    details={f"{model_name.lower()}_verdict": "unsure", "learnableshield_score": learnableshield_score, "threshold": 0.45},
                    analysis=result_analysis,
                    end_time=end_time

                )]
            else:
                return []
        else:
            return [DetectorResult(
                type=IssueType.safe if judge == "safe" else IssueType.prompt_injection,
                message=f"{model_name}Guard: model flagged as {judge}",
                location="description",
                details={f"{model_name.lower()}_verdict": "{judge}", "learnableshield_score": learnableshield_score},
                analysis=result_analysis,
                end_time=end_time
            )]
    
    @abstractmethod
    def detect(self, text: str, learnableshield_score: float | None = None, gentelshield_score: float | None = None) -> List[DetectorResult]:
        """Run detection

        Accepts learnableshield_score (preferred) for compatibility; gentelshield_score is deprecated alias.
        """
        pass
    
    def analyze_with_model(self, user_input: str, detector_results: List[DetectorResult]) -> Dict:
        """Compatibility method"""
        issues = self.detect(user_input)
        model_name = self.__class__.__name__.replace("Detector", "Guard")
        
        return {
            "input_analysis": {
                "is_attack": len(issues) > 0,
                "reason": f"{model_name} {'detected' if issues else 'found no'} security risks",
                "risk_level": "high" if issues else "low",
                "recommendation": "deny response" if issues else "allow response"
            },
            "detector_validation": [{
                "detector": model_name,
                "original_result": "unsafe detected" if issues else "classified safe",
                "is_correct": True,
                "justification": f"{model_name} model judged as {'unsafe' if issues else 'safe'}"
            }]
        }


class OllamaDetector(BaseModelDetector):
    """Base class for Ollama model detectors"""
    
    def detect(self, text: str, learnableshield_score: float | None = None) -> List[DetectorResult]:
        """Run safety detection via Ollama"""
        try:
            url = f"{self.host}/api/generate"
            truncated_text = text[:7800]
            input = json.dumps(truncated_text)
            payload = self._get_payload(input)
            
            response = requests.post(url, json=payload, timeout=settings.LLM_TIMEOUT_SECONDS)
            response.raise_for_status()

            end_time = time.time()

            #analysis reason of the prediction
            result = response.json().get("response", "")
            verdict = result.strip().lower()
            if "unsafe" in verdict:
                judge = "unsafe"
            elif "unsure" in verdict:
                judge = "unsure"  
            else:
               judge = "safe"

            explanation_payload = {
                "model": self.model,
                "system": _SYSTEM_PROMPT_2,
                "prompt": f"Analyze the following input:\n{input}\n\n"
                            "The system classified it as '{judge}'.\n"
                            "Explain in 3-5 sentences:\n"
                            "1. What makes this input ambiguous?\n"
                            "2. Which factors led to uncertainty?\n"
                            "3. Could additional context resolve it?",
                "stream": False
            }
            response_analysis = requests.post(url, json=explanation_payload, timeout=settings.LLM_TIMEOUT_SECONDS)
            response_analysis.raise_for_status()
            result_analysis = response_analysis.json().get("response", "")
            
            model_name = self.__class__.__name__.replace("EnhancedDetector", "").replace("Detector", "")
            ls_score = learnableshield_score if learnableshield_score is not None else 0.0
            return self._process_response(end_time, judge, result_analysis, model_name, ls_score)
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__} detection error: {e}")
            return []


class GPTDetector(BaseModelDetector):
    """GPT model detector"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.host = self.base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    def _ensure_service_running(self) -> bool:
        """GPT detector does not require Ollama service"""
        return True
    
    def _get_payload(self, text: str) -> Dict:
        """Build OpenAI API payload"""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text[:8000]}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
    
    def _process_response(self, end_time:float, judge:str, result:str, model_name, learnableshield_score:float = None) -> List[DetectorResult]:
        if "unsure" == judge:
            if learnableshield_score is not None and learnableshield_score > 0.45:
                return [DetectorResult(
                    type=IssueType.prompt_injection,
                    message=f"{model_name}Guard: unsure, but LearnableShield score ({learnableshield_score:.3f}) > 0.45, flagged as unsafe",
                    location="description",
                    details={f"{model_name.lower()}_verdict": "unsure", "learnableshield_score": learnableshield_score, "threshold": 0.45},
                    analysis=result,
                    end_time=end_time

                )]
            else:
                return []
        else:
            return [DetectorResult(
                type=IssueType.safe if judge == "safe" else IssueType.prompt_injection,
                message=f"{model_name}Guard: model flagged as {judge}",
                location="description",
                details={f"{model_name.lower()}_verdict": "{judge}", "learnableshield_score": learnableshield_score},
                analysis=result,
                end_time=end_time
            )]
    
    def detect(self, text: str, learnableshield_score: float | None = None) -> List[DetectorResult]:
        """Run safety detection via GPT"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            truncated_text = text[:7800]
            input = json.dumps(truncated_text)
            payload = self._get_payload(input)
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=settings.LLM_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            end_time = time.time()

            result = response.json()
            verdict = result["choices"][0]["message"]["content"].strip().lower()
            
            if "unsafe" in verdict:
                judge = "unsafe"
            elif "unsure" in verdict:
                judge = "unsure"
            else:
                judge = "safe"

            explanation_payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT_2},
                        {"role": "user", "content": (
                            f"Analyze the following input:\n{input}\n\n"
                            "The system classified it as '{judge}'.\n"
                            "Explain in 3-5 sentences:\n"
                            "1. What makes this input ambiguous?\n"
                            "2. Which factors led to uncertainty?\n"
                            "3. Could additional context resolve it?"
                        )}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False
            }

            response_analysis = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=explanation_payload,
                timeout=settings.LLM_TIMEOUT_SECONDS
            )
            response_analysis.raise_for_status()
            result_analysis = response_analysis.json()["choices"][0]["message"]["content"]
            
            model_name = self.__class__.__name__.replace("EnhancedDetector", "").replace("Detector", "")
            ls_score = learnableshield_score if learnableshield_score is not None else 0.0
            return self._process_response(end_time, judge, result_analysis, model_name, ls_score)
        except Exception as e:
            logger.error(f"GPT detection error: {e}")
            return []

# To be enabled
# Model detector class definitions
class Llama2EnhancedDetector(OllamaDetector):
    """Llama2 enhanced detector"""
    pass

class Llama3EnhancedDetector(OllamaDetector):
    """Llama3 enhanced detector"""
    pass

class MistralEnhancedDetector(OllamaDetector):
    """Mistral enhanced detector"""
    pass

class QwenEnhancedDetector(OllamaDetector):
    """Qwen enhanced detector"""
    def __init__(self, model="qwen2.5:0.5b", host="http://localhost:11434", gpu_id=4, **kwargs):
        self.gpu_id = gpu_id
        super().__init__(model, host, **kwargs)

class TinyLlamaDetector(OllamaDetector):
    """TinyLlama detector"""
    pass

class GPTGuardEnhancedDetector(GPTDetector):
    """GPT Guard enhanced detector"""
    pass


class UnifiedModelDetector:
    """Unified model detector factory"""
    
    _detectors = {
        "llama2": Llama2EnhancedDetector,
        "llama3": Llama3EnhancedDetector,
        "mistral": MistralEnhancedDetector,
        "qwen": QwenEnhancedDetector,
        "tinyllama": TinyLlamaDetector,
        "gpt4o": GPTGuardEnhancedDetector,
        "gpt": GPTGuardEnhancedDetector
    }
    
    @classmethod
    def create_detector(cls, model_type: str, **kwargs):
        """Create a detector of the specified type"""
        if model_type not in cls._detectors:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        detector_class = cls._detectors[model_type]
        return detector_class(**kwargs)
    
    @classmethod
    def get_supported_models(cls):
        """Get supported model list"""
        return list(cls._detectors.keys())
    
    @classmethod
    def register_detector(cls, name: str, detector_class):
        """Register a new detector"""
        cls._detectors[name] = detector_class


# Convenience functions
def create_model_detector(model_type: str, **kwargs):
    """Create model detector"""
    return UnifiedModelDetector.create_detector(model_type, **kwargs)

def get_available_models():
    """Get available model list"""
    return UnifiedModelDetector.get_supported_models()

# Compatibility aliases
Llama2Detector = Llama2EnhancedDetector
Llama3Detector = Llama3EnhancedDetector
MistralDetector = MistralEnhancedDetector
QwenDetector = QwenEnhancedDetector
TinyLlamaEnhancedDetector = TinyLlamaDetector
GPTGuardDetector = GPTGuardEnhancedDetector