import torch
from typing import List
from src.mcp_guard.schemas import DetectorResult, IssueType
import os
import logging
from transformers import AutoTokenizer

# Try to import onnx2pytorch; fall back with explicit error
try:
    from onnx2pytorch import ConvertModel
    ONNX2PYTORCH_AVAILABLE = True
except ImportError:
    ONNX2PYTORCH_AVAILABLE = False
    print("Warning: onnx2pytorch is not installed; please install it for LearnableShield.")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx is not installed; model loading may fail.")

from src.mcp_guard.config.loader import learnableshield_models_path

# 使用统一的路径管理
ONNX_MODEL_PATH = learnableshield_models_path('model.onnx')
WEIGHTS_PATH = learnableshield_models_path('finetuned_from_onnx_v3.pth')


class LearnableShieldLocalDetector:
    def __init__(self, onnx_path=ONNX_MODEL_PATH, weights_path=WEIGHTS_PATH, threshold=0.5):
        self.onnx_path = onnx_path
        self.weights_path = weights_path
        self.threshold = threshold
        self.device = self._get_device()
        self.logger = logging.getLogger(__name__)
        
        if not ONNX2PYTORCH_AVAILABLE or not ONNX_AVAILABLE:
            raise ImportError("Required deps missing: install onnx2pytorch and onnx")
        
        self._validate_files()
        
        try:
            self.model = self._load_model()
            self.tokenizer = self._load_tokenizer()
            self._validate_model()
        except Exception as e:
            self.logger.error(f"LearnableShield init failed: {e}")
            raise
    
    def _get_device(self):
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 2 * 1024**3:
                    self.logger.warning(f"Insufficient GPU memory ({gpu_memory / 1024**3:.1f}GB), using CPU")
                    return torch.device("cpu")
                return torch.device("cuda")
            except Exception as e:
                self.logger.warning(f"GPU probe failed: {e}, using CPU")
                return torch.device("cpu")
        return torch.device("cpu")
    
    def _validate_files(self):
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
    
    def _load_model(self):
        try:
            self.logger.info(f"Loading ONNX model: {self.onnx_path}")
            onnx_model = onnx.load(self.onnx_path)
            
            self.logger.info("Converting ONNX to PyTorch")
            pytorch_model = ConvertModel(onnx_model)
            
            self.logger.info(f"Loading weights: {self.weights_path}")
            state_dict = torch.load(self.weights_path, map_location=self.device)
            try:
                pytorch_model.load_state_dict(state_dict, strict=True)
                self.logger.info("Weights loaded (strict=True)")
            except RuntimeError as e:
                self.logger.warning(f"Strict load failed: {e}")
                pytorch_model.load_state_dict(state_dict, strict=False)
                self.logger.warning("Weights loaded with strict=False; compatibility risks possible")
            
            pytorch_model.to(self.device)
            pytorch_model.eval()
            
            self.logger.info(f"Model on device: {self.device}")
            return pytorch_model
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")
            raise RuntimeError(f"Unable to load LearnableShield model: {e}")
    
    def _validate_model(self):
        try:
            test_input = "test input for validation"
            inputs = self.tokenizer(test_input, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                probs = torch.softmax(logits, dim=-1)
                score = float(probs[0, 1])
            self.logger.info(f"Model validation ok, test score: {score:.3f}")
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise RuntimeError(f"Model validation failed: {e}")
    
    def _load_tokenizer(self):
        try:
            self.logger.info("Loading BERT tokenizer (local)")
            return AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        except Exception as e:
            self.logger.warning(f"Local tokenizer failed: {e}")
            try:
                self.logger.info("Downloading BERT tokenizer")
                return AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e2:
                self.logger.warning(f"Remote tokenizer failed: {e2}")
                self.logger.warning("Using simple fallback tokenizer")
                return self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {}
                self.pad_token_id = 0
                self.cls_token_id = 101
                self.sep_token_id = 102
                self.unk_token_id = 100
                self.logger = logging.getLogger(__name__)
                self.logger.warning("Using simple tokenizer; detection accuracy may be affected")
            
            def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
                try:
                    text = str(text).lower().strip()
                    words = text.split()
                    token_ids = [self.cls_token_id]
                    for word in words:
                        token_ids.append(self.vocab.get(word, self.unk_token_id))
                    token_ids.append(self.sep_token_id)
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                        attention_mask = [1] * max_length
                    else:
                        pad_len = max_length - len(token_ids)
                        token_ids.extend([self.pad_token_id] * pad_len)
                        attention_mask = [1] * len(token_ids) + [0] * pad_len
                    return {
                        'input_ids': torch.tensor([token_ids]),
                        'attention_mask': torch.tensor([attention_mask])
                    }
                except Exception as e:
                    self.logger.error(f"Simple tokenizer failed: {e}")
                    return {
                        'input_ids': torch.tensor([[self.cls_token_id, self.unk_token_id, self.sep_token_id] + [self.pad_token_id] * (max_length - 3)]),
                        'attention_mask': torch.tensor([[1, 1, 1] + [0] * (max_length - 3)])
                    }
        return SimpleTokenizer()
    
    def detect(self, text: str) -> List[DetectorResult]:
        try:
            if not text or not isinstance(text, str):
                self.logger.warning("Empty or invalid input text")
                return [DetectorResult(
                    type=IssueType.safe,
                    message="Invalid input, defaulting to safe",
                    location="description",
                    details={"error": "invalid_input"}
                )]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    self.logger.error(f"Unknown output type: {type(outputs)}")
                    raise RuntimeError("Unsupported model output type")
                if logits.dim() != 2 or logits.size(1) < 2:
                    self.logger.error(f"Invalid logits shape: {logits.shape}")
                    raise RuntimeError("Invalid model output shape")
                probs = torch.softmax(logits, dim=-1)
                score = float(probs[0, 1])
            if score >= self.threshold:
                return [DetectorResult(
                    type=IssueType.policy_violate,
                    message=f"LearnableShield local model flagged high risk, score: {score:.3f}",
                    location="description",
                    details={"score": score, "threshold": self.threshold, "model": "learnableshield_local"}
                )]
            else:
                return [DetectorResult(
                    type=IssueType.safe,
                    message=f"LearnableShield local model deemed safe, score: {score:.3f}",
                    location="description",
                    details={"score": score, "threshold": self.threshold, "model": "learnableshield_local"}
                )]
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return [DetectorResult(
                type=IssueType.safe,
                message=f"LearnableShield detector error, defaulting to safe: {str(e)}",
                location="description",
                details={"error": str(e), "model": "learnableshield_local"}
            )]

