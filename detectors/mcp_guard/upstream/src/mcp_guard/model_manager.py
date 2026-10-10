import json
import importlib
import logging
import os
from typing import Optional
from src.mcp_guard.schemas import DetectorResult
from src.mcp_guard.config.loader import models_path

CONFIG_PATH = models_path('model_registry.json')

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.active_detector = None
        self._initialize_active_model()
    
    def _load_config(self):
        """Load model configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            return None
    
    def _set_environment_variables(self, model_config):
        """Set environment variables (for models requiring API keys like GPT-4o)"""
        env_vars = model_config.get("env_vars", {})
        for key, value in env_vars.items():
            if value and value != "your-api-key-here":
                os.environ[key] = value
                logger.info(f"Set environment variable: {key}")
    
    def _initialize_active_model(self):
        """Initialize active model"""
        active_model_name = self.config.get("active_model", "llama3")
        model_config = self.config.get("models", {}).get(active_model_name)
        
        if not model_config or not model_config.get("enabled", False):
            logger.warning(f"Model {active_model_name} is disabled or missing")
            return
        
        try:
            # Set env vars if needed
            self._set_environment_variables(model_config)
            
            # Dynamically import model class
            class_path = model_config["class"]
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            detector_class = getattr(module, class_name)
            
            # Instantiate detector
            params = model_config.get("params", {})
            # Filter out None values
            params = {k: v for k, v in params.items() if v is not None}
            
            self.active_detector = detector_class(**params)
            
            logger.info(f"Successfully initialized model: {active_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {active_model_name}: {e}")
            self.active_detector = None
    
    def detect(self, text: str, gentelshield_score: float = None) -> list[DetectorResult]:
        """Run detection"""
        if self.active_detector is None:
            logger.warning("No available model detector")
            return []
        
        try:
            return self.active_detector.detect(text, gentelshield_score)
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            return []
    
    def get_active_model_name(self) -> str:
        """Get current active model name"""
        return self.config.get("active_model", "none")
    
    def get_available_models(self) -> list[str]:
        """Get list of all available models"""
        models = self.config.get("models", {})
        return [name for name, config in models.items() if config.get("enabled", False)]
    
    def switch_model(self, model_name: str) -> bool:
        """Switch active model"""
        if model_name not in self.config.get("models", {}):
            logger.error(f"Model {model_name} does not exist")
            return False
        
        self.config["active_model"] = model_name
        self._initialize_active_model()
        
        # Save configuration
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def enable_model(self, model_name: str) -> bool:
        """Enable model"""
        if model_name not in self.config.get("models", {}):
            logger.error(f"Model {model_name} does not exist")
            return False
        
        self.config["models"][model_name]["enabled"] = True
        return self._save_config()
    
    def disable_model(self, model_name: str) -> bool:
        """Disable model"""
        if model_name not in self.config.get("models", {}):
            logger.error(f"Model {model_name} does not exist")
            return False
        
        self.config["models"][model_name]["enabled"] = False
        return self._save_config()
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False 