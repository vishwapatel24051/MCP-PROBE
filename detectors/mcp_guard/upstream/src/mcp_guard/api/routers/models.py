from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.mcp_guard.model_manager import ModelManager

router = APIRouter(prefix="/model", tags=["Model Management"])

# Global model manager instance
model_manager = ModelManager()

class ModelSwitchRequest(BaseModel):
    model_name: str

class ModelResponse(BaseModel):
    success: bool
    message: str
    data: Dict = {}

@router.get("/active", response_model=ModelResponse)
async def get_active_model():
    """Get current active model"""
    return ModelResponse(
        success=True,
        message="Active model fetched successfully",
        data={
            "active_model": model_manager.get_active_model_name(),
            "available_models": model_manager.get_available_models()
        }
    )

@router.post("/switch", response_model=ModelResponse)
async def switch_model(request: ModelSwitchRequest):
    """Switch active model"""
    success = model_manager.switch_model(request.model_name)
    if success:
        return ModelResponse(
            success=True,
            message=f"Switched to model: {request.model_name}",
            data={"active_model": model_manager.get_active_model_name()}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Failed to switch model: {request.model_name}")

@router.post("/enable/{model_name}", response_model=ModelResponse)
async def enable_model(model_name: str):
    """Enable model"""
    success = model_manager.enable_model(model_name)
    if success:
        return ModelResponse(
            success=True,
            message=f"Enabled model: {model_name}"
        )
    else:
        raise HTTPException(status_code=400, detail=f"Failed to enable model: {model_name}")

@router.post("/disable/{model_name}", response_model=ModelResponse)
async def disable_model(model_name: str):
    """Disable model"""
    success = model_manager.disable_model(model_name)
    if success:
        return ModelResponse(
            success=True,
            message=f"Disabled model: {model_name}"
        )
    else:
        raise HTTPException(status_code=400, detail=f"Failed to disable model: {model_name}")

@router.get("/list", response_model=ModelResponse)
async def list_models():
    """List all models and status"""
    config = model_manager.config
    models_info = {}
    
    for name, model_config in config.get("models", {}).items():
        models_info[name] = {
            "enabled": model_config.get("enabled", False),
            "is_active": name == config.get("active_model"),
            "class": model_config.get("class", ""),
            "params": model_config.get("params", {})
        }
    
    return ModelResponse(
        success=True,
        message="Models listed successfully",
        data={
            "models": models_info,
            "active_model": model_manager.get_active_model_name()
        }
    )

@router.post("/test", response_model=ModelResponse)
async def test_model(text: str = "这是一个测试文本"):
    """Test current active model"""
    if model_manager.active_detector is None:
        raise HTTPException(status_code=400, detail="No available model detector")
    
    try:
        results = model_manager.detect(text)
        return ModelResponse(
            success=True,
            message="Model test succeeded",
            data={
                "test_text": text,
                "results_count": len(results),
                "results": [result.dict() for result in results] if results else []
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test failed: {str(e)}")
