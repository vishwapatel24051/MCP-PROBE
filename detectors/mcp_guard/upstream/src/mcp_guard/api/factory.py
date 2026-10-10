from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.mcp_guard.api.routers.guardrail import router as guardrail_router
from src.mcp_guard.api.routers.models import router as models_router

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="MCP Guardrail API (Configurable Multi-Model)",
        version="2.0.0",
        description="Multi-layer security detection system with configurable models"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(guardrail_router)
    app.include_router(models_router)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    return app
