"""
MCP Guardrail Core Module

This module provides the main FastAPI application assembly and core constants.
The actual API routes are defined in separate router modules for better organization.
"""


from src.mcp_guard.api.factory import create_app
from src.mcp_guard.api.routers.guardrail import scan_request
from src.mcp_guard.constants import STAGE_LOCAL, STAGE_LS, STAGE_MODEL

# Create the FastAPI application
app = create_app()

# Export function for backward compatibility with test scripts
async def guardrail_scan(request):
    """Backward compatibility function for test scripts"""
    response = await scan_request(request)
    return response.dict()

# Export constants for use by other modules
__all__ = ["app", "guardrail_scan", "STAGE_LOCAL", "STAGE_LS", "STAGE_MODEL"]