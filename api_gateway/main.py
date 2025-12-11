"""
Nexus AGI API Gateway
FastAPI application with authentication, billing, and usage tracking
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, get_cors_origins
from tools import ToolRegistry, HTTPFetchTool, WebSearchTool, GitHubRepoTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nexus AGI API",
    description="Artificial General Intelligence API with quantum-enhanced processing",
    version=config.get('APP_VERSION', '4.0.0'),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tool registry
tool_registry = ToolRegistry()

# Security
security = HTTPBearer()


def setup_tools():
    """Initialize and register all available tools"""
    logger.info("Registering tools...")

    # Register HTTP Fetch Tool
    http_tool = HTTPFetchTool(
        timeout=config.get_int('HTTP_TIMEOUT', 30),
        max_retries=config.get_int('HTTP_MAX_RETRIES', 3)
    )
    tool_registry.register('http_fetch', http_tool)

    # Register Web Search Tool
    web_search_tool = WebSearchTool(
        timeout=config.get_int('SEARCH_TIMEOUT', 15),
        max_retries=2
    )
    tool_registry.register('web_search', web_search_tool)

    # Register GitHub Tool
    github_tool = GitHubRepoTool(
        timeout=30,
        max_retries=2
    )
    tool_registry.register('github', github_tool)

    logger.info(f"Registered {len(tool_registry.list_tools())} tools")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Nexus AGI API Gateway...")
    logger.info(f"Environment: {config.get('ENVIRONMENT', 'development')}")
    logger.info(f"Debug mode: {config.get_bool('DEBUG', False)}")

    # Setup tools
    setup_tools()

    logger.info("Nexus AGI API Gateway started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Nexus AGI API Gateway...")


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Nexus AGI API",
        "version": config.get('APP_VERSION', '4.0.0'),
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "tools": "/api/v1/tools",
            "execute": "/api/v1/execute"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": config.get('APP_VERSION', '4.0.0'),
        "environment": config.get('ENVIRONMENT', 'development'),
        "tools_registered": len(tool_registry.list_tools())
    }


@app.get("/api/v1/info")
async def get_api_info():
    """Get API configuration and capabilities"""
    return {
        "version": config.get('APP_VERSION', '4.0.0'),
        "features": config.to_dict()['features'],
        "rate_limits": config.to_dict()['rate_limits'],
        "pricing": config.to_dict()['pricing'],
        "tools_available": tool_registry.list_tools()
    }


# ============================================================================
# TOOLS ENDPOINTS
# ============================================================================

@app.get("/api/v1/tools")
async def list_tools():
    """List all available tools"""
    tools = tool_registry.list_tools()
    return {
        "success": True,
        "count": len(tools),
        "tools": tools
    }


@app.get("/api/v1/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool"""
    info = tool_registry.get_tool_info(tool_name)

    if 'error' in info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {tool_name}"
        )

    return {
        "success": True,
        "tool": info
    }


@app.get("/api/v1/tools/stats")
async def get_tools_stats():
    """Get usage statistics for all tools"""
    return {
        "success": True,
        "stats": tool_registry.get_all_stats()
    }


# ============================================================================
# EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/execute/{tool_name}")
async def execute_tool(
    tool_name: str,
    params: Dict[str, Any],
    request: Request
):
    """
    Execute a tool with given parameters

    Args:
        tool_name: Name of the tool to execute
        params: Tool-specific parameters

    Returns:
        Tool execution result
    """
    # TODO: Add authentication check here
    # user = await get_current_user(credentials)

    # TODO: Add rate limiting check here
    # await check_rate_limit(user)

    # TODO: Log usage for billing
    # await log_usage(user, tool_name)

    logger.info(f"Executing tool: {tool_name}")

    try:
        result = tool_registry.execute(tool_name, params)
        return result

    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )


# ============================================================================
# NEXUS AGI ENDPOINTS (Core functionality)
# ============================================================================

@app.post("/api/v1/nexus/solve")
async def solve_problem(problem: Dict[str, Any], request: Request):
    """
    Solve a complex problem using Nexus AGI

    Args:
        problem: Problem description with domain knowledge and constraints

    Returns:
        Solution with algorithms, strategies, and recommendations
    """
    # TODO: Add authentication
    # TODO: Add rate limiting
    # TODO: This is a placeholder - integrate with actual nexus_agi.py

    return {
        "success": True,
        "message": "Nexus AGI problem solving endpoint (to be implemented)",
        "problem": problem,
        "note": "This endpoint will integrate with the MetaAlgorithm_NexusCore"
    }


# ============================================================================
# AUTHENTICATION ENDPOINTS (Placeholder)
# ============================================================================

@app.post("/api/v1/auth/register")
async def register_user(user_data: Dict[str, Any]):
    """Register a new user"""
    # TODO: Implement user registration
    return {
        "success": True,
        "message": "User registration endpoint (to be implemented)",
        "user_data": user_data
    }


@app.post("/api/v1/auth/login")
async def login_user(credentials: Dict[str, str]):
    """Login and get access token"""
    # TODO: Implement authentication
    return {
        "success": True,
        "message": "Login endpoint (to be implemented)",
        "access_token": "placeholder_token",
        "token_type": "bearer"
    }


# ============================================================================
# BILLING ENDPOINTS (Placeholder)
# ============================================================================

@app.post("/api/v1/billing/subscribe")
async def create_subscription(subscription_data: Dict[str, Any]):
    """Create a new subscription"""
    # TODO: Implement Stripe subscription creation
    return {
        "success": True,
        "message": "Subscription endpoint (to be implemented)",
        "subscription_data": subscription_data
    }


@app.get("/api/v1/billing/usage")
async def get_usage_stats():
    """Get current billing period usage"""
    # TODO: Implement usage tracking
    return {
        "success": True,
        "message": "Usage stats endpoint (to be implemented)",
        "usage": {
            "requests_this_month": 0,
            "limit": 100,
            "tier": "free"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if config.get_bool('DEBUG') else None
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = config.get_int('API_PORT', 8000)
    host = config.get('API_HOST', '0.0.0.0')

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=config.get_bool('DEBUG', False),
        log_level="info"
    )
