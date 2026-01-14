@claude
"""
Nexus AGI API Gateway - Complete Production Version
FastAPI application with full authentication, billing, and usage tracking
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, get_cors_origins
from tools import ToolRegistry, HTTPFetchTool, WebSearchTool, GitHubRepoTool

# Import new modules
from api_gateway.auth import (
    get_current_user,
    get_optional_user,
    create_access_token,
    get_password_hash,
    verify_password,
    create_api_key
)
from api_gateway.database import (
    get_db,
    init_db,
    User,
    create_user,
    get_user_by_email,
    create_subscription,
    get_active_subscription,
    log_usage,
    get_usage_stats
)
from api_gateway.payments import (
    StripePaymentProcessor,
    PRICING_TIERS,
    get_tier_from_price_id
)
from api_gateway.rate_limiter import check_rate_limit_dependency

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nexus AGI API",
    description="Artificial General Intelligence API with quantum-enhanced processing and monetization",
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

# Initialize components
tool_registry = ToolRegistry()
payment_processor = StripePaymentProcessor()


# Pydantic models for request/response
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class SubscriptionCreate(BaseModel):
    tier: str


class ToolExecuteRequest(BaseModel):
    params: Dict[str, Any]


def setup_tools():
    """Initialize and register all available tools"""
    logger.info("Registering tools...")

    http_tool = HTTPFetchTool(
        timeout=config.get_int('HTTP_TIMEOUT', 30),
        max_retries=config.get_int('HTTP_MAX_RETRIES', 3)
    )
    tool_registry.register('http_fetch', http_tool)

    web_search_tool = WebSearchTool(
        timeout=config.get_int('SEARCH_TIMEOUT', 15),
        max_retries=2
    )
    tool_registry.register('web_search', web_search_tool)

    github_tool = GitHubRepoTool(timeout=30, max_retries=2)
    tool_registry.register('github', github_tool)

    logger.info(f"Registered {len(tool_registry.list_tools())} tools")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Nexus AGI API Gateway...")
    logger.info(f"Environment: {config.get('ENVIRONMENT', 'development')}")

    # Initialize database
    init_db()

    # Setup tools
    setup_tools()

    logger.info("✅ Nexus AGI API Gateway started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Nexus AGI API Gateway...")


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db=Depends(get_db)):
    """
    Register a new user

    Creates a user account with free tier subscription
    """
    # Check if user exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = create_user(db, user_data.email, hashed_password, user_data.full_name)

    # Create free subscription
    subscription = create_subscription(db, user.id, tier="free")

    # Create access token
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "tier": "free"}
    )

    return {
        "success": True,
        "message": "User registered successfully",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name
        },
        "subscription": {
            "tier": subscription.tier,
            "requests_limit": subscription.requests_limit
        },
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/api/v1/auth/login")
async def login(credentials: UserLogin, db=Depends(get_db)):
    """
    Login and get access token

    Returns JWT token for authentication
    """
    # Get user
    user = get_user_by_email(db, credentials.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Get subscription
    subscription = get_active_subscription(db, user.id)
    tier = subscription.tier if subscription else "free"

    # Create access token
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "tier": tier,
            "subscription_id": subscription.id if subscription else None
        }
    )

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "tier": tier
        }
    }


@app.get("/api/v1/auth/me")
async def get_me(user=Depends(get_current_user), db=Depends(get_db)):
    """Get current user information"""
    usage_stats = get_usage_stats(db, int(user["user_id"]))

    return {
        "success": True,
        "user": user,
        "usage": usage_stats
    }


@app.post("/api/v1/auth/api-key")
async def create_user_api_key(user=Depends(get_current_user)):
    """Generate API key for user"""
    api_key = create_api_key(user["user_id"], user["tier"])

    return {
        "success": True,
        "api_key": api_key,
        "message": "Store this API key securely. It will not be shown again.",
        "usage": "Add to requests as: Authorization: Bearer <api_key>"
    }


# ============================================================================
# SUBSCRIPTION & BILLING ENDPOINTS
# ============================================================================

@app.get("/api/v1/pricing")
async def get_pricing():
    """Get available pricing tiers"""
    return {
        "success": True,
        "tiers": PRICING_TIERS
    }


@app.post("/api/v1/billing/checkout")
async def create_checkout(
    subscription_data: SubscriptionCreate,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Create Stripe checkout session for subscription

    Returns checkout URL for payment
    """
    tier = subscription_data.tier

    if tier not in PRICING_TIERS or tier == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {tier}"
        )

    price_id = PRICING_TIERS[tier].get("price_id")
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price ID not configured for {tier} tier"
        )

    # Get user
    user_id = int(user["user_id"])
    db_user = db.query(User).filter(User.id == user_id).first()

    # Create checkout session
    try:
        session = payment_processor.create_checkout_session(
            price_id=price_id,
            customer_email=db_user.email,
            success_url=f"{config.get('APP_URL', 'http://localhost:8000')}/billing/success",
            cancel_url=f"{config.get('APP_URL', 'http://localhost:8000')}/billing/cancel"
        )

        return {
            "success": True,
            "checkout_url": session.url,
            "session_id": session.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/billing/usage")
async def get_billing_usage(user=Depends(get_current_user), db=Depends(get_db)):
    """Get current billing period usage"""
    user_id = int(user["user_id"])
    stats = get_usage_stats(db, user_id)

    return {
        "success": True,
        "usage": stats
    }


# ============================================================================
# TOOL EXECUTION ENDPOINTS
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


@app.post("/api/v1/execute/{tool_name}")
async def execute_tool(
    tool_name: str,
    request_data: ToolExecuteRequest,
    user=Depends(get_current_user),
    db=Depends(get_db),
    _=Depends(check_rate_limit_dependency)
):
    """
    Execute a tool with authentication and rate limiting

    Requires authentication and respects rate limits based on subscription tier
    """
    logger.info(f"User {user['user_id']} executing tool: {tool_name}")

    start_time = datetime.utcnow()

    try:
        # Execute tool
        result = tool_registry.execute(tool_name, request_data.params)

        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Log usage
        log_usage(
            db,
            int(user["user_id"]),
            tool_name,
            f"/api/v1/execute/{tool_name}",
            request_data.params,
            "success" if result.get("success") else "error",
            execution_time
        )

        return result

    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Log failed usage
        log_usage(
            db,
            int(user["user_id"]),
            tool_name,
            f"/api/v1/execute/{tool_name}",
            request_data.params,
            "error",
            execution_time
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


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
        "authentication": "Required for tool execution",
        "endpoints": {
            "register": "/api/v1/auth/register",
            "login": "/api/v1/auth/login",
            "pricing": "/api/v1/pricing",
            "tools": "/api/v1/tools"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": config.get('APP_VERSION', '4.0.0'),
        "environment": config.get('ENVIRONMENT', 'development'),
        "tools_registered": len(tool_registry.list_tools()),
        "stripe_configured": payment_processor.enabled
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
        "main_complete:app",
        host=host,
        port=port,
        reload=config.get_bool('DEBUG', False),
        log_level="info"
    )
