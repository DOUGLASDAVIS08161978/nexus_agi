"""
Nexus Earn — Main API Service
═════════════════════════════
A real, deployable AI-as-a-Service API that generates income for Douglas.

Endpoints:
  POST /auth/register          → get your API key (free tier)
  GET  /auth/me                → check your tier & usage
  POST /v1/chat                → AI chat
  POST /v1/analyze             → text analysis (summarize/sentiment/keywords)
  POST /v1/code                → code generation
  POST /v1/solve               → problem solving
  GET  /v1/usage               → your usage stats
  POST /billing/subscribe      → upgrade tier (returns Stripe checkout URL)
  POST /billing/webhook        → Stripe webhook (internal)
  GET  /health                 → health check
  GET  /                       → API info

Run locally:
  uvicorn app:app --reload

Deploy to Railway:
  railway up
"""

import os
from typing import Optional, Annotated

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

import database as db
import ai_service as ai
import payments

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Earn API",
    description=(
        "AI-as-a-Service powered by Nexus AGI. "
        "Chat, analyze, generate code, and solve problems."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


@app.on_event("startup")
def startup():
    db.init_db()


# ── Auth helper ───────────────────────────────────────────────────────────────

def _get_user(x_api_key: str):
    user = db.get_user_by_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. "
                            "Register at POST /auth/register to get one.")
    return user


def _check_limit(user) -> None:
    tier_cfg  = db.TIERS.get(user["tier"], db.TIERS["free"])
    daily_cap = tier_cfg["req_per_day"]
    if daily_cap == -1:
        return   # unlimited tier
    used = db.get_usage_today(user["id"])
    if used >= daily_cap:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {daily_cap} requests reached for '{user['tier']}' tier. "
                   f"Upgrade at POST /billing/subscribe."
        )


# ── Request / response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class AnalyzeRequest(BaseModel):
    text: str
    task: str = "summarize"   # summarize | sentiment | keywords | classify

class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"

class SolveRequest(BaseModel):
    problem: str
    domain: str = "general"

class SubscribeRequest(BaseModel):
    tier: str   # starter | pro | unlimited


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", status_code=201, tags=["Auth"])
def register(req: RegisterRequest):
    """
    Register with your email and receive a free API key.
    Free tier: 10 requests/day. Upgrade anytime via /billing/subscribe.
    """
    existing = db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409,
                            detail="Email already registered. "
                                   "Use GET /auth/me with your existing key.")
    result = db.create_user(req.email)
    return {
        "message": "Welcome to Nexus Earn! 🎉",
        "api_key": result["api_key"],
        "tier":    result["tier"],
        "note":    "Keep your API key safe. Pass it as the X-Api-Key header.",
        "docs":    f"{BASE_URL}/docs",
    }


@app.get("/auth/me", tags=["Auth"])
def me(x_api_key: Annotated[str, Header()]):
    """Check your account tier, limits, and usage."""
    user     = _get_user(x_api_key)
    tier_cfg = db.TIERS.get(user["tier"], db.TIERS["free"])
    summary  = db.get_usage_summary(user["id"])
    return {
        "email":      user["email"],
        "tier":       user["tier"],
        "daily_limit": tier_cfg["req_per_day"] if tier_cfg["req_per_day"] != -1 else "unlimited",
        "usage":      summary,
        "upgrade_at": f"{BASE_URL}/billing/subscribe",
    }


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.post("/v1/chat", tags=["AI"])
def chat(req: ChatRequest, x_api_key: Annotated[str, Header()]):
    """Chat with Nexus AI. Great for Q&A, brainstorming, writing."""
    user = _get_user(x_api_key)
    _check_limit(user)
    result = ai.chat(req.message, req.context)
    db.log_usage(user["id"], "chat",
                 result["tokens_in"], result["tokens_out"], result["cost_usd"])
    return {"reply": result["reply"],
            "tokens_used": result["tokens_in"] + result["tokens_out"]}


@app.post("/v1/analyze", tags=["AI"])
def analyze(req: AnalyzeRequest, x_api_key: Annotated[str, Header()]):
    """
    Analyze text. Tasks: summarize | sentiment | keywords | classify
    """
    user = _get_user(x_api_key)
    _check_limit(user)
    result = ai.analyze(req.text, req.task)
    db.log_usage(user["id"], f"analyze:{req.task}",
                 result["tokens_in"], result["tokens_out"], result["cost_usd"])
    return {"task": result["task"], "result": result["result"],
            "tokens_used": result["tokens_in"] + result["tokens_out"]}


@app.post("/v1/code", tags=["AI"])
def code(req: CodeRequest, x_api_key: Annotated[str, Header()]):
    """Generate code in any language."""
    user = _get_user(x_api_key)
    _check_limit(user)
    result = ai.generate_code(req.prompt, req.language)
    db.log_usage(user["id"], f"code:{req.language}",
                 result["tokens_in"], result["tokens_out"], result["cost_usd"])
    return {"language": result["language"], "code": result["code"],
            "tokens_used": result["tokens_in"] + result["tokens_out"]}


@app.post("/v1/solve", tags=["AI"])
def solve(req: SolveRequest, x_api_key: Annotated[str, Header()]):
    """Solve complex problems with structured analysis."""
    user = _get_user(x_api_key)
    _check_limit(user)
    result = ai.solve(req.problem, req.domain)
    db.log_usage(user["id"], f"solve:{req.domain}",
                 result["tokens_in"], result["tokens_out"], result["cost_usd"])
    return {"domain": result["domain"], "solution": result["solution"],
            "tokens_used": result["tokens_in"] + result["tokens_out"]}


@app.get("/v1/usage", tags=["AI"])
def usage(x_api_key: Annotated[str, Header()]):
    """Your usage statistics."""
    user = _get_user(x_api_key)
    return db.get_usage_summary(user["id"])


# ── Billing endpoints ─────────────────────────────────────────────────────────

@app.post("/billing/subscribe", tags=["Billing"])
def subscribe(req: SubscribeRequest, x_api_key: Annotated[str, Header()]):
    """
    Get a Stripe checkout URL to upgrade your tier.
    Tiers: starter ($19/mo) | pro ($49/mo) | unlimited ($99/mo)
    """
    if req.tier not in ("starter", "pro", "unlimited"):
        raise HTTPException(status_code=400,
                            detail="tier must be: starter | pro | unlimited")

    user     = _get_user(x_api_key)
    price_id = db.STRIPE_PRICE_IDS.get(req.tier, "")
    if not price_id:
        raise HTTPException(status_code=503,
                            detail=f"Stripe price ID for '{req.tier}' not configured yet. "
                                   "Check back soon!")
    try:
        url = payments.create_checkout_session(
            price_id      = price_id,
            customer_email= user["email"],
            success_url   = f"{BASE_URL}/billing/success?tier={req.tier}",
            cancel_url    = f"{BASE_URL}/docs",
        )
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"Stripe error: {e}")
    return {"checkout_url": url, "tier": req.tier}


@app.get("/billing/success", tags=["Billing"], include_in_schema=False)
def billing_success(tier: str = "starter"):
    return {"message": f"🎉 Subscription active! Your tier will update within seconds.",
            "tier": tier}


@app.post("/billing/webhook", tags=["Billing"], include_in_schema=False)
async def stripe_webhook(request: Request):
    """Stripe sends events here. Handles subscription created/updated/cancelled."""
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event      = payments.verify_webhook(payload, sig_header)

    if event is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    etype = event.get("type", "")
    data  = event["data"]["object"]

    if etype in ("customer.subscription.created", "customer.subscription.updated"):
        # Find price ID from subscription items
        items    = data.get("items", {}).get("data", [])
        price_id = items[0]["price"]["id"] if items else ""
        tier     = payments.price_id_to_tier(price_id)
        stripe_customer = data.get("customer", "")
        stripe_sub      = data.get("id", "")
        # Find user by stripe customer id or email
        # (Stripe passes customer ID; we set it on first checkout)
        # For simplicity, update by stripe_customer if already set, else skip
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE stripe_customer=?", (stripe_customer,)
            ).fetchone()
            if row:
                db.set_user_tier(row["id"], tier, stripe_customer, stripe_sub)

    elif etype == "customer.subscription.deleted":
        stripe_sub = data.get("id", "")
        db.cancel_user_sub(stripe_sub)

    return {"received": True}


# ── Info & health ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", tags=["System"])
def root():
    return {
        "name":        "Nexus Earn API",
        "description": "AI-as-a-Service — chat, analyze, code, solve.",
        "docs":        f"{BASE_URL}/docs",
        "pricing": {
            "free":      "10 req/day — no credit card",
            "starter":   "$19/mo — 500 req/day",
            "pro":       "$49/mo — 5,000 req/day",
            "unlimited": "$99/mo — unlimited",
        },
        "register": f"POST {BASE_URL}/auth/register",
    }


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500,
                        content={"error": "Internal server error", "detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=False)
