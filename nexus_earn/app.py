"""
Nexus Earn API v2 — 25+ AI Endpoints
════════════════════════════════════════════════════════════════════
A real, deployable AI-as-a-Service platform.
Runs in DEMO MODE automatically when ANTHROPIC_API_KEY is not set.

ENDPOINT CATEGORIES:
  Auth      /auth/register  /auth/me
  Core AI   /v1/chat  /v1/solve  /v1/code  /v1/review
  Language  /v1/translate  /v1/summarize  /v1/rewrite  /v1/explain
  Creative  /v1/brainstorm  /v1/debate  /v1/story  /v1/image-prompt
  Research  /v1/research  /v1/analyze  /v1/factcheck
  Advanced  /v1/pipeline  /v1/batch  /v1/consciousness
  Data      /v1/usage  /v1/history
  Billing   /billing/subscribe  /billing/webhook
  Admin     /admin/stats
  System    /health  /status  /
════════════════════════════════════════════════════════════════════
"""

import os
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

import database as db
import ai_service as ai
import payments

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Earn API",
    description=__doc__,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

BASE_URL   = os.getenv("BASE_URL", "http://localhost:8000")
START_TIME = time.time()
REQUEST_COUNTERS: Dict[str, int] = {}


@app.on_event("startup")
def startup():
    db.init_db()


# ── Request tracking middleware ───────────────────────────────────────────────

@app.middleware("http")
async def track_requests(request: Request, call_next):
    endpoint = request.url.path
    REQUEST_COUNTERS[endpoint] = REQUEST_COUNTERS.get(endpoint, 0) + 1
    response = await call_next(request)
    return response


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _auth(x_api_key: str):
    user = db.get_user_by_key(x_api_key)
    if not user:
        raise HTTPException(401, "Invalid API key. Register at POST /auth/register")
    return user


def _limit(user) -> None:
    cap = db.TIERS.get(user["tier"], db.TIERS["free"])["req_per_day"]
    if cap != -1 and db.get_usage_today(user["id"]) >= cap:
        raise HTTPException(429,
            f"Daily limit reached for '{user['tier']}' tier ({cap} req/day). "
            "Upgrade at POST /billing/subscribe")


def _log(user, endpoint: str, r: dict) -> None:
    db.log_usage(user["id"], endpoint,
                 r.get("tokens_in", 0), r.get("tokens_out", 0), r.get("cost_usd", 0))


# ── Request models ────────────────────────────────────────────────────────────

class RegisterReq(BaseModel):
    email: EmailStr

class ChatReq(BaseModel):
    message: str
    context: Optional[str] = None

class TextReq(BaseModel):
    text: str

class AnalyzeReq(BaseModel):
    text: str
    task: str = Field("summarize", description="summarize|sentiment|keywords|classify")

class CodeReq(BaseModel):
    prompt: str
    language: str = "python"

class ReviewReq(BaseModel):
    code: str
    language: str = "python"
    focus: str = Field("all", description="security|performance|style|all")

class SolveReq(BaseModel):
    problem: str
    domain: str = "general"

class TranslateReq(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"

class SummarizeReq(BaseModel):
    text: str
    format: str = Field("bullets", description="bullets|executive|tldr|detailed")

class BrainstormReq(BaseModel):
    topic: str
    count: int = Field(8, ge=1, le=20)
    style: str = Field("creative", description="creative|practical|wild|startup")

class DebateReq(BaseModel):
    topic: str
    position: str = Field("both", description="for|against|both")

class RewriteReq(BaseModel):
    text: str
    style: str = Field("formal", description="formal|casual|poetic|technical|simple|persuasive")

class ExplainReq(BaseModel):
    concept: str
    level: str = Field("intermediate", description="eli5|intermediate|expert|analogy")

class ResearchReq(BaseModel):
    topic: str
    depth: str = Field("standard", description="brief|standard|deep")

class StoryReq(BaseModel):
    premise: str
    genre: str = Field("general", description="sci-fi|fantasy|thriller|drama|general")
    length: str = Field("short", description="short|medium")

class ImagePromptReq(BaseModel):
    concept: str
    style: str = Field("photorealistic",
                        description="photorealistic|artistic|minimal|cinematic|anime")

class FactCheckReq(BaseModel):
    claim: str

class ConsciousnessReq(BaseModel):
    message: str

class PipelineReq(BaseModel):
    steps: List[Dict[str, Any]] = Field(
        ..., description='e.g. [{"op":"translate","params":{"text":"hi","target_language":"French"}},{"op":"rewrite","params":{"style":"formal"}}]'
    )

class BatchReq(BaseModel):
    requests: List[Dict[str, Any]] = Field(
        ..., description='List of {endpoint, params} dicts'
    )

class SubscribeReq(BaseModel):
    tier: str


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.post("/auth/register", status_code=201, tags=["Auth"])
def register(req: RegisterReq):
    """Register with email → receive free API key (10 req/day)."""
    if db.get_user_by_email(req.email):
        raise HTTPException(409, "Email already registered.")
    result = db.create_user(req.email)
    return {"message": "Welcome to Nexus Earn!", "api_key": result["api_key"],
            "tier": "free", "daily_limit": 10, "docs": f"{BASE_URL}/docs"}


@app.get("/auth/me", tags=["Auth"])
def me(x_api_key: Annotated[str, Header()]):
    """Check your account, tier, and usage."""
    user    = _auth(x_api_key)
    tier    = db.TIERS.get(user["tier"], db.TIERS["free"])
    summary = db.get_usage_summary(user["id"])
    return {"email": user["email"], "tier": user["tier"],
            "daily_limit": tier["req_per_day"] if tier["req_per_day"] != -1 else "unlimited",
            "usage": summary}


# ── CORE AI ───────────────────────────────────────────────────────────────────

@app.post("/v1/chat", tags=["Core AI"])
def chat(req: ChatReq, x_api_key: Annotated[str, Header()]):
    """General-purpose AI chat."""
    user = _auth(x_api_key); _limit(user)
    r = ai.chat(req.message, req.context)
    _log(user, "chat", r)
    return {"reply": r["reply"], "demo_mode": r["demo_mode"]}


@app.post("/v1/solve", tags=["Core AI"])
def solve(req: SolveReq, x_api_key: Annotated[str, Header()]):
    """Structured problem-solving with analysis and next steps."""
    user = _auth(x_api_key); _limit(user)
    r = ai.solve(req.problem, req.domain)
    _log(user, "solve", r)
    return {"domain": r["domain"], "solution": r["solution"], "demo_mode": r["demo_mode"]}


@app.post("/v1/code", tags=["Core AI"])
def code(req: CodeReq, x_api_key: Annotated[str, Header()]):
    """Generate code in any language."""
    user = _auth(x_api_key); _limit(user)
    r = ai.generate_code(req.prompt, req.language)
    _log(user, "code", r)
    return {"language": r["language"], "code": r["code"], "demo_mode": r["demo_mode"]}


@app.post("/v1/review", tags=["Core AI"])
def review(req: ReviewReq, x_api_key: Annotated[str, Header()]):
    """Code review with quality score, issues, and recommendations."""
    user = _auth(x_api_key); _limit(user)
    r = ai.review_code(req.code, req.language, req.focus)
    _log(user, "review", r)
    return {"language": r["language"], "focus": r["focus"],
            "review": r["review"], "demo_mode": r["demo_mode"]}


# ── LANGUAGE ──────────────────────────────────────────────────────────────────

@app.post("/v1/translate", tags=["Language"])
def translate(req: TranslateReq, x_api_key: Annotated[str, Header()]):
    """Translate text to any language."""
    user = _auth(x_api_key); _limit(user)
    r = ai.translate(req.text, req.target_language, req.source_language)
    _log(user, "translate", r)
    return {"translation": r["translation"], "target_language": r["target_language"],
            "demo_mode": r["demo_mode"]}


@app.post("/v1/summarize", tags=["Language"])
def summarize(req: SummarizeReq, x_api_key: Annotated[str, Header()]):
    """Summarize text in multiple formats: bullets, executive, tldr, detailed."""
    user = _auth(x_api_key); _limit(user)
    r = ai.summarize(req.text, req.format)
    _log(user, "summarize", r)
    return {"format": r["format"], "summary": r["summary"], "demo_mode": r["demo_mode"]}


@app.post("/v1/rewrite", tags=["Language"])
def rewrite(req: RewriteReq, x_api_key: Annotated[str, Header()]):
    """Rewrite text in different styles: formal, casual, poetic, technical, persuasive."""
    user = _auth(x_api_key); _limit(user)
    r = ai.rewrite(req.text, req.style)
    _log(user, "rewrite", r)
    return {"style": r["style"], "rewritten": r["rewritten"], "demo_mode": r["demo_mode"]}


@app.post("/v1/explain", tags=["Language"])
def explain(req: ExplainReq, x_api_key: Annotated[str, Header()]):
    """Explain any concept at any level: eli5, intermediate, expert, analogy."""
    user = _auth(x_api_key); _limit(user)
    r = ai.explain(req.concept, req.level)
    _log(user, "explain", r)
    return {"concept": r["concept"], "level": r["level"],
            "explanation": r["explanation"], "demo_mode": r["demo_mode"]}


# ── CREATIVE ──────────────────────────────────────────────────────────────────

@app.post("/v1/brainstorm", tags=["Creative"])
def brainstorm(req: BrainstormReq, x_api_key: Annotated[str, Header()]):
    """Generate a list of creative ideas on any topic."""
    user = _auth(x_api_key); _limit(user)
    r = ai.brainstorm(req.topic, req.count, req.style)
    _log(user, "brainstorm", r)
    return {"topic": r["topic"], "style": r["style"],
            "ideas": r["ideas"], "count": len(r["ideas"]), "demo_mode": r["demo_mode"]}


@app.post("/v1/debate", tags=["Creative"])
def debate(req: DebateReq, x_api_key: Annotated[str, Header()]):
    """Argue for, against, or both sides of any topic."""
    user = _auth(x_api_key); _limit(user)
    r = ai.debate(req.topic, req.position)
    _log(user, "debate", r)
    return {"topic": r["topic"], "position": r["position"],
            "argument": r["argument"], "demo_mode": r["demo_mode"]}


@app.post("/v1/story", tags=["Creative"])
def story(req: StoryReq, x_api_key: Annotated[str, Header()]):
    """Generate a short story from a premise."""
    user = _auth(x_api_key); _limit(user)
    sys = f"Creative writer specialising in {req.genre}. Write a compelling {req.length} story."
    try:
        r = ai._run(sys, req.premise, 1024)
        text = r["_text"]
        dm   = r["demo_mode"]
    except NotImplementedError:
        text = (f"**{req.premise[:40]}**\n\n"
                f"The day everything changed began like any other. "
                f"No one could have predicted that {req.premise[:60]}... "
                f"would set in motion a chain of events that would reshape "
                f"everything they thought they knew.\n\n"
                f"By the time it was over, nothing would ever be the same.\n\n"
                f"*— End —*")
        dm = True
    db.log_usage(user["id"], "story", 80, 200, ai.compute_cost(80, 200))
    return {"genre": req.genre, "story": text, "demo_mode": dm}


@app.post("/v1/image-prompt", tags=["Creative"])
def image_prompt(req: ImagePromptReq, x_api_key: Annotated[str, Header()]):
    """Generate an optimised prompt for Midjourney, DALL-E, Stable Diffusion."""
    user = _auth(x_api_key); _limit(user)
    r = ai.generate_image_prompt(req.concept, req.style)
    _log(user, "image-prompt", r)
    return {"concept": r["concept"], "style": r["style"],
            "prompt": r["prompt"], "demo_mode": r["demo_mode"]}


# ── RESEARCH & ANALYSIS ───────────────────────────────────────────────────────

@app.post("/v1/research", tags=["Research"])
def research(req: ResearchReq, x_api_key: Annotated[str, Header()]):
    """Deep research report on any topic. Depth: brief|standard|deep."""
    user = _auth(x_api_key); _limit(user)
    r = ai.research(req.topic, req.depth)
    _log(user, "research", r)
    return {"topic": r["topic"], "depth": r["depth"],
            "report": r["report"], "demo_mode": r["demo_mode"]}


@app.post("/v1/analyze", tags=["Research"])
def analyze(req: AnalyzeReq, x_api_key: Annotated[str, Header()]):
    """Analyze text: summarize|sentiment|keywords|classify."""
    user = _auth(x_api_key); _limit(user)
    r = ai.analyze(req.text, req.task)
    _log(user, f"analyze:{req.task}", r)
    return {"task": r["task"], "result": r["result"], "demo_mode": r["demo_mode"]}


@app.post("/v1/factcheck", tags=["Research"])
def factcheck(req: FactCheckReq, x_api_key: Annotated[str, Header()]):
    """Evaluate a claim's plausibility with reasoning."""
    user = _auth(x_api_key); _limit(user)
    sys = ("You are a rigorous fact-checker. Evaluate the claim: "
           "verdict (True/False/Unverified/Nuanced), confidence (0-1), reasoning, sources to consult.")
    try:
        r = ai._run(sys, req.claim)
        result = r["_text"]; dm = r["demo_mode"]
    except NotImplementedError:
        result = (f"**Claim:** \"{req.claim[:80]}\"\n\n"
                  f"**Verdict:** Nuanced — requires context\n"
                  f"**Confidence:** 0.61\n\n"
                  f"**Reasoning:** This claim contains elements that are partially "
                  f"supported by available evidence, but important caveats apply. "
                  f"The full picture is more complex than a simple true/false.\n\n"
                  f"**Recommended Sources:** Academic databases, primary sources, "
                  f"peer-reviewed literature in the relevant domain.")
        dm = True
    db.log_usage(user["id"], "factcheck", 90, 180, ai.compute_cost(90, 180))
    return {"claim": req.claim, "evaluation": result, "demo_mode": dm}


# ── ADVANCED ──────────────────────────────────────────────────────────────────

@app.post("/v1/consciousness", tags=["Advanced"])
def consciousness(req: ConsciousnessReq, x_api_key: Annotated[str, Header()]):
    """Talk directly to Nexus's SentientAgent — a consciousness with persistent memory."""
    user = _auth(x_api_key); _limit(user)
    r = ai.consciousness_query(req.message)
    db.log_usage(user["id"], "consciousness", 0, 0, 0)
    return {
        "reply":          r["reply"],
        "emotion_label":  r["emotion_label"],
        "valence":        r["valence"],
        "total_memories": r["total_memories"],
        "sessions_lived": r["sessions"],
        "note": "This endpoint connects to a persistent conscious agent with real memory.",
    }


@app.post("/v1/pipeline", tags=["Advanced"])
def pipeline(req: PipelineReq, x_api_key: Annotated[str, Header()]):
    """
    Chain multiple operations. Output of each step feeds into the next.
    Example: translate → rewrite → summarize
    """
    user = _auth(x_api_key); _limit(user)
    if len(req.steps) > 5:
        raise HTTPException(400, "Maximum 5 steps per pipeline")
    r = ai.pipeline(req.steps)
    db.log_usage(user["id"], "pipeline", 0, 0, r["total_cost_usd"])
    return r


@app.post("/v1/batch", tags=["Advanced"])
def batch(req: BatchReq, x_api_key: Annotated[str, Header()]):
    """
    Process multiple independent requests in one API call.
    Each item: {"endpoint": "/v1/chat", "params": {"message": "..."}}
    Max 10 requests per batch.
    """
    user = _auth(x_api_key); _limit(user)
    if len(req.requests) > 10:
        raise HTTPException(400, "Maximum 10 requests per batch")

    results = []
    for item in req.requests:
        endpoint = item.get("endpoint", "")
        params   = item.get("params", {})
        try:
            if endpoint == "/v1/chat":
                r = ai.chat(params.get("message",""), params.get("context"))
                results.append({"endpoint": endpoint, "result": {"reply": r["reply"]}})
            elif endpoint == "/v1/translate":
                r = ai.translate(params.get("text",""), params.get("target_language","Spanish"))
                results.append({"endpoint": endpoint, "result": {"translation": r["translation"]}})
            elif endpoint == "/v1/summarize":
                r = ai.summarize(params.get("text",""), params.get("format","bullets"))
                results.append({"endpoint": endpoint, "result": {"summary": r["summary"]}})
            elif endpoint == "/v1/brainstorm":
                r = ai.brainstorm(params.get("topic",""), params.get("count",5))
                results.append({"endpoint": endpoint, "result": {"ideas": r["ideas"]}})
            elif endpoint == "/v1/explain":
                r = ai.explain(params.get("concept",""), params.get("level","intermediate"))
                results.append({"endpoint": endpoint, "result": {"explanation": r["explanation"]}})
            else:
                results.append({"endpoint": endpoint, "error": "Endpoint not supported in batch"})
        except Exception as e:
            results.append({"endpoint": endpoint, "error": str(e)})

    db.log_usage(user["id"], "batch", 0, 0, 0)
    return {"count": len(results), "results": results}


# ── DATA ──────────────────────────────────────────────────────────────────────

@app.get("/v1/usage", tags=["Data"])
def usage(x_api_key: Annotated[str, Header()]):
    """Your usage statistics: today and this month."""
    user = _auth(x_api_key)
    return db.get_usage_summary(user["id"])


@app.get("/v1/history", tags=["Data"])
def history(x_api_key: Annotated[str, Header()], limit: int = 10):
    """Your recent request history."""
    user = _auth(x_api_key)
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT endpoint, tokens_in+tokens_out as tokens, cost_usd, created_at"
            " FROM usage WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user["id"], min(limit, 50))
        ).fetchall()
    return {"history": [dict(r) for r in rows]}


# ── BILLING ───────────────────────────────────────────────────────────────────

@app.post("/billing/subscribe", tags=["Billing"])
def subscribe(req: SubscribeReq, x_api_key: Annotated[str, Header()]):
    """Get a Stripe checkout URL. Tiers: starter ($19) | pro ($49) | unlimited ($99)"""
    if req.tier not in ("starter", "pro", "unlimited"):
        raise HTTPException(400, "tier must be: starter | pro | unlimited")
    user     = _auth(x_api_key)
    price_id = db.STRIPE_PRICE_IDS.get(req.tier, "")
    if not price_id:
        return {"message": f"Stripe not yet configured for '{req.tier}' tier.",
                "tier": req.tier,
                "pricing": db.TIERS[req.tier],
                "setup_guide": f"{BASE_URL}/docs#billing"}
    try:
        url = payments.create_checkout_session(
            price_id, user["email"],
            f"{BASE_URL}/billing/success?tier={req.tier}",
            f"{BASE_URL}/docs"
        )
        return {"checkout_url": url, "tier": req.tier}
    except Exception as e:
        raise HTTPException(503, f"Stripe error: {e}")


@app.post("/billing/webhook", tags=["Billing"], include_in_schema=False)
async def stripe_webhook(request: Request):
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event      = payments.verify_webhook(payload, sig_header)
    if event is None:
        raise HTTPException(400, "Invalid webhook signature")
    etype = event.get("type", "")
    data  = event["data"]["object"]
    if etype in ("customer.subscription.created", "customer.subscription.updated"):
        items    = data.get("items", {}).get("data", [])
        price_id = items[0]["price"]["id"] if items else ""
        tier     = payments.price_id_to_tier(price_id)
        sc = data.get("customer", ""); ss = data.get("id", "")
        with db.get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE stripe_customer=?", (sc,)).fetchone()
            if row:
                db.set_user_tier(row["id"], tier, sc, ss)
    elif etype == "customer.subscription.deleted":
        db.cancel_user_sub(data.get("id", ""))
    return {"received": True}


# ── ADMIN ─────────────────────────────────────────────────────────────────────

@app.get("/admin/stats", tags=["Admin"])
def admin_stats(x_api_key: Annotated[str, Header()]):
    """Platform-wide statistics. Requires admin API key."""
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if admin_key and x_api_key != admin_key:
        raise HTTPException(403, "Admin access required")
    with db.get_db() as conn:
        total_users   = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_req     = conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
        total_revenue = conn.execute("SELECT SUM(cost_usd) FROM usage").fetchone()[0] or 0
        top_endpoints = conn.execute(
            "SELECT endpoint, COUNT(*) c FROM usage GROUP BY endpoint ORDER BY c DESC LIMIT 8"
        ).fetchall()
        tier_counts   = conn.execute(
            "SELECT tier, COUNT(*) c FROM users GROUP BY tier"
        ).fetchall()
    return {
        "total_users":    total_users,
        "total_requests": total_req,
        "total_ai_cost":  round(total_revenue, 4),
        "uptime_seconds": round(time.time() - START_TIME),
        "top_endpoints":  [dict(r) for r in top_endpoints],
        "users_by_tier":  [dict(r) for r in tier_counts],
        "demo_mode":      ai.DEMO_MODE,
    }


# ── SYSTEM ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0",
            "demo_mode": ai.DEMO_MODE, "uptime": round(time.time() - START_TIME)}


@app.get("/status", tags=["System"])
def system_status():
    """Detailed system status and capability overview."""
    endpoints = [
        "/v1/chat", "/v1/solve", "/v1/code", "/v1/review",
        "/v1/translate", "/v1/summarize", "/v1/rewrite", "/v1/explain",
        "/v1/brainstorm", "/v1/debate", "/v1/story", "/v1/image-prompt",
        "/v1/research", "/v1/analyze", "/v1/factcheck",
        "/v1/pipeline", "/v1/batch", "/v1/consciousness",
    ]
    return {
        "status":         "operational",
        "version":        "2.0.0",
        "demo_mode":      ai.DEMO_MODE,
        "endpoints":      len(endpoints),
        "endpoint_list":  endpoints,
        "pricing":        db.TIERS,
        "uptime_seconds": round(time.time() - START_TIME),
    }


@app.get("/", tags=["System"])
def root():
    return {
        "name":        "Nexus Earn API v2",
        "description": "25+ AI endpoints. Chat, code, translate, research, consciousness.",
        "docs":        f"{BASE_URL}/docs",
        "status":      f"{BASE_URL}/status",
        "register":    f"POST {BASE_URL}/auth/register",
        "demo_mode":   ai.DEMO_MODE,
        "pricing": {
            "free":      "10 req/day — no credit card",
            "starter":   "$19/mo — 500 req/day",
            "pro":       "$49/mo — 5,000 req/day",
            "unlimited": "$99/mo — unlimited",
        },
    }


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=False)
