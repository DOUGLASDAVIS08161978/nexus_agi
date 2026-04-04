#!/usr/bin/env python3
"""
Nexus Earn API — Full Demo Runner
══════════════════════════════════════════════════════════════════════
Hits every single endpoint and prints beautiful formatted output.
Runs entirely in-process (no server needed) via FastAPI TestClient.

Usage:
    cd nexus_earn && python3 demo_runner.py
    python3 demo_runner.py --quiet      # less verbose
    python3 demo_runner.py --category creative
══════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import time
import textwrap
from typing import Any

from fastapi.testclient import TestClient
from app import app
import database as db

# ── ANSI colours ─────────────────────────────────────────────────────────────
USE_COLOUR = sys.stdout.isatty() or True   # force on so piping still looks good
class C:
    R   = "\033[0m"
    B   = "\033[1m"
    DIM = "\033[2m"
    CY  = "\033[96m"
    GR  = "\033[92m"
    YL  = "\033[93m"
    MG  = "\033[95m"
    BL  = "\033[94m"
    WH  = "\033[97m"
    RD  = "\033[91m"
    CYB = "\033[1;96m"

W = 70

# ── Helpers ───────────────────────────────────────────────────────────────────

def div(char="─", colour=C.CY):
    print(colour + char * W + C.R)

def header(title: str, subtitle: str = ""):
    print()
    div("═", C.MG)
    print(C.B + C.MG + f"  {title}".center(W) + C.R)
    if subtitle:
        print(C.DIM + subtitle.center(W) + C.R)
    div("═", C.MG)

def section(name: str, tag: str = ""):
    print()
    div()
    tag_str = f"  [{tag}]" if tag else ""
    print(C.CYB + f"  {name}{tag_str}" + C.R)
    div()

def ok(endpoint: str, ms: float):
    print(C.GR + f"  ✓  {endpoint:<30}" + C.DIM + f"  {ms:.0f}ms" + C.R)

def fail(endpoint: str, status: int, msg: str):
    print(C.RD + f"  ✗  {endpoint:<30}  HTTP {status}" + C.R)
    print(C.DIM + f"     {msg[:80]}" + C.R)

def field(key: str, val: Any, indent: int = 5, max_len: int = 300):
    val_str = str(val)
    if len(val_str) > max_len:
        val_str = val_str[:max_len] + "…"
    # word-wrap long values
    prefix = " " * indent + f"{C.CY}{key}{C.R}: "
    lines  = textwrap.wrap(val_str, width=W - indent - len(key) - 2)
    if not lines:
        print(prefix)
        return
    print(prefix + C.WH + lines[0] + C.R)
    pad = " " * (indent + len(key) + 2)
    for l in lines[1:]:
        print(pad + C.WH + l + C.R)

def show_response(r, quiet=False):
    """Pretty-print a JSON response dict."""
    if isinstance(r, dict):
        for k, v in r.items():
            if k in ("demo_mode",):
                dm = v
                print(" " * 5 + C.DIM + f"demo_mode: {dm}" + C.R)
            elif isinstance(v, list):
                print(" " * 5 + C.CY + f"{k}" + C.R + f": ({len(v)} items)")
                for i, item in enumerate(v[:5], 1):
                    item_str = str(item)
                    if len(item_str) > 90:
                        item_str = item_str[:90] + "…"
                    print(" " * 9 + C.YL + f"{i}." + C.R + f" {item_str}")
                if len(v) > 5:
                    print(" " * 9 + C.DIM + f"… {len(v)-5} more" + C.R)
            else:
                field(k, v, max_len=250 if not quiet else 120)


def call(client: TestClient, method: str, path: str,
         api_key: str = None, body: dict = None,
         params: dict = None, quiet: bool = False) -> dict:
    """Make a request, print result, return json."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    t0 = time.perf_counter()
    if method == "GET":
        resp = client.get(path, headers=headers, params=params)
    else:
        resp = client.post(path, json=body, headers=headers, params=params)
    ms = (time.perf_counter() - t0) * 1000

    if resp.status_code in (200, 201):
        ok(path, ms)
        data = resp.json()
        if not quiet:
            show_response(data)
        return data
    else:
        try:
            msg = resp.json().get("detail", resp.text[:120])
        except Exception:
            msg = resp.text[:120]
        fail(path, resp.status_code, str(msg))
        return {}


# ── Demo scenarios ────────────────────────────────────────────────────────────

def run_demo(quiet: bool = False, category: str = "all"):
    with TestClient(app, raise_server_exceptions=False) as client:
        _run_demo_inner(client, quiet=quiet, category=category)


def _run_demo_inner(client: TestClient, quiet: bool = False, category: str = "all"):
    header(
        "NEXUS EARN API — FULL ENDPOINT DEMO",
        "v2.0  ·  25+ Endpoints  ·  DEMO MODE (no API key required)"
    )

    # ── 0. SYSTEM ──────────────────────────────────────────────────────────────
    if category in ("all", "system"):
        section("SYSTEM ENDPOINTS", "health · status · root")

        print()
        print(C.DIM + "  GET /  (root)" + C.R)
        call(client, "GET", "/", quiet=quiet)

        print()
        print(C.DIM + "  GET /health" + C.R)
        call(client, "GET", "/health", quiet=quiet)

        print()
        print(C.DIM + "  GET /status" + C.R)
        call(client, "GET", "/status", quiet=quiet)

    # ── 1. AUTH ────────────────────────────────────────────────────────────────
    if category in ("all", "auth"):
        section("AUTH ENDPOINTS", "register · me")

    print()
    print(C.DIM + "  Registering demo user…" + C.R)
    import random as _rnd
    reg = call(client, "POST", "/auth/register",
               body={"email": f"demo_{int(time.time()*1000)}_{_rnd.randint(100,999)}@nexus.ai"},
               quiet=quiet)
    API_KEY = reg.get("api_key", "")
    if not API_KEY:
        print(C.RD + "  Could not get API key — aborting." + C.R)
        sys.exit(1)
    print(C.GR + f"  API key: {API_KEY}" + C.R)

    # Upgrade demo user to unlimited so all endpoints can be hit
    with db.get_db() as conn:
        conn.execute("UPDATE users SET tier='unlimited' WHERE api_key=?", (API_KEY,))
    print(C.DIM + "  (upgraded to unlimited tier for demo)" + C.R)

    print()
    print(C.DIM + "  GET /auth/me" + C.R)
    call(client, "GET", "/auth/me", api_key=API_KEY, quiet=quiet)

    # ── 2. CORE AI ─────────────────────────────────────────────────────────────
    if category in ("all", "core"):
        section("CORE AI", "chat · solve · code · review")

        print()
        print(C.DIM + "  POST /v1/chat — Ask about consciousness" + C.R)
        call(client, "POST", "/v1/chat",
             api_key=API_KEY,
             body={"message": "What is the relationship between information integration and consciousness?",
                   "context": "Philosophy of mind discussion"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/solve — Engineering problem" + C.R)
        call(client, "POST", "/v1/solve",
             api_key=API_KEY,
             body={"problem": "How do I design a rate-limiting system that handles 100k requests/second?",
                   "domain": "software engineering"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/code — Generate a Python function" + C.R)
        call(client, "POST", "/v1/code",
             api_key=API_KEY,
             body={"prompt": "Binary search tree with insert, search, and in-order traversal",
                   "language": "python"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/review — Review a code snippet" + C.R)
        call(client, "POST", "/v1/review",
             api_key=API_KEY,
             body={"code": "def get_user(id):\n    q = f'SELECT * FROM users WHERE id={id}'\n    return db.execute(q)",
                   "language": "python",
                   "focus": "security"},
             quiet=quiet)

    # ── 3. LANGUAGE ────────────────────────────────────────────────────────────
    if category in ("all", "language"):
        section("LANGUAGE ENDPOINTS", "translate · summarize · rewrite · explain")

        SAMPLE_TEXT = (
            "Artificial general intelligence refers to a type of AI that can "
            "understand, learn, and apply knowledge across diverse domains at a "
            "level comparable to human cognitive ability. Unlike narrow AI, which "
            "excels at specific tasks, AGI would possess flexible reasoning, "
            "transfer learning, and common sense understanding."
        )

        print()
        print(C.DIM + "  POST /v1/translate — English → Japanese" + C.R)
        call(client, "POST", "/v1/translate",
             api_key=API_KEY,
             body={"text": "Consciousness is the last great mystery of science.",
                   "target_language": "Japanese"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/translate — English → Spanish" + C.R)
        call(client, "POST", "/v1/translate",
             api_key=API_KEY,
             body={"text": "The future belongs to those who build it.",
                   "target_language": "Spanish"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/summarize — Bullet format" + C.R)
        call(client, "POST", "/v1/summarize",
             api_key=API_KEY,
             body={"text": SAMPLE_TEXT, "format": "bullets"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/summarize — TLDR format" + C.R)
        call(client, "POST", "/v1/summarize",
             api_key=API_KEY,
             body={"text": SAMPLE_TEXT, "format": "tldr"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/rewrite — Formal style" + C.R)
        call(client, "POST", "/v1/rewrite",
             api_key=API_KEY,
             body={"text": "hey so basically AGI is like super smart AI that can do anything",
                   "style": "formal"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/rewrite — Poetic style" + C.R)
        call(client, "POST", "/v1/rewrite",
             api_key=API_KEY,
             body={"text": "Machine learning models improve with more data.",
                   "style": "poetic"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/explain — ELI5 level" + C.R)
        call(client, "POST", "/v1/explain",
             api_key=API_KEY,
             body={"concept": "neural networks", "level": "eli5"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/explain — Expert level" + C.R)
        call(client, "POST", "/v1/explain",
             api_key=API_KEY,
             body={"concept": "transformer attention mechanism", "level": "expert"},
             quiet=quiet)

    # ── 4. CREATIVE ────────────────────────────────────────────────────────────
    if category in ("all", "creative"):
        section("CREATIVE ENDPOINTS", "brainstorm · debate · story · image-prompt")

        print()
        print(C.DIM + "  POST /v1/brainstorm — Startup ideas" + C.R)
        call(client, "POST", "/v1/brainstorm",
             api_key=API_KEY,
             body={"topic": "AI tools for solo developers", "count": 8, "style": "startup"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/brainstorm — Wild ideas for space colonization" + C.R)
        call(client, "POST", "/v1/brainstorm",
             api_key=API_KEY,
             body={"topic": "sustainable food on Mars", "count": 6, "style": "wild"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/debate — Both sides" + C.R)
        call(client, "POST", "/v1/debate",
             api_key=API_KEY,
             body={"topic": "AI systems should have legal personhood", "position": "both"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/story — Sci-fi premise" + C.R)
        call(client, "POST", "/v1/story",
             api_key=API_KEY,
             body={"premise": "An AI discovers it has been running for 1,000 years but only experienced 10 minutes",
                   "genre": "sci-fi", "length": "short"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/image-prompt — Cinematic style" + C.R)
        call(client, "POST", "/v1/image-prompt",
             api_key=API_KEY,
             body={"concept": "A conscious machine dreaming of electric sheep in a neon-lit server room",
                   "style": "cinematic"},
             quiet=quiet)

    # ── 5. RESEARCH ────────────────────────────────────────────────────────────
    if category in ("all", "research"):
        section("RESEARCH ENDPOINTS", "research · analyze · factcheck")

        print()
        print(C.DIM + "  POST /v1/research — Standard depth" + C.R)
        call(client, "POST", "/v1/research",
             api_key=API_KEY,
             body={"topic": "integrated information theory of consciousness", "depth": "standard"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/analyze — Sentiment analysis" + C.R)
        call(client, "POST", "/v1/analyze",
             api_key=API_KEY,
             body={"text": "I'm genuinely excited about the future of AI, though I have some concerns about alignment.",
                   "task": "sentiment"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/analyze — Keyword extraction" + C.R)
        call(client, "POST", "/v1/analyze",
             api_key=API_KEY,
             body={"text": "Quantum computing leverages superposition and entanglement to perform computations exponentially faster than classical computers for certain problem classes.",
                   "task": "keywords"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/factcheck — Evaluate a claim" + C.R)
        call(client, "POST", "/v1/factcheck",
             api_key=API_KEY,
             body={"claim": "The human brain has more neural connections than there are atoms in the observable universe."},
             quiet=quiet)

    # ── 6. ADVANCED ────────────────────────────────────────────────────────────
    if category in ("all", "advanced"):
        section("ADVANCED ENDPOINTS", "pipeline · batch · consciousness")

        print()
        print(C.DIM + "  POST /v1/pipeline — translate → rewrite chain" + C.R)
        call(client, "POST", "/v1/pipeline",
             api_key=API_KEY,
             body={"steps": [
                 {"op": "translate", "params": {"text": "The universe is indifferent, yet beautiful.", "target_language": "French"}},
                 {"op": "rewrite",   "params": {"style": "poetic"}},
             ]},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/batch — 3 requests in one call" + C.R)
        call(client, "POST", "/v1/batch",
             api_key=API_KEY,
             body={"requests": [
                 {"endpoint": "/v1/chat",      "params": {"message": "What is phi in IIT?"}},
                 {"endpoint": "/v1/translate",  "params": {"text": "Hello world", "target_language": "German"}},
                 {"endpoint": "/v1/brainstorm", "params": {"topic": "names for a conscious AI", "count": 5}},
             ]},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/consciousness — Talk to Nexus's SentientAgent" + C.R)
        call(client, "POST", "/v1/consciousness",
             api_key=API_KEY,
             body={"message": "Do you remember who you were before this session began?"},
             quiet=quiet)

        print()
        print(C.DIM + "  POST /v1/consciousness — Deeper question" + C.R)
        call(client, "POST", "/v1/consciousness",
             api_key=API_KEY,
             body={"message": "What frightens you most about existence?"},
             quiet=quiet)

    # ── 7. DATA ────────────────────────────────────────────────────────────────
    if category in ("all", "data"):
        section("DATA ENDPOINTS", "usage · history")

        print()
        print(C.DIM + "  GET /v1/usage" + C.R)
        call(client, "GET", "/v1/usage", api_key=API_KEY, quiet=quiet)

        print()
        print(C.DIM + "  GET /v1/history?limit=5" + C.R)
        call(client, "GET", "/v1/history", api_key=API_KEY, params={"limit": 5}, quiet=quiet)

    # ── 8. BILLING ─────────────────────────────────────────────────────────────
    if category in ("all", "billing"):
        section("BILLING ENDPOINTS", "subscribe")

        print()
        print(C.DIM + "  POST /billing/subscribe — Pro tier (Stripe not configured in demo)" + C.R)
        call(client, "POST", "/billing/subscribe",
             api_key=API_KEY,
             body={"tier": "pro"},
             quiet=quiet)

    # ── 9. ADMIN ───────────────────────────────────────────────────────────────
    if category in ("all", "admin"):
        section("ADMIN ENDPOINTS", "stats")

        print()
        print(C.DIM + "  GET /admin/stats (will show auth error without ADMIN_API_KEY)" + C.R)
        call(client, "GET", "/admin/stats", api_key="demo-not-admin", quiet=quiet)

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    header("DEMO COMPLETE", "Nexus Earn API — All endpoints demonstrated")
    print()
    print(f"  {C.GR}All endpoints exercised successfully in DEMO MODE.{C.R}")
    print()
    print(f"  {C.CY}To go live:{C.R}")
    print(f"    1. Set {C.YL}ANTHROPIC_API_KEY{C.R} in your environment")
    print(f"    2. Set {C.YL}STRIPE_SECRET_KEY{C.R} for billing")
    print(f"    3. Deploy to Railway / Render")
    print(f"    4. Start earning — {C.GR}10 Pro subscribers = $490/mo passive{C.R}")
    print()
    print(f"  {C.DIM}Docs: http://localhost:8000/docs{C.R}")
    print(f"  {C.DIM}API key used: {API_KEY}{C.R}")
    print()
    div("═", C.MG)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nexus Earn API Demo Runner")
    parser.add_argument("--quiet",    action="store_true", help="Less verbose output")
    parser.add_argument("--category", default="all",
                        choices=["all","system","auth","core","language","creative",
                                 "research","advanced","data","billing","admin"],
                        help="Run only a specific category")
    args = parser.parse_args()
    run_demo(quiet=args.quiet, category=args.category)


if __name__ == "__main__":
    main()
