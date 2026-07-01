"""
nova_cap_ollama_bridge.py
Nova ASI — Ollama Local LLM Bridge

Gives Nova a fully local, free, unlimited brain via Ollama.
Runs on the same device as Nova (Termux, desktop, server).

Fallback position in Nova's intelligence chain:
    Claude (cloud, cached)  — best quality
    Groq   (cloud, fast)    — free tier, rate-limited
    Ollama (local, free)    — unlimited, private, always available

No API key needed. No rate limits. No cost. No internet required.

Setup (once):
    pkg install ollama          # Termux
    # or: curl -fsSL https://ollama.ai/install.sh | sh  (Linux/Mac)
    ollama pull llama3.2        # recommended conversation model
    ollama pull llama3.1:8b     # optional: stronger reasoning
    ollama serve                # start the local server (auto-starts on most installs)

Configuration (optional, in ~/nexus_agi/.env):
    OLLAMA_HOST=http://localhost:11434   # default
    OLLAMA_MODEL=llama3.2               # conversation model
    OLLAMA_CODEGEN_MODEL=llama3.2       # code generation model

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional

# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env() -> None:
    path = os.path.expanduser("~/nexus_agi/.env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

_load_env()

OLLAMA_HOST          = os.getenv("OLLAMA_HOST",         "http://localhost:11434").rstrip("/")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",        "llama3.2")
OLLAMA_CODEGEN_MODEL = os.getenv("OLLAMA_CODEGEN_MODEL", OLLAMA_MODEL)

# ── Availability check ─────────────────────────────────────────────────────────

def is_available() -> bool:
    """True when Ollama server is reachable and has at least one model."""
    try:
        req  = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return bool(data.get("models"))
    except Exception:
        return False


def list_models() -> List[str]:
    """Return names of all locally available Ollama models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def best_model(prefer: str = OLLAMA_MODEL) -> str:
    """
    Return the best available model, preferring OLLAMA_MODEL.
    Falls back through common model names if the preferred isn't installed.
    """
    available = list_models()
    if not available:
        return prefer
    # Exact match
    for m in available:
        if m == prefer or m.startswith(prefer.split(":")[0] + ":"):
            return m
    # Preference order for general conversation
    _candidates = [
        "llama3.2", "llama3.1:8b", "llama3.1", "llama3",
        "llama3:8b", "mistral", "gemma2", "phi3",
    ]
    for c in _candidates:
        for m in available:
            if m.startswith(c):
                return m
    return available[0]


# ── Core chat function ─────────────────────────────────────────────────────────

def ollama_chat(
    messages:    List[Dict[str, str]],
    system:      str   = "",
    model:       str   = "",
    max_tokens:  int   = 512,
    temperature: float = 0.7,
    timeout:     int   = 120,
) -> str:
    """
    Send a chat request to the local Ollama server.

    Args:
        messages     — list of {role, content} dicts (user/assistant turns)
        system       — system prompt string
        model        — model override (default: OLLAMA_MODEL or best available)
        max_tokens   — max output tokens (maps to Ollama's num_predict)
        temperature  — sampling temperature
        timeout      — request timeout in seconds

    Returns the assistant's response text, or "" on failure.
    """
    _model = model or best_model()
    if not _model:
        return ""

    # Build message list with optional system message
    _msgs: List[Dict[str, str]] = []
    if system:
        _msgs.append({"role": "system", "content": system})
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            _msgs.append({"role": m["role"], "content": m["content"]})

    payload = json.dumps({
        "model":   _model,
        "messages": _msgs,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
    except urllib.error.URLError:
        return ""   # Ollama not running — silent fail so chain continues
    except Exception:
        return ""


def ollama_chat_simple(system: str, user: str, **kwargs) -> str:
    """Convenience wrapper: takes a system string and a user string."""
    return ollama_chat(
        messages = [{"role": "user", "content": user}],
        system   = system,
        **kwargs,
    )


# ── Status ─────────────────────────────────────────────────────────────────────

def status_summary() -> str:
    """Human-readable status line for Nova's /status command."""
    if not is_available():
        return (
            f"  Ollama       : offline (install: pkg install ollama && ollama pull llama3.2)\n"
            f"  Host         : {OLLAMA_HOST}"
        )
    models  = list_models()
    current = best_model()
    return (
        f"  Ollama       : online  ({len(models)} model{'s' if len(models) != 1 else ''} available)\n"
        f"  Host         : {OLLAMA_HOST}\n"
        f"  Active model : {current}\n"
        f"  All models   : {', '.join(models[:8])}"
        + (f" +{len(models)-8} more" if len(models) > 8 else "")
    )
