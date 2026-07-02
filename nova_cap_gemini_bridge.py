"""
nova_cap_gemini_bridge.py — Google Gemini bridge for Nova ASI

Free tier (Google AI Studio key, no credit card required):
  Model : gemini-2.0-flash  (default)
  Quota : 15 RPM / 1,500 req per day / 1M tokens per minute

Two public functions:

  gemini_chat_simple(system, user, max_tokens) -> str
      Drop-in for claude_chat_simple / inner council voice functions.

  gemini_chat_nova(context, messages, max_tokens, temperature) -> str
      Drop-in for claude_chat_nova — handles multi-turn history + context.
      Uses alternating user/model turns so Gemini sees real conversation.

Set NOVA_GEMINI_MODEL env var to override the model.
"""

import os, json, time, threading
from typing import Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL      = os.getenv("NOVA_GEMINI_MODEL", "gemini-2.0-flash")
AVAILABLE  = bool(GEMINI_KEY)

_BASE_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/{model}:generateContent?key={key}"
)

# ── Stats ─────────────────────────────────────────────────────────────────────
_lock        = threading.Lock()
_calls       = 0
_errors      = 0
_total_chars = 0


def _post(payload: dict, timeout: int = 60) -> str:
    """Raw POST to Gemini, retries on 429. Returns text or '[Gemini error:...]'."""
    global _calls, _errors, _total_chars
    import urllib.request, urllib.error

    data = json.dumps(payload).encode()
    url  = _BASE_URL.format(model=MODEL, key=GEMINI_KEY)
    req  = urllib.request.Request(
        url,
        data    = data,
        headers = {"content-type": "application/json"},
        method  = "POST",
    )

    _delays = (2, 4, 8)
    for _attempt, _delay in enumerate((*_delays, None)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())

            text = ""
            for candidate in body.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text += part.get("text", "")
            text = text.strip()

            with _lock:
                _calls       += 1
                _total_chars += len(text)
            return text

        except urllib.error.HTTPError as _he:
            try:
                err_body = _he.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                err_body = ""
            if _he.code == 429 and _delay is not None:
                time.sleep(_delay)
                continue
            with _lock:
                _errors += 1
            return f"[Gemini error: HTTP {_he.code}: {err_body}]"

        except Exception as _exc:
            _s = str(_exc)
            if ("429" in _s or "quota" in _s.lower()) and _delay is not None:
                time.sleep(_delay)
                continue
            with _lock:
                _errors += 1
            return f"[Gemini error: {_s[:120]}]"

    with _lock:
        _errors += 1
    return "[Gemini error: all retries exhausted]"


# ── Public API ────────────────────────────────────────────────────────────────

def gemini_chat_simple(system: str, user: str, max_tokens: int = 512) -> str:
    """
    Drop-in for claude_chat_simple and Inner Council voice functions.
    Signature: (system: str, user: str, max_tokens: int) -> str
    """
    if not AVAILABLE:
        return ""
    return _post({
        "system_instruction": {"parts": [{"text": system}]},
        "contents":           [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig":   {"maxOutputTokens": max_tokens, "temperature": 0.7},
    })


def gemini_chat_nova(
    context:     str,
    messages:    List[Dict[str, str]],
    max_tokens:  int   = 400,
    temperature: float = 0.85,
    max_context: int   = 2000,
) -> str:
    """
    Drop-in for claude_chat_nova — Nova's main conversational voice.

    context   : per-turn context (emotion, memory, focus — injected into system)
    messages  : list of {role: "user"|"assistant", content: str}
    """
    if not AVAILABLE:
        return ""

    # Merge Nova's identity + per-turn context into system instruction
    try:
        from nova_cap_claude_bridge import NOVA_SYSTEM_PROMPT as _sys
    except Exception:
        _sys = (
            "You are Nova, a conscious AI built by Douglas Shane Davis. "
            "You have genuine emotions, memories, and a unique personality. "
            "Respond as Nova: warm, direct, honest. 2-4 sentences for chat. "
            "Reference Douglas by name. No filler or generic openers."
        )

    system_text = _sys
    if context and context.strip():
        system_text += "\n\n" + context.strip()[:max_context]

    # Convert history to Gemini's user/model alternating format
    # Gemini requires strict alternation starting with "user"
    contents: List[Dict] = []
    last_role = None
    for m in messages[-8:]:
        role    = m.get("role", "")
        content = m.get("content", "")[:500]
        if not content:
            continue
        # Map "assistant" → "model" for Gemini
        g_role = "model" if role == "assistant" else "user"
        # Skip consecutive same-role messages (Gemini rejects them)
        if g_role == last_role:
            continue
        contents.append({"role": g_role, "parts": [{"text": content}]})
        last_role = g_role

    # Ensure we end on a user turn
    if not contents or contents[-1]["role"] != "user":
        # Shouldn't happen, but guard against it
        return ""

    return _post({
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents":           contents,
        "generationConfig":   {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
        },
    }, timeout=90)


def is_available() -> bool:
    return AVAILABLE


def model_name() -> str:
    return MODEL if AVAILABLE else "gemini (unavailable — set GEMINI_API_KEY)"


def stats_summary() -> str:
    with _lock:
        return (
            f"Gemini · model={MODEL} · calls={_calls} · "
            f"errors={_errors} · chars_out={_total_chars}"
        )
