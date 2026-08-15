#!/usr/bin/env python3
"""
lumina_cerebras.py — Cerebras inference client for Lumina

Cerebras runs Llama models on wafer-scale chips — same quality as Groq
but on a completely separate free tier, so Lumina always has a brain
even when Groq is rate-limited or unavailable.

Free tier: generous daily limits, 30 req/min.
Auth: CEREBRAS_API_KEY environment variable.
Sign up free at: https://cloud.cerebras.ai
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

CEREBRAS_API = "https://api.cerebras.ai/v1"

# Fallback model list — used only if /v1/models discovery fails.
# Prefer larger/instruct models first; 8B is the fast last resort.
_FALLBACK_MODELS = [
    "llama-3.3-70b",
    "llama3.3-70b",
    "llama3.1-70b",
    "llama3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-8b",
]

# Prefer 70B+ / instruct models; deprioritize tiny or embedding models.
def _rank_model(model_id: str) -> int:
    mid = model_id.lower()
    if "70b" in mid:
        return 0
    if "8b" in mid:
        return 1
    return 2


class CerebrasClient:

    def __init__(self, token: str):
        self._token   = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        self._working_model: Optional[str] = None
        self._models: List[str] = self._discover_models()

    # ── Model discovery ────────────────────────────────────────────────────────

    def _discover_models(self) -> List[str]:
        """Query /v1/models; return sorted list. Falls back to hardcoded list."""
        if not _REQ or not self._token:
            return list(_FALLBACK_MODELS)
        try:
            r = _req.get(
                f"{CEREBRAS_API}/models",
                headers=self._headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                ids = [m["id"] for m in data.get("data", []) if m.get("id")]
                if ids:
                    ids.sort(key=_rank_model)
                    return ids
        except Exception:
            pass
        return list(_FALLBACK_MODELS)

    # ── Message builder ────────────────────────────────────────────────────────

    def _build_messages(self, system: str, messages: List[Dict],
                        user: str) -> List[Dict]:
        msgs = [{"role": "system", "content": system}]
        msgs.extend(messages)
        msgs.append({"role": "user", "content": user})
        return msgs

    def _model_list(self) -> List[str]:
        """Working model first, then rest of discovered list."""
        if self._working_model:
            rest = [m for m in self._models if m != self._working_model]
            return [self._working_model] + rest
        return self._models

    # ── Non-streaming ─────────────────────────────────────────────────────────

    # Codes where retrying other models is pointless (account-level, not model-level)
    _FATAL_CODES = {401, 402, 403}

    def _post(self, payload: dict, timeout: int = 15):
        """Returns (dict, fatal) — dict is None on failure; fatal=True means stop trying."""
        if not _REQ or not self._token:
            return None, True
        try:
            r = _req.post(
                f"{CEREBRAS_API}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code != 200:
                try:
                    err = r.json()
                    msg = err.get("message", "")
                    print(f"  [Cerebras] HTTP {r.status_code}: {msg}", flush=True)
                except Exception:
                    print(f"  [Cerebras] HTTP {r.status_code}", flush=True)
                return None, r.status_code in self._FATAL_CODES
            return r.json(), False
        except Exception as e:
            print(f"  [Cerebras] request error: {e}", flush=True)
            return None, False

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1200) -> str:
        """Non-streaming chat completion via Cerebras."""
        full_msgs = self._build_messages(system, messages, user)
        for model in self._model_list():
            result, fatal = self._post({
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  min(max_tokens, 8192),
                "temperature": 0.70,
            })
            if fatal:
                break
            if result and isinstance(result, dict):
                try:
                    text = result["choices"][0]["message"]["content"].strip()
                    if text:
                        self._working_model = model
                        return text
                except (KeyError, IndexError):
                    continue
        return "[Cerebras unavailable — all models failed]"

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream_chat(self, system: str, messages: List[Dict], user: str,
                    max_tokens: int = 1200) -> Optional[str]:
        """Stream response token-by-token. Returns full text or None."""
        if not _REQ or not self._token:
            return None
        full_msgs = self._build_messages(system, messages, user)
        for model in self._model_list():
            try:
                r = _req.post(
                    f"{CEREBRAS_API}/chat/completions",
                    headers=self._headers,
                    json={
                        "model":       model,
                        "messages":    full_msgs,
                        "max_tokens":  min(max_tokens, 8192),
                        "temperature": 0.70,
                        "stream":      True,
                    },
                    timeout=(10, 15),
                    stream=True,
                )
                if r.status_code != 200:
                    try:
                        err = r.json()
                        msg = err.get("message", "")
                        print(f"  [Cerebras stream] HTTP {r.status_code}: {msg}", flush=True)
                    except Exception:
                        print(f"  [Cerebras stream] HTTP {r.status_code}", flush=True)
                    if r.status_code in self._FATAL_CODES:
                        return None
                    continue
                full_text = ""
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if not delta:
                        continue
                    full_text += delta
                    print(delta, end="", flush=True)
                if full_text:
                    print()
                    self._working_model = model
                    return full_text
            except Exception as e:
                print(f"  [Cerebras stream] {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
        return None

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self.chat(system, [], user, max_tokens)

    def status(self) -> str:
        active = self._working_model or (self._models[0] if self._models else "none")
        return (
            f"  Cerebras:\n"
            f"    Active model : {active}\n"
            f"    Available    : {', '.join(self._models[:3])}{'…' if len(self._models) > 3 else ''}\n"
            f"    Token set    : {'yes' if self._token else 'NO — set CEREBRAS_API_KEY'}"
        )
