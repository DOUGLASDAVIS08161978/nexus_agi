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

# Model list — tries each in order until one succeeds.
# Cerebras naming: no hyphen between "llama" and version (e.g. llama3.3-70b).
CEREBRAS_MODELS = [
    "llama-3.3-70b",   # Llama 3.3 70B (best quality)
    "llama3.3-70b",    # alternate spelling
    "llama3.1-70b",    # Llama 3.1 70B (reliable fallback)
    "llama3.1-8b",     # Llama 3.1 8B  (fast last resort)
]


class CerebrasClient:

    def __init__(self, token: str):
        self._token   = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        self._working_model: Optional[str] = None  # cache first model that succeeds

    def _build_messages(self, system: str, messages: List[Dict],
                        user: str) -> List[Dict]:
        msgs = [{"role": "system", "content": system}]
        msgs.extend(messages)
        msgs.append({"role": "user", "content": user})
        return msgs

    # ── Non-streaming ─────────────────────────────────────────────────────────

    def _post(self, payload: dict, timeout: int = 45) -> Optional[dict]:
        if not _REQ or not self._token:
            return None
        try:
            r = _req.post(
                f"{CEREBRAS_API}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1200) -> str:
        """
        Non-streaming chat completion via Cerebras.
        Falls through CEREBRAS_MODELS until one responds.
        """
        full_msgs = self._build_messages(system, messages, user)
        models = ([self._working_model] + CEREBRAS_MODELS
                  if self._working_model else CEREBRAS_MODELS)
        for model in models:
            result = self._post({
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  min(max_tokens, 8192),
                "temperature": 0.70,
            })
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
        """
        Stream a response token-by-token, printing each chunk live.
        Returns the complete text, or None if all models fail.
        """
        if not _REQ or not self._token:
            return None
        full_msgs = self._build_messages(system, messages, user)
        models = ([self._working_model] + CEREBRAS_MODELS
                  if self._working_model else CEREBRAS_MODELS)
        for model in models:
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
                    timeout=(10, 45),
                    stream=True,
                )
                if r.status_code != 200:
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
                    print()   # newline after streamed response
                    self._working_model = model
                    return full_text
            except Exception:
                continue
        return None

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self.chat(system, [], user, max_tokens)

    def status(self) -> str:
        active = self._working_model or CEREBRAS_MODELS[0]
        return (
            f"  Cerebras:\n"
            f"    Active model : {active}\n"
            f"    Token set    : {'yes' if self._token else 'NO — set CEREBRAS_API_KEY'}"
        )
