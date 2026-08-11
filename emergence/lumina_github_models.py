#!/usr/bin/env python3
"""
lumina_cerebras.py — Cerebras inference fallback for Lumina

Cerebras runs the same Llama models as Groq but on a separate free tier,
giving Lumina a second brain when Groq is rate-limited.

Free tier: generous daily limits, 30 req/min.
Auth: CEREBRAS_API_KEY environment variable.
Sign up free at: https://cloud.cerebras.ai
"""

from __future__ import annotations
from typing import Dict, List, Optional

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

CEREBRAS_API = "https://api.cerebras.ai/v1"

# Same Llama quality as Groq — separate daily quota
CEREBRAS_MODELS = [
    "llama-3.3-70b",    # best quality, same as Groq primary
    "llama3.1-8b",      # fast fallback
]


class CerebrasClient:

    def __init__(self, token: str):
        self._token   = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

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
                print(f"  [DBG-CB] {payload.get('model','?')} → HTTP {r.status_code}", flush=True)
                return None
            return r.json()
        except Exception as e:
            print(f"  [DBG-CB] {payload.get('model','?')} → {type(e).__name__}: {e}", flush=True)
            return None

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1024) -> str:
        """
        Chat completion via Cerebras.
        Falls back through CEREBRAS_MODELS until one responds.
        """
        full_msgs = (
            [{"role": "system", "content": system}]
            + list(messages)
            + [{"role": "user", "content": user}]
        )
        for model in CEREBRAS_MODELS:
            result = self._post({
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  min(max_tokens, 8192),
                "temperature": 0.70,
            })
            if result and isinstance(result, dict):
                try:
                    return result["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    continue
        return "[Cerebras unavailable — all models failed]"

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self.chat(system, [], user, max_tokens)

    def status(self) -> str:
        names = ", ".join(CEREBRAS_MODELS)
        return (
            f"  Cerebras:\n"
            f"    Models    : {names}\n"
            f"    Token set : {'yes' if self._token else 'NO — set CEREBRAS_API_KEY'}"
        )
