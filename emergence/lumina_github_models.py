#!/usr/bin/env python3
"""
lumina_github_models.py — GitHub Models fallback LLM for Lumina

Uses GitHub's free model inference API (OpenAI-compatible endpoint).
Auth: GITHUB_TOKEN — the same token already used for repo operations.

Free tier limits (per model per day):
  Low-tier  models: 150 requests/day  (gpt-4o-mini, Phi-3.5, Llama-8B)
  High-tier models:  50 requests/day  (Llama-70B, Mistral-large)

Fallback order tries best quality first, drops to smaller models on failure.
"""

from __future__ import annotations
from typing import Dict, List, Optional

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

GH_API = "https://models.inference.ai.azure.com"

# Try best quality first; smaller models as safety net
GH_MODELS = [
    "gpt-4o-mini",                    # best free-tier model, 150 req/day
    "Meta-Llama-3.3-70B-Instruct",    # 70B quality,           50 req/day
    "Meta-Llama-3.1-8B-Instruct",     # fast reliable,        150 req/day
    "Phi-3.5-mini-instruct",          # solid small,          150 req/day
]


class GitHubModelsClient:

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
                f"{GH_API}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code != 200:
                print(f"  [DBG-GH] {payload.get('model','?')} → HTTP {r.status_code}", flush=True)
                return None
            return r.json()
        except Exception as e:
            print(f"  [DBG-GH] {payload.get('model','?')} → {type(e).__name__}: {e}", flush=True)
            return None

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1024) -> str:
        """
        Chat completion via GitHub Models.
        Falls back through GH_MODELS until one responds.
        """
        full_msgs = (
            [{"role": "system", "content": system}]
            + list(messages)
            + [{"role": "user", "content": user}]
        )
        for model in GH_MODELS:
            result = self._post({
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  min(max_tokens, 4096),
                "temperature": 0.70,
            })
            if result and isinstance(result, dict):
                try:
                    return result["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    continue
        return "[GitHub Models unavailable — all models failed]"

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self.chat(system, [], user, max_tokens)

    def status(self) -> str:
        names = ", ".join(m.split("/")[-1] for m in GH_MODELS)
        return (
            f"  GitHub Models:\n"
            f"    Models    : {names}\n"
            f"    Token set : {'yes' if self._token else 'NO — set GITHUB_TOKEN'}"
        )
