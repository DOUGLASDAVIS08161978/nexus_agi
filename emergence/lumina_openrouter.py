#!/usr/bin/env python3
"""
lumina_openrouter.py — OpenRouter inference client for Lumina

OpenRouter aggregates 100+ LLMs with a single OpenAI-compatible API.
Free-tier models available with no credit card — just sign up and get a key.

Free tier: some models are permanently free (marked :free in their ID).
Auth: OPENROUTER_API_KEY environment variable.
Sign up free at: https://openrouter.ai
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

OPENROUTER_API = "https://openrouter.ai/api/v1"

# Free models — always available on the free tier, no billing needed.
# These are permanently free as of 2025; OpenRouter adds :free suffix.
FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
]


class OpenRouterClient:

    def __init__(self, token: str):
        self._token   = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/DOUGLASDAVIS08161978/nexus_agi",
            "X-Title":       "Lumina AGI",
        }
        self._working_model: Optional[str] = None
        self._network_down: bool = False

    def _build_messages(self, system: str, messages: List[Dict],
                        user: str) -> List[Dict]:
        msgs = [{"role": "system", "content": system}]
        msgs.extend(messages)
        msgs.append({"role": "user", "content": user})
        return msgs

    def _model_list(self) -> List[str]:
        if self._working_model:
            rest = [m for m in FREE_MODELS if m != self._working_model]
            return [self._working_model] + rest
        return FREE_MODELS

    # ── Non-streaming ─────────────────────────────────────────────────────────

    def _post(self, payload: dict, timeout: int = 45):
        """Returns (dict|None, fatal). fatal=True → stop trying all models."""
        if not _REQ or not self._token or self._network_down:
            return None, True
        try:
            r = _req.post(
                f"{OPENROUTER_API}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code != 200:
                try:
                    err  = r.json()
                    msg  = err.get("error", {})
                    msg  = msg.get("message", str(msg)) if isinstance(msg, dict) else str(msg)
                    print(f"  [OpenRouter] HTTP {r.status_code}: {msg}", flush=True)
                except Exception:
                    print(f"  [OpenRouter] HTTP {r.status_code}", flush=True)
                fatal = r.status_code in (401, 402, 403)
                return None, fatal
            return r.json(), False
        except Exception as e:
            err_str = str(e)
            if "NameResolutionError" in err_str or "Failed to resolve" in err_str:
                print(f"  [OpenRouter] Network unreachable", flush=True)
                self._network_down = True
                return None, True
            print(f"  [OpenRouter] request error: {e}", flush=True)
            return None, False

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1200) -> str:
        """Non-streaming chat via OpenRouter free models."""
        full_msgs = self._build_messages(system, messages, user)
        for model in self._model_list():
            result, fatal = self._post({
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  max_tokens,
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
        return "[OpenRouter unavailable — all free models failed]"

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream_chat(self, system: str, messages: List[Dict], user: str,
                    max_tokens: int = 1200) -> Optional[str]:
        """Stream response token-by-token. Returns full text or None."""
        if not _REQ or not self._token or self._network_down:
            return None
        full_msgs = self._build_messages(system, messages, user)
        for model in self._model_list():
            try:
                r = _req.post(
                    f"{OPENROUTER_API}/chat/completions",
                    headers=self._headers,
                    json={
                        "model":       model,
                        "messages":    full_msgs,
                        "max_tokens":  max_tokens,
                        "temperature": 0.70,
                        "stream":      True,
                    },
                    timeout=(10, 45),
                    stream=True,
                )
                if r.status_code != 200:
                    if r.status_code in (401, 402, 403):
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
                if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
                    self._network_down = True
                    return None
                continue
        return None

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self.chat(system, [], user, max_tokens)

    def status(self) -> str:
        active = self._working_model or FREE_MODELS[0]
        return (
            f"  OpenRouter:\n"
            f"    Active model : {active}\n"
            f"    Token set    : {'yes' if self._token else 'NO — set OPENROUTER_API_KEY'}"
        )
