"""
Nexus Earn — AI Service
Wraps the Anthropic Claude API to power all Nexus Earn endpoints.
"""

import os
import httpx
from typing import Optional

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")  # cheapest fast model
MAX_TOKENS        = 1024


def _api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return key


def _call(system: str, user_msg: str,
          max_tokens: int = MAX_TOKENS) -> dict:
    """Make a raw call to the Anthropic Messages API."""
    headers = {
        "x-api-key":         _api_key(),
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    body = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": user_msg}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(ANTHROPIC_API_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def _extract(response: dict) -> tuple[str, int, int]:
    """Return (text, input_tokens, output_tokens)."""
    text = response["content"][0]["text"]
    usage = response.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# Cost per 1M tokens (claude-haiku-4-5 pricing — update if model changes)
_COST_IN  = 0.80 / 1_000_000
_COST_OUT = 4.00 / 1_000_000


def compute_cost(tokens_in: int, tokens_out: int) -> float:
    return round(tokens_in * _COST_IN + tokens_out * _COST_OUT, 6)


# ── Public service functions ──────────────────────────────────────────────────

def chat(message: str, context: Optional[str] = None) -> dict:
    system = (
        "You are Nexus, an advanced AI assistant. "
        "Be helpful, clear, and concise."
        + (f"\n\nContext provided by user:\n{context}" if context else "")
    )
    resp = _call(system, message)
    text, ti, to = _extract(resp)
    return {"reply": text, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to)}


def analyze(text: str, task: str = "summarize") -> dict:
    tasks = {
        "summarize":  "Summarize the following text concisely.",
        "sentiment":  "Analyze the sentiment of the following text. Return: sentiment (positive/negative/neutral), confidence (0-1), and a brief explanation.",
        "keywords":   "Extract the 10 most important keywords/phrases from the following text. Return as a numbered list.",
        "classify":   "Classify the topic/category of the following text. Be specific.",
    }
    instruction = tasks.get(task, tasks["summarize"])
    system = f"You are a text analysis expert. {instruction}"
    resp = _call(system, text)
    result, ti, to = _extract(resp)
    return {"task": task, "result": result, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to)}


def generate_code(prompt: str, language: str = "python") -> dict:
    system = (
        f"You are an expert {language} developer. "
        f"Write clean, well-commented {language} code for the request. "
        "Return only the code block, no extra explanation unless asked."
    )
    resp = _call(system, prompt, max_tokens=2048)
    code, ti, to = _extract(resp)
    return {"language": language, "code": code, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to)}


def solve(problem: str, domain: str = "general") -> dict:
    system = (
        f"You are an expert problem solver specializing in {domain}. "
        "Analyse the problem thoroughly and provide a clear, actionable solution. "
        "Structure your response with: 1) Problem Analysis, 2) Solution, 3) Next Steps."
    )
    resp = _call(system, problem, max_tokens=2048)
    solution, ti, to = _extract(resp)
    return {"domain": domain, "solution": solution, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to)}
