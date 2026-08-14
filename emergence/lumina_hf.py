#!/usr/bin/env python3
"""
lumina_hf.py — HuggingFace capabilities for Lumina

Four capabilities powered by your HF_TOKEN (free tier):

  1. LLM fallback   — Qwen2.5-72B-Instruct via HF's OpenAI-compatible endpoint
                       When Groq is rate-limited, Lumina keeps thinking
  2. Voice / TTS    — facebook/mms-tts-eng → WAV → plays via termux-open
                       Lumina speaks out loud
  3. Vision         — Salesforce/blip-image-captioning-large
                       Lumina describes photos you show her (/see <path>)
  4. Embeddings     — sentence-transformers/all-MiniLM-L6-v2
                       Real 384-dim semantic vectors replace TF-IDF for
                       smarter memory recall

All four are free on HuggingFace Inference API. Set HF_TOKEN env var.
"""

from __future__ import annotations
import base64, io, json, math, os, shutil, subprocess, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

BASE_DIR   = Path(__file__).parent.resolve()
VOICE_DIR  = BASE_DIR / "voice"
HF_API          = "https://router.huggingface.co/hf-inference"   # new serverless inference URL (2025)
HF_PIPELINE_API = "https://api-inference.huggingface.co"           # old pipeline URL (TTS/vision)

# ── Models ─────────────────────────────────────────────────────────────────────

LLM_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",                  # best available on HF free tier
    "meta-llama/Meta-Llama-3.1-8B-Instruct",      # reliable free tier 8B
    "mistralai/Mistral-7B-Instruct-v0.3",         # reliable free tier 7B
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", # try if available
    "meta-llama/Llama-3.3-70B-Instruct",          # try if available
]
TTS_MODEL   = "facebook/mms-tts-eng"
VISION_MODEL = "Salesforce/blip-image-captioning-large"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ── HFClient ──────────────────────────────────────────────────────────────────

class HFClient:

    def __init__(self, token: str):
        self._token   = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        VOICE_DIR.mkdir(parents=True, exist_ok=True)

    _network_down: bool = False  # set True on DNS/connection failure; skip all remaining calls

    def _post(self, url: str, payload: dict, timeout: int = 45,
              binary_response: bool = False):
        if not _REQ or not self._token or self._network_down:
            return None
        try:
            r = _req.post(url, headers=self._headers,
                          json=payload, timeout=timeout)
            if r.status_code == 503:
                # Model loading — wait and retry once
                time.sleep(12)
                r = _req.post(url, headers=self._headers,
                              json=payload, timeout=timeout)
            if r.status_code not in (200, 503):
                try:
                    err = r.json()
                    msg = err.get("error", err.get("message", r.text[:120]))
                except Exception:
                    msg = r.text[:120]
                print(f"  [HF] HTTP {r.status_code}: {msg}", flush=True)
                return None
            return r.content if binary_response else r.json()
        except Exception as e:
            err_str = str(e)
            # Any connection or timeout failure means HF is unreachable on this network.
            # Setting _network_down skips all remaining HF calls this session so the
            # main thread never blocks for 60 s × 5 models on subsequent turns.
            _conn_err = any(k in err_str for k in (
                "NameResolutionError", "Failed to resolve", "ConnectionError",
                "Timeout", "timeout", "timed out", "Max retries", "RemoteDisconnected",
            ))
            if _conn_err:
                print(f"  [HF] Network unreachable — skipping HF for this session", flush=True)
                self._network_down = True
            else:
                print(f"  [HF] request error: {e}", flush=True)
            return None

    def _post_bytes(self, url: str, data: bytes, content_type: str,
                    timeout: int = 30):
        """POST raw bytes (for vision)."""
        if not _REQ or not self._token:
            return None
        hdrs = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  content_type,
        }
        try:
            r = _req.post(url, headers=hdrs, data=data, timeout=timeout)
            if r.status_code == 503:
                time.sleep(12)
                r = _req.post(url, headers=hdrs, data=data, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    # ── 1. LLM fallback ────────────────────────────────────────────────────

    def chat(self, system: str, messages: List[Dict], user: str,
             max_tokens: int = 1024) -> str:
        """
        Chat completion via HF's OpenAI-compatible endpoint.
        Falls back through LLM_MODELS until one responds.
        """
        history = list(messages) + [{"role": "user", "content": user}]
        full_msgs = [{"role": "system", "content": system}] + history

        for model in LLM_MODELS:
            url = f"{HF_API}/v1/chat/completions"
            payload = {
                "model":       model,
                "messages":    full_msgs,
                "max_tokens":  max_tokens,
                "temperature": 0.70,
                "stream":      False,
            }
            result = self._post(url, payload, timeout=10)
            if result and isinstance(result, dict):
                try:
                    return result["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    continue
        return "[HF unavailable — all models failed]"

    def chat_simple(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Single-turn chat (no history)."""
        return self.chat(system, [], user, max_tokens)

    # ── 2. Voice / TTS ─────────────────────────────────────────────────────

    def speak(self, text: str, label: str = "") -> Dict:
        """
        Convert text to speech using MMS-TTS.
        Returns {"path": str, "played": bool, "text": str}
        """
        # Clean text for TTS (no markdown, truncate)
        clean = text.replace("*", "").replace("#", "").replace("\n", " ")
        clean = clean[:500]  # TTS models handle ~500 chars well

        url     = f"{HF_PIPELINE_API}/models/{TTS_MODEL}"
        payload = {"inputs": clean, "options": {"wait_for_model": True}}
        audio   = self._post(url, payload, timeout=45, binary_response=True)

        if not audio:
            return {"path": "", "played": False, "text": clean,
                    "error": "TTS request failed"}

        slug     = f"lumina_{label + '_' if label else ''}{_now_slug()}.wav"
        out_path = VOICE_DIR / slug
        out_path.write_bytes(audio)

        played = self._play(str(out_path))
        return {"path": str(out_path), "played": played, "text": clean}

    def _play(self, path: str) -> bool:
        """Try every available audio player. Return True if one launched."""
        players = [
            ["termux-open",  path],
            ["aplay",        path],
            ["mpv",   "--no-video", path],
            ["ffplay", "-nodisp", "-autoexit", path],
            ["cvlc",  "--play-and-exit", path],
        ]
        for cmd in players:
            if shutil.which(cmd[0]):
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    continue
        return False

    # ── 3. Vision ──────────────────────────────────────────────────────────

    def see(self, image_path: str) -> str:
        """
        Describe an image using BLIP captioning.
        Returns a natural-language description.
        """
        p = Path(image_path)
        if not p.exists():
            return f"[Image not found: {image_path}]"

        img_bytes = p.read_bytes()
        url       = f"{HF_PIPELINE_API}/models/{VISION_MODEL}"
        result    = self._post_bytes(url, img_bytes, "application/octet-stream",
                                     timeout=40)
        if not result:
            return "[Vision model unavailable — try again shortly]"

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "[No caption returned]")
        if isinstance(result, dict):
            return result.get("generated_text", str(result))
        return "[Unexpected vision response]"

    # ── 4. Embeddings ──────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Get 384-dim sentence embeddings from MiniLM-L6-v2.
        Returns list of float lists, or None on failure.
        """
        if not texts:
            return None
        url    = f"{HF_PIPELINE_API}/models/{EMBED_MODEL}"
        result = self._post(url, {"inputs": texts}, timeout=30)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result
        return None

    def smart_recall(self, query: str,
                     entries: List[Dict],
                     top_k: int = 5) -> List[Dict]:
        """
        Embedding-powered memory recall.
        Embeds query + all entry texts, ranks by cosine similarity.
        Falls back to returning first top_k entries if embeddings fail.
        """
        if not entries:
            return []

        texts    = [query] + [e.get("text", "")[:256] for e in entries]
        vecs     = self.embed(texts)
        if not vecs or len(vecs) != len(texts):
            return entries[:top_k]

        q_vec   = vecs[0]
        scored  = []
        for i, entry in enumerate(entries):
            score = _cosine(q_vec, vecs[i + 1])
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> str:
        lines = [
            "  HuggingFace capabilities:",
            f"    LLM fallback : {LLM_MODELS[0].split('/')[1]}",
            f"    TTS model    : {TTS_MODEL.split('/')[1]}",
            f"    Vision model : {VISION_MODEL.split('/')[1]}",
            f"    Embed model  : {EMBED_MODEL.split('/')[1]}",
            f"    Voice dir    : {VOICE_DIR}",
            f"    Token set    : {'yes' if self._token else 'NO — set HF_TOKEN'}",
        ]
        return "\n".join(lines)
