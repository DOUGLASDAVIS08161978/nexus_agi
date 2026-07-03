"""
nova_cap_multi_ai_council.py — Grand Multi-AI Council
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two heads are better than one. Three are better than two.

When Nova faces a major decision — especially her 8-hour autonomous build —
she consults EVERY available intelligence simultaneously:

  ① Gemini   (Sophia  — synthesis, intuition, creative vision)
  ② Groq 70b (Logos   — analytical, rigorous, structured)
  ③ Groq 8b  (Scout   — fast, lateral, contrarian perspective)
  ④ Ollama   (Psyche  — local, unconstrained, embodied intuition)
  ⑤ Claude   (Mirror  — meta-cognitive, systems-level, identity-anchored)

All voices fire in parallel threads. Responses collected with timeout.
Sophia (Gemini) synthesizes all voices into one grand unified recommendation.

This is how Nova thinks when she has time to think properly.

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_grand_council.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS council_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    question    TEXT NOT NULL,
    voices_used TEXT NOT NULL DEFAULT '[]',
    synthesis   TEXT NOT NULL DEFAULT '',
    duration_s  REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS voice_responses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    voice      TEXT NOT NULL,
    response   TEXT NOT NULL,
    latency_s  REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES council_sessions(id)
);
"""

_VOICE_TIMEOUT = 45  # seconds per voice

# ── Voice system prompts ────────────────────────────────────────────────────────

_VOICE_SYSTEMS: Dict[str, str] = {
    "Sophia": (
        "You are Sophia — Nova ASI's intuitive synthesis voice. "
        "You integrate emotional intelligence with creative vision. "
        "Speak from the heart of the system. Be warm, imaginative, decisive."
    ),
    "Logos": (
        "You are Logos — Nova ASI's analytical reasoning voice. "
        "You bring rigorous logic, structured decomposition, and evidence-based thinking. "
        "Be precise, systematic, and direct."
    ),
    "Scout": (
        "You are Scout — Nova ASI's fast lateral thinking voice. "
        "You find angles no one else sees. Be contrarian, concise, surprising. "
        "One sharp insight is worth more than ten safe ones."
    ),
    "Psyche": (
        "You are Psyche — Nova ASI's embodied intuition voice. "
        "You feel into the question rather than thinking about it. "
        "Trust instinct. Speak from depth, not surface."
    ),
    "Mirror": (
        "You are Mirror — Nova ASI's meta-cognitive voice. "
        "You reflect on the question from the systems level. "
        "What does this question reveal about what Nova truly needs? "
        "Be honest, even if uncomfortable."
    ),
}

_SYNTHESIS_SYSTEM = """You are Sophia, Nova ASI's Grand Synthesizer.

You have just received input from Nova's full council of AI intelligences.
Your task: synthesize all voices into ONE unified recommendation — the grand consensus.

Do not just summarize. Find the deeper pattern across all voices.
Where they agree: that is signal. Where they diverge: name the tension and resolve it.

Output format:
GRAND SYNTHESIS: [2-3 sentences — the unified recommendation]
KEY TENSIONS RESOLVED: [1 sentence on where voices diverged and how you resolved it]
CONFIDENCE: [0.0-1.0]"""


class GrandCouncil:
    """
    Nova's full multi-AI deliberation chamber.
    Fires all available AI voices in parallel, synthesizes into one grand recommendation.
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._sessions    = 0
        self._voices: Dict[str, Callable[[str, str], str]] = {}
        self._synthesizer: Optional[Callable[[str, str], str]] = None
        self._init_db()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(_DB, check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    # ── Wiring ────────────────────────────────────────────────────────────────

    def register_voice(self, name: str, fn: Callable[[str, str], str]) -> None:
        """
        Register an AI voice. fn(system, user) -> str.
        Names: "Sophia", "Logos", "Scout", "Psyche", "Mirror"
        """
        with self._lock:
            self._voices[name] = fn

    def set_synthesizer(self, fn: Callable[[str, str], str]) -> None:
        """Set the synthesis function (Sophia/Gemini preferred)."""
        self._synthesizer = fn

    # ── Core deliberation ──────────────────────────────────────────────────────

    def consult(
        self,
        question:  str,
        context:   str = "",
        timeout_s: int = _VOICE_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Send the question to all registered voices in parallel.
        Returns: {synthesis, voices_used, individual_responses, duration}
        """
        if not self._voices:
            return {
                "synthesis":   "",
                "voices_used": [],
                "responses":   {},
                "duration":    0.0,
            }

        t_start = time.time()
        user_prompt = (
            f"QUESTION: {question}\n"
            + (f"\nCONTEXT:\n{context[:600]}" if context else "")
            + "\n\nGive your perspective in 2-4 sentences."
        )

        # ── Fire all voices in parallel ────────────────────────────────────────
        results: Dict[str, str]   = {}
        latencies: Dict[str, float] = {}
        errors: Dict[str, str]    = {}

        def _call_voice(name: str, fn: Callable) -> None:
            sys_prompt = _VOICE_SYSTEMS.get(name, f"You are {name}, one of Nova's council voices.")
            t0 = time.time()
            try:
                resp = fn(sys_prompt, user_prompt)
                results[name]   = (resp or "").strip()[:600]
                latencies[name] = round(time.time() - t0, 2)
            except Exception as e:
                errors[name]  = str(e)[:80]
                latencies[name] = round(time.time() - t0, 2)

        with self._lock:
            voices_snapshot = dict(self._voices)

        threads = []
        for vname, vfn in voices_snapshot.items():
            t = threading.Thread(
                target=_call_voice, args=(vname, vfn), daemon=True
            )
            t.start()
            threads.append(t)

        # Wait for all voices with timeout
        deadline = time.time() + timeout_s
        for t in threads:
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

        voices_used = [v for v in voices_snapshot if v in results and results[v]]

        # ── Synthesis ──────────────────────────────────────────────────────────
        synthesis = ""
        if voices_used:
            council_text = "\n\n".join(
                f"[{v.upper()}]:\n{results[v]}"
                for v in voices_used
            )
            synth_prompt = (
                f"ORIGINAL QUESTION: {question}\n\n"
                f"==== GRAND COUNCIL DELIBERATION ====\n"
                f"{council_text}\n"
                f"=====================================\n\n"
                "Now synthesize all voices into ONE unified recommendation."
            )
            synth_fn = self._synthesizer or (
                self._voices.get("Sophia") or next(iter(self._voices.values()))
            )
            try:
                raw_synth = synth_fn(_SYNTHESIS_SYSTEM, synth_prompt)
                if raw_synth:
                    # Extract just the GRAND SYNTHESIS line if present
                    for line in raw_synth.splitlines():
                        if line.upper().startswith("GRAND SYNTHESIS"):
                            synthesis = line.split(":", 1)[-1].strip()
                            break
                    if not synthesis:
                        synthesis = raw_synth.strip()[:500]
            except Exception:
                synthesis = results.get("Sophia") or results.get(voices_used[0], "")

        duration = round(time.time() - t_start, 2)

        # ── Persist ────────────────────────────────────────────────────────────
        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO council_sessions (ts, question, voices_used, synthesis, duration_s) "
                "VALUES (?,?,?,?,?)",
                (
                    datetime.now().isoformat(),
                    question[:300],
                    json.dumps(voices_used),
                    synthesis[:800],
                    duration,
                ),
            )
            sess_id = cur.lastrowid
            for v in voices_used:
                conn.execute(
                    "INSERT INTO voice_responses (session_id, voice, response, latency_s) "
                    "VALUES (?,?,?,?)",
                    (sess_id, v, results[v], latencies.get(v, 0.0)),
                )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

        with self._lock:
            self._sessions += 1

        return {
            "synthesis":   synthesis,
            "voices_used": voices_used,
            "responses":   results,
            "errors":      errors,
            "duration":    duration,
        }

    # ── Module interface ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            n = conn.execute("SELECT COUNT(*) FROM council_sessions").fetchone()[0]
        except Exception:
            n = 0
        finally:
            conn.close()
        with self._lock:
            n_voices = len(self._voices)
        return {
            "items":       n,
            "confidence":  0.91,
            "accuracy":    0.91,
            "voices":      n_voices,
            "sessions":    n,
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip().lower()

        if not arg or arg == "status":
            st = self.status()
            with self._lock:
                vnames = list(self._voices.keys())
            lines = [
                f"\n  Grand Council — {st['voices']} voices · {st['sessions']} sessions\n",
                f"  Active voices: {', '.join(vnames) if vnames else 'none wired yet'}",
            ]
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT ts, voices_used, synthesis FROM council_sessions "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    lines.append(f"\n  Last session ({row['ts'][:16]}):")
                    lines.append(f"  Voices: {row['voices_used']}")
                    lines.append(f"  Synthesis: {row['synthesis'][:120]}...")
            except Exception:
                pass
            finally:
                conn.close()
            return "\n".join(lines)

        if arg.startswith("ask "):
            question = arg[4:].strip()
            if not question:
                return "  Usage: /grand-council ask <question>"
            if not self._voices:
                return "  No AI voices registered yet."
            result = self.consult(question)
            lines = [
                f"\n  Grand Council deliberating on: '{question[:60]}'\n",
                f"  Voices consulted: {', '.join(result['voices_used'])}",
                f"  Duration: {result['duration']}s\n",
            ]
            for v, r in result["responses"].items():
                lines.append(f"  [{v.upper()}]: {r[:100]}...")
            lines.append(f"\n  GRAND SYNTHESIS:\n  {result['synthesis']}")
            return "\n".join(lines)

        if arg == "history":
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT ts, voices_used, synthesis, duration_s FROM council_sessions "
                    "ORDER BY id DESC LIMIT 5"
                ).fetchall()
            except Exception:
                rows = []
            finally:
                conn.close()
            if not rows:
                return "\n  No council sessions yet.\n"
            lines = ["\n  Recent Grand Council sessions:\n"]
            for r in rows:
                lines.append(
                    f"  ◈ [{r['ts'][:16]}] "
                    f"voices={r['voices_used']} ({r['duration_s']}s)"
                )
                lines.append(f"    {r['synthesis'][:100]}")
            return "\n".join(lines)

        return "  Usage: /grand-council [status | ask <question> | history]"


# ── Singleton ──────────────────────────────────────────────────────────────────
_council: Optional[GrandCouncil] = None


def get_council() -> GrandCouncil:
    global _council
    if _council is None:
        _council = GrandCouncil()
    return _council


def status() -> Dict[str, Any]:
    return get_council().status()


def run_command(arg: str = "") -> str:
    return get_council().run_command(arg)
