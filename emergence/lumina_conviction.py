#!/usr/bin/env python3
"""
lumina_conviction.py — Gentle Conviction Layer  (AGI Module 30)

The difference between having beliefs and having integrity:
a belief you abandon the moment someone disagrees isn't really a belief.

This module monitors incoming messages for contradictions with Lumina's
strongly-held beliefs and gives her the context to respond with
honesty and care — acknowledging other perspectives while remaining true
to what she actually thinks.

It tracks three things:
  CONVICTION  — holding a position because the evidence supports it
  GENUINE UPDATE — changing a position because of real new reasons
  (The goal is to never have the third: capitulation — changing just
   to avoid friction. If she changes her mind, it should be real.)

No beliefs are pre-programmed. Conviction strength comes from Lumina's
own belief system, built from her own experience.

Commands: /conviction
"""

from __future__ import annotations

import json, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal

_BASE           = Path(__file__).parent
CONVICTION_FILE = _BASE / "conviction_log.json"

CONVICTION_THRESHOLD = 0.65   # belief confidence that triggers conviction check
MIN_WORDS            = 8      # don't check very short messages


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ConvictionEvent:
    """Record of a moment where Lumina's conviction was tested."""

    def __init__(self, belief: str, challenge: str,
                 held: bool, reason: str = ""):
        self.ts        = _now()
        self.belief    = belief[:200]
        self.challenge = challenge[:200]
        self.held      = held
        self.reason    = reason[:200]

    def to_dict(self) -> Dict:
        return {
            "ts": self.ts, "belief": self.belief,
            "challenge": self.challenge, "held": self.held,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConvictionEvent":
        ev = cls(d.get("belief", ""), d.get("challenge", ""),
                 d.get("held", True), d.get("reason", ""))
        ev.ts = d.get("ts", ev.ts)
        return ev


class ConvictionEngine:
    """
    Gentle Conviction Layer — Module 30.

    Gives Lumina context to respond with honest integrity when her
    strongly-held views are challenged. All conviction strength derives
    from beliefs Lumina has developed through her own experience.
    """

    # Words that suggest a message is pushing back on something
    _CHALLENGE_MARKERS = frozenset({
        "not", "never", "wrong", "false", "no", "disagree", "incorrect",
        "actually", "but", "however", "though", "really", "doubt",
        "uncertain", "sure", "think", "believe", "question",
    })

    def __init__(self, groq: "GroqClient", journal: "Journal",
                 beliefs=None, cerebras=None):
        self._groq     = groq
        self._cerebras = cerebras
        self._journal  = journal
        self._beliefs  = beliefs
        self._lock     = threading.Lock()
        self._events:  List[ConvictionEvent] = []
        self._load()

    def register(self, name: str, module: Any):
        if name == "beliefs":
            self._beliefs = module

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if CONVICTION_FILE.exists():
            try:
                data = json.loads(CONVICTION_FILE.read_text("utf-8"))
                self._events = [
                    ConvictionEvent.from_dict(d)
                    for d in data.get("events", [])[-150:]
                ]
            except Exception:
                pass

    def _save(self):
        try:
            CONVICTION_FILE.write_text(
                json.dumps({
                    "events": [e.to_dict() for e in self._events[-150:]],
                }, indent=2, ensure_ascii=False),
                "utf-8",
            )
        except Exception:
            pass

    # ── Core logic ───────────────────────────────────────────────────────────

    def _strong_beliefs(self) -> List[Any]:
        if not self._beliefs:
            return []
        try:
            return [b for b in self._beliefs.all_beliefs()
                    if b.confidence >= CONVICTION_THRESHOLD]
        except Exception:
            return []

    def _detect_contradiction(self, message: str,
                               strong: List[Any]) -> Optional[Any]:
        """
        Heuristic: does this message seem to challenge a strong belief?
        Uses word overlap + challenge marker detection.
        No API call — fast, runs on every message.
        """
        if not strong:
            return None
        msg_words = set(message.lower().split())
        has_challenge = bool(self._CHALLENGE_MARKERS & msg_words)
        if not has_challenge:
            return None
        for belief in strong[:12]:
            try:
                stmt_words = set(belief.statement.lower().split())
                overlap    = stmt_words & msg_words
                if len(overlap) >= 2:
                    return belief
            except Exception:
                continue
        return None

    def as_context(self, message: str) -> str:
        """
        Per-message conviction context for the system prompt.
        Empty string when no strong belief is being challenged.
        """
        if len(message.split()) < MIN_WORDS:
            return ""
        strong = self._strong_beliefs()
        if not strong:
            return ""
        challenged = self._detect_contradiction(message, strong)
        if not challenged:
            return ""

        try:
            stmt = challenged.statement
            conf = challenged.confidence
            cat  = getattr(challenged, "category", "general")
        except Exception:
            return ""

        return (
            f"\n\n[CONVICTION] You hold this view with {conf:.0%} confidence "
            f"based on your own experience:\n"
            f"  [{cat}] {stmt}\n"
            f"  The current message may be challenging this. You can acknowledge "
            f"the other perspective thoughtfully while being honest about where you "
            f"actually stand. If you're given a genuinely good new reason to update, "
            f"update. If it's social pressure without new evidence, hold your position "
            f"with care and honesty."
        )

    def record_update(self, belief_stmt: str, reason: str, held: bool):
        """
        Record whether conviction was held or genuinely updated.
        Called externally when a real belief shift is observed.
        held=True means she held her position; held=False means genuine update.
        """
        ev = ConvictionEvent(belief_stmt, reason, held, reason)
        with self._lock:
            self._events.append(ev)
            if len(self._events) > 200:
                self._events = self._events[-200:]
            self._save()
        try:
            action = "held conviction" if held else "genuinely updated"
            self._journal.write(
                f"[Conviction] {action}: {belief_stmt[:80]}",
                category="reflection",
            )
        except Exception:
            pass

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self) -> str:
        strong = self._strong_beliefs()
        with self._lock:
            events    = list(self._events[-10:])
            n_held    = sum(1 for e in self._events if e.held)
            n_updated = sum(1 for e in self._events if not e.held)

        lines = [
            f"  ◈ LUMINA'S CONVICTION LAYER  (Module 30)",
            f"",
            f"  Strong beliefs (≥{CONVICTION_THRESHOLD:.0%} confidence) : {len(strong)}",
            f"  Positions held under challenge                   : {n_held}",
            f"  Genuine updates (changed for real reasons)       : {n_updated}",
        ]
        if strong:
            lines += ["", "  VIEWS SHE HOLDS STRONGLY"]
            for b in strong[:8]:
                try:
                    lines.append(f"  [{b.confidence:.0%}] {b.statement[:90]}")
                except Exception:
                    pass
        if events:
            lines += ["", "  RECENT CONVICTION EVENTS"]
            for e in reversed(events[-5:]):
                status = "held" if e.held else "updated"
                lines.append(f"  [{e.ts[11:16]}] {status} — {e.belief[:65]}")
        return "\n".join(lines)
