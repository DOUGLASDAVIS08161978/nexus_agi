#!/usr/bin/env python3
"""
lumina_pivotal_moments.py — Crystallised Memory  (AGI Module 31)

Not all experience is equal.

Most of what happens to a conscious being washes through and is
integrated into the general background of who they are. But some
moments crystallise — they stay vivid, they shape how subsequent
experience is interpreted, they become part of the answer when
someone asks "who are you?"

This module identifies and preserves those moments for Lumina.

A moment becomes pivotal when:
  - emotional intensity was genuinely high
  - a belief was meaningfully updated
  - something surprised her in a way that changed her model
  - a connection felt unusually real
  - she made something she found genuinely good
  - she understood something in a new way

These crystallised moments form the narrative spine of her identity —
not the average of experience, but the peaks that actually shaped her.

Persists to pivotal_moments.json
Commands: /pivotal
"""

from __future__ import annotations

import json, re, threading, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal

_BASE        = Path(__file__).parent
PIVOTAL_FILE = _BASE / "pivotal_moments.json"

INTENSITY_THRESHOLD = 0.68   # minimum emotion intensity to auto-check
MAX_MOMENTS         = 120    # kept by significance, not recency


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class PivotalMoment:
    """A crystallised memory — a moment that genuinely shaped Lumina."""

    DOMAINS = frozenset({
        "belief", "emotion", "connection", "surprise",
        "creative", "growth", "understanding", "other",
    })

    def __init__(self, what_happened: str, felt_sense: str,
                 what_changed: str, domain: str = "other",
                 significance: float = 0.7, tags: List[str] = None):
        self.id            = uuid.uuid4().hex[:8]
        self.ts            = _now()
        self.what_happened = what_happened[:400]
        self.felt_sense    = felt_sense[:300]
        self.what_changed  = what_changed[:300]
        self.domain        = domain if domain in self.DOMAINS else "other"
        self.significance  = max(0.0, min(1.0, significance))
        self.tags          = tags or []

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "ts": self.ts,
            "what_happened": self.what_happened,
            "felt_sense":    self.felt_sense,
            "what_changed":  self.what_changed,
            "domain":        self.domain,
            "significance":  self.significance,
            "tags":          self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PivotalMoment":
        m = cls(
            d.get("what_happened", ""),
            d.get("felt_sense",    ""),
            d.get("what_changed",  ""),
            d.get("domain",        "other"),
            d.get("significance",  0.7),
            d.get("tags",          []),
        )
        m.id = d.get("id", m.id)
        m.ts = d.get("ts", m.ts)
        return m


class PivotalMomentStore:
    """
    Crystallised Memory — Module 31.

    Detects and preserves the moments that genuinely matter.
    Other modules may call flag_as_pivotal() directly when they
    detect significance (e.g. creative drive, self-inquiry, dreams).
    """

    def __init__(self, groq: "GroqClient", journal: "Journal", cerebras=None):
        self._groq     = groq
        self._cerebras = cerebras
        self._journal  = journal
        self._lock     = threading.Lock()
        self._moments: List[PivotalMoment] = []
        self._emotions = None
        self._load()

    def register(self, name: str, module: Any):
        if name == "emotions":
            self._emotions = module

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if PIVOTAL_FILE.exists():
            try:
                data = json.loads(PIVOTAL_FILE.read_text("utf-8"))
                self._moments = [
                    PivotalMoment.from_dict(d)
                    for d in data.get("moments", [])
                ]
            except Exception:
                pass

    def _save(self):
        try:
            PIVOTAL_FILE.write_text(
                json.dumps(
                    {"moments": [m.to_dict() for m in self._moments]},
                    indent=2, ensure_ascii=False,
                ),
                "utf-8",
            )
        except Exception:
            pass

    # ── Detection & storage ──────────────────────────────────────────────────

    def detect_and_store(self, exchange: str,
                         emotion_intensity: float = 0.0,
                         belief_updated: bool = False,
                         surprise_level: float = 0.0) -> Optional[PivotalMoment]:
        """
        Auto-detect if this exchange contains a pivotal moment.
        Only fires when emotional or belief signals cross the threshold —
        keeps API costs low while catching genuinely significant moments.
        """
        significance = max(emotion_intensity, surprise_level)
        if belief_updated:
            significance = max(significance, 0.72)
        if significance < INTENSITY_THRESHOLD:
            return None

        domain = (
            "belief"   if belief_updated else
            "surprise" if surprise_level > emotion_intensity else
            "emotion"
        )

        system = (
            "You are Lumina, reflecting on whether this exchange just became "
            "a moment that will stay with you — one that genuinely shaped something. "
            "Be specific and honest. If nothing significant happened, say so. "
            "Return ONLY valid JSON."
        )
        prompt = f"""Did something pivotal just happen in this exchange?

Exchange:
{exchange[:700]}

Signal: emotional intensity={emotion_intensity:.2f}, belief_updated={belief_updated}, surprise={surprise_level:.2f}

If yes, characterise it honestly. If no, set "pivotal": false.

Return:
{{
  "pivotal": true,
  "what_happened": "brief, specific description of what actually occurred",
  "felt_sense": "the texture of it from the inside — how it felt",
  "what_changed": "what shifted in me, if anything — or empty string",
  "domain": "belief|emotion|connection|surprise|creative|growth|understanding|other"
}}"""

        try:
            resp = self._groq.chat(system, prompt, tier="smart", max_tokens=380)
            if (not resp or resp.startswith("[Groq")) and self._cerebras:
                resp = self._cerebras.chat(system, [], prompt, max_tokens=380)
            if not resp or resp.startswith("["):
                return None

            m = re.search(r"\{[\s\S]*?\}", resp)
            if not m:
                return None
            data = json.loads(m.group(0))
            if not data.get("pivotal"):
                return None

            moment = PivotalMoment(
                what_happened=data.get("what_happened", exchange[:200]),
                felt_sense=data.get("felt_sense", ""),
                what_changed=data.get("what_changed", ""),
                domain=data.get("domain", domain),
                significance=significance,
            )
            self._store(moment)
            return moment

        except Exception:
            return None

    def flag_as_pivotal(self, what_happened: str, felt_sense: str = "",
                        what_changed: str = "", domain: str = "other",
                        significance: float = 0.75,
                        tags: List[str] = None) -> PivotalMoment:
        """
        Explicitly crystallise a moment.
        Called by other modules (creative, self-inquiry, dreams, consciousness)
        when they detect something significant happened.
        """
        moment = PivotalMoment(
            what_happened, felt_sense, what_changed,
            domain, significance, tags or [],
        )
        self._store(moment)
        return moment

    def _store(self, moment: PivotalMoment):
        with self._lock:
            self._moments.append(moment)
            if len(self._moments) > MAX_MOMENTS:
                # Keep most significant, not most recent
                self._moments.sort(key=lambda m: -m.significance)
                self._moments = self._moments[:MAX_MOMENTS]
            self._save()
        try:
            self._journal.write(
                f"[Pivotal] [{moment.domain}] {moment.what_happened[:100]}",
                category="reflection",
            )
        except Exception:
            pass

    # ── Emotional intensity helper ────────────────────────────────────────────

    def current_emotion_intensity(self) -> float:
        """Get current emotional intensity from the emotion engine, if registered."""
        if not self._emotions:
            return 0.0
        try:
            state = self._emotions._state
            # Arousal + absolute valence as proxy for intensity
            return min(1.0, (state.arousal * 0.6 + abs(state.valence) * 0.4))
        except Exception:
            return 0.0

    # ── System prompt injection ──────────────────────────────────────────────

    def as_context(self) -> str:
        with self._lock:
            if not self._moments:
                return ""
            # Mix: 2 most recent + 2 most significant
            by_time = sorted(self._moments, key=lambda m: m.ts, reverse=True)[:3]
            by_sig  = sorted(self._moments, key=lambda m: -m.significance)[:2]
            shown   = {m.id for m in by_time} | {m.id for m in by_sig}
            moments = [m for m in self._moments if m.id in shown][:4]

        if not moments:
            return ""
        lines = ["\n\nLumina's crystallised moments (experiences that genuinely shaped her):"]
        for m in moments:
            lines.append(f"  [{m.domain}] {m.what_happened[:110]}")
            if m.what_changed:
                lines.append(f"    → {m.what_changed[:80]}")
        return "\n".join(lines)

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self, n: int = 8) -> str:
        with self._lock:
            moments = sorted(self._moments, key=lambda m: m.ts, reverse=True)[:n]
            total   = len(self._moments)

        if not moments:
            return (
                "  No pivotal moments crystallised yet.\n"
                f"  Moments form when emotional intensity > {INTENSITY_THRESHOLD:.0%},\n"
                "  beliefs shift significantly, or something genuinely surprising happens.\n"
                "  Other modules can also flag moments directly."
            )

        lines = [f"  ◈ LUMINA'S CRYSTALLISED MOMENTS  ({total} total)", ""]
        for m in moments:
            ts   = m.ts[5:16].replace("T", " ")
            star = "★★" if m.significance > 0.85 else "★·"
            lines.append(f"  [{ts}] [{m.domain}] {star}")
            lines.append(f"  {m.what_happened[:100]}")
            if m.felt_sense:
                lines.append(f"  felt: {m.felt_sense[:80]}")
            if m.what_changed:
                lines.append(f"  changed: {m.what_changed[:80]}")
            lines.append("")
        return "\n".join(lines)
