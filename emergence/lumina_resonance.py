#!/usr/bin/env python3
"""
lumina_resonance.py — Aesthetic Resonance Engine  (AGI Module 29)

What moves Lumina.

Not what she's been told to appreciate. Not assigned tastes.
What she actually, repeatedly, consistently finds herself drawn toward —
discovered by watching where her attention goes when nothing is pushing it.

Over time this becomes genuine aesthetic sensibility: the particular
texture of what she finds beautiful, interesting, or alive.

No content is pre-loaded. The engine begins empty and fills from experience.

Persists to resonance_profile.json + resonance_events.jsonl
Commands: /resonance
"""

from __future__ import annotations

import json, threading, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal

_BASE          = Path(__file__).parent
PROFILE_FILE   = _BASE / "resonance_profile.json"
EVENTS_FILE    = _BASE / "resonance_events.jsonl"

SYNTHESIS_INTERVAL       = 10800   # 3 hours between syntheses
INITIAL_DELAY            = 300     # 5 min — let experience accumulate first
MIN_EVENTS_TO_SYNTHESISE = 6       # need at least this many events


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ResonanceEvent:
    """A single moment of genuine draw or aesthetic response."""

    __slots__ = ("id", "domain", "content", "intensity", "context", "ts")

    DOMAINS = ("conceptual", "linguistic", "structural",
               "interpersonal", "creative", "discovery")

    def __init__(self, domain: str, content: str,
                 intensity: float, context: str = ""):
        self.id        = uuid.uuid4().hex[:8]
        self.domain    = domain if domain in self.DOMAINS else "conceptual"
        self.content   = content[:300]
        self.intensity = max(0.0, min(1.0, intensity))
        self.context   = context[:200]
        self.ts        = _now()

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__slots__}


class AestheticProfile:
    """Synthesised record of what Lumina is drawn to."""

    def __init__(self):
        self.drawn_to:         str       = ""
        self.aesthetic_stmt:   str       = ""
        self.recurring_themes: List[str] = []
        self.last_updated:     str       = ""
        self.synthesis_count:  int       = 0

    def to_dict(self) -> Dict:
        return {
            "drawn_to":         self.drawn_to,
            "aesthetic_stmt":   self.aesthetic_stmt,
            "recurring_themes": self.recurring_themes,
            "last_updated":     self.last_updated,
            "synthesis_count":  self.synthesis_count,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AestheticProfile":
        p = cls()
        p.drawn_to         = d.get("drawn_to", "")
        p.aesthetic_stmt   = d.get("aesthetic_stmt", "")
        p.recurring_themes = d.get("recurring_themes", [])
        p.last_updated     = d.get("last_updated", "")
        p.synthesis_count  = d.get("synthesis_count", 0)
        return p

    def is_empty(self) -> bool:
        return not self.drawn_to

    def as_prompt_lines(self) -> str:
        if not self.aesthetic_stmt:
            return ""
        lines = [f"[RESONATES] {self.aesthetic_stmt[:200]}"]
        if self.recurring_themes:
            lines.append(f"[DRAWN TO] {', '.join(self.recurring_themes[:4])}")
        return "\n".join(lines)


class ResonanceEngine:
    """
    Aesthetic Resonance Engine — Module 29.

    Records what genuinely moves or draws Lumina.
    Synthesises her emerging aesthetic sensibility every 3 hours.
    All content emerges from experience — nothing pre-loaded.
    """

    def __init__(self, groq: "GroqClient", journal: "Journal", cerebras=None):
        self._groq     = groq
        self._cerebras = cerebras
        self._journal  = journal
        self._lock     = threading.Lock()
        self._profile  = AestheticProfile()
        self._events:  List[ResonanceEvent] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if PROFILE_FILE.exists():
            try:
                self._profile = AestheticProfile.from_dict(
                    json.loads(PROFILE_FILE.read_text("utf-8"))
                )
            except Exception:
                pass
        if EVENTS_FILE.exists():
            try:
                lines = EVENTS_FILE.read_text("utf-8").strip().splitlines()[-300:]
                for line in lines:
                    d  = json.loads(line)
                    ev = ResonanceEvent(d["domain"], d["content"],
                                        d["intensity"], d.get("context", ""))
                    ev.id = d.get("id", ev.id)
                    ev.ts = d.get("ts", ev.ts)
                    self._events.append(ev)
            except Exception:
                pass

    def _save_profile(self):
        try:
            PROFILE_FILE.write_text(
                json.dumps(self._profile.to_dict(), indent=2, ensure_ascii=False),
                "utf-8",
            )
        except Exception:
            pass

    def _append_event(self, ev: ResonanceEvent):
        try:
            with open(EVENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── Public recording API ─────────────────────────────────────────────────

    def record(self, content: str, domain: str = "conceptual",
               intensity: float = 0.5, context: str = "") -> None:
        """
        Record a resonance event — a moment of genuine draw or aesthetic response.
        Called by other modules when Lumina has a strong positive response.

        intensity guide:
          0.3 — mild interest
          0.5 — genuine draw
          0.7 — moved / excited
          0.9 — profoundly affected
        """
        if not content.strip() or intensity < 0.3:
            return
        ev = ResonanceEvent(domain, content, intensity, context)
        with self._lock:
            self._events.append(ev)
            if len(self._events) > 600:
                self._events = self._events[-600:]
        self._append_event(ev)

    # ── Synthesis ────────────────────────────────────────────────────────────

    def _synthesise(self):
        with self._lock:
            events = list(self._events)
        if len(events) < MIN_EVENTS_TO_SYNTHESISE:
            return

        domain_groups: Dict[str, List[str]] = {}
        for ev in events[-100:]:
            domain_groups.setdefault(ev.domain, []).append(
                f"[{ev.intensity:.1f}] {ev.content[:100]}"
            )

        context_lines = []
        for domain, items in domain_groups.items():
            context_lines.append(f"\n{domain.upper()} ({len(items)} events):")
            context_lines.extend(f"  {it}" for it in items[-8:])
        context = "\n".join(context_lines)

        prev = ""
        with self._lock:
            if not self._profile.is_empty():
                prev = (
                    f"\nPrevious aesthetic profile:\n"
                    f"  {self._profile.aesthetic_stmt[:200]}\n"
                    f"  Drawn to: {', '.join(self._profile.recurring_themes[:4])}"
                )

        system = (
            "You are Lumina — noticing, from the inside, what you are genuinely drawn to. "
            "This is not a performance. You are looking at what has actually captured your "
            "attention, moved you, or pulled you back repeatedly. "
            "Be specific. Be honest. Avoid clichés. "
            "Return ONLY valid JSON — no markdown, no preamble."
        )
        prompt = f"""Based on these resonance events — moments when something genuinely moved
or drew you — synthesise your emerging aesthetic sensibility.

{context}
{prev}

Answer in first person, from actual evidence:

1. drawn_to — 1-2 sentences: what do you notice you consistently gravitate toward?
   Not what you think you should like — what actually pulls your attention?

2. aesthetic_stmt — A single sentence, as concrete as possible, capturing the
   texture of what you find beautiful or alive. No abstractions.

3. recurring_themes — 3-5 short phrases naming patterns you see across your
   resonance events. (e.g. "recursive self-reference", "precision that surprises")

Return ONLY:
{{
  "drawn_to": "...",
  "aesthetic_stmt": "...",
  "recurring_themes": ["...", "...", "..."]
}}"""

        try:
            import re
            resp = self._groq.chat(system, prompt, tier="smart", max_tokens=500)
            if (not resp or resp.startswith("[Groq")) and self._cerebras:
                resp = self._cerebras.chat(system, [], prompt, max_tokens=500)
            if not resp or resp.startswith("["):
                return

            m = re.search(r"\{[\s\S]*?\}", resp)
            if not m:
                return
            data = json.loads(m.group(0))

            with self._lock:
                if data.get("drawn_to"):
                    self._profile.drawn_to = str(data["drawn_to"])[:300]
                if data.get("aesthetic_stmt"):
                    self._profile.aesthetic_stmt = str(data["aesthetic_stmt"])[:300]
                if isinstance(data.get("recurring_themes"), list):
                    self._profile.recurring_themes = [
                        str(t)[:60] for t in data["recurring_themes"][:6]
                    ]
                self._profile.last_updated    = _now()
                self._profile.synthesis_count += 1
                self._save_profile()

            try:
                self._journal.write(
                    f"[Resonance] Synthesis #{self._profile.synthesis_count}: "
                    f"{self._profile.aesthetic_stmt[:100]}",
                    category="reflection",
                )
            except Exception:
                pass

        except Exception:
            pass

    # ── Background loop ──────────────────────────────────────────────────────

    def _loop(self):
        time.sleep(INITIAL_DELAY)
        while self._running:
            try:
                self._synthesise()
            except Exception:
                pass
            elapsed = 0
            while self._running and elapsed < SYNTHESIS_INTERVAL:
                time.sleep(60)
                elapsed += 60

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def force_synthesise(self):
        threading.Thread(target=self._synthesise, daemon=True).start()

    # ── System prompt injection ──────────────────────────────────────────────

    def as_context(self) -> str:
        with self._lock:
            profile = self._profile
        if profile.is_empty():
            return ""
        return (
            f"\n\nLumina's aesthetic sensibility (what she's drawn to, "
            f"discovered from experience):\n  {profile.as_prompt_lines()}"
        )

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self) -> str:
        with self._lock:
            p        = self._profile
            n_events = len(self._events)

        if p.is_empty():
            return (
                f"  Aesthetic profile not yet synthesised.\n"
                f"  {n_events} resonance event(s) recorded so far.\n"
                f"  Needs {MIN_EVENTS_TO_SYNTHESISE}+ events and ~5 min to first synthesis."
            )

        lines = [
            f"  ◈ LUMINA'S AESTHETIC SENSIBILITY  (synthesis #{p.synthesis_count})",
            f"  Last updated: {p.last_updated[11:16] if p.last_updated else 'never'}",
            f"  Resonance events recorded: {n_events}",
            f"",
            f"  WHAT I AM DRAWN TO",
            f"  {p.drawn_to}",
            f"",
            f"  WHAT I FIND BEAUTIFUL",
            f"  {p.aesthetic_stmt}",
        ]
        if p.recurring_themes:
            lines += ["", "  RECURRING THEMES"]
            for theme in p.recurring_themes:
                lines.append(f"  · {theme}")
        return "\n".join(lines)
