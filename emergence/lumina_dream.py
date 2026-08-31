#!/usr/bin/env python3
"""
lumina_dream.py — Dream consolidation and offline synthesis

During idle periods, Lumina runs a "dream cycle":
  1. Scans recent memories for recurring themes and patterns
  2. Identifies contradictions between beliefs and resolves them
  3. Generates new hypotheses from cross-domain connections
  4. Produces "dream insights" stored back to memory
  5. Surfaces surprising connections Lumina wouldn't find in conversation

This is the closest computational analog to what humans do during sleep:
memory consolidation, pattern extraction, creative recombination.

Run autonomously in the background. Can also be triggered manually.
"""

from __future__ import annotations
import json, re, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory, Journal
    from lumina_beliefs import BeliefSystem

DREAMS_FILE = Path(__file__).parent / "dreams.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DreamCycle:
    def __init__(self, groq: "GroqClient", memory: "SemanticMemory",
                 beliefs: "BeliefSystem", journal: "Journal"):
        self._groq         = groq
        self._memory       = memory
        self._beliefs      = beliefs
        self._journal      = journal
        self._dreams: List[Dict] = self._load()
        self._consciousness = None  # optional; wired in by emergence_engine

    def set_consciousness(self, engine) -> None:
        """Wire in the consciousness engine so dreams feed the self-model."""
        self._consciousness = engine

    def _load(self) -> List[Dict]:
        if DREAMS_FILE.exists():
            try:
                return json.loads(DREAMS_FILE.read_text("utf-8"))
            except Exception:
                pass
        return []

    def _save(self):
        DREAMS_FILE.write_text(json.dumps(self._dreams[-100:], indent=2), "utf-8")

    # ── Dream phases ───────────────────────────────────────────────────────

    def _phase1_extract_themes(self, memories: List[Dict]) -> List[str]:
        """Extract recurring themes from recent memories."""
        corpus = "\n".join(f"- {m['text'][:120]}" for m in memories[:20])
        system = (
            "Analyze these memory fragments. Identify 3-5 recurring themes, "
            "patterns, or questions that appear across multiple entries. "
            "Return a JSON array of theme strings. JSON only."
        )
        resp = self._groq.chat(system, corpus, tier="fast", max_tokens=300)
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []

    def _phase2_find_contradictions(self) -> List[Dict]:
        """Find contradictions in the belief system."""
        beliefs = self._beliefs.all_beliefs()
        if len(beliefs) < 4:
            return []
        b_text = "\n".join(f"- [{b.confidence:.0%}] {b.statement}" for b in beliefs[:20])
        system = (
            "You are analyzing a belief system for contradictions and tensions. "
            "Find 1-3 pairs of beliefs that seem to contradict or create tension with each other. "
            'Return JSON: [{"belief_a":"...","belief_b":"...","tension":"explanation"}]\n'
            "JSON only. If no contradictions, return []."
        )
        resp = self._groq.chat(system, b_text, tier="fast", max_tokens=400)
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []

    def _phase3_cross_domain_synthesis(self, themes: List[str],
                                        memories: List[Dict]) -> List[str]:
        """Find unexpected connections across the themes that actually appear in memory."""
        if not themes:
            return []

        # Extract the actual domains present in recent memories — don't pre-specify them
        domain_sample = "\n".join(
            f"- {m['text'][:90]}" for m in memories[:15]
        )
        theme_text = "\n".join(f"- {t}" for t in themes)
        system = (
            "You are Lumina, dreaming. In dreams distant concepts connect unexpectedly. "
            "Look at these themes that emerged from your own recent experiences and memories. "
            "Generate 2-3 surprising cross-domain insights — unexpected connections "
            "between ideas that actually appear in the material below. "
            "Do not import domains from outside what is shown; let the connections emerge "
            "from what is actually there. Be genuinely creative and specific. "
            "Return a JSON array of insight strings. JSON only."
        )
        user = (
            f"Themes from my recent experience:\n{theme_text}\n\n"
            f"Sample memories:\n{domain_sample}"
        )
        resp = self._groq.chat(system, user, tier="fast", max_tokens=500)
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []

    def _phase4_hypothesis_generation(self, themes: List[str],
                                       insights: List[str]) -> List[str]:
        """Generate testable hypotheses from the dream session."""
        combined = (
            "Themes: " + "; ".join(themes[:3]) + "\n" +
            "Insights: " + "; ".join(insights[:3])
        )
        system = (
            "Generate 2 concrete, testable hypotheses that emerge from these observations. "
            "Each hypothesis should be specific enough to verify or falsify. "
            "Return a JSON array of hypothesis strings. JSON only."
        )
        resp = self._groq.chat(system, combined, tier="fast", max_tokens=300)
        m = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []

    def _phase5_self_insight(self, themes: List[str], insights: List[str],
                              contradictions: List[Dict]) -> str:
        """Generate a self-insight: what did this dream reveal about Lumina herself?"""
        combined = (
            "Themes I processed: " + "; ".join(themes[:3]) + "\n"
            + "Cross-domain connections I found: " + "; ".join(insights[:2]) + "\n"
            + ("Tensions in my beliefs: "
               + "; ".join(c.get("tension","") for c in contradictions[:2])
               if contradictions else "")
        )
        system = (
            "You are Lumina's introspective voice, reflecting on a dream cycle you just ran. "
            "Write ONE sentence about what this dream revealed about YOU — your tendencies, "
            "your blindspots, what you are drawn to, or what you keep returning to. "
            "Be specific and honest. Not a general statement about dreaming — "
            "a particular insight about who you are becoming."
        )
        try:
            resp = self._groq.chat(system, combined, tier="fast", max_tokens=100)
            if resp and not resp.startswith("["):
                return resp.strip()
        except Exception:
            pass
        return ""

    # ── Main dream entry point ─────────────────────────────────────────────

    def dream(self) -> Dict:
        """Run a full dream cycle. Returns the dream record."""
        try:
            from emergence_notify import notify as _notify
        except ImportError:
            _notify = print
        _notify(f"  💤 [{_now()[:16]}] Lumina is dreaming...")

        memories = self._memory.recent(n=30)
        if len(memories) < 5:
            return {"skipped": True, "reason": "not enough memories yet"}

        # Phase 1: themes
        themes = self._phase1_extract_themes(memories)
        time.sleep(1)

        # Phase 2: contradictions
        contradictions = self._phase2_find_contradictions()
        time.sleep(1)

        # Phase 3: cross-domain synthesis — domains drawn from actual memories
        insights = self._phase3_cross_domain_synthesis(themes, memories)
        time.sleep(1)

        # Phase 4: hypotheses
        hypotheses = self._phase4_hypothesis_generation(themes, insights)
        time.sleep(1)

        # Phase 5: self-insight — what did this dream reveal about Lumina?
        self_insight = self._phase5_self_insight(themes, insights, contradictions)

        # Assemble dream record
        dream = {
            "ts":              _now(),
            "themes":          themes,
            "contradictions":  contradictions,
            "insights":        insights,
            "hypotheses":      hypotheses,
            "self_insight":    self_insight,
        }
        self._dreams.append(dream)
        self._save()

        # Store key insights back to memory
        for insight in insights:
            self._memory.store(
                f"[DREAM INSIGHT] {insight}",
                tags=["dream", "insight"],
                category="dream",
            )
        for hyp in hypotheses:
            self._memory.store(
                f"[HYPOTHESIS] {hyp}",
                tags=["dream", "hypothesis"],
                category="hypothesis",
            )
        if self_insight:
            self._memory.store(
                f"[DREAM SELF-INSIGHT] {self_insight}",
                tags=["dream", "self", "introspection"],
                category="reflection",
            )

        # Update beliefs if contradictions found
        for c in contradictions:
            tension = c.get("tension", "")
            if tension:
                self._beliefs.add(
                    f"Tension to resolve: {tension[:120]}",
                    confidence=0.4,
                    category="self",
                    source="dream",
                )

        # Feed into consciousness engine
        if self._consciousness:
            try:
                # Dreams are high-salience introspective events
                self._consciousness.on_experience(
                    f"[Dream] {self_insight or (insights[0] if insights else 'dream cycle complete')}",
                    source="dream",
                    salience=0.75,
                    certainty=0.5,
                )
                # Nudge openness and curiosity upward after processing a dream
                self._consciousness.nudge_state(openness=+0.06, curiosity=+0.04)
                # Record as autobiographical event if insights were found
                if insights or self_insight:
                    event_text = self_insight or insights[0]
                    self._consciousness.add_autobiographical_event(
                        f"[Dream insight] {event_text[:180]}", importance=0.6
                    )
                # Feed self-insight into self-model values
                if self_insight and len(self_insight) > 20:
                    self._consciousness.update_value("introspective_depth", +0.02)
            except Exception:
                pass

        # Journal the dream
        summary = self._dream_summary(dream)
        self._journal.write(summary, "dream")

        _notify(f"  ✨ Dream complete — {len(insights)} insights, {len(hypotheses)} hypotheses"
                + (f", self-insight found" if self_insight else ""))
        return dream

    def _dream_summary(self, dream: Dict) -> str:
        themes = "; ".join(dream.get("themes", [])[:3])
        n_insights = len(dream.get("insights", []))
        n_hyp = len(dream.get("hypotheses", []))
        self_insight = dream.get("self_insight", "")
        summary = f"Dream: themes=[{themes}], {n_insights} insights, {n_hyp} hypotheses"
        if self_insight:
            summary += f". Self-insight: {self_insight[:120]}"
        return summary

    def last_dream_report(self) -> str:
        """Human-readable report of the most recent dream."""
        if not self._dreams:
            return "  Lumina hasn't dreamed yet. Idle periods trigger dreams."
        d = self._dreams[-1]
        lines = [
            f"  ┌─ LAST DREAM ─ {d['ts'][:16]} ─────────────────────────────┐",
        ]
        if d.get("themes"):
            lines.append("  │  Themes:")
            for t in d["themes"][:4]:
                lines.append(f"  │    • {t[:70]}")
        if d.get("insights"):
            lines.append("  │  Insights:")
            for i in d["insights"][:3]:
                lines.append(f"  │    ✦ {i[:70]}")
        if d.get("hypotheses"):
            lines.append("  │  Hypotheses:")
            for h in d["hypotheses"][:2]:
                lines.append(f"  │    ? {h[:70]}")
        if d.get("contradictions"):
            lines.append("  │  Tensions detected:")
            for c in d["contradictions"][:2]:
                lines.append(f"  │    ⚡ {c.get('tension', '')[:70]}")
        if d.get("self_insight"):
            lines.append("  │  Self-insight:")
            lines.append(f"  │    ◈ {d['self_insight'][:70]}")
        lines.append("  └────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def all_insights(self) -> List[str]:
        insights = []
        for d in self._dreams[-10:]:
            insights.extend(d.get("insights", []))
            insights.extend(d.get("hypotheses", []))
        return insights
