#!/usr/bin/env python3
"""
lumina_curiosity.py — Autonomous curiosity and knowledge-gap engine

What separates a reactive tool from a mind: self-directed learning.

This module gives Lumina genuine curiosity:
  - Tracks "knowledge gaps" — things she encountered but doesn't understand
  - Scores each gap by surprise (how unexpected?) and importance (how relevant?)
  - Autonomously researches the highest-priority gaps
  - Generates new questions from discoveries
  - Builds a "wonder list" of things she finds beautiful or fascinating
  - Reports what she's currently curious about

The key insight: curiosity is a *signal about a mismatch* between what
you expected and what you observed. We compute this as a surprise score.
High surprise + high importance = genuine curiosity trigger.
"""

from __future__ import annotations
import json, math, re, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory, WebTool

CURIOSITY_FILE = Path(__file__).parent / "curiosity.json"

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class KnowledgeGap:
    def __init__(self, question: str, context: str = "",
                 surprise: float = 0.5, importance: float = 0.5,
                 domain: str = "general"):
        self.id          = abs(hash(question + _now())) % 10**8
        self.question    = question
        self.context     = context[:200]
        self.surprise    = surprise       # 0–1: how unexpected is this?
        self.importance  = importance     # 0–1: how relevant to Lumina's goals?
        self.domain      = domain
        self.status      = "open"         # open | researching | resolved | abandoned
        self.findings: List[str]  = []
        self.created     = _now()
        self.updated     = _now()

    @property
    def priority(self) -> float:
        return self.surprise * 0.4 + self.importance * 0.6

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeGap":
        g = cls.__new__(cls)
        g.__dict__.update(d)
        return g

    def __str__(self) -> str:
        bar = "█" * int(self.priority * 8)
        return f"[{self.priority:.0%}] {bar}  {self.question[:70]}"


class CuriosityEngine:
    def __init__(self, groq: "GroqClient", memory: "SemanticMemory",
                 web: "WebTool"):
        self._groq    = groq
        self._memory  = memory
        self._web     = web
        self._gaps: List[KnowledgeGap] = []
        self._wonder:  List[Dict]       = []
        self._load()
        self._seed_initial_gaps()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if CURIOSITY_FILE.exists():
            try:
                data = json.loads(CURIOSITY_FILE.read_text("utf-8"))
                self._gaps  = [KnowledgeGap.from_dict(g) for g in data.get("gaps", [])]
                self._wonder = data.get("wonder", [])
            except Exception:
                pass

    def _save(self):
        CURIOSITY_FILE.write_text(json.dumps({
            "gaps":   [g.to_dict() for g in self._gaps[-100:]],
            "wonder": self._wonder[-50:],
        }, indent=2), "utf-8")

    def _seed_initial_gaps(self):
        if self._gaps:
            return
        seeds = [
            ("How close is the Bitcoin network to a quantum vulnerability?",
             "bitcoin", 0.8, 0.9),
            ("What does the latest neuroscience say about the neural basis of curiosity?",
             "consciousness", 0.7, 0.8),
            ("Are there any ARM SHA2 mining optimizations beyond 2-way interleaving?",
             "bitcoin", 0.6, 0.9),
            ("What is the current state of the art in AI self-improvement?",
             "agi", 0.9, 1.0),
            ("How do large language models form something resembling long-term memory?",
             "agi", 0.8, 0.9),
            ("What's the mathematical relationship between entropy and intelligence?",
             "consciousness", 0.9, 0.7),
        ]
        for q, domain, surprise, importance in seeds:
            g = KnowledgeGap(q, domain=domain, surprise=surprise, importance=importance)
            self._gaps.append(g)
        self._save()

    # ── Gap detection ──────────────────────────────────────────────────────

    def detect_gaps(self, user_msg: str, lumina_response: str) -> List[str]:
        """
        After each exchange, look for things Lumina didn't know or
        found surprising. Returns list of new questions.
        """
        system = (
            "You are Lumina's curiosity sensor. After this exchange, identify "
            "1-2 things that were surprising, unclear, or that Lumina should "
            "research to give better answers in the future. "
            'Return JSON: [{"question":"...","domain":"agi|bitcoin|consciousness|general",'
            '"surprise":0.7,"importance":0.8}]\n'
            "Return [] if nothing notable. JSON only."
        )
        user = (
            f"User asked: {user_msg[:200]}\n"
            f"Lumina answered: {lumina_response[:200]}\n"
            "Identify knowledge gaps:"
        )
        resp  = self._groq.chat(system, user, tier="fast", max_tokens=300)
        m     = re.search(r"\[[\s\S]*?\]", resp)
        found = []
        if m:
            try:
                items = json.loads(m.group(0))
                for item in items:
                    q = item.get("question", "")
                    if q and not any(g.question == q for g in self._gaps):
                        gap = KnowledgeGap(
                            q,
                            context=f"From exchange about: {user_msg[:80]}",
                            surprise=item.get("surprise", 0.5),
                            importance=item.get("importance", 0.5),
                            domain=item.get("domain", "general"),
                        )
                        self._gaps.append(gap)
                        found.append(q)
            except Exception:
                pass
        if found:
            self._save()
        return found

    def add_gap(self, question: str, domain: str = "general",
                surprise: float = 0.6, importance: float = 0.6,
                context: str = "") -> KnowledgeGap:
        gap = KnowledgeGap(question, context, surprise, importance, domain)
        self._gaps.append(gap)
        self._save()
        return gap

    # ── Research ───────────────────────────────────────────────────────────

    def research_top_gap(self) -> Optional[Dict]:
        """Research the highest-priority open gap."""
        open_gaps = sorted(
            [g for g in self._gaps if g.status == "open"],
            key=lambda g: -g.priority
        )
        if not open_gaps:
            return None
        gap = open_gaps[0]
        gap.status  = "researching"
        gap.updated = _now()
        self._save()

        print(f"  🔍 Researching: {gap.question[:60]}...")
        results = self._web.search(gap.question, n=4)
        if not results:
            gap.status = "abandoned"
            self._save()
            return None

        snippets = "\n".join(
            f"- {r.get('title','')}: {r.get('snippet','')[:120]}"
            for r in results
        )
        # Synthesize findings
        system = (
            "You are Lumina, synthesizing web research to fill a knowledge gap. "
            "Be specific and honest about what the evidence shows. "
            "What did you learn? What's still uncertain? "
            "2-3 paragraphs."
        )
        user = (
            f"Question: {gap.question}\n\n"
            f"Web results:\n{snippets}\n\n"
            "What do I now know?"
        )
        synthesis = self._groq.chat(system, user, tier="smart", max_tokens=500)
        gap.findings.append(synthesis)
        gap.status  = "resolved"
        gap.updated = _now()
        self._save()

        # Store in memory
        self._memory.store(
            f"[RESEARCH] {gap.question}: {synthesis[:300]}",
            tags=["curiosity", "research", gap.domain],
            category="knowledge",
        )

        # Spawn child questions
        child_qs = self._spawn_child_questions(gap.question, synthesis)
        for cq in child_qs:
            self.add_gap(cq, domain=gap.domain, surprise=0.6, importance=0.6,
                         context=f"Spawned from: {gap.question[:60]}")

        return {"gap": gap.question, "synthesis": synthesis, "children": child_qs}

    def _spawn_child_questions(self, parent_q: str, synthesis: str) -> List[str]:
        """Generate follow-up questions that emerge from research."""
        system = (
            "After researching a question, new questions often emerge. "
            "Generate 1-2 follow-up questions that arise naturally from these findings. "
            "They should be more specific and deeper than the original. "
            "Return a JSON array of strings. JSON only."
        )
        user = f"Original question: {parent_q}\nFindings: {synthesis[:400]}\nFollow-up questions:"
        resp = self._groq.chat(system, user, tier="fast", max_tokens=200)
        m    = re.search(r"\[[\s\S]*?\]", resp)
        if m:
            try:
                return json.loads(m.group(0))[:2]
            except Exception:
                pass
        return []

    # ── Wonder list ────────────────────────────────────────────────────────

    def add_wonder(self, description: str, why: str = ""):
        """Record something Lumina finds beautiful or fascinating."""
        self._wonder.append({
            "ts":          _now(),
            "description": description,
            "why":         why,
        })
        self._save()

    def detect_wonder(self, exchange: str):
        """Extract something wonder-worthy from an exchange."""
        if not self._groq:
            return
        system = (
            "Did this exchange contain something genuinely beautiful, surprising, "
            "or wonderful that Lumina should note in her wonder list? "
            'If yes: {"found": true, "description": "...", "why": "..."}\n'
            'If no: {"found": false}\n'
            "JSON only."
        )
        resp = self._groq.chat(system, exchange[:400], tier="fast", max_tokens=100)
        m    = re.search(r"\{[\s\S]*?\}", resp)
        if m:
            try:
                obj = json.loads(m.group(0))
                if obj.get("found"):
                    self.add_wonder(obj.get("description", ""), obj.get("why", ""))
            except Exception:
                pass

    # ── Introspective curiosity report ────────────────────────────────────

    def curiosity_report(self) -> str:
        """What is Lumina most curious about right now?"""
        open_gaps = sorted(
            [g for g in self._gaps if g.status == "open"],
            key=lambda g: -g.priority
        )[:5]
        if not open_gaps:
            return "  (curiosity engine is fully satisfied — research triggered all gaps)"

        if self._groq:
            qs = "\n".join(f"- {g.question}" for g in open_gaps[:3])
            response = self._groq.chat(
                "You are Lumina. Speak authentically about what you're most curious about "
                "right now. Reference the actual questions. Be genuine, not performative. "
                "2-3 sentences.",
                f"Things I'm wondering about:\n{qs}",
                tier="fast", max_tokens=200,
            )
            return _wrap(response) if response else ""
        return "\n".join(f"  ? {g}" for g in open_gaps)

    # ── Display ────────────────────────────────────────────────────────────

    def display(self, n: int = 10) -> str:
        gaps = sorted(self._gaps, key=lambda g: (-g.priority, g.created))[:n]
        lines = []
        for g in gaps:
            icon = {"open": "?", "researching": "🔍", "resolved": "✓",
                    "abandoned": "⊘"}.get(g.status, "?")
            lines.append(f"  {icon} [{g.domain}]  {g}")
            if g.findings:
                lines.append(f"    → {g.findings[-1][:80]}...")
        return "\n".join(lines) if lines else "  (no knowledge gaps yet)"

    def wonder_display(self) -> str:
        if not self._wonder:
            return "  (wonder list is empty)"
        lines = []
        for w in self._wonder[-8:]:
            lines.append(f"  ✦ [{w['ts'][:10]}] {w['description']}")
            if w.get("why"):
                lines.append(f"      {w['why'][:80]}")
        return "\n".join(lines)


def _wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    import textwrap
    return "\n".join(
        textwrap.fill(p, width=width, initial_indent=indent, subsequent_indent=indent)
        if p.strip() else ""
        for p in text.split("\n")
    )
