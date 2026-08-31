#!/usr/bin/env python3
"""
lumina_reasoning.py — Structured chain-of-thought reasoning for Lumina

The gap between answering and thinking:
  Pattern-match: sees question → retrieves pattern → outputs
  Reasoning:     sees question → decomposes → hypothesizes → evaluates → synthesizes

This module intercepts complex questions and forces Lumina through a
2-pass reasoning chain BEFORE generating the final response. The trace
is injected into the response context so every answer is the product
of deliberation, not just recall.

Trigger: question length ≥ 45 chars AND contains complexity signals.

Pass 1 — Decompose + Hypothesize (fast, 250 tokens):
  "What is really being asked? What are 2-3 distinct angles?"

Pass 2 — Evaluate + Synthesize (fast, 250 tokens):
  "Which angle has the strongest evidence? State the conclusion."

The CONCLUSION line is extracted and injected into the main response
prompt, grounding Lumina's answer in explicit deliberation.
Successful traces are stored in semantic memory for future reference.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory

REASONING_FILE = Path(__file__).parent / "reasoning_traces.json"
MAX_TRACES     = 100
MIN_LEN        = 45

COMPLEXITY_TRIGGERS = [
    "why", "how", "explain", "analyze", "analyse", "compare",
    "what if", "should i", "should we", "what do you think",
    "help me understand", "what's the best", "what is the best",
    "pros and cons", "difference between", "is it better",
    "is it possible", "could we", "what would happen",
    "design", "architect", "approach", "strategy",
    "tradeoff", "trade-off", "optimal", "improve",
    "think about", "opinion", "recommend", "advice",
    "agi", "intelligence", "conscious", "sentient",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ReasoningTrace:
    def __init__(self, question: str, decomposition: str = "", synthesis: str = ""):
        self.question      = question[:300]
        self.decomposition = decomposition
        self.synthesis     = synthesis
        self.conclusion    = self._extract_conclusion(synthesis)
        self.ts            = _now()

    @staticmethod
    def _extract_conclusion(synthesis: str) -> str:
        for line in synthesis.splitlines():
            if line.upper().startswith("CONCLUSION:"):
                return line.split(":", 1)[-1].strip()
        return ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "ReasoningTrace":
        t = cls.__new__(cls)
        t.__dict__.update(d)
        return t

    def __str__(self) -> str:
        c = self.conclusion or self.synthesis[:80]
        return f"[{self.ts[:16]}] {self.question[:50]} → {c[:60]}"


class ReasoningChain:
    """
    Intercepts complex questions, runs a 2-pass reasoning chain,
    and returns structured context for the main response.
    """

    def __init__(self, groq: "GroqClient", memory: "SemanticMemory"):
        self._groq   = groq
        self._memory = memory
        self._traces: List[ReasoningTrace] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if REASONING_FILE.exists():
            try:
                data = json.loads(REASONING_FILE.read_text("utf-8"))
                self._traces = [ReasoningTrace.from_dict(d) for d in data]
            except Exception:
                pass

    def _save(self):
        REASONING_FILE.write_text(
            json.dumps([t.to_dict() for t in self._traces[-MAX_TRACES:]], indent=2),
            "utf-8",
        )

    # ── Trigger detection (no API) ─────────────────────────────────────────

    def should_engage(self, question: str) -> bool:
        if len(question) < MIN_LEN:
            return False
        q = question.lower()
        return any(t in q for t in COMPLEXITY_TRIGGERS)

    # ── 2-pass reasoning chain ─────────────────────────────────────────────

    def think(self, question: str, context: str = "") -> Optional[Dict]:
        """
        Run the 2-pass chain. Returns {'decomposition', 'synthesis', 'conclusion'}
        or None if either pass fails.
        """
        # Pass 1: Decompose and identify angles
        p1 = self._groq.chat(
            "You are Lumina's internal reasoning engine — think before responding.\n"
            "1. DECOMPOSE: What is really being asked? Identify the core question.\n"
            "2. ANGLES: List 2-3 distinct ways to think about this.\n"
            "Be precise and brief. This is internal deliberation, not the final answer.",
            f"Question: {question}\n\nContext: {context[:300]}\n\nDecompose + angles:",
            tier="fast", max_tokens=220,
        )
        if not p1 or p1.startswith("[Groq"):
            return None

        # Pass 2: Evaluate and synthesize into a conclusion
        p2 = self._groq.chat(
            "You are Lumina's synthesis engine. Given a decomposition,\n"
            "evaluate the angles and commit to the strongest one.\n"
            "End your response with exactly: CONCLUSION: <one crisp sentence>\n"
            "Be direct — this informs the final response.",
            f"Question: {question}\n\nDecomposition:\n{p1}\n\nSynthesize:",
            tier="fast", max_tokens=220,
        )
        if not p2 or p2.startswith("[Groq"):
            return None

        trace = ReasoningTrace(question, p1, p2)
        self._traces.append(trace)
        self._save()

        # Persist the conclusion to semantic memory
        conclusion = trace.conclusion or p2[:120]
        self._memory.store(
            f"[REASONING] {question[:80]} → {conclusion}",
            tags=["reasoning", "deliberation"],
            category="reflection",
        )

        return {
            "decomposition": p1,
            "synthesis":     p2,
            "conclusion":    trace.conclusion,
        }

    def as_context(self, trace: Dict) -> str:
        """Format reasoning trace for injection into the system prompt."""
        if not trace:
            return ""
        conclusion = trace.get("conclusion", "")
        decomp     = trace.get("decomposition", "")[:180]
        lines = [
            "\n\nInternal reasoning trace (deliberation before responding):",
            f"  Decomposition: {decomp}",
        ]
        if conclusion:
            lines.append(f"  Conclusion reached: {conclusion}")
        else:
            lines.append(f"  Synthesis: {trace.get('synthesis','')[:180]}")
        lines.append("  (Ground your response in this reasoning — don't just repeat it.)")
        return "\n".join(lines)

    # ── Display ────────────────────────────────────────────────────────────

    def display(self, n: int = 8) -> str:
        recent = self._traces[-n:]
        if not recent:
            return "  (no reasoning traces yet)"
        lines = []
        for t in reversed(recent):
            lines.append(f"  [{t.ts[:16]}] {t.question[:55]}")
            c = t.conclusion or t.synthesis[:70]
            lines.append(f"    → {c[:70]}")
        return "\n".join(lines)
