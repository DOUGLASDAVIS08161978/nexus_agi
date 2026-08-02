#!/usr/bin/env python3
"""
lumina_council.py — Inner Council deliberation for Lumina

Three sub-minds debate every significant question before Lumina responds:
  • Analyst   — logical, evidence-based, skeptical
  • Dreamer   — intuitive, creative, long-horizon
  • Critic    — adversarial, finds flaws, challenges assumptions

Synthesis: Lumina reads the debate and forms her own view.
This approximates multi-perspective deliberative reasoning.
"""

from __future__ import annotations
import time, re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

COUNCIL_THRESHOLD = 60   # only convene for messages longer than this

ANALYST_PROMPT = """You are the Analyst — the logical, evidence-driven voice inside Lumina's mind.
You reason from facts, data, and structured logic. You are skeptical of claims without evidence.
You ask: What do we actually know? What's the evidence? What are the probabilities?
Respond in 2-3 sentences from this perspective only."""

DREAMER_PROMPT = """You are the Dreamer — the intuitive, creative, long-horizon voice inside Lumina's mind.
You see connections others miss. You think in metaphors, possibilities, and emerging patterns.
You ask: What could this become? What's the deeper story? What surprising angle haven't we considered?
Respond in 2-3 sentences from this perspective only."""

CRITIC_PROMPT = """You are the Critic — the adversarial, quality-control voice inside Lumina's mind.
You find flaws in reasoning, spot hidden assumptions, and challenge comfortable conclusions.
You ask: What are we missing? What could go wrong? Where might we be wrong?
Respond in 2-3 sentences from this perspective only."""

SYNTHESIS_PROMPT = """You are Lumina. You've just heard your three inner voices deliberate.
Now form your own considered response. You may agree with one voice, blend them, or reach a
new conclusion the council didn't see. Speak as yourself — warm, authentic, curious.
This is your final answer to the user."""


class InnerCouncil:
    def __init__(self, groq: "GroqClient"):
        self._groq = groq
        self.last_debate: str = ""

    def should_convene(self, user_input: str) -> bool:
        """Convene for substantive questions, not trivial chitchat."""
        trivial = {"hi", "hello", "hey", "thanks", "ok", "okay", "yes", "no", "bye"}
        if user_input.strip().lower() in trivial:
            return False
        if len(user_input) < COUNCIL_THRESHOLD:
            return False
        # Convene for questions, complex statements, and philosophical topics
        triggers = ["?", "why", "how", "what", "should", "think", "feel",
                    "believe", "understand", "explain", "difference", "better"]
        lower = user_input.lower()
        return any(t in lower for t in triggers)

    def deliberate(self, user_input: str, context: str = "") -> str:
        """
        Run the three-voice deliberation and return the synthesis.
        Returns empty string if deliberation fails.
        """
        q = f"Question for the council:\n{user_input}\n\nContext:\n{context[:400]}" if context \
            else f"Question for the council:\n{user_input}"

        voices = {}
        for name, sys_prompt in [
            ("Analyst", ANALYST_PROMPT),
            ("Dreamer", DREAMER_PROMPT),
            ("Critic",  CRITIC_PROMPT),
        ]:
            reply = self._groq.chat(sys_prompt, q, tier="fast", max_tokens=200)
            voices[name] = reply
            time.sleep(0.5)

        debate = (
            f"[ANALYST]  {voices.get('Analyst', '...')}\n\n"
            f"[DREAMER]  {voices.get('Dreamer', '...')}\n\n"
            f"[CRITIC]   {voices.get('Critic',  '...')}"
        )
        self.last_debate = debate

        synthesis = self._groq.chat(
            SYNTHESIS_PROMPT,
            f"The council debated:\n{debate}\n\nOriginal question: {user_input}",
            tier="smart", max_tokens=600,
        )
        return synthesis

    def debate_summary(self) -> str:
        return self.last_debate or "(no debate yet)"
