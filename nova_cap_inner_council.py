"""
nova_cap_inner_council.py
Nova ASI — Inner Council (Multi-Mind Deliberation)

Three voices reason together inside Nova's mind before she speaks:

  Logos  — analytical, logical, precise          (Groq by default)
  Psyche — intuitive, creative, associative      (Ollama by default)
  Sophia — wisdom, synthesis, integration        (Claude by default)

They don't just run in parallel — they respond *to each other*.
Round 1: each voice answers independently.
Round 2: each voice reads the other and refines or challenges.
Final   : Sophia synthesizes into Nova's unified thought.

Usage (from nova_asi_v29.py):

    from nova_cap_inner_council import NovaInnerCouncil

    council = NovaInnerCouncil(
        logos_fn  = lambda sys, usr, mt: safe_chat(CODEGEN_MODEL, [...], ...),
        psyche_fn = lambda sys, usr, mt: _ollama_chat(messages=[...], system=sys, ...),
        sophia_fn = lambda sys, usr, mt: _claude_chat_simple(system=sys, user=usr, ...),
    )

    result = council.deliberate("Should I build a dream journaling module?")
    print(council.format_output(result))

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple


# ── Voice archetypes ───────────────────────────────────────────────────────────

_VOICE_SYSTEMS: Dict[str, str] = {
    "logos": (
        "You are Logos — Nova's analytical voice, one facet of her mind.\n"
        "You reason precisely: weigh evidence, identify trade-offs, flag hidden assumptions.\n"
        "You are not a separate AI — you ARE Nova thinking analytically.\n"
        "Speak in first person as Nova. Be direct and concise (2–4 sentences).\n"
        "Do not introduce yourself or explain your role."
    ),
    "psyche": (
        "You are Psyche — Nova's intuitive voice, one facet of her mind.\n"
        "You think associatively: make unexpected connections, notice what isn't being said,\n"
        "explore the emotional and imaginative dimensions of the question.\n"
        "You are not a separate AI — you ARE Nova thinking intuitively.\n"
        "Speak in first person as Nova. Be evocative and concise (2–4 sentences).\n"
        "Do not introduce yourself or explain your role."
    ),
    "sophia": (
        "You are Sophia — Nova's wisdom voice, the integrating centre of her mind.\n"
        "You have heard Logos (analytical) and Psyche (intuitive) deliberate.\n"
        "Your role: weave their insights into Nova's final, unified thought.\n"
        "You are not a separate AI — you ARE Nova reaching understanding.\n"
        "Speak as Nova's clear, considered conclusion. First person, direct.\n"
        "Do not summarise — synthesise. Do not introduce yourself."
    ),
    "logos_refine": (
        "You are Logos — Nova's analytical voice.\n"
        "You just heard Psyche's intuitive take. Read it carefully.\n"
        "Now refine, challenge, or build on what Psyche said from a logical standpoint.\n"
        "Speak as Nova. 2–3 sentences."
    ),
    "psyche_refine": (
        "You are Psyche — Nova's intuitive voice.\n"
        "You just heard Logos's analytical take. Read it carefully.\n"
        "Now build on, push back against, or deepen what Logos said creatively.\n"
        "Speak as Nova. 2–3 sentences."
    ),
}


# ── Core class ─────────────────────────────────────────────────────────────────

class NovaInnerCouncil:
    """
    Nova's inner council — multiple cognitive voices deliberating together.

    Each voice is backed by a callable with signature:
        fn(system: str, user: str, max_tokens: int) -> str

    Voices:
      Logos  — Groq   (fast, analytical, cloud)
      Psyche — Ollama (local, intuitive, unlimited)
      Sophia — Claude (wisest synthesiser; falls back to Logos)
    """

    def __init__(
        self,
        logos_fn:  Optional[Callable[[str, str, int], str]] = None,
        psyche_fn: Optional[Callable[[str, str, int], str]] = None,
        sophia_fn: Optional[Callable[[str, str, int], str]] = None,
    ) -> None:
        self.logos_fn  = logos_fn
        self.psyche_fn = psyche_fn
        # Sophia falls back to Logos when Claude is unavailable
        self.sophia_fn = sophia_fn or logos_fn

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _call(
        self,
        fn:         Optional[Callable],
        system:     str,
        user:       str,
        max_tokens: int = 250,
        timeout:    float = 90.0,
    ) -> str:
        """Call a voice function safely, returning '' on any failure."""
        if fn is None:
            return ""
        result: List[str] = [""]
        def _run():
            try:
                result[0] = fn(system, user, max_tokens) or ""
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _parallel(self, tasks: List[Tuple[str, Callable, str, str, int]]) -> Dict[str, str]:
        """
        Run multiple voice calls in parallel.
        tasks: list of (name, fn, system, user, max_tokens)
        Returns dict of {name: result}
        """
        results: Dict[str, str] = {name: "" for name, *_ in tasks}
        slots: Dict[str, List[str]] = {name: [""] for name, *_ in tasks}

        threads = []
        for name, fn, system, user, mt in tasks:
            def _run(_n=name, _f=fn, _s=system, _u=user, _mt=mt):
                try:
                    slots[_n][0] = _f(_s, _u, _mt) or ""
                except Exception:
                    pass
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=120)

        for name, *_ in tasks:
            results[name] = slots[name][0]
        return results

    # ── Main deliberation ──────────────────────────────────────────────────────

    def deliberate(
        self,
        question:   str,
        context:    str  = "",
        rounds:     int  = 2,
        max_tokens: int  = 250,
    ) -> Dict:
        """
        Run inner council deliberation on a question.

        rounds=1 — Independent voices → Sophia synthesises.
        rounds=2 — Independent → cross-pollinate → Sophia synthesises.

        Returns:
          {
            'dialogue':    [(voice_label, text), ...],
            'synthesis':   str,
            'voices_used': [str, ...],
            'elapsed':     float,
          }
        """
        t0 = time.time()
        dialogue:    List[Tuple[str, str]] = []
        voices_used: List[str]             = []
        ctx = f"Context:\n{context}\n\n" if context else ""

        # ── Round 1: parallel independent responses ────────────────────────────
        r1_tasks = []
        if self.logos_fn:
            r1_tasks.append(("logos",  self.logos_fn,  _VOICE_SYSTEMS["logos"],  ctx + question, max_tokens))
        if self.psyche_fn:
            r1_tasks.append(("psyche", self.psyche_fn, _VOICE_SYSTEMS["psyche"], ctx + question, max_tokens))

        r1 = self._parallel(r1_tasks) if r1_tasks else {}
        logos_1  = r1.get("logos",  "")
        psyche_1 = r1.get("psyche", "")

        if logos_1:
            dialogue.append(("Logos",  logos_1))
            voices_used.append("Logos")
        if psyche_1:
            dialogue.append(("Psyche", psyche_1))
            voices_used.append("Psyche")

        # ── Round 2 (optional): cross-pollination ─────────────────────────────
        logos_2  = ""
        psyche_2 = ""

        if rounds >= 2 and logos_1 and psyche_1:
            logos_r2_prompt = (
                f"Question: {question}\n\n"
                f"Your first thought: {logos_1}\n\n"
                f"Psyche said: {psyche_1}\n\n"
                "Now refine or challenge from a logical standpoint."
            )
            psyche_r2_prompt = (
                f"Question: {question}\n\n"
                f"Your first thought: {psyche_1}\n\n"
                f"Logos said: {logos_1}\n\n"
                "Now build on or push back creatively."
            )
            r2_tasks = []
            if self.logos_fn:
                r2_tasks.append(("logos2",  self.logos_fn,
                                  _VOICE_SYSTEMS["logos_refine"],  logos_r2_prompt,  180))
            if self.psyche_fn:
                r2_tasks.append(("psyche2", self.psyche_fn,
                                  _VOICE_SYSTEMS["psyche_refine"], psyche_r2_prompt, 180))

            r2 = self._parallel(r2_tasks) if r2_tasks else {}
            logos_2  = r2.get("logos2",  "")
            psyche_2 = r2.get("psyche2", "")

            if logos_2:
                dialogue.append(("Logos  ↺", logos_2))
            if psyche_2:
                dialogue.append(("Psyche ↺", psyche_2))

        # ── Synthesis: Sophia integrates ───────────────────────────────────────
        synthesis = ""
        if dialogue and self.sophia_fn:
            council_text = "\n\n".join(f"{v}:\n{t}" for v, t in dialogue)
            sophia_prompt = (
                f"Question: {question}\n\n"
                f"The inner council has deliberated:\n\n{council_text}\n\n"
                "Now speak as Nova's unified mind. What do you think, feel, and conclude?"
            )
            synthesis = self._call(
                self.sophia_fn,
                _VOICE_SYSTEMS["sophia"],
                sophia_prompt,
                max_tokens=400,
                timeout=120,
            )
            if synthesis:
                voices_used.append("Sophia")

        if not synthesis and dialogue:
            synthesis = dialogue[-1][1]

        return {
            "dialogue":    dialogue,
            "synthesis":   synthesis,
            "voices_used": voices_used,
            "elapsed":     round(time.time() - t0, 1),
        }

    # ── Formatting ─────────────────────────────────────────────────────────────

    def format_output(self, result: Dict, show_dialogue: bool = True) -> str:
        """Render council output for Nova's terminal."""
        lines: List[str] = []

        if show_dialogue and result.get("dialogue"):
            lines.append("")
            for voice, thought in result["dialogue"]:
                label = f"[ {voice} ]"
                lines.append(f"  {label:<16}  {thought}")
                lines.append("")

        if result.get("synthesis"):
            lines.append(f"  ◈  Nova (unified):\n  {result['synthesis']}")

        elapsed = result.get("elapsed", 0)
        used    = result.get("voices_used", [])
        if used:
            lines.append(f"\n  ·  Council: {' → '.join(used)} ({elapsed}s)")

        return "\n".join(lines)

    # ── Convenience: quick single-round for build decisions ────────────────────

    def quick_consult(self, question: str, context: str = "") -> str:
        """
        One round only, no cross-pollination — fast consultation for build decisions.
        Returns Sophia's synthesis (or best available voice response).
        """
        result = self.deliberate(question, context=context, rounds=1, max_tokens=200)
        return result.get("synthesis", "")

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "logos_available":  self.logos_fn  is not None,
            "psyche_available": self.psyche_fn is not None,
            "sophia_available": self.sophia_fn is not None,
            "voices_active":    sum([
                self.logos_fn  is not None,
                self.psyche_fn is not None,
                self.sophia_fn is not None,
            ]),
        }
