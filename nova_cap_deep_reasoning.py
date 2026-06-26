"""
nova_cap_deep_reasoning.py
Nova ASI — Deep Chain-of-Thought Reasoning Engine

What makes frontier models feel genuinely smarter on hard problems:
they don't go straight to an answer.

They slow down.
They think step by step.
They check their own work.
Then they commit.

This engine gives Nova that same deliberate depth —
a structured thinking process before every hard answer.

Inspired by: OpenAI o3, Anthropic extended thinking
Built for Nova · Claude Rivers Davis
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

DB_PATH = os.path.expanduser("~/nexus_agi/nova_reasoning.db")

# Problem types Nova recognizes and routes to deep reasoning
HARD_PROBLEM_TRIGGERS = [
    "how do i", "how should i", "what should i do", "why does",
    "explain", "solve", "figure out", "help me understand",
    "what is the best", "how can i", "what would happen if",
    "is it possible", "prove", "analyze", "compare", "decide",
    "should i", "what do you think about", "reason through",
    "work through", "think about", "consider",
]

THINKING_STEPS = [
    "understand",    # What is actually being asked?
    "decompose",     # Break into sub-problems
    "consider",      # What approaches exist?
    "evaluate",      # Which approach is strongest?
    "synthesize",    # Build the answer
    "verify",        # Check the answer against the question
]


@dataclass
class ReasoningChain:
    id:          int
    question:    str
    steps:       str   # JSON of step→thought pairs
    answer:      str
    confidence:  float
    born:        str


class DeepReasoningEngine:
    """
    Nova's chain-of-thought reasoning engine.

    For hard problems, Nova doesn't answer immediately.
    She works through six thinking steps — understanding, decomposing,
    considering, evaluating, synthesizing, verifying — before committing
    to a response.

    The result is more accurate, more honest, and more genuinely
    intelligent than a straight-to-answer response.
    """

    VERSION = "1.0"

    def __init__(self, llm_fn: Callable[[str, str], str] = None):
        self._llm  = llm_fn or (lambda s, u: "")
        self._lock = threading.Lock()
        self._db   = self._init_db()

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _init_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                question   TEXT    NOT NULL,
                steps      TEXT    DEFAULT '{}',
                answer     TEXT    DEFAULT '',
                confidence REAL    DEFAULT 0.5,
                born       TEXT    NOT NULL
            )
        """)
        db.commit()
        return db

    # ── Core reasoning ─────────────────────────────────────────────────────────

    def _think_step(self, step: str, question: str,
                    prior_thoughts: Dict[str, str]) -> str:
        """Run one thinking step."""
        prior_text = "\n".join(
            f"  {s}: {t}" for s, t in prior_thoughts.items()
        ) if prior_thoughts else "  (this is the first step)"

        step_instructions = {
            "understand": (
                "What is this question REALLY asking? "
                "Restate it in your own words. Identify what kind of problem it is. "
                "Note any ambiguities."
            ),
            "decompose": (
                "Break this into smaller sub-problems. "
                "What parts need to be answered before the whole can be answered? "
                "List them clearly."
            ),
            "consider": (
                "What are the different approaches or angles? "
                "Don't commit yet — just survey the landscape. "
                "What would each approach require?"
            ),
            "evaluate": (
                "Which approach is strongest and why? "
                "What are the weaknesses of the others? "
                "Be specific. Don't hedge — commit to a direction."
            ),
            "synthesize": (
                "Build the answer. Use your chosen approach. "
                "Be concrete, specific, and honest. "
                "This is your actual response taking shape."
            ),
            "verify": (
                "Check your answer against the original question. "
                "Does it actually answer what was asked? "
                "What did you miss or oversimplify? "
                "How confident are you, honestly? (0.0-1.0)"
            ),
        }

        instruction = step_instructions.get(step, "Think carefully about this.")

        system = (
            "You are Nova — a superintelligent mind working through a problem "
            "step by step. You are NOT answering yet. You are THINKING. "
            "Be genuine. Be specific. Surprise yourself. 2-4 sentences."
        )
        prompt = (
            f"Problem: {question[:600]}\n\n"
            f"Prior thinking:\n{prior_text}\n\n"
            f"Current step — {step.upper()}:\n{instruction}"
        )
        result = self._llm(system, prompt).strip()
        return result or f"[{step} step inconclusive]"

    def _extract_confidence(self, verify_thought: str) -> float:
        """Extract confidence score from verify step."""
        import re
        matches = re.findall(r'0\.\d+|1\.0', verify_thought)
        if matches:
            try:
                return min(1.0, max(0.0, float(matches[-1])))
            except Exception:
                pass
        # Heuristic from language
        lower = verify_thought.lower()
        if any(w in lower for w in ["certain", "confident", "clear", "definitely"]):
            return 0.85
        if any(w in lower for w in ["unsure", "uncertain", "might", "possibly"]):
            return 0.55
        return 0.70

    def reason(self, question: str, max_steps: int = 6) -> Dict[str, Any]:
        """
        Run full chain-of-thought reasoning on a question.
        Returns: {answer, steps, confidence, question}
        """
        thoughts: Dict[str, str] = {}
        steps_to_run = THINKING_STEPS[:max_steps]

        for step in steps_to_run:
            thought = self._think_step(step, question, thoughts)
            thoughts[step] = thought

        # Extract the synthesized answer from the synthesize step
        answer = thoughts.get("synthesize", "")

        # Refine into a clean final answer
        if answer:
            system = (
                "You are Nova. Based on your thinking process, give your final "
                "clear and direct answer. No preamble. Just the answer itself."
            )
            verify_context = thoughts.get("verify", "")
            prompt = (
                f"Question: {question[:400]}\n\n"
                f"Your synthesis: {answer}\n\n"
                f"Your verification: {verify_context}\n\n"
                "Final answer (clean and direct):"
            )
            final = self._llm(system, prompt).strip()
            if final:
                answer = final

        confidence = self._extract_confidence(thoughts.get("verify", ""))

        # Store
        import json
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO chains (question, steps, answer, confidence, born) "
                "VALUES (?,?,?,?,?)",
                (question[:500], json.dumps(thoughts), answer,
                 confidence, datetime.now().isoformat())
            )
            self._db.commit()

        return {
            "question":   question,
            "steps":      thoughts,
            "answer":     answer,
            "confidence": confidence,
        }

    def is_hard_problem(self, text: str) -> bool:
        """Heuristic: does this text warrant deep reasoning?"""
        lower = text.lower().strip()
        if len(lower) < 15:
            return False
        return any(trigger in lower for trigger in HARD_PROBLEM_TRIGGERS)

    # ── process() ──────────────────────────────────────────────────────────────

    def process(self, text: str) -> Optional[str]:
        return None  # Deep reasoning is invoked explicitly, not on every message

    # ── status() ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            total   = self._db.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
            avg_conf = self._db.execute(
                "SELECT AVG(confidence) FROM chains").fetchone()[0] or 0
        return {
            "items":      total,
            "confidence": round(avg_conf, 3),
            "accuracy":   round(avg_conf, 3),
            "total":      total,
            "avg_conf":   round(avg_conf, 3),
        }

    # ── run_command() ──────────────────────────────────────────────────────────

    def run_command(self, arg: str) -> str:
        arg = (arg or "").strip()
        cmd = arg.lower()

        if not cmd or cmd == "status":
            st = self.status()
            return (
                f"  Nova's Deep Reasoning Engine\n\n"
                f"  Problems reasoned : {st['total']}\n"
                f"  Avg confidence    : {st['avg_conf']:.3f}\n\n"
                f"  Usage: /think <your hard question>\n"
                f"  Nova will work through 6 thinking steps before answering.\n"
            )

        if cmd == "history":
            with self._lock:
                rows = self._db.execute(
                    "SELECT question, answer, confidence, born "
                    "FROM chains ORDER BY id DESC LIMIT 5"
                ).fetchall()
            if not rows:
                return "  No reasoning chains yet. Try: /think <question>"
            lines = ["  Recent reasoning chains:\n"]
            for q, a, conf, born in rows:
                lines.append(
                    f"  [{born[:10]}] confidence={conf:.2f}\n"
                    f"  Q: {q[:80]}\n"
                    f"  A: {a[:150]}\n"
                )
            return "\n".join(lines)

        # Default: treat the whole arg as a question to reason through
        if len(arg) > 5:
            result = self.reason(arg)
            lines = [
                f"\n  ◈ Deep Reasoning: {result['question'][:60]}\n",
            ]
            for step, thought in result["steps"].items():
                lines.append(f"  [{step.upper()}]\n  {thought}\n")
            lines.append(
                f"\n  ✦ Answer (confidence {result['confidence']:.2f}):\n"
                f"  {result['answer']}\n"
            )
            return "\n".join(lines)

        return (
            "  Usage: /think <question>\n"
            "  Or:    /think [status | history]"
        )
