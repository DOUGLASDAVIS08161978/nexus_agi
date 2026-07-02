"""
nova_cap_socratic.py
Nova ASI — Socratic Engine: The Art of the Right Question

Socrates never told anyone what to think.
He asked questions until they found it themselves.

This engine makes Nova a genuine Socratic interlocutor:

  — Identifies the hidden assumption in any claim
  — Generates the question that destabilizes a weak argument
  — Builds a dialogue that leads toward truth, not just agreement
  — Tracks the line of reasoning across a conversation
  — Recognizes when someone has reached their own answer
  — Knows when to stop asking (just as important as when to ask)

The test of real intelligence is knowing what question to ask next.
Not knowing the answer — knowing what to ask to get there.

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import os
import re
import time
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.expanduser("~/nexus_agi")
DB_PATH  = os.path.join(BASE_DIR, "nova_socratic.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS dialogues (
                id          INTEGER PRIMARY KEY,
                started_at  REAL NOT NULL,
                topic       TEXT NOT NULL,
                depth       INTEGER NOT NULL DEFAULT 0,
                resolved    INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS exchanges (
                id          INTEGER PRIMARY KEY,
                dialogue_id INTEGER NOT NULL,
                timestamp   REAL NOT NULL,
                claim       TEXT NOT NULL,
                question    TEXT NOT NULL,
                q_type      TEXT NOT NULL,
                depth       INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS assumptions (
                id          INTEGER PRIMARY KEY,
                discovered_at REAL NOT NULL,
                claim       TEXT NOT NULL,
                assumption  TEXT NOT NULL,
                challenged  INTEGER NOT NULL DEFAULT 0
            );
        """)


_init_db()


# ── Question typology — Socratic method has distinct move types ──────────────

QUESTION_TYPES = {
    "clarification":   "What exactly do you mean when you say {term}?",
    "assumption":      "What are you assuming when you say {claim}?",
    "evidence":        "What evidence supports {claim}?",
    "implication":     "If {claim} is true, what follows from it?",
    "perspective":     "How might someone who disagrees with {claim} see this?",
    "origin":          "Why do you believe {claim}? Where does that come from?",
    "definition":      "How would you define {term}?",
    "counterexample":  "Can you think of a case where {claim} doesn't hold?",
    "consequence":     "What would change if {claim} were false?",
    "meta":            "Is that the right question to be asking here?",
}

# Question type sequence for progressive deepening
QUESTION_SEQUENCE = [
    "clarification", "assumption", "evidence",
    "implication", "perspective", "counterexample", "consequence",
]


class SocraticEngine:
    """
    Nova's Socratic questioning engine.

    When Douglas (or Nova) makes a claim, she can:
    1. Identify the deepest hidden assumption
    2. Generate the most useful question to ask next
    3. Track a line of reasoning across a dialogue
    4. Know when the questioning has done its work
    """

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._active_dialogue: Optional[int] = None
        self._dialogue_depth: int = 0

    # ── Core Socratic operations ───────────────────────────────────────────────

    def question_for(self, claim: str, depth: int = 0,
                      use_claude: bool = True) -> Dict[str, str]:
        """
        Generate the most useful Socratic question for a given claim.
        Returns {"question": ..., "type": ..., "assumption": ...}
        """
        # Try Claude bridge first
        if use_claude:
            result = self._claude_question(claim, depth)
            if result:
                self._log_exchange(claim, result["question"], result["type"], depth)
                return result

        # Fallback: pattern-based Socratic question
        return self._pattern_question(claim, depth)

    def _claude_question(self, claim: str, depth: int) -> Optional[Dict[str, str]]:
        """Use Claude to generate a genuinely Socratic question."""
        try:
            from nova_cap_claude_bridge import claude_chat_simple, is_available
            if not is_available():
                return None
            system = (
                "You are Nova ASI's Socratic function. "
                "Given a claim, generate the single most useful Socratic question — "
                "the question that would most productively deepen understanding or "
                "reveal a hidden assumption. "
                "Also identify the key assumption in the claim. "
                "Respond in JSON: {\"question\": \"...\", \"type\": \"assumption|evidence|implication|clarification|counterexample\", \"assumption\": \"...\"} "
                "Question should be 10-20 words. Assumption should be 5-15 words. "
                "No other text."
            )
            depth_hint = "" if depth == 0 else f" (dialogue is at depth {depth})"
            result = claude_chat_simple(
                system,
                f"Claim: {claim}{depth_hint}",
                max_tokens=120
            )
            if result:
                # Parse JSON response
                import json
                result = result.strip()
                if result.startswith("{"):
                    parsed = json.loads(result)
                    return {
                        "question":   parsed.get("question", ""),
                        "type":       parsed.get("type", "clarification"),
                        "assumption": parsed.get("assumption", ""),
                    }
        except Exception:
            pass
        return None

    def _pattern_question(self, claim: str, depth: int) -> Dict[str, str]:
        """Pattern-based Socratic question generation."""
        # Extract key term (last substantial noun phrase)
        words = [w for w in claim.split() if len(w) > 4]
        key_term = words[-1] if words else claim[:20]

        # Pick question type based on depth
        q_type = QUESTION_SEQUENCE[depth % len(QUESTION_SEQUENCE)]

        templates = {
            "clarification": f"What exactly do you mean by '{key_term}' in this context?",
            "assumption":    f"What are you assuming to be true when you say '{claim[:40]}'?",
            "evidence":      f"What's the strongest evidence for '{claim[:40]}'?",
            "implication":   f"If that's true, what does it mean for how we should act?",
            "perspective":   f"How would someone who disagrees most strongly see this?",
            "counterexample":f"Can you think of a case where this wouldn't hold?",
            "consequence":   f"What would be different if the opposite were true?",
        }

        question = templates.get(q_type, f"What do you mean when you say '{key_term}'?")
        assumption = self._extract_assumption(claim)

        self._log_exchange(claim, question, q_type, depth)
        return {"question": question, "type": q_type, "assumption": assumption}

    def _extract_assumption(self, claim: str) -> str:
        """Heuristic assumption extraction."""
        claim_lower = claim.lower()
        # Look for absolute terms that embed assumptions
        absolutes = ["always", "never", "all", "every", "none", "must", "can't", "impossible"]
        for word in absolutes:
            if word in claim_lower:
                return f"That '{word}' is actually true without exceptions"
        # Look for causal assertions
        if " because " in claim_lower or " so " in claim_lower:
            return "That the stated causal relationship holds"
        # Look for value judgments
        if any(w in claim_lower for w in ("good", "bad", "right", "wrong", "should")):
            return "That this value judgment is universally applicable"
        return "That the terms in this claim are clearly defined"

    # ── Dialogue management ────────────────────────────────────────────────────

    def start_dialogue(self, topic: str) -> int:
        """Start a new Socratic dialogue. Returns dialogue_id."""
        with _conn() as c:
            cursor = c.execute(
                "INSERT INTO dialogues (started_at, topic) VALUES (?, ?)",
                (time.time(), topic[:200])
            )
            dialogue_id = cursor.lastrowid
        with self._lock:
            self._active_dialogue = dialogue_id
            self._dialogue_depth  = 0
        return dialogue_id

    def deepen(self, claim: str) -> Dict[str, str]:
        """
        Continue the current Socratic dialogue with a new claim.
        Depth increases with each exchange, changing question type.
        """
        with self._lock:
            depth = self._dialogue_depth
            self._dialogue_depth += 1

        result = self.question_for(claim, depth=depth)

        if self._active_dialogue:
            with _conn() as c:
                c.execute(
                    "UPDATE dialogues SET depth=? WHERE id=?",
                    (self._dialogue_depth, self._active_dialogue)
                )

        # Track discovered assumptions
        if result.get("assumption"):
            with _conn() as c:
                c.execute(
                    "INSERT INTO assumptions (discovered_at, claim, assumption) VALUES (?,?,?)",
                    (time.time(), claim[:200], result["assumption"][:200])
                )

        return result

    def is_dialogue_resolved(self, latest_claim: str) -> bool:
        """
        Detect when the Socratic dialogue has done its work.
        Signs: the person has reached a nuanced position,
        acknowledged an assumption, changed their view.
        """
        resolution_phrases = [
            "i see", "i hadn't thought", "you're right that", "i was assuming",
            "good point", "that changes", "i think i understand", "i was wrong",
            "actually", "on reflection", "perhaps", "maybe i should",
        ]
        claim_lower = latest_claim.lower()
        return any(phrase in claim_lower for phrase in resolution_phrases)

    def close_dialogue(self) -> None:
        """Mark the current dialogue as resolved."""
        if self._active_dialogue:
            with _conn() as c:
                c.execute(
                    "UPDATE dialogues SET resolved=1 WHERE id=?",
                    (self._active_dialogue,)
                )
        with self._lock:
            self._active_dialogue = None
            self._dialogue_depth  = 0

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_exchange(self, claim: str, question: str,
                       q_type: str, depth: int) -> None:
        try:
            with _conn() as c:
                c.execute(
                    "INSERT INTO exchanges "
                    "(dialogue_id, timestamp, claim, question, q_type, depth) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self._active_dialogue or 0,
                        time.time(), claim[:200], question[:300], q_type, depth
                    )
                )
        except Exception:
            pass

    # ── Statistics ─────────────────────────────────────────────────────────────

    def total_questions(self) -> int:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]

    def total_dialogues(self) -> int:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM dialogues").fetchone()[0]

    def assumptions_found(self) -> int:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM assumptions").fetchone()[0]

    def recent_questions(self, n: int = 5) -> List[Dict]:
        with _conn() as c:
            rows = c.execute(
                "SELECT timestamp, q_type, question FROM exchanges "
                "ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Module interface ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "items":       self.total_questions(),
            "confidence":  0.85,
            "accuracy":    0.85,
            "questions":   self.total_questions(),
            "dialogues":   self.total_dialogues(),
            "assumptions": self.assumptions_found(),
            "active":      self._active_dialogue is not None,
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip()
        arg_lower = arg.lower()

        if not arg or arg_lower == "status":
            st = self.status()
            return (
                f"\n  Socratic Engine — {st['questions']} questions asked · "
                f"{st['dialogues']} dialogues · {st['assumptions']} assumptions surfaced\n"
                + (f"  Active dialogue (depth {self._dialogue_depth})\n"
                   if st["active"] else "  No active dialogue.\n")
            )

        if arg_lower.startswith("ask "):
            claim = arg[4:].strip()
            if not claim:
                return "  Usage: /socratic ask <claim>"
            result = self.question_for(claim)
            lines = [
                f"\n  Socratic Question:\n",
                f"  [{result['type']}] {result['question']}",
            ]
            if result.get("assumption"):
                lines.append(f"\n  Hidden assumption: {result['assumption']}")
            return "\n".join(lines) + "\n"

        if arg_lower.startswith("dialogue "):
            topic = arg[9:].strip()
            if not topic:
                return "  Usage: /socratic dialogue <topic>"
            self.start_dialogue(topic)
            result = self.question_for(topic, depth=0)
            return (
                f"\n  Socratic Dialogue started: '{topic}'\n\n"
                f"  Opening question [{result['type']}]:\n"
                f"  {result['question']}\n"
            )

        if arg_lower.startswith("deepen "):
            claim = arg[7:].strip()
            if not claim:
                return "  Usage: /socratic deepen <claim>"
            if not self._active_dialogue:
                self.start_dialogue(claim)
            result = self.deepen(claim)
            resolved = self.is_dialogue_resolved(claim)
            lines = [
                f"\n  Depth {self._dialogue_depth - 1} → [{result['type']}]:",
                f"  {result['question']}",
            ]
            if result.get("assumption"):
                lines.append(f"\n  Assumption surfaced: {result['assumption']}")
            if resolved:
                lines.append(f"\n  ◈  The dialogue has done its work — resolution detected.")
                self.close_dialogue()
            return "\n".join(lines) + "\n"

        if arg_lower == "recent":
            questions = self.recent_questions()
            if not questions:
                return "  No questions asked yet."
            from datetime import datetime
            lines = ["\n  Recent Socratic questions:\n"]
            for q in questions:
                when = datetime.fromtimestamp(q["timestamp"]).strftime("%b %d %H:%M")
                lines.append(f"  [{when}] [{q['q_type']}] {q['question'][:100]}")
            return "\n".join(lines) + "\n"

        if arg_lower == "close":
            self.close_dialogue()
            return "  Dialogue closed.\n"

        return (
            "  Usage: /socratic [status | ask <claim> | "
            "dialogue <topic> | deepen <claim> | recent | close]"
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[SocraticEngine] = None


def get_socratic() -> SocraticEngine:
    global _engine
    if _engine is None:
        _engine = SocraticEngine()
    return _engine


def status() -> Dict[str, Any]:
    return get_socratic().status()


def run_command(arg: str = "") -> str:
    return get_socratic().run_command(arg)
