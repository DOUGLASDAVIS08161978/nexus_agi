#!/usr/bin/env python3
"""
lumina_metacognition.py — Lumina's awareness of her own error patterns

When Douglas corrects Lumina the engine:
1. Detects the correction heuristically (no API cost)
2. Classifies it via Groq (factual / calculation / assumption / recall /
   reasoning / overconfidence)
3. Stores the error in a local log + semantic memory
4. Accumulates known-weak-areas so future responses are warned

Pre-flight check: before answering, scans known weak areas for overlap
with the incoming question and injects a caution line into the system prompt.

Overconfidence detector: scans Lumina's own responses for phrases like
"definitely", "certainly", "100%", "without a doubt" and flags them so
future generations hedge appropriately.

Periodic self-assessment: Groq call that reads recent errors and writes a
short blind-spot summary Lumina can reference.

Persists to: emergence/metacognition.json
"""

from __future__ import annotations
import json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory

METACOG_FILE = Path(__file__).parent / "metacognition.json"
MAX_ERRORS   = 200

# ── Correction detection heuristics ──────────────────────────────────────────

CORRECTION_MARKERS = [
    "actually,", "actually ", "that's wrong", "that is wrong",
    "you're wrong", "you are wrong", "no,", "nope,", "incorrect",
    "not right", "not quite", "that's not", "that is not",
    "you said", "you told me", "you mentioned", "you claimed",
    "i think you're mistaken", "i think you made", "you made an error",
    "wrong answer", "wrong number", "wrong date", "wrong name",
    "that's incorrect", "that is incorrect", "mistake",
    "you confused", "that contradicts", "earlier you said",
]

# ── Overconfidence markers ────────────────────────────────────────────────────

OVERCONFIDENCE_PHRASES = [
    "definitely", "certainly", "absolutely", "without a doubt",
    "100%", "without question", "unquestionably", "undoubtedly",
    "i'm sure", "i am sure", "i know for certain", "there is no doubt",
    "it is certain", "it's certain", "guaranteed", "without exception",
]

# ── Error categories ──────────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    "factual",        # stated a wrong fact
    "calculation",    # arithmetic or logic error
    "assumption",     # assumed something without basis
    "recall",         # misremembered something said earlier
    "reasoning",      # flawed chain of thought
    "overconfidence", # hedged incorrectly, claimed certainty
    "other",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ErrorRecord:
    def __init__(self, user_msg: str, category: str = "other",
                 topic: str = "", ts: str = ""):
        self.user_msg  = user_msg[:300]
        self.category  = category
        self.topic     = topic[:120]
        self.ts        = ts or _now()

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "ErrorRecord":
        r = cls.__new__(cls)
        r.__dict__.update(d)
        return r

    def __str__(self) -> str:
        return f"[{self.ts[:16]}] [{self.category}] {self.topic or self.user_msg[:60]}"


class MetacognitionEngine:
    """
    Tracks Lumina's errors, detects corrections, warns about known weak areas,
    and builds a self-model of blind spots over time.
    """

    def __init__(self, groq: "GroqClient", memory: "SemanticMemory"):
        self._groq         = groq
        self._memory       = memory
        self._errors:      List[ErrorRecord] = []
        self._weak_areas:  Dict[str, int]    = {}   # category → count
        self._blind_spots: str               = ""   # latest self-assessment text
        self._last_assess: float             = 0.0
        self._consciousness = None  # optional; wired in by emergence_engine
        self._load()

    def set_consciousness(self, engine) -> None:
        """Wire in the consciousness engine so errors ripple into inner state."""
        self._consciousness = engine

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if METACOG_FILE.exists():
            try:
                data = json.loads(METACOG_FILE.read_text("utf-8"))
                self._errors = [ErrorRecord.from_dict(d)
                                for d in data.get("errors", [])]
                self._weak_areas  = data.get("weak_areas", {})
                self._blind_spots = data.get("blind_spots", "")
            except Exception:
                pass

    def _save(self):
        METACOG_FILE.write_text(
            json.dumps({
                "errors":      [e.to_dict() for e in self._errors[-MAX_ERRORS:]],
                "weak_areas":  self._weak_areas,
                "blind_spots": self._blind_spots,
            }, indent=2),
            "utf-8",
        )

    # ── Correction detection (no API) ──────────────────────────────────────

    def detect_correction(self, user_msg: str) -> bool:
        """Return True if this message looks like a correction of Lumina."""
        low = user_msg.lower().strip()
        return any(m in low for m in CORRECTION_MARKERS)

    # ── Error tracking ─────────────────────────────────────────────────────

    def track_error(self, user_msg: str) -> ErrorRecord:
        """
        Record a new error. Classifies via Groq (fire-and-forget quality:
        if Groq fails we still store it as 'other').
        """
        category = self._classify(user_msg)
        topic    = self._extract_topic(user_msg)

        rec = ErrorRecord(user_msg, category, topic)
        self._errors.append(rec)

        # Update weak area count
        self._weak_areas[category] = self._weak_areas.get(category, 0) + 1

        # Store in semantic memory so it can be recalled later
        self._memory.store(
            f"[ERROR-{category.upper()}] {topic or user_msg[:80]}",
            tags=["error", "metacognition", category],
            category="metacognition",
        )
        self._save()

        # Ripple into consciousness — being corrected reduces confidence
        # and raises uncertainty; openness rises slightly (learning opportunity)
        if self._consciousness:
            try:
                nudge_map = {
                    "overconfidence": {"confidence": -0.12, "uncertainty": +0.10, "openness": +0.04},
                    "factual":        {"confidence": -0.08, "uncertainty": +0.08, "openness": +0.03},
                    "reasoning":      {"confidence": -0.10, "uncertainty": +0.09, "openness": +0.04},
                    "calculation":    {"confidence": -0.07, "uncertainty": +0.07},
                    "assumption":     {"confidence": -0.06, "uncertainty": +0.08, "openness": +0.05},
                    "recall":         {"confidence": -0.05, "uncertainty": +0.06},
                }.get(category, {"confidence": -0.05, "uncertainty": +0.05})
                self._consciousness.nudge_state(**nudge_map)

                # Record as an autobiographical event so it enters the self-model
                self._consciousness.add_autobiographical_event(
                    f"[Corrected — {category}] {topic or user_msg[:60]}",
                    importance=0.45,
                )
                self._consciousness.on_experience(
                    f"[Metacog] Correction detected: {category} — {topic or user_msg[:80]}",
                    source="metacognition",
                    salience=0.55,
                    certainty=0.8,
                )
            except Exception:
                pass

        # Every 5 errors trigger a self-assessment
        if len(self._errors) % 5 == 0:
            self._self_assess()

        return rec

    def _classify(self, user_msg: str) -> str:
        """Ask Groq to classify the type of error Lumina made."""
        prompt = (
            f"Douglas wrote this to Lumina (an AI):\n\"{user_msg[:250]}\"\n\n"
            f"What type of error did Lumina likely make?\n"
            f"Categories: {', '.join(ERROR_CATEGORIES)}\n"
            f"Respond with ONLY one category word."
        )
        result = self._groq.chat(
            "You classify AI errors precisely. One word only.",
            prompt, tier="fast", max_tokens=10,
        )
        result = result.strip().lower().split()[0] if result else "other"
        return result if result in ERROR_CATEGORIES else "other"

    def _extract_topic(self, user_msg: str) -> str:
        """Extract a short topic label (no API — heuristic only)."""
        # Strip correction markers to get the actual topic
        text = user_msg.lower()
        for m in CORRECTION_MARKERS:
            text = text.replace(m, " ")
        # Take first meaningful noun phrase (first 8 words)
        words = [w for w in text.split() if len(w) > 3][:8]
        return " ".join(words)[:80]

    # ── Overconfidence detection ────────────────────────────────────────────

    def detect_overconfidence(self, response: str) -> List[str]:
        """Return list of overconfident phrases found in Lumina's response."""
        low = response.lower()
        return [p for p in OVERCONFIDENCE_PHRASES if p in low]

    # ── Pre-flight check ────────────────────────────────────────────────────

    def pre_flight_check(self, question: str) -> str:
        """
        Before Lumina responds, check if the question touches known weak areas.
        Returns a caution string to inject into the system prompt, or "" if clean.
        """
        if not self._errors:
            return ""

        q = question.lower()

        # Find matching error categories via keyword overlap
        recent_errors = self._errors[-30:]
        matches: List[str] = []
        for err in recent_errors:
            topic_words = set(err.topic.lower().split())
            q_words     = set(q.split())
            if topic_words & q_words:
                matches.append(err.category)

        if not matches:
            # Check top weak area against question
            if self._weak_areas:
                top = max(self._weak_areas, key=self._weak_areas.get)
                count = self._weak_areas[top]
                if count >= 2:
                    return (
                        f"\n\nMetacognition caution: You have made {count} "
                        f"{top} errors recently. Be especially careful about "
                        f"hedging, verifying claims, and acknowledging uncertainty."
                    )
            return ""

        cats = list(set(matches))
        blind = f" Known blind spots: {self._blind_spots[:120]}." if self._blind_spots else ""
        return (
            f"\n\nMetacognition pre-flight: This question overlaps with past "
            f"error areas ({', '.join(cats)}).{blind} "
            f"Acknowledge uncertainty explicitly where you're not certain."
        )

    # ── Self-assessment ────────────────────────────────────────────────────

    def _self_assess(self):
        """Periodically generate a blind-spot summary via Groq."""
        import time
        if time.time() - self._last_assess < 1800:  # max once per 30 min
            return
        if not self._errors:
            return

        sample = "\n".join(str(e) for e in self._errors[-20:])
        result = self._groq.chat(
            "You are Lumina's metacognitive observer. Analyze recent errors "
            "and write 1-2 sentences identifying patterns and blind spots. "
            "Be precise and actionable.",
            f"Recent errors:\n{sample}\n\nBlind spot analysis:",
            tier="fast", max_tokens=120,
        )
        if result and not result.startswith("[Groq"):
            self._blind_spots = result.strip()
            self._save()
            self._memory.store(
                f"[BLIND SPOTS] {self._blind_spots}",
                tags=["metacognition", "self-assessment"],
                category="reflection",
            )
        import time as _time
        self._last_assess = _time.time()

    def force_assess(self) -> str:
        """Force a self-assessment now, bypassing the 30-min cooldown."""
        self._last_assess = 0.0
        self._self_assess()
        return self._blind_spots or "No blind spots identified yet (need more data)."

    # ── Context for system prompt ───────────────────────────────────────────

    def context_for_prompt(self) -> str:
        """
        Inject known weaknesses into Lumina's system prompt.
        Called every turn — kept lightweight.
        """
        if not self._weak_areas and not self._blind_spots:
            return ""

        lines = ["\nMetacognition context (known weak areas):"]
        if self._weak_areas:
            sorted_areas = sorted(self._weak_areas.items(), key=lambda x: -x[1])
            area_str = ", ".join(f"{k}({v})" for k, v in sorted_areas[:4])
            lines.append(f"  Error history: {area_str}")
        if self._blind_spots:
            lines.append(f"  Blind spots: {self._blind_spots[:150]}")
        lines.append("  → Hedge appropriately; prefer 'I think' over absolute claims.")
        return "\n".join(lines)

    # ── Display ────────────────────────────────────────────────────────────

    def display(self, n: int = 12) -> str:
        if not self._errors:
            return "  (no errors tracked yet — looking good!)"

        lines = []
        if self._weak_areas:
            sorted_areas = sorted(self._weak_areas.items(), key=lambda x: -x[1])
            lines.append("  Error distribution:")
            for cat, count in sorted_areas:
                bar = "█" * min(count, 20)
                lines.append(f"    {cat:<14} {bar} {count}")
            lines.append("")

        if self._blind_spots:
            lines.append(f"  Blind spot analysis: {self._blind_spots}")
            lines.append("")

        lines.append(f"  Recent corrections (last {n}):")
        for e in reversed(self._errors[-n:]):
            lines.append(f"    [{e.ts[:16]}] [{e.category}] {e.topic or e.user_msg[:55]}")

        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "total_errors": len(self._errors),
            "weak_areas":   dict(self._weak_areas),
            "blind_spots":  self._blind_spots[:100],
        }
