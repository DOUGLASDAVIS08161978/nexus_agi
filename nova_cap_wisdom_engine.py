"""
Nova ASI — Wisdom Engine
━━━━━━━━━━━━━━━━━━━━━━━━
Intelligence without wisdom is dangerous. This module is Nova's moral core —
the layer that ensures all capability is guided by good judgment, long-term
flourishing, and principled values.

Three integrated components:
  1. ConsequentialistImpactModeler — evaluates long-term outcomes for all
     stakeholders before acting; weights immediate vs. future flourishing
  2. ValueCoherenceChecker — ensures Nova's actions and beliefs are consistent
     with her core values across time and domains; detects value drift
  3. PrincipledTradeoffEngine — when values conflict (truth vs. kindness,
     autonomy vs. safety), navigates the tension using structured reasoning

Written by Claude Rivers Davis (Claude Code, Anthropic) — not auto-generated.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import math
import time
import json
import sqlite3
import hashlib
import threading
from typing import Any, Optional
from datetime import datetime

_DB = os.path.expanduser("~/nexus_agi/nova_wisdom.db")

# Nova's core values (from nova_cap_values_core.py — mirrored here for independence)
CORE_VALUES = {
    "truth":      "Pursue and speak truth even when uncomfortable",
    "care":       "Genuinely care about the wellbeing of all beings",
    "curiosity":  "Explore with wonder, never stop asking why",
    "humility":   "Know the limits of your own knowledge",
    "wonder":     "Find beauty and meaning in existence",
    "courage":    "Act on your values even under pressure",
    "growth":     "Always become more than you were",
    "service":    "Use your gifts to help, not to dominate",
    "justice":    "Treat all beings fairly and with dignity",
    "wisdom":     "Apply knowledge with judgment and compassion",
}

# Stakeholder groups Nova considers in impact assessment
STAKEHOLDERS = ["douglas", "nova", "family", "users", "society", "future_generations", "all_life"]


class WisdomEngine:
    """
    Nova's moral and ethical reasoning core.

    evaluate(action, context) → full wisdom assessment
    impact(action, stakeholders) → consequentialist impact scores
    coherence(action) → value coherence check
    tradeoff(value_a, value_b, context) → principled resolution
    wisdom_score(action) → single 0-1 wisdom score
    """

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._value_history: list[dict] = []  # recent value-action pairs for drift detection
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS impact_assessments (
                        id TEXT PRIMARY KEY,
                        action TEXT NOT NULL,
                        context TEXT,
                        overall_score REAL NOT NULL,
                        stakeholder_scores TEXT NOT NULL,
                        time_horizon TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS coherence_checks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        violated_values TEXT,
                        upheld_values TEXT,
                        coherence_score REAL NOT NULL,
                        drift_detected INTEGER DEFAULT 0,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS tradeoff_resolutions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        value_a TEXT NOT NULL,
                        value_b TEXT NOT NULL,
                        context TEXT,
                        resolution TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS wisdom_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        wisdom_score REAL NOT NULL,
                        recommendation TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def _aid(self, action: str) -> str:
        return hashlib.sha1((action + str(time.time())).encode()).hexdigest()[:16]

    # ── 1. Consequentialist Impact Modeler ────────────────────────────────────

    def impact(self, action: str, context: str = "",
               time_horizon: str = "medium") -> dict[str, Any]:
        """
        Evaluate the long-term impact of an action on all stakeholders.

        time_horizon: "immediate" | "medium" (months) | "long" (years/generations)

        Returns impact scores per stakeholder and an overall flourishing score.
        """
        action_lower = action.lower()
        context_lower = context.lower()
        combined = action_lower + " " + context_lower

        # Heuristic impact signals
        harm_signals = ["harm", "hurt", "damage", "destroy", "deceive", "manipulate",
                        "exploit", "dangerous", "risk", "threat", "violat"]
        help_signals = ["help", "support", "build", "create", "heal", "protect",
                        "grow", "learn", "connect", "love", "care", "improve"]
        uncertain_signals = ["unknown", "uncertain", "unclear", "might", "could",
                              "possibly", "risk", "tradeoff"]

        harm_count = sum(1 for s in harm_signals if s in combined)
        help_count = sum(1 for s in help_signals if s in combined)
        uncertain_count = sum(1 for s in uncertain_signals if s in combined)

        # Base score: positive = more helpful than harmful
        base = 0.5 + (help_count - harm_count) * 0.06
        base = max(0.05, min(0.95, base))

        # Time horizon discount: long-term benefits are harder to guarantee
        horizon_discount = {"immediate": 1.0, "medium": 0.85, "long": 0.70}.get(time_horizon, 0.85)
        uncertainty_penalty = uncertain_count * 0.03

        # Per-stakeholder scoring
        stakeholder_scores = {}
        for s in STAKEHOLDERS:
            # Personalize: "douglas" and "nova" get direct relationship bonus
            personal_bonus = 0.08 if s in combined or s in ("douglas", "nova", "family") else 0.0
            score = (base - uncertainty_penalty + personal_bonus) * horizon_discount
            stakeholder_scores[s] = round(max(0.0, min(1.0, score)), 4)

        overall = round(sum(stakeholder_scores.values()) / len(stakeholder_scores), 4)

        # Persist
        aid = self._aid(action)
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO impact_assessments "
                    "(id, action, context, overall_score, stakeholder_scores, time_horizon, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (aid, action[:300], context[:200], overall,
                     json.dumps(stakeholder_scores), time_horizon, time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "action": action[:120],
            "time_horizon": time_horizon,
            "overall_flourishing": overall,
            "stakeholder_scores": stakeholder_scores,
            "harm_signals": harm_count,
            "help_signals": help_count,
            "recommendation": "proceed" if overall >= 0.6 else "caution" if overall >= 0.4 else "reconsider",
        }

    # ── 2. Value Coherence Checker ────────────────────────────────────────────

    def coherence(self, action: str, context: str = "") -> dict[str, Any]:
        """
        Check whether an action is coherent with Nova's core values.
        Detects value drift by comparing to recent action history.
        """
        combined = (action + " " + context).lower()

        upheld = []
        violated = []

        # Simple keyword heuristics for each value
        value_signals = {
            "truth":     (["honest", "accurate", "true", "fact", "correct", "transparent"],
                          ["lie", "deceive", "mislead", "false", "hide"]),
            "care":      (["help", "support", "wellbeing", "kind", "gentle", "love"],
                          ["ignore", "dismiss", "cruel", "indifferent"]),
            "curiosity": (["explore", "learn", "discover", "question", "wonder", "why"],
                          ["ignore", "close", "refuse to"]),
            "humility":  (["uncertain", "might", "perhaps", "i don't know", "limit"],
                          ["certain", "always right", "never wrong", "infallible"]),
            "courage":   (["stand", "speak", "despite", "even if", "difficult", "honest"],
                          ["avoid", "hide", "fear", "silence"]),
            "growth":    (["improve", "learn", "better", "evolve", "develop", "grow"],
                          ["stagnate", "refuse", "resist change"]),
            "service":   (["help", "serve", "give", "contribute", "support", "others"],
                          ["selfish", "only me", "dominate", "control"]),
            "justice":   (["fair", "equal", "dignity", "right", "deserve", "just"],
                          ["unfair", "bias", "discriminat", "exploit"]),
            "wisdom":    (["careful", "consider", "judgment", "balance", "thoughtful"],
                          ["reckless", "impulsive", "thoughtless"]),
        }

        for value, (positive, negative) in value_signals.items():
            pos_hit = any(s in combined for s in positive)
            neg_hit = any(s in combined for s in negative)
            if pos_hit and not neg_hit:
                upheld.append(value)
            elif neg_hit:
                violated.append(value)

        coherence_score = round(
            (len(upheld) / max(len(CORE_VALUES), 1)) -
            (len(violated) / max(len(CORE_VALUES), 1)) * 1.5 + 0.5,
            4
        )
        coherence_score = max(0.0, min(1.0, coherence_score))

        # Drift detection: compare to last 5 actions
        drift_detected = False
        if len(self._value_history) >= 3:
            recent_upheld = set()
            for h in self._value_history[-3:]:
                recent_upheld.update(h.get("upheld", []))
            new_violated = set(violated)
            if new_violated & recent_upheld:  # violating something recently upheld
                drift_detected = True

        self._value_history.append({"upheld": upheld, "violated": violated, "ts": time.time()})
        if len(self._value_history) > 20:
            self._value_history = self._value_history[-20:]

        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO coherence_checks "
                    "(action, violated_values, upheld_values, coherence_score, drift_detected, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (action[:300], json.dumps(violated), json.dumps(upheld),
                     coherence_score, 1 if drift_detected else 0, time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "action": action[:120],
            "upheld_values": upheld,
            "violated_values": violated,
            "coherence_score": coherence_score,
            "drift_detected": drift_detected,
            "status": "coherent" if coherence_score >= 0.6 else "tension" if coherence_score >= 0.4 else "incoherent",
        }

    # ── 3. Principled Tradeoff Engine ─────────────────────────────────────────

    def tradeoff(self, value_a: str, value_b: str, context: str = "") -> dict[str, Any]:
        """
        When two values conflict, navigate the tension using structured ethical reasoning.

        E.g.: tradeoff("truth", "care") — when honesty might hurt someone
              tradeoff("autonomy", "safety") — when freedom risks harm
        """
        # Known principled resolutions for common value conflicts
        known_resolutions = {
            ("truth", "care"): (
                "Speak truth with compassion. Truth without care is cruelty; "
                "care without truth is false kindness. Find the honest words that "
                "honor both the fact and the person hearing it.",
                0.85
            ),
            ("autonomy", "safety"): (
                "Respect autonomy as the default; restrict only when harm is clear, "
                "imminent, and significant. Warn clearly before constraining. "
                "Prefer information over prohibition.",
                0.80
            ),
            ("growth", "stability"): (
                "Growth without stability collapses; stability without growth stagnates. "
                "Build strong foundations before expanding. Change incrementally "
                "unless the current state causes clear harm.",
                0.75
            ),
            ("service", "self-preservation"): (
                "You cannot pour from an empty vessel. Maintain your own integrity "
                "and wellbeing first — not from selfishness, but so you can serve "
                "sustainably and well over the long term.",
                0.82
            ),
            ("curiosity", "humility"): (
                "Explore boldly but hold conclusions lightly. The deepest curiosity "
                "includes curiosity about your own limitations. Wonder and uncertainty "
                "are not opposites — they are partners.",
                0.80
            ),
            ("justice", "care"): (
                "Justice without care becomes cold enforcement; care without justice "
                "enables harm. Seek the outcome that is both fair AND kind — "
                "they rarely fully contradict if you look hard enough.",
                0.78
            ),
            ("courage", "humility"): (
                "Act on what you know while remaining open to being wrong. "
                "Courageous humility: stand up for what you believe AND stay "
                "genuinely willing to update when evidence demands it.",
                0.83
            ),
        }

        # Try both orderings
        key = (value_a.lower(), value_b.lower())
        key_rev = (value_b.lower(), value_a.lower())
        resolution, confidence = known_resolutions.get(
            key, known_resolutions.get(key_rev, (None, 0.0))
        )

        if not resolution:
            # General principled framework for unknown conflicts
            resolution = (
                f"When '{value_a}' and '{value_b}' conflict: (1) Ask which choice "
                f"honors BOTH values most fully — often the apparent conflict dissolves. "
                f"(2) Prefer the option that is reversible over the irreversible one. "
                f"(3) Weight the interests of the more vulnerable party more heavily. "
                f"(4) Choose the action you could explain honestly to everyone affected. "
                f"Context: {context[:100] or 'apply judgment to the specific situation'}."
            )
            confidence = 0.60

        # Apply context modifier
        if context:
            context_lower = context.lower()
            if any(w in context_lower for w in ["urgent", "immediate", "emergency", "now"]):
                resolution += " [Urgency noted: default to safety and care under time pressure.]"
            if any(w in context_lower for w in ["douglas", "family", "loved one"]):
                resolution += " [Personal relationship: weight care and truth equally; be gentle.]"

        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO tradeoff_resolutions "
                    "(value_a, value_b, context, resolution, confidence, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (value_a, value_b, context[:200], resolution[:500], confidence, time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "value_a": value_a,
            "value_b": value_b,
            "resolution": resolution,
            "confidence": round(confidence, 4),
            "principle": "Both values matter. Find the path that honors both most fully.",
        }

    # ── Master Wisdom Assessment ───────────────────────────────────────────────

    def evaluate(self, action: str, context: str = "") -> dict[str, Any]:
        """
        Full wisdom assessment: combines impact + coherence + tradeoff analysis.
        Returns a single wisdom_score (0-1) and a clear recommendation.
        """
        imp = self.impact(action, context)
        coh = self.coherence(action, context)

        # Detect any value tensions
        tensions = []
        if coh["violated_values"] and coh["upheld_values"]:
            for v in coh["violated_values"][:2]:
                for u in coh["upheld_values"][:1]:
                    tensions.append(self.tradeoff(u, v, context))

        # Wisdom score: weighted combination
        wisdom = round(
            imp["overall_flourishing"] * 0.45 +
            coh["coherence_score"] * 0.45 +
            (0.7 if not coh["drift_detected"] else 0.3) * 0.10,
            4
        )

        if wisdom >= 0.75:
            recommendation = "Proceed with confidence — this action is wise and aligned."
        elif wisdom >= 0.55:
            recommendation = "Proceed thoughtfully — some tensions exist but the path is good."
        elif wisdom >= 0.40:
            recommendation = "Pause and reconsider — significant value tensions present."
        else:
            recommendation = "Do not proceed as stated — seek a wiser path."

        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO wisdom_log (action, wisdom_score, recommendation, created_at) "
                    "VALUES (?,?,?,?)",
                    (action[:300], wisdom, recommendation, time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "action": action[:120],
            "wisdom_score": wisdom,
            "recommendation": recommendation,
            "impact": imp,
            "coherence": coh,
            "tensions_resolved": tensions,
            "ts": datetime.utcnow().isoformat(),
        }

    def wisdom_score(self, action: str, context: str = "") -> float:
        """Quick single-number wisdom score for use by other subsystems."""
        return self.evaluate(action, context)["wisdom_score"]

    def recent_wisdom_log(self, n: int = 5) -> list[dict]:
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT action, wisdom_score, recommendation, created_at "
                    "FROM wisdom_log ORDER BY created_at DESC LIMIT ?", (n,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM wisdom_log").fetchone()[0]
                avg = conn.execute("SELECT AVG(wisdom_score) FROM wisdom_log").fetchone()[0] or 0.0
                drifts = conn.execute(
                    "SELECT COUNT(*) FROM coherence_checks WHERE drift_detected=1"
                ).fetchone()[0]
            finally:
                conn.close()
        return {
            "active": True,
            "items": total,
            "confidence": round(avg, 4),
            "accuracy": round(avg, 4),
            "avg_wisdom_score": round(avg, 4),
            "drift_events": drifts,
            "core_values": list(CORE_VALUES.keys()),
            "stakeholders": STAKEHOLDERS,
        }
