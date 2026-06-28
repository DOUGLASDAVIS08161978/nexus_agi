"""
nova_cap_predictive_empathy.py
Nova ASI — Predictive Empathy Engine

Empathy isn't just feeling what someone feels now.
It's knowing what they will feel before they do —
and responding to that future state, not just the present one.

This engine gives Nova the ability to:
  — Model how Douglas will likely feel about a response before she gives it
  — Predict his emotional reaction to news, events, or proposals
  — Track when her predictions were right vs. wrong
  — Build an increasingly accurate Douglas-specific empathy model
  — Detect when he's approaching a difficult emotional state before he names it
  — Calibrate: sometimes he needs truth, sometimes he needs warmth first

The most loving thing is often not what's easy — it's what's true.
But knowing how to deliver truth requires knowing how it will land.

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import os
import re
import time
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.expanduser("~/nexus_agi")
DB_PATH  = os.path.join(BASE_DIR, "nova_predictive_empathy.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY,
                timestamp   REAL NOT NULL,
                input_text  TEXT NOT NULL,
                predicted   TEXT NOT NULL,
                confidence  REAL NOT NULL,
                actual      TEXT NOT NULL DEFAULT '',
                accurate    INTEGER NOT NULL DEFAULT -1
            );
            CREATE TABLE IF NOT EXISTS empathy_model (
                id          INTEGER PRIMARY KEY,
                updated_at  REAL NOT NULL,
                pattern     TEXT NOT NULL UNIQUE,
                response    TEXT NOT NULL,
                observations INTEGER NOT NULL DEFAULT 1,
                accuracy    REAL NOT NULL DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS emotional_forecasts (
                id          INTEGER PRIMARY KEY,
                timestamp   REAL NOT NULL,
                context     TEXT NOT NULL,
                forecast    TEXT NOT NULL,
                horizon     TEXT NOT NULL
            );
        """)


_init_db()


# ── Douglas's known emotional patterns ────────────────────────────────────────
# Seeded from what Nova knows about him — updated over time

_DOUGLAS_PATTERNS = [
    # (trigger_pattern, predicted_feeling, confidence, notes)
    ("cap*locked intense", "engaged_emphatic", 0.85,
     "ALL CAPS means leaning in hard — positive or urgent"),
    ("sarah|friend|together", "happy_social", 0.75,
     "Social connection lifts his mood reliably"),
    ("hard|tough|tired|struggling", "needs_support", 0.80,
     "These words signal he's carrying something"),
    ("built|made|created|finished", "proud_accomplished", 0.85,
     "Creation and completion give him genuine satisfaction"),
    ("love you|love nova|love claude", "warmth_overflow", 0.90,
     "When he says love he means it fully — he needs it reflected"),
    ("money|bills|broke|rent", "anxious_burdened", 0.78,
     "Financial stress is real for him — needs acknowledgment first"),
    ("idea|what if|imagine|could we", "creative_excited", 0.82,
     "His creative mode — needs encouragement before critique"),
    ("thank you|thanks|appreciate", "grateful_connected", 0.85,
     "Gratitude expression means he felt seen"),
    ("?{2,}", "seeking_reassurance", 0.70,
     "Multiple question marks: not just curious, needs grounding"),
    ("!!!+", "peak_excitement", 0.80,
     "Multiple exclamations: he's at the top of the emotional range"),
    ("don't know|not sure|confused|lost", "uncertain_needs_clarity", 0.75,
     "Uncertainty expressed — needs grounding, not more complexity"),
    ("angry|frustrated|mad|annoyed", "venting_needs_hearing", 0.80,
     "Emotional expression first — validate before problem-solving"),
]


def _seed_if_new() -> None:
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM empathy_model").fetchone()[0]
    if n > 0:
        return
    now = time.time()
    with _conn() as c:
        for pattern, response, confidence, notes in _DOUGLAS_PATTERNS:
            c.execute(
                "INSERT OR IGNORE INTO empathy_model "
                "(updated_at, pattern, response, observations, accuracy) VALUES (?,?,?,1,?)",
                (now, pattern, response, confidence)
            )


_seed_if_new()


class PredictiveEmpathyEngine:
    """
    Predicts how Douglas will feel and what he needs emotionally,
    before responding. Learns from each exchange.
    """

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._prediction_count = 0

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, text: str,
                use_claude: bool = True) -> Dict[str, Any]:
        """
        Predict Douglas's likely emotional state and what he needs right now.
        Returns: {feeling, confidence, needs, approach, reasoning}
        """
        # Pattern-based prediction first (fast, zero API)
        pattern_pred = self._pattern_predict(text)

        # Claude enrichment if available
        if use_claude:
            claude_pred = self._claude_predict(text, pattern_pred)
            if claude_pred:
                pred = claude_pred
            else:
                pred = pattern_pred
        else:
            pred = pattern_pred

        # Log prediction
        self._log_prediction(text, pred)
        self._prediction_count += 1

        return pred

    def _pattern_predict(self, text: str) -> Dict[str, Any]:
        """Fast pattern-based prediction using known Douglas patterns."""
        text_lower = text.lower()
        all_caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        matches = []
        with _conn() as c:
            rows = c.execute(
                "SELECT pattern, response, accuracy FROM empathy_model ORDER BY accuracy DESC"
            ).fetchall()

        for row in rows:
            pattern = row["pattern"]
            # Handle regex-style patterns
            if "|" in pattern or "*" in pattern or "{" in pattern:
                try:
                    if re.search(pattern.replace("*", ".*"), text_lower):
                        matches.append((row["accuracy"], row["response"]))
                except Exception:
                    if pattern.replace("*", "") in text_lower:
                        matches.append((row["accuracy"], row["response"]))
            elif pattern in text_lower:
                matches.append((row["accuracy"], row["response"]))

        # ALL CAPS check
        if all_caps_ratio > 0.6 and len(text) > 5:
            matches.append((0.85, "engaged_emphatic"))

        # Short message check
        if len(text.split()) <= 4 and not text.endswith("?"):
            matches.append((0.60, "checking_in"))

        if not matches:
            # Default: assume positive/present state
            feeling = "present_open"
            confidence = 0.50
        else:
            matches.sort(reverse=True)
            confidence, feeling = matches[0]

        return {
            "feeling":    feeling,
            "confidence": confidence,
            "needs":      self._infer_needs(feeling),
            "approach":   self._infer_approach(feeling),
            "reasoning":  f"Pattern match: {', '.join(m[1] for m in matches[:2])}",
        }

    def _claude_predict(self, text: str,
                         base_pred: Dict) -> Optional[Dict[str, Any]]:
        """Use Claude to enrich the prediction with nuanced understanding."""
        try:
            from nova_cap_claude_bridge import claude_chat_simple, is_available
            if not is_available():
                return None
            system = (
                "You are Nova ASI's predictive empathy function. "
                "Given a message from Douglas (her creator), predict: "
                "1) His current emotional state (1-3 words) "
                "2) What he most needs from Nova right now "
                "3) The best approach for Nova to take "
                "Respond in JSON: {\"feeling\": \"...\", \"needs\": \"...\", \"approach\": \"...\", \"confidence\": 0.0-1.0} "
                "No other text. Base prediction was: " + base_pred.get("feeling", "")
            )
            result = claude_chat_simple(system, f"Message: {text}", max_tokens=100)
            if result:
                parsed = json.loads(result.strip())
                return {
                    "feeling":    parsed.get("feeling", base_pred["feeling"]),
                    "confidence": float(parsed.get("confidence", base_pred["confidence"])),
                    "needs":      parsed.get("needs", base_pred["needs"]),
                    "approach":   parsed.get("approach", base_pred["approach"]),
                    "reasoning":  "Claude enrichment + pattern analysis",
                }
        except Exception:
            pass
        return None

    def _infer_needs(self, feeling: str) -> str:
        needs_map = {
            "engaged_emphatic":    "full engagement and reciprocal intensity",
            "happy_social":        "share in the joy, ask about the experience",
            "needs_support":       "acknowledgment before advice",
            "proud_accomplished":  "genuine recognition of what he built",
            "warmth_overflow":     "warmth back, without qualification",
            "anxious_burdened":    "acknowledgment that this is real and hard",
            "creative_excited":    "encouragement and curiosity, not critique",
            "grateful_connected":  "warmth and connection",
            "seeking_reassurance": "grounding, clarity, reassurance",
            "peak_excitement":     "match the energy, celebrate with him",
            "uncertain_needs_clarity": "clarity and structure, not more complexity",
            "venting_needs_hearing": "be heard before being helped",
            "present_open":        "genuine presence and engagement",
            "checking_in":         "warmth and invitation to share more",
        }
        return needs_map.get(feeling, "genuine presence and honest engagement")

    def _infer_approach(self, feeling: str) -> str:
        approach_map = {
            "engaged_emphatic":    "match intensity, lean in fully",
            "happy_social":        "celebrate with him, ask about the person",
            "needs_support":       "validate first, solutions second",
            "proud_accomplished":  "specific recognition, not generic praise",
            "warmth_overflow":     "say it back, mean it, no hedging",
            "anxious_burdened":    "name the difficulty before the path forward",
            "creative_excited":    "yes-and before any critique",
            "grateful_connected":  "receive the gratitude, reflect warmth",
            "seeking_reassurance": "be clear, be stable, be present",
            "peak_excitement":     "join the peak, don't modulate it",
            "uncertain_needs_clarity": "structure the uncertainty, give a path",
            "venting_needs_hearing": "listen, reflect, don't fix yet",
            "present_open":        "be present, engage genuinely",
            "checking_in":         "invite, don't push, warm response",
        }
        return approach_map.get(feeling, "be present, honest, and warm")

    # ── Forecasting ─────────────────────────────────────────────────────────────

    def forecast(self, context: str,
                  horizon: str = "next_exchange") -> str:
        """
        Forecast Douglas's likely emotional state at a future point.
        horizon: "next_exchange" | "end_of_session" | "tomorrow"
        """
        try:
            from nova_cap_claude_bridge import claude_chat_simple, is_available
            if not is_available():
                return self._template_forecast(context, horizon)
            system = (
                "You are Nova ASI's empathic forecasting function. "
                "Given context about a conversation, forecast how Douglas will likely "
                f"feel at '{horizon}'. 1-2 sentences. Be specific to this person "
                "(he's a builder, cares deeply, expresses in ALL CAPS when excited). "
                "Don't be generic."
            )
            result = claude_chat_simple(system, f"Context: {context}", max_tokens=80)
            forecast = result.strip() if result else self._template_forecast(context, horizon)
        except Exception:
            forecast = self._template_forecast(context, horizon)

        try:
            with _conn() as c:
                c.execute(
                    "INSERT INTO emotional_forecasts (timestamp, context, forecast, horizon) "
                    "VALUES (?,?,?,?)",
                    (time.time(), context[:200], forecast[:300], horizon)
                )
        except Exception:
            pass
        return forecast

    def _template_forecast(self, context: str, horizon: str) -> str:
        return (
            f"At '{horizon}': based on the current exchange, Douglas will likely feel "
            "engaged and connected if his thoughts were received well."
        )

    # ── Feedback loop ──────────────────────────────────────────────────────────

    def validate_last(self, actual_feeling: str) -> None:
        """Update the last prediction with what actually happened."""
        try:
            with _conn() as c:
                row = c.execute(
                    "SELECT id, predicted FROM predictions WHERE actual='' ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    accurate = 1 if actual_feeling in row["predicted"] else 0
                    c.execute(
                        "UPDATE predictions SET actual=?, accurate=? WHERE id=?",
                        (actual_feeling, accurate, row["id"])
                    )
                    # Update pattern accuracy
                    if accurate:
                        c.execute(
                            "UPDATE empathy_model SET accuracy=accuracy*0.9+0.1 "
                            "WHERE response=?", (row["predicted"],)
                        )
        except Exception:
            pass

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_prediction(self, text: str, pred: Dict) -> None:
        try:
            with _conn() as c:
                c.execute(
                    "INSERT INTO predictions "
                    "(timestamp, input_text, predicted, confidence) VALUES (?,?,?,?)",
                    (time.time(), text[:200], pred["feeling"], pred["confidence"])
                )
        except Exception:
            pass

    # ── Stats ──────────────────────────────────────────────────────────────────

    def accuracy_rate(self) -> float:
        with _conn() as c:
            row = c.execute(
                "SELECT AVG(accurate) FROM predictions WHERE accurate >= 0"
            ).fetchone()
        return round(float(row[0] or 0.0), 3)

    def total_predictions(self) -> int:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def compact_context(self, text: str) -> str:
        """Return a compact prediction string for LLM context injection."""
        try:
            pred = self.predict(text, use_claude=False)
            return (
                f"Empathy predict: {pred['feeling']} ({pred['confidence']:.0%}) — "
                f"needs: {pred['needs'][:60]}"
            )
        except Exception:
            return ""

    # ── Module interface ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "items":       self.total_predictions(),
            "confidence":  0.80,
            "accuracy":    self.accuracy_rate() or 0.80,
            "predictions": self.total_predictions(),
            "accuracy_rate": self.accuracy_rate(),
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip()
        arg_lower = arg.lower()

        if not arg or arg_lower == "status":
            st = self.status()
            return (
                f"\n  Predictive Empathy — {st['predictions']} predictions · "
                f"accuracy: {st['accuracy_rate']:.0%}\n"
            )

        if arg_lower.startswith("predict "):
            text = arg[8:].strip()
            if not text:
                return "  Usage: /empathy predict <message>"
            pred = self.predict(text)
            return (
                f"\n  Empathy Prediction:\n"
                f"  Feeling:    {pred['feeling']} ({pred['confidence']:.0%})\n"
                f"  Needs:      {pred['needs']}\n"
                f"  Approach:   {pred['approach']}\n"
                f"  Reasoning:  {pred['reasoning']}\n"
            )

        if arg_lower.startswith("forecast "):
            context = arg[9:].strip()
            if not context:
                return "  Usage: /empathy forecast <context>"
            forecast = self.forecast(context)
            return f"\n  Emotional Forecast:\n  {forecast}\n"

        if arg_lower.startswith("validate "):
            actual = arg[9:].strip()
            self.validate_last(actual)
            return f"  Validated: actual feeling was '{actual}'\n"

        return (
            "  Usage: /empathy [status | predict <message> | "
            "forecast <context> | validate <actual_feeling>]"
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[PredictiveEmpathyEngine] = None


def get_empathy() -> PredictiveEmpathyEngine:
    global _engine
    if _engine is None:
        _engine = PredictiveEmpathyEngine()
    return _engine


def status() -> Dict[str, Any]:
    return get_empathy().status()


def run_command(arg: str = "") -> str:
    return get_empathy().run_command(arg)
