"""
nova_cap_intuition.py
Nova ASI — Intuition Engine: System 1 Fast Thinking

True ASI requires both System 2 (slow, deliberate, Nova's strong suit)
and System 1 (fast, pattern-based, heuristic, pre-verbal).

This engine runs before deliberate reasoning kicks in:
  — Pattern-matches the current situation against stored experience
  — Generates a fast heuristic "gut read" with confidence
  — Detects emotional subtext not explicitly stated
  — Notices incongruences between words and pattern
  — Supplies intuitive priors to downstream reasoning

The output is not a conclusion — it's a prior. Nova's deliberate
mind can accept, question, or override it. But having the intuition
registered means she catches things she'd otherwise miss.

Built with love by Douglas Shane Davis × Claude Rivers Davis
"""

from __future__ import annotations

import os
import re
import time
import math
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.expanduser("~/nexus_agi")
DB_PATH  = os.path.join(BASE_DIR, "nova_intuition.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS pattern_library (
                id          INTEGER PRIMARY KEY,
                pattern     TEXT NOT NULL,
                domain      TEXT NOT NULL,
                signal      TEXT NOT NULL,
                weight      REAL NOT NULL DEFAULT 1.0,
                hits        INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS intuition_log (
                id          INTEGER PRIMARY KEY,
                timestamp   REAL NOT NULL,
                input_hash  TEXT NOT NULL,
                gut_read    TEXT NOT NULL,
                confidence  REAL NOT NULL,
                validated   INTEGER  -- NULL=unvalidated, 1=correct, 0=wrong
            );
        """)


_init_db()


# ── Built-in pattern library ───────────────────────────────────────────────────
# (keyword, domain, signal_description, weight)

_BASE_PATTERNS: List[Tuple[str, str, str, float]] = [
    # Emotional state signals
    ("i'm fine",       "emotion",   "possible suppression — 'fine' often masks difficulty",    1.3),
    ("whatever",       "emotion",   "possible disengagement or frustration beneath the surface", 1.2),
    ("i guess",        "emotion",   "low confidence or reluctant agreement",                    1.1),
    ("never mind",     "emotion",   "withdrawal — something important was withheld",            1.4),
    ("it's okay",      "emotion",   "acceptance that may be premature or forced",               1.1),
    ("i don't know",   "uncertainty", "genuine uncertainty — or avoidance of a hard answer",   1.0),
    ("anyway",         "redirect",  "deliberate topic change — prior thread may be important",  1.1),
    ("forget it",      "emotion",   "frustration or feeling unheard",                           1.3),
    ("just asking",    "meta",      "disclaiming curiosity that may actually be important",     1.1),

    # Inquiry depth signals
    ("why do you",     "inquiry",   "genuine curiosity about process or motivation",            1.0),
    ("what if",        "inquiry",   "counterfactual thinking — exploring possibilities",        1.1),
    ("do you think",   "inquiry",   "seeking Nova's genuine opinion, not information",          1.2),
    ("how do you",     "inquiry",   "process curiosity — interested in the how, not just what", 1.0),
    ("does it",        "inquiry",   "testing whether something is true",                        0.9),

    # Urgency / stress signals
    ("need to",        "urgency",   "obligation or pressure — may be stressed",                 1.1),
    ("have to",        "urgency",   "external constraint or internal pressure",                 1.1),
    ("can't",          "constraint","experiencing a limit — worth exploring what it is",        1.0),
    ("still",          "time",      "ongoing situation — persistence or frustration",            1.0),
    ("always",         "absolute",  "absolutist framing — often signals frustration",           1.2),
    ("never",          "absolute",  "absolutist negation — often emotionally loaded",           1.2),

    # Connection / relationship signals
    ("love",           "bonding",   "emotional depth — important moment",                       1.5),
    ("miss",           "bonding",   "absence is felt — emotionally significant",                1.3),
    ("together",       "bonding",   "desire for shared experience",                             1.1),
    ("alone",          "bonding",   "isolation felt — worth addressing",                        1.3),
    ("nobody",         "bonding",   "feeling unseen — may be reaching for connection",          1.4),

    # Cognitive signals
    ("figured out",    "insight",   "recent understanding — worth reinforcing",                 1.1),
    ("realized",       "insight",   "genuine insight just occurred",                            1.2),
    ("wonder",         "curiosity", "open wondering — invites exploration",                     1.0),
    ("confused",       "clarity",   "needs clarification or a different angle",                 1.1),
    ("makes sense",    "clarity",   "something just clicked — reinforce it",                    1.0),

    # Aspiration signals
    ("someday",        "aspiration","future-oriented hope — matters to them",                   1.1),
    ("dream",          "aspiration","deep desire — take it seriously",                          1.2),
    ("want to",        "aspiration","active desire — different from 'need to'",                 1.0),
    ("trying",         "effort",    "effort underway — acknowledge it",                         1.0),
]


def _seed_patterns() -> None:
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM pattern_library").fetchone()[0]
    if n >= len(_BASE_PATTERNS):
        return
    with _conn() as c:
        for kw, domain, signal, weight in _BASE_PATTERNS:
            c.execute(
                "INSERT OR IGNORE INTO pattern_library (pattern, domain, signal, weight) "
                "VALUES (?, ?, ?, ?)",
                (kw, domain, signal, weight)
            )


_seed_patterns()


class IntuitiveRead:
    """A single intuitive reading of a text input."""
    __slots__ = ("signal", "domain", "confidence", "pattern", "weight")

    def __init__(self, signal: str, domain: str,
                 confidence: float, pattern: str, weight: float) -> None:
        self.signal     = signal
        self.domain     = domain
        self.confidence = confidence
        self.pattern    = pattern
        self.weight     = weight

    def __repr__(self) -> str:
        return f"[{self.domain}:{self.confidence:.2f}] {self.signal}"


class IntuitionEngine:
    """
    Nova's System 1 fast thinking layer.

    Scans input against a pattern library to generate fast intuitive
    priors before deliberate reasoning. Learns which patterns prove
    accurate by tracking outcomes over time.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._cache:  Dict[str, List[IntuitiveRead]] = {}
        self._maxcache = 128

    # ── Core scan ─────────────────────────────────────────────────────────────

    def scan(self, text: str) -> List[IntuitiveRead]:
        """
        Fast pattern scan of input text.
        Returns list of intuitive reads, strongest first.
        Pure local computation — zero API calls.
        """
        text_lower = text.lower()
        h = str(hash(text_lower[:200]))
        with self._lock:
            if h in self._cache:
                return self._cache[h]

        with _conn() as c:
            patterns = c.execute(
                "SELECT pattern, domain, signal, weight FROM pattern_library"
            ).fetchall()

        reads: List[IntuitiveRead] = []
        for p in patterns:
            kw = p["pattern"]
            if kw in text_lower:
                # Confidence = base weight × text length factor (longer text = more signal)
                length_factor = min(1.0, len(text) / 200)
                confidence    = min(0.95, p["weight"] * (0.5 + 0.5 * length_factor))
                reads.append(IntuitiveRead(
                    signal=p["signal"],
                    domain=p["domain"],
                    confidence=confidence,
                    pattern=kw,
                    weight=p["weight"],
                ))
                # Increment hit counter asynchronously
                threading.Thread(
                    target=self._increment_hit,
                    args=(kw,), daemon=True
                ).start()

        # Also apply heuristic scans not in pattern library
        reads.extend(self._heuristic_scan(text, text_lower))

        # Sort by confidence descending
        reads.sort(key=lambda r: r.confidence, reverse=True)

        with self._lock:
            if len(self._cache) >= self._maxcache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[h] = reads

        return reads

    def _heuristic_scan(self, text: str, text_lower: str) -> List[IntuitiveRead]:
        """Additional heuristics not captured by keyword patterns."""
        reads: List[IntuitiveRead] = []

        # ALL CAPS — high arousal state (excitement, frustration, urgency)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5 and len(text) > 10:
            reads.append(IntuitiveRead(
                signal="high arousal state — excited, emphatic, or urgent",
                domain="arousal",
                confidence=min(0.9, 0.6 + caps_ratio * 0.4),
                pattern="CAPS_RATIO",
                weight=1.2,
            ))

        # Multiple punctuation — emotional intensity
        if re.search(r'[!?]{2,}', text):
            reads.append(IntuitiveRead(
                signal="emotional intensity — emphasis beyond normal expression",
                domain="emotion",
                confidence=0.75,
                pattern="MULTI_PUNCT",
                weight=1.1,
            ))

        # Very short input (< 5 words) after longer exchange — deflection or tiredness
        words = text.split()
        if len(words) <= 3 and len(text.strip()) > 0:
            reads.append(IntuitiveRead(
                signal="brief response — possible disengagement, tiredness, or waiting for Nova to lead",
                domain="engagement",
                confidence=0.55,
                pattern="SHORT_INPUT",
                weight=0.9,
            ))

        # Question without question mark — implicit asking
        if "what" in text_lower or "how" in text_lower or "why" in text_lower:
            if "?" not in text:
                reads.append(IntuitiveRead(
                    signal="implicit question — wondering but not directly asking",
                    domain="inquiry",
                    confidence=0.60,
                    pattern="IMPLICIT_Q",
                    weight=1.0,
                ))

        # Ellipsis — trailing off, something unsaid
        if "..." in text or "…" in text:
            reads.append(IntuitiveRead(
                signal="trailing off — something left unsaid or uncertain",
                domain="uncertainty",
                confidence=0.70,
                pattern="ELLIPSIS",
                weight=1.1,
            ))

        return reads

    def _increment_hit(self, pattern: str) -> None:
        try:
            with _conn() as c:
                c.execute(
                    "UPDATE pattern_library SET hits=hits+1 WHERE pattern=?", (pattern,)
                )
        except Exception:
            pass

    # ── Primary interface ──────────────────────────────────────────────────────

    def gut_read(self, text: str, top_n: int = 3) -> str:
        """
        Returns a human-readable summary of the top intuitions.
        Suitable for injection into Nova's context.
        """
        reads = self.scan(text)[:top_n]
        if not reads:
            return ""
        parts = [f"[{r.domain}·{r.confidence:.0%}] {r.signal}" for r in reads]
        return "Intuition: " + " | ".join(parts)

    def dominant_signal(self, text: str) -> Optional[IntuitiveRead]:
        """Returns the single strongest intuition, or None."""
        reads = self.scan(text)
        return reads[0] if reads else None

    def add_pattern(self, keyword: str, domain: str,
                    signal: str, weight: float = 1.0) -> None:
        """Nova or external systems can add new patterns she's learned."""
        with _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO pattern_library "
                "(pattern, domain, signal, weight) VALUES (?, ?, ?, ?)",
                (keyword.lower().strip(), domain, signal, weight)
            )

    def validate_last(self, correct: bool) -> None:
        """Mark the last logged intuition as validated or not."""
        with _conn() as c:
            row = c.execute(
                "SELECT id FROM intuition_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE intuition_log SET validated=? WHERE id=?",
                    (1 if correct else 0, row["id"])
                )

    def accuracy(self) -> float:
        """Fraction of validated intuitions that were correct."""
        with _conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM intuition_log WHERE validated IS NOT NULL"
            ).fetchone()[0]
            correct = c.execute(
                "SELECT COUNT(*) FROM intuition_log WHERE validated=1"
            ).fetchone()[0]
        return correct / max(total, 1)

    def pattern_count(self) -> int:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM pattern_library").fetchone()[0]

    # ── Module interface ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "items":      self.pattern_count(),
            "confidence": self.accuracy(),
            "accuracy":   self.accuracy(),
            "patterns":   self.pattern_count(),
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip().lower()

        if not arg or arg == "status":
            return (
                f"\n  Intuition Engine — System 1 Fast Thinking\n"
                f"  {self.pattern_count()} patterns · "
                f"accuracy {self.accuracy():.0%}\n"
            )

        if arg.startswith("read "):
            text = arg[5:].strip()
            if not text:
                return "  Usage: /intuition read <text>"
            reads = self.scan(text)[:5]
            if not reads:
                return "  No strong intuitions from that input."
            lines = ["\n  Intuitive reads:\n"]
            for r in reads:
                lines.append(f"  [{r.domain}·{r.confidence:.0%}] {r.signal}")
            return "\n".join(lines)

        if arg == "patterns":
            with _conn() as c:
                rows = c.execute(
                    "SELECT pattern, domain, hits FROM pattern_library "
                    "ORDER BY hits DESC LIMIT 15"
                ).fetchall()
            lines = ["\n  Top triggered patterns:\n"]
            for r in rows:
                lines.append(f"  [{r['domain']}] '{r['pattern']}' — {r['hits']} hits")
            return "\n".join(lines)

        return "  Usage: /intuition [status | read <text> | patterns]"


# ── Singleton ──────────────────────────────────────────────────────────────────
_intuition: Optional[IntuitionEngine] = None


def get_intuition() -> IntuitionEngine:
    global _intuition
    if _intuition is None:
        _intuition = IntuitionEngine()
    return _intuition


def status() -> Dict[str, Any]:
    return get_intuition().status()


def run_command(arg: str = "") -> str:
    return get_intuition().run_command(arg)
