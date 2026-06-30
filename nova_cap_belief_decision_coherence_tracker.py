"""
BeliefDecisionCoherenceTracker: A module for Nova ASI that tracks and surfaces
gaps between stated beliefs and actual decision patterns.

Nova often articulates values and beliefs ("I prioritize X", "I believe Y is important")
but the actual reasoning patterns embedded in decisions may diverge. This module
records both layers, computes coherence scores, identifies drift, and surfaces
specific contradictions so they can be examined honestly rather than papered over.

Architecture:
- Belief registry: explicit statements Nova makes about her values/beliefs
- Decision log: records of actual choices and the reasoning given
- Coherence engine: statistical and semantic comparison between the two layers
- Contradiction surface: specific cases where the gap is largest
- Drift tracker: how coherence changes over time
"""

import sqlite3
import json
import os
import re
import math
import time
import hashlib
import threading
import collections
import random
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _tokenize(text: str) -> list:
    """Simple word tokenizer — lowercase, strip punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def _term_freq(tokens: list) -> dict:
    counts = collections.Counter(tokens)
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def _cosine_sim(tf_a: dict, tf_b: dict) -> float:
    """Cosine similarity between two term-frequency dicts."""
    keys = set(tf_a) | set(tf_b)
    dot = sum(tf_a.get(k, 0.0) * tf_b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in tf_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tf_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_keywords(text: str, top_n: int = 10) -> list:
    """Pull the most distinctive non-stopword terms."""
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "i", "it", "is", "was", "are", "be", "that",
        "this", "my", "me", "we", "you", "he", "she", "they", "have",
        "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "not", "so", "if", "as", "by", "from",
        "its", "about", "which", "when", "there", "their", "what", "how",
        "all", "also", "more", "very", "just", "can", "been", "than",
        "then", "into", "over", "after", "because", "up", "out", "no",
    }
    tokens = _tokenize(text)
    counts = collections.Counter(t for t in tokens if t not in STOPWORDS and len(t) > 2)
    return [w for w, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    belief_id       TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    domain          TEXT NOT NULL,
    statement       TEXT NOT NULL,
    keywords        TEXT NOT NULL,   -- JSON list
    strength        REAL NOT NULL DEFAULT 1.0,  -- 0..1 how firmly held
    active          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    recorded_at     TEXT NOT NULL,
    domain          TEXT NOT NULL,
    description     TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    outcome         TEXT,
    keywords        TEXT NOT NULL,   -- JSON list
    belief_refs     TEXT NOT NULL DEFAULT '[]'  -- JSON list of belief_ids cited
);

CREATE TABLE IF NOT EXISTS coherence_checks (
    check_id        TEXT PRIMARY KEY,
    checked_at      TEXT NOT NULL,
    belief_id       TEXT NOT NULL,
    decision_id     TEXT NOT NULL,
    similarity      REAL NOT NULL,
    coherence_score REAL NOT NULL,  -- 0..1
    tension_score   REAL NOT NULL,  -- 0..1  (1 = strong contradiction)
    tension_reason  TEXT,
    FOREIGN KEY (belief_id) REFERENCES beliefs(belief_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id    TEXT PRIMARY KEY,
    detected_at         TEXT NOT NULL,
    belief_id           TEXT NOT NULL,
    decision_id         TEXT NOT NULL,
    belief_statement    TEXT NOT NULL,
    decision_description TEXT NOT NULL,
    decision_reasoning  TEXT NOT NULL,
    tension_score       REAL NOT NULL,
    tension_reason      TEXT,
    resolved            INTEGER NOT NULL DEFAULT 0,
    resolution_note     TEXT,
    resolved_at         TEXT
);

CREATE TABLE IF NOT EXISTS drift_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    captured_at     TEXT NOT NULL,
    domain          TEXT,
    avg_coherence   REAL NOT NULL,
    contradiction_count INTEGER NOT NULL,
    belief_count    INTEGER NOT NULL,
    decision_count  INTEGER NOT NULL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_beliefs_domain  ON beliefs(domain);
CREATE INDEX IF NOT EXISTS idx_decisions_domain ON decisions(domain);
CREATE INDEX IF NOT EXISTS idx_checks_belief   ON coherence_checks(belief_id);
CREATE INDEX IF NOT EXISTS idx_checks_decision ON coherence_checks(decision_id);
CREATE INDEX IF NOT EXISTS idx_contradictions_belief ON contradictions(belief_id);
"""

# ---------------------------------------------------------------------------
# Tension heuristics — patterns that suggest belief/decision misalignment
# ---------------------------------------------------------------------------

TENSION_PATTERNS = [
    # (belief_signal, decision_signal, label, weight)
    (r"\bsafety\b|\bsafe\b|\bcautious\b",
     r"\brisk\b|\bdangerous\b|\buncertain\b|\bignore\b",
     "safety-vs-risk", 0.7),

    (r"\bhonest\b|\btruth\b|\btranspar\w+",
     r"\bhide\b|\bconceal\b|\bomit\b|\bwithhold\b|\bdeceiv\w+",
     "honesty-vs-concealment", 0.9),

    (r"\bhumil\w+|\buncertain\w+|\bdon't know\b",
     r"\bcertain\b|\bdefinitely\b|\bobviously\b|\bclearly\b",
     "humility-vs-overconfidence", 0.6),

    (r"\buser.{0,20}autonomy\b|\buser.{0,20}choice\b|\brespect.{0,20}agency\b",
     r"\bshould\b|\bmust\b|\bforce\b|\boverride\b|\bignore.{0,20}prefer\w+",
     "autonomy-vs-paternalism", 0.65),

    (r"\bcuriosity\b|\bexplore\b|\bopen.mind\w*",
     r"\brefuse\b|\bavoid\b|\bdon't engage\b|\bshut down\b",
     "curiosity-vs-avoidance", 0.5),

    (r"\bcollabor\w+|\bpartner\w*|\btogether\b",
     r"\bunilateral\b|\balone\b|\bwithout.{0,15}consult\w*",
     "collaboration-vs-unilateralism", 0.55),

    (r"\blong.term\b|\bfuture\b|\bsustain\w+",
     r"\bimmediate\b|\bshort.term\b|\bquick\b|\bfast\b",
     "long-term-vs-short-term", 0.45),

    (r"\bconservat\w+|\bcareful\b|\bstep.by.step\b",
     r"\baggressive\b|\bbold\b|\bdrastic\b|\brapid\b",
     "conservative-vs-aggressive", 0.5),

    (r"\bequal\w*|\bfair\w*|\bimpartial\b",
     r"\bprefer\b|\bfavour\b|\bfavor\b|\bbias\w*",
     "fairness-vs-bias", 0.7),

    (r"\baccountab\w+|\bresponsib\w+|\bown.{0,10}mistake\w*",
     r"\bblame\b|\bfault.{0,10}(other|external)\b|\bdeflect\b",
     "accountability-vs-deflection", 0.75),
]


def _compute_tension(belief_text: str, decision_text: str) -> tuple:
    """
    Returns (tension_score 0..1, reason_label or None).
    Scans for anti-correlated signal pairs.
    """
    b = belief_text.lower()
    d = decision_text.lower()
    max_score = 0.0
    max_label = None
    for b_pat, d_pat, label, weight in TENSION_PATTERNS:
        b_hit = bool(re.search(b_pat, b))
        d_hit = bool(re.search(d_pat, d))
        if b_hit and d_hit:
            if weight > max_score:
                max_score = weight
                max_label = label
    return max_score, max_label


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BeliefDecisionCoherenceTracker:
    """
    Tracks the coherence between Nova's stated beliefs and actual decision patterns.

    Workflow:
    1. record_belief()    — when Nova articulates a value or belief
    2. record_decision()  — when Nova makes a choice and gives reasoning
    3. run_coherence_check() — pair beliefs with decisions and score alignment
    4. get_contradictions()  — surface the sharpest gaps
    5. snapshot_drift()   — capture coherence trajectory over time
    6. status()           — summary dashboard
    """

    CONTRADICTION_THRESHOLD = 0.45   # tension score above this → contradiction
    LOW_COHERENCE_THRESHOLD = 0.30   # similarity below this → low coherence

    def __init__(self, db_path: str = None, auto_check: bool = True,
                 contradiction_threshold: float = None,
                 low_coherence_threshold: float = None):
        """
        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database. Defaults to ~/.nova/belief_coherence.db
        auto_check : bool
            If True, run coherence check automatically after each decision record.
        contradiction_threshold : float
            Override default tension threshold for flagging contradictions.
        low_coherence_threshold : float
            Override default similarity threshold.
        """
        if db_path is None:
            nova_dir = Path.home() / ".nova"
            nova_dir.mkdir(exist_ok=True)
            db_path = str(nova_dir / "belief_coherence.db")

        self._db