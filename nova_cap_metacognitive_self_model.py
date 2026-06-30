"""
nova_cap_metacognitive_self_model.py

A metacognitive self-modeling module for Nova ASI that tracks reasoning moves
in real-time, identifies productive vs. unproductive patterns, learns cognitive
strengths and blind spots, and predicts optimal approaches for novel problems.

The module maintains a persistent SQLite store of reasoning episodes, extracts
structural patterns from them, and builds a continuously-updated self-model that
can advise on strategy selection before Nova commits to an approach.
"""

import sqlite3
import json
import os
import re
import math
import time
import threading
import hashlib
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("NOVA_DATA_DIR", ".")) / "nova_metacognitive_self_model.db"

OUTCOME_DISCOVERY   = "discovery"
OUTCOME_DEAD_END    = "dead_end"
OUTCOME_PARTIAL     = "partial"
OUTCOME_UNKNOWN     = "unknown"

MOVE_TYPES = [
    "hypothesis_generation",
    "analogy_mapping",
    "decomposition",
    "synthesis",
    "counter_example_search",
    "constraint_relaxation",
    "abstraction_lift",
    "concretization",
    "pattern_matching",
    "causal_inference",
    "meta_reflection",
    "exhaustive_enumeration",
    "probabilistic_estimate",
    "reframing",
    "assumption_audit",
]

DOMAIN_TAGS = [
    "mathematics", "language", "logic", "science", "planning",
    "creativity", "ethics", "code", "perception", "memory",
    "social", "strategy", "unknown",
]

CONFIDENCE_DECAY   = 0.92   # per-episode decay for old pattern scores
MIN_EPISODES       = 3      # minimum episodes before pattern is trusted
PATTERN_WINDOW     = 200    # episodes kept in rolling analysis window
BLIND_SPOT_THRESH  = 0.30   # success rate below this → blind spot
STRENGTH_THRESH    = 0.70   # success rate above this → strength


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cosine_sim(a: dict, b: dict) -> float:
    """Sparse cosine similarity between two frequency dicts."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na  = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    nb  = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (na * nb)


def _entropy(counts: list) -> float:
    total = sum(counts) or 1
    return -sum((c / total) * math.log2(c / total + 1e-12) for c in counts if c > 0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetacognitiveSelfModel:
    """
    Nova's metacognitive self-model.

    Lifecycle
    ---------
    1. Call ``begin_episode(problem, domain)`` when Nova starts working on
       something.  Returns an episode_id.
    2. Call ``log_move(episode_id, move_type, description, confidence)``
       for each reasoning step.
    3. Call ``close_episode(episode_id, outcome, notes)`` when done.
    4. Call ``predict_approach(problem_description, domain)`` before tackling
       a novel problem to get a recommended strategy profile.
    5. Call ``reflect()`` periodically (or it runs automatically in the
       background) to update the self-model.
    """

    def __init__(self, *, db_path: str | Path = DB_PATH,
                 auto_reflect_interval: int = 300,
                 verbose: bool = False):
        self._db_path   = Path(db_path)
        self._verbose   = verbose
        self._lock      = threading.RLock()
        self._active    : dict[str, dict] = {}   # episode_id → live episode
        self._model_cache: dict | None    = None
        self._model_ts  : float           = 0.0

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_model_cache()

        # Background reflection thread
        self._stop_event = threading.Event()
        if auto_reflect_interval > 0:
            self._reflect_thread = threading.Thread(
                target=self._background_reflect,
                args=(auto_reflect_interval,),
                daemon=True,
                name="nova-metacog-reflect",
            )
            self._reflect_thread.start()

    # ------------------------------------------------------------------
    # DB bootstrap
    # ------------------------------------------------------------------

    def _init_db(self):
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id          TEXT PRIMARY KEY,
                    problem     TEXT,
                    domain      TEXT,
                    started_at  TEXT,
                    closed_at   TEXT,
                    outcome     TEXT DEFAULT 'unknown',
                    notes       TEXT,
                    duration_s  REAL,
                    move_count  INTEGER DEFAULT 0,
                    fingerprint TEXT
                );

                CREATE TABLE IF NOT EXISTS moves (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id  TEXT,
                    seq         INTEGER,
                    move_type   TEXT,
                    description TEXT,
                    confidence  REAL,
                    ts          TEXT,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id)
                );

                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_key     TEXT PRIMARY KEY,
                    domain          TEXT,
                    move_sequence   TEXT,
                    outcome_counts  TEXT,
                    total_seen      INTEGER DEFAULT 0,
                    success_rate    REAL DEFAULT 0.0,
                    last_updated    TEXT
                );

                CREATE TABLE IF NOT EXISTS self_model (
                    key     TEXT PRIMARY KEY,
                    value   TEXT,
                    updated TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_moves_ep  ON moves(episode_id);
                CREATE INDEX IF NOT EXISTS idx_ep_domain ON episodes(domain);
                CREATE INDEX IF NOT EXISTS idx_ep_outcome ON episodes(outcome);
            """)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def begin_episode(self, problem: str, domain: str = "unknown") -> str:
        """
        Start tracking a new reasoning episode.

        Parameters
        ----------
        problem : natural-language description of what Nova is working on.
        domain  : one of DOMAIN_TAGS (auto-detected if omitted).

        Returns
        -------
        episode_id : str  — pass this to log_move / close_episode.
        """
        domain = self._normalise_domain(domain, problem)
        ep_id  = _fingerprint(f"{problem}{time.time()}{random.random()}")

        episode = {
            "id"        : ep_id,
            "problem"   : problem,
            "domain"    : domain,
            "started_at": _now_iso(),
            "moves"     : [],
            "seq"       : 0,
        }

        with self._lock:
            self._active[ep_id] = episode

        with self._connect() as con:
            con.execute(
                "INSERT INTO episodes(id, problem, domain, started_at, fingerprint) "
                "VALUES (?,?,?,?,?)",
                (ep_id, problem, domain, episode["started_at"], _fingerprint(problem)),
            )

        if self._verbose:
            print(f"[metacog] begin_episode {ep_id[:8]} domain={domain}")
        return ep_id

    def log_move(self, episode_id: str, move_type: str,
                 description: str = "", confidence: float = 0.5) -> int:
        """
        Record one reasoning move inside an active episode.

        Parameters
        ----------
        episode_id  : returned by begin_episode.
        move_type   : one of MOVE_TYPES.
        description : free-text description of what Nova did.
        confidence  : 0–1, how confident Nova felt about this move.

        Returns
        -------
        seq : sequence number of this move within the episode.
        """
        move_type  = self._normalise_move(move_type)
        confidence = max(0.0, min(1.0, confidence))

        with self._lock:
            ep = self._active.get(episode_id)
            if ep is None:
                return -1
            ep["seq"] += 1
            seq = ep["seq"]
            ep["moves"].append({
                "seq"        : seq,
                "move_type"  : move_type,
                "description": description,
                "confidence" : confidence,
                "ts"         : _now_iso(),
            })

        with self._connect() as con:
            con.execute(
                "INSERT INTO moves(episode_id, seq, move_type, description, confidence, ts) "
                "VALUES (?,?,?,?,?,?)",
                (episode_id, seq, move_type, description, confidence, _now_iso()),
            )

        return seq


    def _background_reflect(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _load_model_cache(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _normalise_domain(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _normalise_move(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _normalise_outcome(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _update_patterns_for_episode(self):
        """Auto-stub: referenced but not generated."""
        pass
    def close_episode(self, episode_id: str, outcome: str = OUTCOME_UNKNOWN,
                      notes: str = "") -> dict:
        """
        Finalise an episode and trigger incremental pattern learning.

        Parameters
        ----------
        outcome : one of OUTCOME_DISCOVERY, OUTCOME_DEAD_END,
                  OUTCOME_PARTIAL, OUTCOME_UNKNOWN.
        notes   : optional free-text summary.

        Returns
        -------
        summary dict with stats about the episode.
        """
        outcome = self._normalise_outcome(outcome)

        with self._lock:
            ep = self._active.pop(episode_id, None)

        if ep is None:
            return {"error": "episode not found", "episode_id": episode_id}

        closed_at  = _now_iso()
        started_dt = datetime.fromisoformat(ep["started_at"])
        closed_dt  = datetime.fromisoformat(closed_at)
        duration   = (closed_dt - started_dt).total_seconds()
        move_count = len(ep["moves"])

        with self._connect() as con:
            con.execute(
                "UPDATE episodes SET closed_at=?, outcome=?, notes=?, "
                "duration_s=?, move_count=? WHERE id=?",
                (closed_at, outcome, notes, duration, move_count, episode_id),
            )

        # Incremental pattern update (non-blocking)
        threading.Thread(
            target=self._update_patterns_for_episode,
            args=(ep, outcome),
            daemon=True,
        ).start()

        summary = {
            "episode_id": episode_id,
            "domain"    : ep["domain"],
            "outcome"   : outcome,
            "duration_s": round(duration, 2),
            "move_count": move_count,
            "move_types": [m["move_type"] for m in ep["moves"]],
        }
        if self._verbose:
            print(f"[metacog] close_episode {episode_id[:8]} outcome={outcome} "
                  f"moves={move_count} dur={duration:.1f}s")
        return