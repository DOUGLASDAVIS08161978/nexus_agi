"""
Nova ASI — Recursive Self-Critique
Proposed autonomously via /evolve
"""

"""
CritiqueEngine: Recursive self-critique module for Nova.

Scans recent responses for vagueness, brevity, and unsupported claims using
probabilistic scoring, Bayesian fault-weight updates, and EMA-tracked quality
trends. Recommends targeted improvements based on the dominant fault pattern.
"""

import sqlite3
import time
import math
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class CritiqueEngine:
    """Recursive self-critique: records responses, scores faults, recommends fixes."""

    # ── fault taxonomy ────────────────────────────────────────────────────────
    _VAGUE_MARKERS: List[str] = [
        r"\bsomething\b", r"\bsomehow\b", r"\bvarious\b", r"\bthings\b",
        r"\bstuff\b", r"\bkind of\b", r"\bsort of\b", r"\bmaybe\b",
        r"\bperhaps\b", r"\bgenerally\b", r"\busually\b", r"\boften\b",
    ]
    _CLAIM_TRIGGERS: List[str] = [
        r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bproven\b",
        r"\bfact\b", r"\bobviously\b", r"\bclearly\b", r"\bundeniably\b",
    ]
    _SUPPORT_SIGNALS: List[str] = [
        r"\bbecause\b", r"\bsince\b", r"\bdue to\b", r"\bfor example\b",
        r"\bsuch as\b", r"\baccording to\b", r"\bdata show\b", r"\bstudy\b",
        r"\bresearch\b", r"\bsource\b", r"\bcitation\b",
    ]

    _BREVITY_THRESHOLD: int   = 60    # words below which a response is "brief"
    _CAPACITY: int            = 200
    _DECAY_RATE: float        = 0.0001
    _EMA_ALPHA: float         = 0.2

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._build_schema()
        # Bayesian fault-weight priors (fault → probability of being the main issue)
        self._fault_priors: Dict[str, float] = {
            "vagueness": 0.33,
            "brevity":   0.33,
            "unsupported": 0.34,
        }
        # EMA of overall quality score (0 = terrible, 1 = perfect)
        self._quality_ema: float = 0.5
        # Compile regex patterns once
        self._vague_re    = [re.compile(p, re.I) for p in self._VAGUE_MARKERS]
        self._claim_re    = [re.compile(p, re.I) for p in self._CLAIM_TRIGGERS]
        self._support_re  = [re.compile(p, re.I) for p in self._SUPPORT_SIGNALS]

    # ── schema ────────────────────────────────────────────────────────────────
    def _build_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT    NOT NULL,
                ts          REAL    NOT NULL,
                importance  REAL    NOT NULL DEFAULT 1.0,
                access_cnt  INTEGER NOT NULL DEFAULT 0,
                fault_json  TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS critique_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                fault       TEXT    NOT NULL,
                score       REAL    NOT NULL,
                quality_ema REAL    NOT NULL
            )
        """)
        self._conn.commit()

    # ── public API ────────────────────────────────────────────────────────────
    def record(self, response_text: str) -> int:
        """Store a response; evict least-important if over capacity."""
        self._evict_if_needed()
        cur = self._conn.execute(
            "INSERT INTO responses (text, ts, importance, access_cnt) VALUES (?,?,?,0)",
            (response_text, time.time(), 1.0),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def critique(self) -> Dict[str, Any]:
        """Scan last 5 responses for vagueness, brevity, and unsupported claims."""
        rows = self._conn.execute(
            "SELECT id, text, ts, importance, access_cnt FROM responses "
            "ORDER BY ts DESC LIMIT 5"
        ).fetchall()
        if not rows:
            return {"error": "no responses recorded yet"}

        aggregate: Dict[str, float] = defaultdict(float)
        details: List[Dict[str, Any]] = []

        for rid, text, ts, importance, access_cnt in rows:
            # ── decay importance by elapsed time ──────────────────────────
            elapsed   = time.time() - ts
            decayed   = importance * math.exp(-self._DECAY_RATE * elapsed)
            # ── boost importance on access ────────────────────────────────
            new_imp   = min(decayed * 1.05, 10.0)
            new_acc   = access_cnt + 1
            self._conn.execute(
                "UPDATE responses SET importance=?, access_cnt=? WHERE id=?",
                (new_imp, new_acc, rid),
            )
            # ── score individual faults ───────────────────────────────────
            faults    = self._score_faults(text)
            self._conn.execute(
                "UPDATE responses SET fault_json=? WHERE id=?",
                (json.dumps(faults), rid),
            )
            weight    = new_imp  # higher-importance responses count more
            for fault, score in faults.items():
                aggregate[fault] += score * weight
            details.append({"id": rid, "faults": faults, "weight": round(new_imp, 3)})

        self._conn.commit()

        # ── Bayesian update of fault priors ───────────────────────────────
        total_agg = sum(aggregate.values()) or 1.0
        for fault in self._fault_priors:
            likelihood = aggregate[fault] / total_agg
            prior      = self._fault_priors[fault]
            posterior  = (likelihood * prior) / max(
                likelihood * prior + (1 - likelihood) * (1 - prior), 1e-9
            )
            self._fault_priors[fault] = round(
                0.9 * prior + 0.1 * posterior, 4
            )  # soft update to avoid collapse

        # ── EMA quality update ────────────────────────────────────────────
        raw_quality   = 1.0 - (sum(aggregate.values()) / (len(rows) * 3.0))
        raw_quality   = max(0.0, min(1.0, raw_quality))
        self._quality_ema = (
            self._EMA_ALPHA * raw_quality + (1 - self._EMA_ALPHA) * self._quality_ema
        )
        dominant_fault = max(self._fault_priors, key=lambda f: self._fault_priors[f])
        self._conn.execute(
            "INSERT INTO critique_log (ts, fault, score, quality_ema) VALUES (?,?,?,?)",
            (time.time(), dominant_fault,
             round(aggregate[dominant_fault], 4),
             round(self._quality_ema, 4)),
        )
        self._conn.commit()

        return {
            "responses_scanned": len(rows),
            "fault_scores":      {k: round(v, 4) for k, v in aggregate.items()},
            "fault_posteriors":  dict(self._fault_priors),
            "dominant_fault":    dominant_fault,
            "quality_ema":       round(self._quality_ema, 4),
            "details":           details,
        }

    def recommend(self) -> str:
        """Return a targeted improvement tip based on the dominant fault posterior."""
        dominant = max(self._fault_priors, key=lambda f: self._fault_priors[f])
        tips: Dict[str, str] = {
            "vagueness":   (
                "Replace hedging words ('something', 'kind of', 'generally') with "
                "precise nouns, quantities, or named concepts."
            ),
            "brevity":     (
                f"Responses below {self._BREVITY_THRESHOLD} words lack depth — "
                "expand with at least one concrete example or step-by-step reasoning."
            ),
            "unsupported": (
                "Strong claims ('always', 'proven', 'clearly') require evidence — "
                "cite a source, add 'for example', or qualify with a confidence level."
            ),
        }
        quality_note = (
            f" (Current quality EMA: {self._quality_ema:.2f}/1.00 — "
            + ("improving." if self._quality_ema > 0.6 else "needs work.")
            + ")"
        )
        return tips.get(dominant, "No recommendation available.") + quality_note

    def quality_trend(self) -> List[Dict[str, Any]]:
        """Return the last 10 EMA quality snapshots for trend analysis."""
        rows = self._conn.execute(
            "SELECT ts, fault, score, quality_ema FROM critique_log "
            "ORDER BY ts DESC LIMIT 10"
        ).fetchall()
        return [
            {"ts": r[0], "dominant_fault": r[1],
             "fault_score": r[2], "quality_ema": r[3]}
            for r in rows
        ]

    def capacity_used(self) -> Dict[str, Any]:
        """Report how many responses are stored vs. the LRU capacity ceiling."""
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return {"stored": count, "capacity": self._CAPACITY,
                "pct_full": round(100 * count / self._CAPACITY, 1)}

    def forget_least_important(self) -> Dict[str, Any]:
        """Manually evict the single least-important (decayed) response."""
        row = self._conn.execute(
            "SELECT id, importance FROM responses ORDER BY importance ASC LIMIT 1"
        ).fetchone()
        if not row:
            return {"evicted": None}
        self._conn.execute("DELETE FROM responses WHERE id=?", (row[0],))
        self._conn.commit()
        return {"evicted_id": row[0], "importance_was": round(row[1], 4)}

    # ── private helpers ───────────────────────────────────────────────────────
    def _score_faults(self, text: str) -> Dict[str, float]:
        words       = text.split()
        word_count  = max(len(words), 1)

        vague_hits  = sum(1 for p in self._vague_re if p.search(text))
        vagueness   = min(vague_hits / max(len(self._VAGUE_MARKERS) * 0.3, 1), 1.0)

        brevity     = 1.0 if word_count < self._BREVITY_THRESHOLD else max(
            0.0, 1.0 - (word_count - self._BREVITY_THRESHOLD) / 200
        )

        claim_hits   = sum(1 for p in self._claim_re if p.search(text))
        support_hits = sum(1 for p in self._support_re if p.search(text))
        unsupported  = (
            min(claim_hits / max(len(self._CLAIM_TRIGGERS) * 0.25, 1), 1.0)
            * (1.0 - min(support_hits / max(claim_hits, 1), 1.0))
            if claim_hits > 0 else 0.0
        )

        return {
            "vagueness":   round(vagueness, 4),
            "brevity":     round(brevity, 4),
            "unsupported": round(unsupported, 4),
        }

    def _evict_if_needed(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        while count >= self._CAPACITY:
            self.forget_least_important()
            count -= 1

# Usage: obj = CritiqueEngine() | obj.record("Some response text") | obj.critique() | obj.recommend()