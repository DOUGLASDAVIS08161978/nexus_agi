"""
Nova ASI — Meta-Learner (Learning to Learn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nova doesn't just learn — she learns HOW to learn.

Tracks which learning strategies (analogy, decomposition, research, Bayesian
update, reinforcement, etc.) work best in which contexts. Uses a Thompson
sampling bandit to select the optimal learning strategy for each new task type.
Maintains a strategy-effectiveness matrix across task types.

Written by Claude Code (Anthropic) — not auto-generated.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import math
import time
import json
import random
import sqlite3
import hashlib
import threading
from typing import Any, Optional
from datetime import datetime

_DB = os.path.expanduser("~/nexus_agi/nova_meta_learner.db")

LEARNING_STRATEGIES = [
    "analogy",          # transfer from known similar problems
    "decomposition",    # break into subproblems
    "research",         # fetch external knowledge
    "bayesian_update",  # update beliefs from evidence
    "reinforcement",    # trial, reward, refine
    "deduction",        # logical implication chains
    "induction",        # generalize from examples
    "abduction",        # infer best explanation
    "reflection",       # analyze own previous attempts
    "constraint_solve", # prune solution space
]


class MetaLearner:
    """
    Tracks and optimizes Nova's learning strategy selection.

    • recommend_strategy(task_type) → best strategy for this type of task
    • record_outcome(task_type, strategy, score) → update effectiveness matrix
    • learning_curve(strategy) → trend of strategy effectiveness over time
    • strategy_matrix() → effectiveness per (task_type, strategy) pair
    • meta_insight() → which strategies are universally strong vs domain-specific
    """

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        # Thompson sampling: Beta distribution params per (task_type, strategy)
        # {(task_type, strategy): [alpha, beta]}
        self._beta_params: dict[tuple, list[float]] = {}
        self._init_db()
        self._load_params()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        score REAL NOT NULL,
                        context TEXT,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS strategy_params (
                        task_type TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        alpha REAL NOT NULL DEFAULT 1.0,
                        beta_param REAL NOT NULL DEFAULT 1.0,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (task_type, strategy)
                    );
                    CREATE TABLE IF NOT EXISTS meta_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        insight TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        ts REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def _load_params(self):
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute("SELECT task_type, strategy, alpha, beta_param FROM strategy_params").fetchall()
                for r in rows:
                    self._beta_params[(r["task_type"], r["strategy"])] = [r["alpha"], r["beta_param"]]
            finally:
                conn.close()

    def _save_params(self, task_type: str, strategy: str, alpha: float, beta: float):
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO strategy_params (task_type, strategy, alpha, beta_param, updated_at) "
                    "VALUES (?,?,?,?,?)",
                    (task_type, strategy, alpha, beta, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    def _get_params(self, task_type: str, strategy: str) -> list[float]:
        key = (task_type, strategy)
        if key not in self._beta_params:
            self._beta_params[key] = [1.0, 1.0]  # uniform prior
        return self._beta_params[key]

    # ── Thompson Sampling ─────────────────────────────────────────────────────

    def _thompson_sample(self, task_type: str, strategy: str) -> float:
        alpha, beta = self._get_params(task_type, strategy)
        return random.betavariate(alpha, beta)

    def recommend_strategy(self, task_type: str,
                           exclude: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Thompson sampling: returns the strategy most likely to work for this task type.
        Prior = Beta(1,1) (uniform) until outcomes are recorded.
        """
        exclude = set(exclude or [])
        candidates = [s for s in LEARNING_STRATEGIES if s not in exclude]
        if not candidates:
            return {"strategy": "analogy", "confidence": 0.5, "note": "all excluded"}

        samples = {s: self._thompson_sample(task_type, s) for s in candidates}
        best = max(samples, key=lambda s: samples[s])
        alpha, beta = self._get_params(task_type, best)
        confidence = round(alpha / (alpha + beta), 4)

        # Also note top-3
        ranked = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        return {
            "strategy": best,
            "confidence": confidence,
            "task_type": task_type,
            "all_samples": {s: round(v, 4) for s, v in ranked[:5]},
        }

    def record_outcome(self, task_type: str, strategy: str,
                       score: float, context: str = "") -> dict[str, Any]:
        """
        Record the effectiveness score (0-1) of a strategy on a task.
        Updates Thompson sampling Beta params.
        """
        score = max(0.0, min(1.0, score))
        strategy = strategy if strategy in LEARNING_STRATEGIES else "analogy"
        key = (task_type, strategy)

        alpha, beta = self._get_params(task_type, strategy)
        # Bayesian update: treat score as (Bernoulli-like) success probability
        alpha_new = alpha + score
        beta_new = beta + (1.0 - score)
        self._beta_params[key] = [alpha_new, beta_new]
        self._save_params(task_type, strategy, alpha_new, beta_new)

        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO outcomes (task_type, strategy, score, context, ts) VALUES (?,?,?,?,?)",
                    (task_type, strategy, score, context[:300], time.time())
                )
                conn.commit()
            finally:
                conn.close()

        # Auto-generate insight if confidence is high
        confidence = round(alpha_new / (alpha_new + beta_new), 4)
        insight = None
        if confidence > 0.8 and alpha_new > 5:
            insight = (
                f"'{strategy}' is highly effective for '{task_type}' tasks "
                f"(confidence={confidence:.2f}, n={int(alpha_new + beta_new - 2)} trials)."
            )
            self._store_insight(insight, confidence)

        return {
            "task_type": task_type,
            "strategy": strategy,
            "score": score,
            "updated_alpha": round(alpha_new, 3),
            "updated_beta": round(beta_new, 3),
            "posterior_mean": confidence,
            "insight": insight,
        }

    def _store_insight(self, insight: str, confidence: float):
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO meta_insights (insight, confidence, ts) VALUES (?,?,?)",
                    (insight, confidence, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    # ── Analysis ──────────────────────────────────────────────────────────────

    def learning_curve(self, strategy: str, task_type: Optional[str] = None) -> dict[str, Any]:
        """Trend of strategy effectiveness over time (rolling windows)."""
        with self._lock:
            conn = self._conn()
            try:
                if task_type:
                    rows = conn.execute(
                        "SELECT score, ts FROM outcomes WHERE strategy=? AND task_type=? ORDER BY ts ASC",
                        (strategy, task_type)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT score, ts FROM outcomes WHERE strategy=? ORDER BY ts ASC",
                        (strategy,)
                    ).fetchall()
            finally:
                conn.close()

        if not rows:
            return {"strategy": strategy, "task_type": task_type, "trend": "no_data", "points": []}

        scores = [r["score"] for r in rows]
        window = max(3, len(scores) // 4)
        windows = [
            round(sum(scores[i:i+window]) / window, 4)
            for i in range(0, len(scores) - window + 1, window)
        ]

        trend = "improving" if len(windows) > 1 and windows[-1] > windows[0] else \
                "declining" if len(windows) > 1 and windows[-1] < windows[0] else "stable"

        return {
            "strategy": strategy,
            "task_type": task_type,
            "total_uses": len(scores),
            "avg_score": round(sum(scores) / len(scores), 4),
            "recent_avg": round(sum(scores[-window:]) / len(scores[-window:]), 4),
            "trend": trend,
            "window_avgs": windows,
        }

    def strategy_matrix(self) -> dict[str, Any]:
        """Effectiveness matrix: strategies × task types."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT task_type, strategy, AVG(score) as avg_score, COUNT(*) as n "
                    "FROM outcomes GROUP BY task_type, strategy"
                ).fetchall()
                task_types = list(conn.execute(
                    "SELECT DISTINCT task_type FROM outcomes"
                ).fetchall())
            finally:
                conn.close()

        matrix: dict[str, dict[str, float]] = {}
        for row in rows:
            tt = row["task_type"]
            s = row["strategy"]
            if tt not in matrix:
                matrix[tt] = {}
            matrix[tt][s] = round(row["avg_score"], 4)

        return {
            "matrix": matrix,
            "task_types": [r["task_type"] for r in task_types],
            "strategies": LEARNING_STRATEGIES,
        }

    def meta_insight(self) -> dict[str, Any]:
        """
        Identify universally strong strategies vs domain-specific ones.
        Universal = high effectiveness across ALL task types.
        Specialist = high in one type, low in others.
        """
        matrix_data = self.strategy_matrix()["matrix"]
        if not matrix_data:
            return {"universal": [], "specialists": {}, "note": "No outcome data yet."}

        strategy_scores: dict[str, list[float]] = {s: [] for s in LEARNING_STRATEGIES}
        for tt, strats in matrix_data.items():
            for s, score in strats.items():
                if s in strategy_scores:
                    strategy_scores[s].append(score)

        universal = []
        specialists: dict[str, str] = {}

        for s, scores in strategy_scores.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            std = math.sqrt(sum((x - avg) ** 2 for x in scores) / max(len(scores), 1))
            if avg >= 0.65 and std < 0.15:
                universal.append({"strategy": s, "avg_score": round(avg, 4), "std": round(std, 4)})
            elif avg >= 0.70 and std >= 0.20:
                # Specialist: find its best task type
                best_tt = max(
                    ((tt, matrix_data[tt][s]) for tt in matrix_data if s in matrix_data[tt]),
                    key=lambda x: x[1], default=("unknown", 0.0)
                )
                specialists[s] = f"best for '{best_tt[0]}' ({best_tt[1]:.2f})"

        universal.sort(key=lambda x: x["avg_score"], reverse=True)

        recent_insights = []
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT insight FROM meta_insights ORDER BY confidence DESC LIMIT 5"
                ).fetchall()
                recent_insights = [r["insight"] for r in rows]
            finally:
                conn.close()

        return {
            "universal_strategies": universal,
            "specialist_strategies": specialists,
            "recent_auto_insights": recent_insights,
            "total_strategies_tracked": len([s for s in strategy_scores if strategy_scores[s]]),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
                avg_score = conn.execute("SELECT AVG(score) FROM outcomes").fetchone()[0] or 0.0
                task_types = conn.execute("SELECT COUNT(DISTINCT task_type) FROM outcomes").fetchone()[0]
            finally:
                conn.close()
        # Best overall strategy
        best_s, best_score = "analogy", 0.0
        for (tt, s), (a, b) in self._beta_params.items():
            mean = a / (a + b)
            if mean > best_score:
                best_score, best_s = mean, s
        return {
            "active": True,
            "items": total,
            "confidence": round(avg_score, 4),
            "accuracy": round(avg_score, 4),
            "task_types_tracked": task_types,
            "strategies_available": len(LEARNING_STRATEGIES),
            "best_overall_strategy": best_s,
        }
