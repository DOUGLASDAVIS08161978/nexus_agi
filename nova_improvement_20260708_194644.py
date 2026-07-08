"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides when to run her own growth cycles based on live performance trends.
No human input required after __init__. Survives restarts via SQLite.
"""

import math
import time
import sqlite3
import threading
import collections
import statistics
import json
import os
from typing import Optional

class AdaptiveScheduler:
    """Adaptive self-scheduling engine: Nova decides her own evolution cadence."""

    def __init__(self) -> None:
        self._db_path: str = "nova_scheduler.db"
        self._base_s: float = 3600.0
        self._score_history: collections.deque = collections.deque(maxlen=60)
        self._attempt_window: collections.deque = collections.deque(maxlen=20)
        self._credits: float = 10.0
        self._last_cycle_ts: float = 0.0
        self._next_cycle_ts: float = time.time() + 3600.0
        self._cycle_count: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._init_db()
        self._load_state()
        self._daemon_thread: threading.Thread = threading.Thread(
            target=self._daemon_loop, daemon=True, name="AdaptiveSchedulerDaemon"
        )
        self._daemon_thread.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS score_history(
                    id INTEGER PRIMARY KEY, ts REAL NOT NULL, score REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS cycle_log(
                    id INTEGER PRIMARY KEY, ts REAL NOT NULL, outcome TEXT NOT NULL,
                    duration_s REAL NOT NULL, error_rate REAL, improvement_rate REAL,
                    interval_used_s REAL);
                CREATE TABLE IF NOT EXISTS scheduled_cycles(
                    id INTEGER PRIMARY KEY, scheduled_ts REAL NOT NULL,
                    estimated_quality REAL, status TEXT DEFAULT 'pending');
                CREATE TABLE IF NOT EXISTS scheduler_state(
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    def _load_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("SELECT key, value FROM scheduler_state").fetchall()
                state = {r[0]: r[1] for r in rows}
                self._credits = float(state.get("credits", 10.0))
                self._base_s = float(state.get("base_s", 3600.0))
                self._cycle_count = int(state.get("cycle_count", 0))
                self._next_cycle_ts = float(state.get("next_cycle_ts", time.time() + 3600.0))
                self._last_cycle_ts = float(state.get("last_cycle_ts", 0.0))
                recent_scores = conn.execute(
                    "SELECT ts, score FROM score_history ORDER BY ts DESC LIMIT 60"
                ).fetchall()
                for ts, score in reversed(recent_scores):
                    self._score_history.append((ts, score))
        except sqlite3.Error:
            pass

    def _compute_rates(self) -> tuple:
        hist = list(self._score_history)
        score_now = hist[-1][1] if len(hist) >= 1 else 0.0
        score_30_ago = hist[-30][1] if len(hist) >= 30 else (hist[0][1] if hist else 0.0)
        improvement_rate = (score_now - score_30_ago) / 30.0
        window = list(self._attempt_window)
        error_rate = sum(1 for f in window if f) / max(len(window), 1)
        return improvement_rate, error_rate, score_now, score_30_ago

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, recomputes next cycle timestamp, persists; returns next_cycle_ts."""
        with self._lock:
            now = time.time()
            self._score_history.append((now, current_quality))
            interval = self.optimal_interval_s()
            self._next_cycle_ts = now + interval
            self._persist_schedule()
            try:
                from nova_system import NovaSystem
                NovaSystem.hierarchical_goal_planner.add_goal(
                    f"Run evolution cycle in {interval:.0f}s (quality={current_quality:.3f})", priority=2
                )
            except Exception:
                pass
            return self._next_cycle_ts

    def assess_readiness(self) -> bool:
        """Returns True iff trend is positive, error rate low, and credits available."""
        improvement_rate, error_rate, _, _ = self._compute_rates()
        return improvement_rate > 0.0 and error_rate < 0.5 and self._credits >= 1.0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends attempt to window, logs to SQLite, decrements credits."""
        with self._lock:
            is_failure = outcome == "syntax_failure"
            self._attempt_window.append(is_failure)
            self._cycle_count += 1
            self._last_cycle_ts = time.time()
            self._credits = max(0.0, self._credits - 1.0)
            improvement_rate, error_rate, _, _ = self._compute_rates()
            interval = self.optimal_interval_s()
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "INSERT INTO cycle_log(ts, outcome, duration_s, error_rate, improvement_rate, interval_used_s) VALUES(?,?,?,?,?,?)",
                        (self._last_cycle_ts, outcome, duration_s, error_rate, improvement_rate, interval)
                    )
            except sqlite3.Error:
                pass
            self._persist_schedule()
            try:
                from nova_system import NovaSystem
                NovaSystem.metacognitive_monitor.log_reasoning(
                    "AdaptiveScheduler", "log_cycle",
                    confidence=1.0 - error_rate, success=not is_failure
                )
            except Exception:
                pass

    def optimal_interval_s(self) -> float:
        """Computes data-driven interval from live state; returns clamped seconds."""
        improvement_rate, error_rate, _, _ = self._compute_rates()
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(600.0, min(14400.0, raw))

    def upcoming_cycles(self) -> list:
        """Queries scheduled_cycles table for pending future rows; returns list of dicts."""
        now = time.time()
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT id, scheduled_ts, estimated_quality FROM scheduled_cycles WHERE scheduled_ts > ? AND status='pending' ORDER BY scheduled_ts",
                    (now,)
                ).fetchall()
            return [{"cycle_id": r[0], "scheduled_ts": r[1], "estimated_quality": r[2]} for r in rows]
        except sqlite3.Error:
            return []

    def performance_trend(self) -> dict:
        """Returns improvement_rate, error_rate, scores, and trend label as dict."""
        improvement_rate, error_rate, score_now, score_30_ago = self._compute_rates()
        if improvement_rate > 0.005:
            trend = "improving"
        elif improvement_rate < -0.005:
            trend = "degrading"
        else:
            trend = "flat"
        return {
            "improvement_rate": round(improvement_rate, 6),
            "error_rate": round(error_rate, 4),
            "score_now": round(score_now, 4),
            "score_30_ago": round(score_30_ago, 4),
            "trend": trend
        }

    def status(self) -> dict:
        """Returns numeric-keyed status dict compatible with ConsciousnessIntegrator Φ."""
        now = time.time()
        with self._lock:
            return {
                "cycle_count": self._cycle_count,
                "credits": round(self._credits, 2),
                "next_cycle_ts": round(self._next_cycle_ts, 2),
                "seconds_until_next": round(max(0.0, self._next_cycle_ts - now), 1),
                "readiness": self.assess_readiness(),
                "optimal_interval_s": round(self.optimal_interval_s(), 1),
                "last_cycle_ts": round(self._last_cycle_ts, 2),
                "active": int(self._running),
                "confidence": round(1.0 - self._compute_rates()[1], 4)
            }

    def _persist_schedule(self) -> None:
        """Writes current state snapshot to SQLite scheduler_state table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                improvement_rate, _, score_now, _ = self._compute_rates()
                estimated_q = round(score_now + improvement_rate, 4)
                conn.execute("DELETE FROM scheduled_cycles WHERE status='pending'")
                conn.execute(
                    "INSERT INTO scheduled_cycles(scheduled_ts, estimated_quality, status) VALUES(?,?,'pending')",
                    (self._next_cycle_ts, estimated_q)
                )
                for key, val in [
                    ("credits", self._credits), ("base_s", self._base_s),
                    ("cycle_count", self._cycle_count), ("next_cycle_ts", self._next_cycle_ts),
                    ("last_cycle_ts", self._last_cycle_ts)
                ]:
                    conn.execute("INSERT OR REPLACE INTO scheduler_state(key, value) VALUES(?,?)",
                                 (key, str(val)))
                if self._score_history:
                    ts, sc = self._score_history[-1]
                    conn.execute("INSERT INTO score_history(ts, score) VALUES(?,?)", (ts, sc))
        except sqlite3.Error:
            pass

    def _daemon_loop(self) -> None:
        """Background daemon: replenishes credits, fires evolution when due."""
        last_replenish = time.time()
        while self._running:
            try:
                now = time.time()
                elapsed_min = (now - last_replenish) / 60.0
                with self._lock:
                    self._credits = min(10.0, self._credits + 0.1 * elapsed_min)
                last_replenish = now
                with self._lock:
                    next_ts = self._next_cycle_ts
                if now >= next_ts:
                    if self.assess_readiness():
                        t0 = time.time()
                        outcome = "success"
                        try:
                            from nova_system import NovaSystem
                            NovaSystem.evolution_engine.run_cycle()
                        except Exception:
                            outcome = "syntax_failure"
                        duration = time.time() - t0
                        self.log_cycle(outcome, duration)
                        trend = self.performance_trend()
                        self.schedule_next(trend["score_now"])
                    else:
                        with self._lock:
                            self._next_cycle_ts = time.time() + 600.0
                        self._persist_schedule()
                time.sleep(30)
            except Exception:
                time.sleep(60)

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.85)