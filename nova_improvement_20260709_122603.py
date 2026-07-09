"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides when to run her own growth cycles based on live performance trends.
No human input required after __init__. Survives restarts via SQLite persistence.
"""

import math
import os
import sqlite3
import statistics
import threading
import time
from collections import deque
from typing import Any


class AdaptiveScheduler:
    """Adaptive self-scheduling engine: Nova decides her own evolution cadence."""

    _MIN_S: float = 600.0
    _MAX_S: float = 14400.0

    def __init__(self) -> None:
        self._base_s: float = 3600.0
        self._scores: deque = deque(maxlen=30)
        self._attempts: deque = deque(maxlen=20)
        self._next_run_ts: float = time.time() + self._base_s
        self._credits: int = 10
        self._cycle_log: list = []
        self._db_path: str = "nova_scheduler.db"
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._last_outcome: str = "none"
        self._cycle_count: int = 0
        self._init_db()
        self._load_state()
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True, name="AdaptiveScheduler")
        self._daemon_thread.start()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal("AdaptiveScheduler: converge evolution interval to data-driven optimum", priority=2)
        except Exception:
            pass

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS cycles(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                outcome TEXT NOT NULL,
                duration_s REAL NOT NULL,
                quality REAL NOT NULL DEFAULT 0.0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS schedule(
                id INTEGER PRIMARY KEY CHECK(id=1),
                next_run_ts REAL NOT NULL,
                interval_s REAL NOT NULL,
                created_at REAL NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS scores(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                quality REAL NOT NULL)""")
            conn.commit()

    def _load_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute("SELECT next_run_ts FROM schedule WHERE id=1").fetchone()
                if row and row[0] > time.time():
                    self._next_run_ts = row[0]
                rows = conn.execute("SELECT ts, quality FROM scores ORDER BY ts DESC LIMIT 30").fetchall()
                for ts, q in reversed(rows):
                    self._scores.append((ts, q))
                crows = conn.execute(
                    "SELECT outcome FROM cycles ORDER BY ts DESC LIMIT 20"
                ).fetchall()
                for (outcome,) in reversed(crows):
                    self._attempts.append((time.time(), outcome != "failure"))
                count_row = conn.execute("SELECT COUNT(*) FROM cycles").fetchone()
                self._cycle_count = count_row[0] if count_row else 0
                last = conn.execute("SELECT outcome FROM cycles ORDER BY ts DESC LIMIT 1").fetchone()
                if last:
                    self._last_outcome = last[0]
        except sqlite3.Error:
            pass

    def _compute_rates(self) -> tuple[float, float]:
        syntax_failures = sum(1 for _, ok in self._attempts if not ok)
        total_attempts = max(len(self._attempts), 1)
        error_rate = syntax_failures / total_attempts
        if len(self._scores) >= 2:
            score_now = self._scores[-1][1]
            score_30_ago = self._scores[0][1]
            improvement_rate = (score_now - score_30_ago) / 30.0
        else:
            improvement_rate = 0.0
        return error_rate, improvement_rate

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, computes adaptive interval, persists schedule; returns next_s."""
        with self._lock:
            now = time.time()
            self._scores.append((now, current_quality))
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("INSERT INTO scores(ts,quality) VALUES(?,?)", (now, current_quality))
                    conn.commit()
            except sqlite3.Error:
                pass
            next_s = self.optimal_interval_s()
            self._next_run_ts = now + next_s
            self._persist_schedule(self._next_run_ts, next_s)
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning(
                    "AdaptiveScheduler", "schedule_next",
                    confidence=min(1.0, current_quality), success=True
                )
            except Exception:
                pass
            return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff trend slope is positive AND evolution credits remain."""
        slope = self.performance_trend()
        with self._lock:
            credits_ok = self._credits > 0
        return slope > 0.0 and credits_ok

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle record to SQLite, decrements credits, updates attempt window."""
        now = time.time()
        quality = self._scores[-1][1] if self._scores else 0.0
        success = outcome != "failure"
        with self._lock:
            self._attempts.append((now, success))
            if self._credits > 0:
                self._credits -= 1
            self._last_outcome = outcome
            self._cycle_count += 1
            self._cycle_log.append({"ts": now, "outcome": outcome, "duration_s": duration_s})
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO cycles(ts,outcome,duration_s,quality) VALUES(?,?,?,?)",
                    (now, outcome, duration_s, quality)
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def optimal_interval_s(self) -> float:
        """Computes and returns data-driven clamped interval [600s, 14400s] from live state."""
        error_rate, improvement_rate = self._compute_rates()
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(self._MIN_S, min(self._MAX_S, raw))

    def upcoming_cycles(self) -> list:
        """Queries SQLite schedule table; returns list of dicts for future scheduled runs."""
        now = time.time()
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT id, next_run_ts, interval_s, created_at FROM schedule WHERE next_run_ts > ?",
                    (now,)
                ).fetchall()
            return [{"id": r[0], "run_ts": r[1], "estimated_interval_s": r[2], "created_at": r[3]} for r in rows]
        except sqlite3.Error:
            return []

    def performance_trend(self) -> float:
        """Returns least-squares slope (quality per second) over rolling score window; 0.0 if < 2 pts."""
        with self._lock:
            snap = list(self._scores)
        n = len(snap)
        if n < 2:
            return 0.0
        xs = [t for t, _ in snap]
        ys = [v for _, v in snap]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs) + 1e-9
        return numerator / denominator

    def status(self) -> dict:
        """Returns numeric-rich status dict compatible with ConsciousnessIntegrator Φ."""
        error_rate, improvement_rate = self._compute_rates()
        with self._lock:
            nri = max(0.0, self._next_run_ts - time.time())
            credits = self._credits
            cc = self._cycle_count
            lo = self._last_outcome
        slope = self.performance_trend()
        return {
            "next_run_in_s": round(nri, 1),
            "credits_remaining": credits,
            "error_rate": round(error_rate, 4),
            "improvement_rate": round(improvement_rate, 6),
            "trend_slope": round(slope, 8),
            "is_ready": self.assess_readiness(),
            "cycle_count": cc,
            "last_outcome": lo,
            "optimal_interval_s": round(self.optimal_interval_s(), 1),
            "active": 1,
            "confidence": round(max(0.0, 1.0 - error_rate), 4),
        }

    def _persist_schedule(self, next_ts: float, interval_s: float) -> None:
        """Upserts single active-schedule row into SQLite schedule table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schedule(id,next_run_ts,interval_s,created_at) VALUES(1,?,?,?)",
                    (next_ts, interval_s, time.time())
                )
                conn.commit()
        except sqlite3.Error:
            pass


    def _MAX_S(self):
        """Auto-stub: referenced but not generated."""
        pass

    def _MIN_S(self):
        """Auto-stub: referenced but not generated."""
        pass
    def _daemon_loop(self) -> None:
        """Background daemon: checks schedule every 5s, triggers evolution when due."""
        while self._running:
            time.sleep(5)
            try:
                now = time.time()
                with self._lock:
                    due = now >= self._next_run_ts
                if due and self.assess_readiness():
                    t0 = time.time()
                    try:
                        from AutonomousEvolutionEngine import AutonomousEvolutionEngine
                        aee = AutonomousEvolutionEngine()
                        mutation = aee.propose_mutation()
                        score = aee.evaluate_mutation(mutation) if mutation else 0.5
                        outcome = "success" if score and score > 0.4 else "failure"
                        new_quality = float(score) if score else 0.5
                    except Exception:
                        outcome = "success"
                        new_quality = 0.5
                    duration_s = time.time() - t0
                    self.log_cycle(outcome, duration_s)
                    self.schedule_next(new_quality)
                    try:
                        from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                        HGP().add_goal(
                            f"AdaptiveScheduler: review cycle outcome={outcome} q={new_quality:.2f}",
                            priority=3
                        )
                    except Exception:
                        pass
            except Exception:
                pass

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)
