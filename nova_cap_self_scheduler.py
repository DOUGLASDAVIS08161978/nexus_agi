"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-29
"""

"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides when to run her own growth cycles based on live performance trends.
No human input required after __init__. Persists schedule across restarts via SQLite.
"""

import collections
import math
import os
import sqlite3
import statistics
import threading
import time
from typing import Any

class AdaptiveScheduler:
    """Nova self-schedules her own evolution cycles using adaptive exponential interval control."""

    def __init__(self) -> None:
        self._base_s: float = 3600.0
        self._min_s: float = 600.0
        self._max_s: float = 14400.0
        self._score_history: collections.deque = collections.deque(maxlen=30)
        self._attempt_history: collections.deque = collections.deque(maxlen=20)
        self._next_run_ts: float = time.time() + 3600.0
        self._cycle_log: list = []
        self._credits: int = 10
        self._db_path: str = "scheduler.db"
        self._daemon_thread: threading.Thread = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._ema_trend: float = 0.5
        self._init_db()
        self._restore_state()
        self._running = True
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self._daemon_thread.start()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal("Self-schedule evolution cycles autonomously", priority=2)
        except Exception:
            pass

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, outcome TEXT, duration_s REAL, interval_used REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheduled_ts REAL, predicted_interval REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY, value REAL)""")
            conn.commit()

    def _restore_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute("SELECT value FROM state WHERE key='next_run_ts'").fetchone()
                if row and row[0] > time.time():
                    self._next_run_ts = row[0]
                row = conn.execute("SELECT value FROM state WHERE key='credits'").fetchone()
                if row:
                    self._credits = int(row[0])
                row = conn.execute("SELECT value FROM state WHERE key='ema_trend'").fetchone()
                if row:
                    self._ema_trend = row[0]
        except sqlite3.Error:
            pass

    def _persist_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                for key, val in [("next_run_ts", self._next_run_ts),
                                  ("credits", float(self._credits)),
                                  ("ema_trend", self._ema_trend)]:
                    conn.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, val))
                conn.commit()
        except sqlite3.Error:
            pass

    def _improvement_rate(self) -> float:
        """Returns per-sample quality slope over score_history, clamped to [-1.0, 1.0]."""
        with self._lock:
            hist = list(self._score_history)
        if len(hist) < 2:
            return 0.0
        rate = (hist[-1] - hist[0]) / max(len(hist) - 1, 1)
        return max(-1.0, min(1.0, rate))

    def _error_rate(self) -> float:
        """Returns fraction of failed attempts in rolling window of 20."""
        with self._lock:
            attempts = list(self._attempt_history)
        if not attempts:
            return 0.0
        return sum(1 for a in attempts if not a) / max(len(attempts), 1)

    def schedule_next(self, current_quality: float) -> float:
        """Records quality score, recomputes adaptive interval, persists next_run_ts; returns seconds until next cycle."""
        with self._lock:
            self._score_history.append(current_quality)
            self._ema_trend = 0.2 * current_quality + 0.8 * self._ema_trend
        er = self._error_rate()
        ir = self._improvement_rate()
        raw_s = self._base_s * math.exp(er - ir)
        next_s = max(self._min_s, min(self._max_s, raw_s))
        with self._lock:
            self._next_run_ts = time.time() + next_s
        self._persist_state()
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT INTO schedule(scheduled_ts, predicted_interval) VALUES(?,?)",
                             (self._next_run_ts, next_s))
                conn.commit()
        except sqlite3.Error:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("scheduling", "adaptive_exp", 1.0 - er, ir > 0)
        except Exception:
            pass
        return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff improvement trend is positive AND credits remain."""
        with self._lock:
            hist = list(self._score_history)
            credits = self._credits
        if len(hist) < 2:
            return credits > 0
        trend_positive = (hist[-1] - hist[0]) > 0.0
        return trend_positive and credits > 0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle record to DB; decrements credits on success."""
        ts = time.time()
        er = self._error_rate()
        ir = self._improvement_rate()
        raw_s = self._base_s * math.exp(er - ir)
        interval_used = max(self._min_s, min(self._max_s, raw_s))
        success = outcome.lower() in ("success", "ok", "pass", "complete")
        with self._lock:
            self._attempt_history.append(success)
            if success and self._credits > 0:
                self._credits -= 1
            self._cycle_log.append({"ts": ts, "outcome": outcome, "duration_s": duration_s})
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT INTO cycles(ts,outcome,duration_s,interval_used) VALUES(?,?,?,?)",
                             (ts, outcome, duration_s, interval_used))
                conn.commit()
        except sqlite3.Error:
            pass
        self._persist_state()

    def optimal_interval_s(self) -> float:
        """Returns data-driven optimal interval in seconds using live error and improvement rates."""
        er = self._error_rate()
        ir = self._improvement_rate()
        raw = self._base_s * math.exp(er - ir)
        return max(self._min_s, min(self._max_s, raw))

    def upcoming_cycles(self) -> list:
        """Queries DB schedule table; returns next 5 scheduled timestamps with predicted intervals."""
        now = time.time()
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT scheduled_ts, predicted_interval FROM schedule WHERE scheduled_ts > ? ORDER BY scheduled_ts LIMIT 5",
                    (now,)).fetchall()
            return [{"scheduled_ts": r[0], "in_seconds": round(r[0] - now, 1),
                     "predicted_interval_s": r[1]} for r in rows]
        except sqlite3.Error:
            return []

    def performance_trend(self) -> float:
        """Returns EMA-smoothed slope: positive means Nova is improving."""
        with self._lock:
            hist = list(self._score_history)
        if len(hist) < 2:
            return 0.0
        return (hist[-1] - hist[0]) / max(len(hist), 1)

    def status(self) -> dict:
        """Returns snapshot dict with numeric keys suitable for ConsciousnessIntegrator Φ."""
        with self._lock:
            credits = self._credits
            nrt = self._next_run_ts
        er = self._error_rate()
        ir = self._improvement_rate()
        opt = self.optimal_interval_s()
        trend = self.performance_trend()
        ready = self.assess_readiness()
        return {
            "next_run_ts": round(nrt, 2),
            "credits": credits,
            "error_rate": round(er, 4),
            "improvement_rate": round(ir, 4),
            "optimal_interval_s": round(opt, 1),
            "trend": round(trend, 4),
            "is_ready": ready,
            "cycles": len(self._cycle_log),
            "confidence": round(1.0 - er, 4),
            "active": int(self._running),
        }

    def auto_cycle(self) -> None:
        """Daemon entry: sleeps until next_run_ts, fires evolution cycle if ready, reschedules."""
        pass

    def _daemon_loop(self) -> None:
        while self._running:
            with self._lock:
                wait = max(10.0, self._next_run_ts - time.time())
            time.sleep(min(wait, 60.0))
            if not self._running:
                break
            if time.time() < self._next_run_ts:
                continue
            ready = self.assess_readiness()
            if ready:
                t0 = time.time()
                try:
                    from AutonomousEvolutionEngine import AutonomousEvolutionEngine
                    aee = AutonomousEvolutionEngine()
                    aee.auto_cycle()
                    outcome = "success"
                except Exception:
                    outcome = "error"
                duration = time.time() - t0
                self.log_cycle(outcome, duration)
                with self._lock:
                    latest = list(self._score_history)
                q = latest[-1] if latest else 0.5
                self.schedule_next(q)
            else:
                self.schedule_next(self._ema_trend)
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal("Review evolution cycle outcomes and adapt interval", priority=1)
            except Exception:
                pass

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)
