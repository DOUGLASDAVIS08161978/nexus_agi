"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-30
"""

"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides autonomously when to run growth cycles based on live performance trends.
No human input required after __init__. Survives restarts via SQLite persistence.
"""
import collections
import math
import os
import sqlite3
import statistics
import threading
import time
from typing import Any

DB_PATH = "nova_scheduler.db"


class AdaptiveScheduler:
    """Nova's adaptive self-scheduling engine: paces her own evolution without human tuning."""

    def __init__(self) -> None:
        self._base_s: float = 3600.0
        self._min_s: float = 600.0
        self._max_s: float = 14400.0
        self._score_history: collections.deque = collections.deque(maxlen=60)
        self._attempt_window: collections.deque = collections.deque(maxlen=20)
        self._next_run_ts: float = time.time()
        self._cycle_count: int = 0
        self._credits: float = 10.0
        self._credit_regen_rate: float = 1.0
        self._last_credit_ts: float = time.time()
        self._db_path: str = DB_PATH
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._init_db()
        self._load_state()
        self._daemon_thread: threading.Thread = threading.Thread(
            target=self._daemon_loop, daemon=True, name="AdaptiveScheduler-daemon"
        )
        self._daemon_thread.start()

    # ── persistence ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scheduler_state
                   (key TEXT PRIMARY KEY, value REAL)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cycles
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, outcome INTEGER, duration_s REAL)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS score_history
                   (ts REAL, quality REAL)"""
            )
            conn.commit()

    def _load_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM scheduler_state")}
                if "next_run_ts" in rows:
                    self._next_run_ts = rows["next_run_ts"]
                if "cycle_count" in rows:
                    self._cycle_count = int(rows["cycle_count"])
                if "credits" in rows:
                    self._credits = rows["credits"]
                if "last_credit_ts" in rows:
                    self._last_credit_ts = rows["last_credit_ts"]
                for ts, q in conn.execute(
                    "SELECT ts, quality FROM score_history ORDER BY ts DESC LIMIT 60"
                ):
                    self._score_history.appendleft((ts, q))
                for (outcome,) in conn.execute(
                    "SELECT outcome FROM cycles ORDER BY id DESC LIMIT 20"
                ):
                    self._attempt_window.appendleft(bool(outcome))
        except sqlite3.Error:
            pass

    def _persist_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                for k, v in [
                    ("next_run_ts", self._next_run_ts),
                    ("cycle_count", float(self._cycle_count)),
                    ("credits", self._credits),
                    ("last_credit_ts", self._last_credit_ts),
                ]:
                    conn.execute(
                        "INSERT OR REPLACE INTO scheduler_state(key,value) VALUES(?,?)", (k, v)
                    )
                conn.commit()
        except sqlite3.Error:
            pass

    # ── credit regeneration ───────────────────────────────────────────────────

    def _regen_credits(self) -> None:
        elapsed_hours = (time.time() - self._last_credit_ts) / 3600.0
        self._credits = min(10.0, self._credits + elapsed_hours * self._credit_regen_rate)
        self._last_credit_ts = time.time()

    # ── core algorithm helpers ────────────────────────────────────────────────

    def _compute_rates(self) -> tuple[float, float]:
        scores = [s for (_, s) in list(self._score_history)]
        score_now = scores[-1] if scores else 0.0
        score_30_ago = scores[-30] if len(scores) >= 30 else (scores[0] if scores else 0.0)
        improvement_rate = (score_now - score_30_ago) / 30.0
        error_rate = sum(1 for x in self._attempt_window if not x) / max(len(self._attempt_window), 1)
        return improvement_rate, error_rate

    def _adaptive_interval(self, improvement_rate: float, error_rate: float) -> float:
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(self._min_s, min(self._max_s, raw))

    # ── public methods ────────────────────────────────────────────────────────

    def schedule_next(self, current_quality: float) -> float:
        """Records quality score, recomputes adaptive interval, persists next_run_ts; returns seconds until next cycle."""
        with self._lock:
            now = time.time()
            self._score_history.append((now, current_quality))
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("INSERT INTO score_history(ts,quality) VALUES(?,?)", (now, current_quality))
                    conn.commit()
            except sqlite3.Error:
                pass
            improvement_rate, error_rate = self._compute_rates()
            interval = self._adaptive_interval(improvement_rate, error_rate)
            self._next_run_ts = now + interval
            self._persist_state()
            try:
                from nova_tools import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning(
                    "AdaptiveScheduler", "schedule_next",
                    confidence=max(0.0, 1.0 - error_rate), success=True
                )
            except Exception:
                pass
            return interval

    def assess_readiness(self) -> bool:
        """Returns True iff performance trend slope > 0 and credits >= 1.0 after regeneration."""
        with self._lock:
            self._regen_credits()
            slope = self._performance_trend_locked()
            return slope > 0.0 and self._credits >= 1.0

    def log_cycle(self, outcome: bool, duration_s: float) -> None:
        """Appends outcome to attempt window, decrements credits, writes to SQLite, increments cycle count."""
        with self._lock:
            self._attempt_window.append(outcome)
            self._credits = max(0.0, self._credits - 1.0)
            self._cycle_count += 1
            now = time.time()
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "INSERT INTO cycles(ts,outcome,duration_s) VALUES(?,?,?)",
                        (now, int(outcome), duration_s),
                    )
                    conn.commit()
            except sqlite3.Error:
                pass
            self._persist_state()
            try:
                from nova_tools import HierarchicalGoalPlanner
                if self._cycle_count % 5 == 0:
                    HierarchicalGoalPlanner().add_goal(
                        f"Review AdaptiveScheduler performance after {self._cycle_count} cycles",
                        priority=3,
                    )
            except Exception:
                pass

    def optimal_interval_s(self) -> float:
        """Returns data-driven optimal interval in seconds using live error and improvement rates."""
        with self._lock:
            improvement_rate, error_rate = self._compute_rates()
            return self._adaptive_interval(improvement_rate, error_rate)

    def upcoming_cycles(self, n: int = 5) -> list:
        """Returns list of n projected unix timestamps for future cycles based on optimal interval."""
        with self._lock:
            step = self._adaptive_interval(*self._compute_rates())
            base = self._next_run_ts
            result = []
            for i in range(n):
                base += step
                result.append(round(base, 2))
            return result

    def _performance_trend_locked(self) -> float:
        history = list(self._score_history)
        if len(history) < 2:
            return 0.0
        t0 = history[0][0]
        xs = [t - t0 for (t, _) in history]
        ys = [s for (_, s) in history]
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)
        denom = n * sum_x2 - sum_x ** 2 + 1e-9
        return (n * sum_xy - sum_x * sum_y) / denom

    def performance_trend(self) -> float:
        """Computes OLS slope of quality vs elapsed seconds; positive = improving, negative = degrading."""
        with self._lock:
            return self._performance_trend_locked()

    def status(self) -> dict:
        """Returns snapshot dict with scheduling state, credits, error/improvement rates, and daemon health."""
        with self._lock:
            self._regen_credits()
            improvement_rate, error_rate = self._compute_rates()
            interval = self._adaptive_interval(improvement_rate, error_rate)
            slope = self._performance_trend_locked()
            return {
                "next_run_ts": round(self._next_run_ts, 2),
                "credits": round(self._credits, 3),
                "cycle_count": self._cycle_count,
                "error_rate": round(error_rate, 4),
                "improvement_rate": round(improvement_rate, 6),
                "trend_slope": round(slope, 8),
                "interval_s": round(interval, 1),
                "daemon_alive": self._daemon_thread is not None and self._daemon_thread.is_alive(),
                "items": self._cycle_count,
                "confidence": round(max(0.0, 1.0 - error_rate), 4),
                "active": 1 if self._running else 0,
                "entropy": round(
                    -sum(
                        p * math.log2(p + 1e-12)
                        for p in [error_rate, max(1e-9, 1.0 - error_rate)]
                    ),
                    4,
                ),
            }

    def auto_cycle(self) -> None:
        """Daemon-callable cycle: regenerates credits, checks readiness, schedules next run if due."""
        now = time.time()
        with self._lock:
            self._regen_credits()
            due = now >= self._next_run_ts
        if due:
            ready = self.assess_readiness()
            if ready:
                t0 = time.time()
                quality_estimate = 0.5 + 0.1 * self._performance_trend_locked()
                quality_estimate = max(0.0, min(1.0, quality_estimate))
                self.log_cycle(outcome=True, duration_s=time.time() - t0)
                self.schedule_next(quality_estimate)
            else:
                with self._lock:
                    improvement_rate, error_rate = self._compute_rates()
                    interval = self._adaptive_interval(improvement_rate, error_rate)
                    self._next_run_ts = now + interval
                    self._persist_state()
            try:
                from nova_tools import MetacognitiveMonitor
                improvement_rate, error_rate = self._compute_rates()
                MetacognitiveMonitor().log_reasoning(
                    "AdaptiveScheduler", "auto_cycle",
                    confidence=max(0.0, 1.0 - error_rate), success=ready if due else True
                )
            except Exception:
                pass

    def _daemon_loop(self) -> None:
        while self._running:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(60)

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)
