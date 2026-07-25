"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Computes optimal cycle intervals from live performance trends; no human input required.
Persists schedule and cycle history in SQLite; daemon thread drives autonomous operation.
"""

import math
import time
import sqlite3
import threading
import collections
import statistics
from datetime import datetime, timezone
from typing import Optional


class AdaptiveScheduler:
    """Adaptive self-scheduling engine: Nova decides when to run her own evolution cycles."""

    def __init__(self) -> None:
        """Initialise state, restore persisted schedule, launch daemon thread."""
        self._base_s: float = 3600.0
        self._min_s: float = 600.0
        self._max_s: float = 14400.0
        self._score_history: collections.deque = collections.deque(maxlen=60)
        self._attempt_log: collections.deque = collections.deque(maxlen=20)
        self._next_run_ts: float = time.time() + 3600.0
        self._credits: int = 10
        self._db_path: str = "nova_scheduler.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._running: bool = True
        self._lock: threading.Lock = threading.Lock()
        self._last_cycle_outcome: str = "none"
        self._total_cycles: int = 0
        self._ema_trend: float = 0.0
        self._init_db()
        self._restore_state()
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True, name="AdaptiveScheduler")
        self._daemon_thread.start()

    def _init_db(self) -> None:
        """Create SQLite tables if they do not exist."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                outcome TEXT,
                duration_s REAL,
                ts REAL
            );
            CREATE TABLE IF NOT EXISTS scheduled_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts REAL,
                iso_time TEXT
            );
            CREATE TABLE IF NOT EXISTS scheduler_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._conn.commit()

    def _restore_state(self) -> None:
        """Load persisted next_run_ts, credits, total_cycles from DB."""
        cur = self._conn.cursor()
        for key, attr, cast in [
            ("next_run_ts", "_next_run_ts", float),
            ("credits", "_credits", int),
            ("total_cycles", "_total_cycles", int),
            ("ema_trend", "_ema_trend", float),
        ]:
            row = cur.execute("SELECT value FROM scheduler_state WHERE key=?", (key,)).fetchone()
            if row:
                try:
                    setattr(self, f"_{attr.lstrip('_')}", cast(row[0]))
                except (ValueError, TypeError):
                    pass
        rows = cur.execute("SELECT outcome, duration_s FROM cycles ORDER BY ts DESC LIMIT 20").fetchall()
        for outcome, _ in reversed(rows):
            self._attempt_log.append((time.time(), outcome == "success"))

    def _persist_state(self) -> None:
        """Write current scheduler state to DB (must be called under lock)."""
        cur = self._conn.cursor()
        for key, val in [
            ("next_run_ts", self._next_run_ts),
            ("credits", self._credits),
            ("total_cycles", self._total_cycles),
            ("ema_trend", self._ema_trend),
        ]:
            cur.execute("INSERT OR REPLACE INTO scheduler_state(key,value) VALUES(?,?)", (key, str(val)))
        self._conn.commit()

    def _compute_interval(self) -> float:
        """Compute adaptive interval from current score and error state."""
        window = list(self._score_history)[-30:]
        if len(window) >= 2:
            improvement_rate = (window[-1][1] - window[0][1]) / max(len(window) - 1, 1)
        else:
            improvement_rate = 0.0
        error_rate = sum(1 for _, ok in self._attempt_log if not ok) / max(len(self._attempt_log), 1)
        raw_s = self._base_s * math.exp(error_rate - improvement_rate)
        return max(self._min_s, min(self._max_s, raw_s)), improvement_rate, error_rate

    def schedule_next(self, current_quality: float) -> float:
        """Records current_quality, recomputes next_s, persists next_run_ts; returns next_s."""
        with self._lock:
            self._score_history.append((time.time(), current_quality))
            next_s, improvement_rate, _ = self._compute_interval()
            self._ema_trend = 0.15 * improvement_rate + 0.85 * self._ema_trend
            self._next_run_ts = time.time() + next_s
            iso = datetime.fromtimestamp(self._next_run_ts, tz=timezone.utc).isoformat()
            cur = self._conn.cursor()
            cur.execute("INSERT INTO scheduled_cycles(run_ts, iso_time) VALUES(?,?)", (self._next_run_ts, iso))
            self._persist_state()
            self._conn.commit()
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(f"Run evolution cycle at {iso}", priority=5)
            except Exception:
                pass
            return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff EMA trend is positive AND credits remain; thread-safe."""
        with self._lock:
            return self._ema_trend > 0.0 and self._credits > 0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle record to DB, updates attempt log, increments total_cycles, decrements credits."""
        with self._lock:
            ts = time.time()
            cur = self._conn.cursor()
            cur.execute("INSERT INTO cycles(outcome, duration_s, ts) VALUES(?,?,?)", (outcome, duration_s, ts))
            self._conn.commit()
            self._attempt_log.append((ts, outcome == "success"))
            self._last_cycle_outcome = outcome
            self._total_cycles += 1
            self._credits = max(0, self._credits - 1)
            self._persist_state()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            confidence = 1.0 if outcome == "success" else 0.3
            MetacognitiveMonitor().log_reasoning("AdaptiveScheduler", "adaptive_interval", confidence, outcome == "success")
        except Exception:
            pass

    def optimal_interval_s(self) -> float:
        """Evaluates adaptive formula at current live state; returns clamped data-driven optimum."""
        with self._lock:
            result, _, _ = self._compute_interval()
            return result

    def upcoming_cycles(self) -> list:
        """Queries scheduled_cycles for future rows; returns list of {id, run_ts, iso_time} dicts ascending."""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT id, run_ts, iso_time FROM scheduled_cycles WHERE run_ts > ? ORDER BY run_ts ASC", (now,)
            ).fetchall()
        return [{"id": r[0], "run_ts": r[1], "iso_time": r[2]} for r in rows]

    def performance_trend(self) -> float:
        """Returns EMA-smoothed improvement_rate reflecting directional momentum."""
        with self._lock:
            return self._ema_trend

    def status(self) -> dict:
        """Returns snapshot dict with numeric keys for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            now = time.time()
            _, _, error_rate = self._compute_interval()
            return {
                "next_run_ts": self._next_run_ts,
                "seconds_until_next": max(0.0, self._next_run_ts - now),
                "credits": self._credits,
                "total_cycles": self._total_cycles,
                "ema_trend": round(self._ema_trend, 6),
                "error_rate": round(error_rate, 4),
                "last_outcome": self._last_cycle_outcome,
                "is_running": self._running,
                "active": 1 if self._running else 0,
                "confidence": max(0.0, min(1.0, 0.5 + self._ema_trend * 10)),
                "cycles": self._total_cycles,
                "entropy": round(-sum(
                    p * math.log2(p + 1e-12)
                    for p in ([error_rate, 1.0 - error_rate])
                ), 4),
            }

    def auto_cycle(self) -> None:
        """Daemon loop: sleeps until next_run_ts, assesses readiness, triggers cycle, reschedules."""
        while self._running:
            with self._lock:
                sleep_s = max(1.0, self._next_run_ts - time.time())
            time.sleep(min(sleep_s, 30.0))
            if not self._running:
                break
            if time.time() < self._next_run_ts:
                continue
            ready = self.assess_readiness()
            if ready:
                t0 = time.time()
                outcome = "success"
                try:
                    from AutonomousEvolutionEngine import AutonomousEvolutionEngine
                    AutonomousEvolutionEngine().auto_cycle()
                except Exception:
                    outcome = "skipped"
                duration = time.time() - t0
                self.log_cycle(outcome, duration)
                with self._lock:
                    last_score = self._score_history[-1][1] if self._score_history else 0.5
                self.schedule_next(last_score)
            else:
                self.schedule_next(self._score_history[-1][1] if self._score_history else 0.5)

    def _daemon_loop(self) -> None:
        """Internal entry point for daemon thread calling auto_cycle."""
        try:
            self.auto_cycle()
        except Exception:
            pass

    def add_credits(self, n: int) -> None:
        """Increments self._credits by n under lock, persists updated balance to DB."""
        with self._lock:
            self._credits += max(0, n)
            self._persist_state()

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)
