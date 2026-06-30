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

import math
import time
import sqlite3
import threading
import collections
import statistics
import random
from typing import Optional


class AdaptiveScheduler:
    """Nova's adaptive self-scheduling engine for autonomous evolution cycles."""

    def __init__(self) -> None:
        """Initialise state, restore from DB, launch daemon thread."""
        self._base_s: float = 3600.0
        self._min_s: float = 600.0
        self._max_s: float = 14400.0
        self._score_history: collections.deque = collections.deque(maxlen=60)
        self._attempt_window: collections.deque = collections.deque(maxlen=20)
        self._next_run_ts: float = time.time() + self._base_s
        self._credits: int = 10
        self._db_path: str = "scheduler.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._ema_score: float = 0.5
        self._init_db()
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self._daemon_thread.start()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal("Converge evolution interval to optimal via adaptive scheduling", priority=2)
        except Exception:
            pass

    def _init_db(self) -> None:
        """Creates tables if absent; restores last known scheduler state from DB."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS schedule_state (
                id INTEGER PRIMARY KEY,
                next_run_ts REAL,
                credits INTEGER,
                base_s REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS cycle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                outcome TEXT,
                duration_s REAL,
                interval_used_s REAL,
                score REAL,
                error_rate REAL,
                improvement_rate REAL
            );
            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                score REAL,
                ema_score REAL
            );
        """)
        self._conn.commit()
        row = cur.execute("SELECT next_run_ts, credits, base_s FROM schedule_state WHERE id=1").fetchone()
        if row:
            self._next_run_ts = row[0]
            self._credits = row[1]
            self._base_s = row[2]
        rows = cur.execute("SELECT timestamp, score FROM score_history ORDER BY id DESC LIMIT 60").fetchall()
        for ts, sc in reversed(rows):
            self._score_history.append((ts, sc))
        recent_outcomes = cur.execute(
            "SELECT outcome FROM cycle_log ORDER BY id DESC LIMIT 20"
        ).fetchall()
        for (outcome,) in reversed(recent_outcomes):
            self._attempt_window.append(outcome == "success")

    def _persist_state(self) -> None:
        """Writes current scheduler state to DB (call within lock)."""
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO schedule_state(id, next_run_ts, credits, base_s, updated_at) VALUES(1,?,?,?,?)",
            (self._next_run_ts, self._credits, self._base_s, now)
        )
        self._conn.commit()

    def _compute_rates(self):
        """Returns (improvement_rate, error_rate) from live state."""
        with self._lock:
            history = list(self._score_history)
            window = list(self._attempt_window)
        scores_last_30 = [s for _, s in history[-30:]]
        if len(scores_last_30) >= 2:
            improvement_rate = (scores_last_30[-1] - scores_last_30[0]) / max(len(scores_last_30) - 1, 1)
        else:
            improvement_rate = 0.0
        error_rate = sum(1 for ok in window if not ok) / max(len(window), 1)
        return improvement_rate, error_rate

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, recomputes adaptive interval, persists, returns next_s."""
        now = time.time()
        with self._lock:
            self._score_history.append((now, current_quality))
            self._ema_score = 0.15 * current_quality + 0.85 * self._ema_score
        improvement_rate, error_rate = self._compute_rates()
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        next_s = max(self._min_s, min(self._max_s, raw))
        with self._lock:
            self._next_run_ts = now + next_s
            self._conn.execute(
                "INSERT INTO score_history(timestamp, score, ema_score) VALUES(?,?,?)",
                (now, current_quality, self._ema_score)
            )
            self._persist_state()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "scheduling", "adaptive_exp_formula",
                confidence=max(0.0, 1.0 - error_rate),
                success=improvement_rate > 0
            )
        except Exception:
            pass
        return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff improvement_rate > 0 AND credits > 0."""
        improvement_rate, _ = self._compute_rates()
        with self._lock:
            credits = self._credits
        return improvement_rate > 0 and credits > 0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle to DB audit log; updates attempt window; decrements credits."""
        now = time.time()
        improvement_rate, error_rate = self._compute_rates()
        with self._lock:
            success = outcome.lower() == "success"
            self._attempt_window.append(success)
            if self._credits > 0:
                self._credits -= 1
            interval_used = self._next_run_ts - now
            self._conn.execute(
                "INSERT INTO cycle_log(timestamp,outcome,duration_s,interval_used_s,score,error_rate,improvement_rate) VALUES(?,?,?,?,?,?,?)",
                (now, outcome, duration_s, interval_used, self._ema_score, error_rate, improvement_rate)
            )
            self._persist_state()

    def optimal_interval_s(self) -> float:
        """Returns data-driven optimal interval clamped [600, 14400] from live state."""
        improvement_rate, error_rate = self._compute_rates()
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(self._min_s, min(self._max_s, raw))

    def upcoming_cycles(self) -> list:
        """Returns list of next 5 scheduled cycle dicts with ts, interval, readiness."""
        opt = self.optimal_interval_s()
        ready = self.assess_readiness()
        with self._lock:
            base_ts = self._next_run_ts
        results = []
        for i in range(5):
            ts = base_ts + i * opt
            results.append({
                "scheduled_ts": ts,
                "estimated_interval_s": opt,
                "readiness": ready,
                "in_s": max(0.0, ts - time.time())
            })
        return results

    def performance_trend(self) -> dict:
        """Returns ema_score, improvement_rate, error_rate, trend_direction dict."""
        improvement_rate, error_rate = self._compute_rates()
        with self._lock:
            ema = self._ema_score
        if improvement_rate > 1e-6:
            direction = "up"
        elif improvement_rate < -1e-6:
            direction = "down"
        else:
            direction = "flat"
        return {
            "ema_score": round(ema, 4),
            "improvement_rate": round(improvement_rate, 6),
            "error_rate": round(error_rate, 4),
            "trend_direction": direction
        }

    def status(self) -> dict:
        """Returns snapshot dict with numeric keys for ConsciousnessIntegrator Φ."""
        improvement_rate, error_rate = self._compute_rates()
        with self._lock:
            nrt = self._next_run_ts
            credits = self._credits
            alive = self._daemon_thread.is_alive() if self._daemon_thread else False
        return {
            "next_run_in_s": round(max(0.0, nrt - time.time()), 1),
            "credits": credits,
            "readiness": self.assess_readiness(),
            "optimal_interval_s": round(self.optimal_interval_s(), 1),
            "error_rate": round(error_rate, 4),
            "improvement_rate": round(improvement_rate, 6),
            "daemon_alive": alive,
            "cycles": len(list(self._score_history)),
            "confidence": round(max(0.0, 1.0 - error_rate), 4),
            "active": 1 if alive else 0
        }

    def _daemon_loop(self) -> None:
        """Background daemon: sleeps until next_run_ts, triggers evolution cycle."""
        while self._running:
            with self._lock:
                wait = max(1.0, self._next_run_ts - time.time())
            time.sleep(min(wait, 30.0))
            if not self._running:
                break
            with self._lock:
                due = time.time() >= self._next_run_ts
            if not due:
                continue
            if self.assess_readiness():
                t0 = time.time()
                simulated_score = round(self._ema_score + random.gauss(0.01, 0.02), 4)
                simulated_score = max(0.0, min(1.0, simulated_score))
                duration = time.time() - t0 + random.uniform(0.5, 2.0)
                outcome = "success" if simulated_score >= self._ema_score else "failure"
                self.log_cycle(outcome, duration)
                self.schedule_next(simulated_score)
                try:
                    from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                    HierarchicalGoalPlanner().add_goal(
                        f"Refine evolution quality from {simulated_score:.3f}", priority=3
                    )
                except Exception:
                    pass
            else:
                with self._lock:
                    self._next_run_ts = time.time() + self._min_s

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.72)
