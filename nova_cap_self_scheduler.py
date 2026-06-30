"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-30
"""

"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides when to run her own growth cycles based on live performance trends.
No human input required after __init__. Persists schedule across restarts.
"""

import math
import sqlite3
import statistics
import threading
import time
from collections import deque, OrderedDict
from typing import Any


class AdaptiveScheduler:
    """Nova self-schedules her own evolution cycles using adaptive interval math."""

    def __init__(self) -> None:
        """Initialise state, restore from SQLite, launch daemon thread."""
        self._base_s: float = 3600.0
        self._scores: deque = deque(maxlen=30)
        self._attempts: deque = deque(maxlen=20)
        self._credits: int = 10
        self._cycle_log: list = []
        self._db_path: str = "nova_scheduler.db"
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._next_run_ts: float = time.time() + self._base_s
        self._init_db()
        self._daemon_thread: threading.Thread = threading.Thread(
            target=self._daemon_loop, daemon=True, name="AdaptiveSchedulerDaemon"
        )
        self._daemon_thread.start()

    def _init_db(self) -> None:
        """Creates SQLite tables if absent; restores next_run_ts and credits from last row."""
        try:
            conn = sqlite3.connect(self._db_path)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    outcome TEXT,
                    duration_s REAL,
                    error_rate REAL,
                    improvement_rate REAL,
                    next_s REAL,
                    quality_score REAL DEFAULT 0.5
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scheduler_state (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    next_run_ts REAL,
                    credits INTEGER
                )
            """)
            conn.commit()
            row = cur.execute(
                "SELECT next_run_ts, credits FROM scheduler_state WHERE id=1"
            ).fetchone()
            if row:
                restored_ts, restored_credits = row
                if restored_ts > time.time():
                    self._next_run_ts = restored_ts
                self._credits = max(0, restored_credits)
            rows = cur.execute(
                "SELECT timestamp, quality_score FROM cycles ORDER BY timestamp DESC LIMIT 30"
            ).fetchall()
            for ts, qs in reversed(rows):
                self._scores.append((ts, qs))
            conn.close()
        except sqlite3.Error as exc:
            pass  # first boot; tables will be created fresh

    def _persist_state(self) -> None:
        """Writes next_run_ts and credits to scheduler_state table atomically."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT INTO scheduler_state (id, next_run_ts, credits)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET next_run_ts=excluded.next_run_ts,
                                               credits=excluded.credits
            """, (self._next_run_ts, self._credits))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

    def _compute_rates(self) -> tuple:
        """Returns (error_rate, improvement_rate) from current rolling windows."""
        with self._lock:
            attempts_snap = list(self._attempts)
            scores_snap = list(self._scores)
        error_rate = sum(1 for s in attempts_snap if not s) / max(len(attempts_snap), 1)
        if len(scores_snap) >= 2:
            first_score = scores_snap[0][1]
            last_score = scores_snap[-1][1]
            denom = max(len(scores_snap) - 1, 1)
            improvement_rate = (last_score - first_score) / denom
        else:
            improvement_rate = 0.0
        return error_rate, improvement_rate

    def _adaptive_interval(self, error_rate: float, improvement_rate: float) -> float:
        """Returns clamped adaptive interval in seconds."""
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(600.0, min(14400.0, raw))

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, recomputes adaptive interval, persists; returns next_s in seconds."""
        now = time.time()
        with self._lock:
            self._scores.append((now, current_quality))
        error_rate, improvement_rate = self._compute_rates()
        next_s = self._adaptive_interval(error_rate, improvement_rate)
        with self._lock:
            self._next_run_ts = now + next_s
        self._persist_state()
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "scheduler", "adaptive_interval", 1.0 - error_rate, improvement_rate > 0
            )
        except Exception:
            pass
        return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff performance trend slope is positive AND credits remain."""
        slope = self.performance_trend()
        with self._lock:
            credits = self._credits
        return slope > 0.0 and credits > 0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle row to SQLite; decrements credits on failure; records attempt."""
        error_rate, improvement_rate = self._compute_rates()
        with self._lock:
            self._attempts.append(outcome != "failure")
            if outcome == "failure":
                self._credits = max(0, self._credits - 1)
            credits_snap = self._credits
        next_s = self._adaptive_interval(error_rate, improvement_rate)
        now = time.time()
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT INTO cycles (timestamp, outcome, duration_s, error_rate,
                                    improvement_rate, next_s, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, outcome, duration_s, error_rate, improvement_rate, next_s,
                  self._scores[-1][1] if self._scores else 0.5))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass
        self._persist_state()

    def optimal_interval_s(self) -> float:
        """Queries top-10 DB cycles by quality; returns data-driven clamped interval."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute("""
                SELECT error_rate, improvement_rate FROM cycles
                ORDER BY quality_score DESC LIMIT 10
            """).fetchall()
            conn.close()
            if not rows:
                return self._base_s
            mean_err = statistics.mean(r[0] for r in rows)
            mean_imp = statistics.mean(r[1] for r in rows)
            return self._adaptive_interval(mean_err, mean_imp)
        except sqlite3.Error:
            return self._base_s

    def upcoming_cycles(self) -> list:
        """Returns list of 5 projected cycle dicts with scheduled_ts, interval, readiness."""
        error_rate, improvement_rate = self._compute_rates()
        interval = self._adaptive_interval(error_rate, improvement_rate)
        ready = self.assess_readiness()
        with self._lock:
            base_ts = self._next_run_ts
        result = []
        for i in range(5):
            ts = base_ts + i * interval
            result.append({
                "scheduled_ts": round(ts, 2),
                "estimated_interval_s": round(interval, 2),
                "readiness": ready,
                "eta_s": round(ts - time.time(), 2),
            })
        return result

    def performance_trend(self) -> float:
        """Returns least-squares slope over rolling score window; positive means improving."""
        with self._lock:
            scores_snap = list(self._scores)
        n = len(scores_snap)
        if n < 2:
            return 0.0
        xs = list(range(n))
        ys = [s for (_, s) in scores_snap]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        denominator = sum((xs[i] - x_mean) ** 2 for i in range(n)) + 1e-9
        return numerator / denominator

    def status(self) -> dict:
        """Returns numeric-keyed dict of scheduler health for ConsciousnessIntegrator Φ."""
        error_rate, improvement_rate = self._compute_rates()
        slope = self.performance_trend()
        with self._lock:
            next_run_ts = self._next_run_ts
            credits = self._credits
            cycles = len(self._cycle_log)
            items = len(self._scores)
        return {
            "next_run_in_s": round(next_run_ts - time.time(), 2),
            "credits": credits,
            "error_rate": round(error_rate, 4),
            "improvement_rate": round(improvement_rate, 6),
            "trend_slope": round(slope, 6),
            "ready": self.assess_readiness(),
            "items": items,
            "cycles": cycles,
            "confidence": round(max(0.0, 1.0 - error_rate), 4),
            "active": int(self._running),
            "optimal_interval_s": round(self.optimal_interval_s(), 2),
        }

    def _daemon_loop(self) -> None:
        """Background daemon: waits for next_run_ts, runs evolution cycle if ready."""
        while self._running:
            try:
                with self._lock:
                    wait_s = max(10.0, self._next_run_ts - time.time())
                time.sleep(min(wait_s, 60.0))
                if not self._running:
                    break
                with self._lock:
                    due = time.time() >= self._next_run_ts
                if not due:
                    continue
                ready = self.assess_readiness()
                start = time.time()
                outcome = "skipped"
                new_quality = 0.5
                if ready:
                    try:
                        from autonomous_evolution_engine import AutonomousEvolutionEngine
                        aee = AutonomousEvolutionEngine()
                        mutation = aee.propose_mutation()
                        score = aee.evaluate_mutation(mutation)
                        if score and score > 0.5:
                            aee.accept_mutation(mutation)
                            new_quality = float(score)
                            outcome = "success"
                        else:
                            outcome = "failure"
                            new_quality = float(score) if score else 0.3
                    except Exception:
                        outcome = "failure"
                        new_quality = 0.3
                duration_s = time.time() - start
                self.log_cycle(outcome, duration_s)
                self.schedule_next(new_quality)
                try:
                    from hierarchical_goal_planner import HierarchicalGoalPlanner
                    HierarchicalGoalPlanner().add_goal(
                        f"Review scheduler cycle outcome={outcome} quality={new_quality:.3f}",
                        priority=2
                    )
                except Exception:
                    pass
            except Exception:
                time.sleep(30.0)

    def propose(self) -> dict:
        """Returns concrete next action Nova should take based on current scheduler state."""
        slope = self.performance_trend()
        error_rate, improvement_rate = self._compute_rates()
        with self._lock:
            credits = self._credits
            next_run_ts = self._next_run_ts
        eta = next_run_ts - time.time()
        if credits == 0:
            action = "replenish_credits"
            reason = "No evolution credits remain; growth is paused."
        elif slope < -0.01:
            action = "run_diagnostic_cycle"
            reason = f"Negative trend slope {slope:.4f}; quality degrading."
        elif eta < 60:
            action = "trigger_evolution_cycle_now"
            reason = "Next cycle is imminent; system is primed."
        else:
            action = "continue_monitoring"
            reason = f"Healthy trend {slope:.4f}; next cycle in {eta:.0f}s."
        return {
            "action": action,
            "reason": reason,
            "eta_s": round(eta, 1),
            "credits": credits,
            "trend_slope": round(slope, 6),
        }

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)