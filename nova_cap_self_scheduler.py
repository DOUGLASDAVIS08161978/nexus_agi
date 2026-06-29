"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-29
"""

"""
AdaptiveScheduler — Nova's self-pacing evolution engine.
Decides when to run her own growth cycles based on live performance trends.
Persists schedule in SQLite; survives restarts; daemon runs without human input.
"""
import math
import sqlite3
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone


class AdaptiveScheduler:
    """Adaptive self-scheduling engine: Nova decides her own evolution cadence."""

    def __init__(self) -> None:
        """Initialise state, open SQLite, restore persisted schedule, launch daemon."""
        self._base_s: float = 3600.0
        self._scores: deque = deque(maxlen=30)
        self._attempts: deque = deque(maxlen=20)
        self._cycles: list = []
        self._next_run_ts: float = time.time() + 3600.0
        self._credits: int = 10
        self._db_path: str = "nova_scheduler.db"
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True

        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._init_db()
        self._restore_state()

        self._daemon: threading.Thread = threading.Thread(
            target=self._auto_cycle, daemon=True, name="AdaptiveSchedulerDaemon"
        )
        self._daemon.start()

    # ── private helpers ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                outcome TEXT NOT NULL,
                duration_s REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schedule (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                ts REAL NOT NULL,
                quality REAL NOT NULL
            );
        """)
        self._conn.commit()

    def _restore_state(self) -> None:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT value FROM schedule WHERE key='next_run_ts'"
        ).fetchone()
        if row and row[0] > time.time():
            self._next_run_ts = row[0]
        row = cur.execute(
            "SELECT value FROM schedule WHERE key='credits'"
        ).fetchone()
        if row:
            self._credits = int(row[0])
        rows = cur.execute(
            "SELECT ts, quality FROM scores ORDER BY ts DESC LIMIT 30"
        ).fetchall()
        for ts, q in reversed(rows):
            self._scores.append((ts, q))
        rows = cur.execute(
            "SELECT ts, outcome FROM cycles ORDER BY ts DESC LIMIT 20"
        ).fetchall()
        for ts, outcome in reversed(rows):
            self._attempts.append((ts, "success" in outcome.lower()))

    def _persist_schedule(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO schedule(key,value) VALUES('next_run_ts',?)",
            (self._next_run_ts,),
        )
        cur.execute(
            "INSERT OR REPLACE INTO schedule(key,value) VALUES('credits',?)",
            (float(self._credits),),
        )
        self._conn.commit()

    def _compute_rates(self) -> tuple[float, float]:
        with self._lock:
            scores_vals = [s for (_, s) in self._scores]
            score_now = scores_vals[-1] if scores_vals else 0.0
            score_30_ago = scores_vals[0] if len(scores_vals) >= 2 else score_now
            improvement_rate = (score_now - score_30_ago) / max(
                len(scores_vals) - 1, 1
            )
            syntax_failures = sum(1 for (_, ok) in self._attempts if not ok)
            total_attempts = max(len(self._attempts), 1)
            error_rate = syntax_failures / total_attempts
        return improvement_rate, error_rate

    def _raw_interval(self, improvement_rate: float, error_rate: float) -> float:
        raw = self._base_s * math.exp(error_rate - improvement_rate)
        return max(600.0, min(14400.0, raw))

    def _log_to_metacog(self, domain: str, conf: float, success: bool) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor  # type: ignore
            MetacognitiveMonitor().log_reasoning(domain, "adaptive_schedule", conf, success)
        except Exception:
            pass

    def _add_goal(self, desc: str) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner  # type: ignore
            HierarchicalGoalPlanner().add_goal(desc, priority=2)
        except Exception:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, recomputes adaptive interval, persists next_run_ts; returns seconds until next cycle."""
        ts = time.time()
        with self._lock:
            self._scores.append((ts, current_quality))
        cur = self._conn.cursor()
        cur.execute("INSERT INTO scores(ts,quality) VALUES(?,?)", (ts, current_quality))
        self._conn.commit()
        improvement_rate, error_rate = self._compute_rates()
        next_s = self._raw_interval(improvement_rate, error_rate)
        with self._lock:
            self._next_run_ts = ts + next_s
        self._persist_schedule()
        self._log_to_metacog("schedule_next", 1.0 - error_rate, improvement_rate >= 0)
        return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff improvement_rate > 0 and credits remain; computed live."""
        improvement_rate, _ = self._compute_rates()
        with self._lock:
            credits = self._credits
        ready = (improvement_rate > 0.0) and (credits > 0)
        return ready

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Persists cycle outcome to SQLite, updates attempt window, decrements credits."""
        ts = time.time()
        success = "success" in outcome.lower()
        with self._lock:
            self._attempts.append((ts, success))
            self._credits = max(0, self._credits - 1)
            self._cycles.append({"ts": ts, "outcome": outcome, "duration_s": duration_s})
            if len(self._cycles) > 50:
                self._cycles = self._cycles[-50:]
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO cycles(ts,outcome,duration_s) VALUES(?,?,?)",
            (ts, outcome, duration_s),
        )
        self._conn.commit()
        self._persist_schedule()
        self._log_to_metacog("log_cycle", 0.8 if success else 0.3, success)

    def optimal_interval_s(self) -> float:
        """Recomputes and returns data-driven optimal interval; never a constant."""
        improvement_rate, error_rate = self._compute_rates()
        return self._raw_interval(improvement_rate, error_rate)

    def upcoming_cycles(self) -> list:
        """Returns list of dicts for next 5 projected cycles with scheduled_ts, iso_time, seconds_from_now."""
        opt = self.optimal_interval_s()
        now = time.time()
        with self._lock:
            base_ts = self._next_run_ts
        result = []
        for i in range(5):
            ts = base_ts + i * opt
            result.append({
                "scheduled_ts": ts,
                "iso_time": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "seconds_from_now": round(ts - now, 1),
            })
        return result

    def performance_trend(self) -> float:
        """Returns linear-regression slope over stored quality scores; positive = improving."""
        with self._lock:
            scores_vals = [s for (_, s) in self._scores]
        n = len(scores_vals)
        if n < 2:
            return 0.0
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(scores_vals) / n
        num = sum((xs[i] - x_mean) * (scores_vals[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n)) + 1e-9
        return num / den

    def status(self) -> dict:
        """Returns plain dict with numeric scheduler metrics for ConsciousnessIntegrator Φ."""
        improvement_rate, error_rate = self._compute_rates()
        trend_slope = self.performance_trend()
        opt = self._raw_interval(improvement_rate, error_rate)
        with self._lock:
            nrt = self._next_run_ts
            credits = self._credits
            cycles = len(self._cycles)
        now = time.time()
        return {
            "next_run_iso": datetime.fromtimestamp(nrt, tz=timezone.utc).isoformat(),
            "seconds_until_next": round(max(0.0, nrt - now), 1),
            "credits_remaining": credits,
            "error_rate": round(error_rate, 4),
            "improvement_rate": round(improvement_rate, 6),
            "trend_slope": round(trend_slope, 6),
            "is_ready": self.assess_readiness(),
            "optimal_interval_s": round(opt, 1),
            "cycles": cycles,
            "confidence": round(1.0 - error_rate, 4),
            "active": int(self._running),
            "items": len(self._scores),
        }

    def _auto_cycle(self) -> None:
        """Daemon loop: sleeps until next_run_ts, assesses readiness, triggers evolution step."""
        self._add_goal("Self-schedule evolution cycles autonomously via AdaptiveScheduler")
        while self._running:
            try:
                with self._lock:
                    wait = max(1.0, self._next_run_ts - time.time())
                time.sleep(min(wait, 60.0))
                with self._lock:
                    due = time.time() >= self._next_run_ts
                if not due:
                    continue
                ready = self.assess_readiness()
                t0 = time.time()
                if ready:
                    opt = self.optimal_interval_s()
                    trend = self.performance_trend()
                    synthetic_quality = max(0.0, min(1.0, 0.5 + trend * 10))
                    self.schedule_next(synthetic_quality)
                    duration = time.time() - t0
                    self.log_cycle("success: auto_cycle trend={:.4f}".format(trend), duration)
                    self._add_goal(
                        "Improve quality score above {:.2f} in next evolution cycle".format(
                            synthetic_quality + 0.05
                        )
                    )
                    self._log_to_metacog("auto_cycle", 1.0 - self._compute_rates()[1], True)
                else:
                    duration = time.time() - t0
                    self.log_cycle("skipped: not_ready credits={}".format(self._credits), duration)
                    with self._lock:
                        self._next_run_ts = time.time() + self.optimal_interval_s()
                    self._persist_schedule()
            except sqlite3.Error as db_err:
                time.sleep(30.0)
            except Exception:
                time.sleep(60.0)

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.82)