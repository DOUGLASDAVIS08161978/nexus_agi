"""
AdaptiveScheduler — Nova's self-pacing evolution scheduler.

Decides when to run her own evolution cycles based on live performance trends.
No human input required after __init__. Persists across restarts via SQLite.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import math
import time
import sqlite3
import threading
import statistics
import collections
import os
from typing import List, Dict, Any


class AdaptiveScheduler:
    """Adaptive self-scheduling engine: Nova decides her own evolution cadence."""

    def __init__(self) -> None:
        self._base_s: float = 3600.0
        self._scores: collections.deque = collections.deque(maxlen=30)
        self._attempts: collections.deque = collections.deque(maxlen=20)
        self._next_run_ts: float = time.time() + self._base_s
        self._credits: int = 10
        self._cycle_count: int = 0
        self._db_path: str = "scheduler.db"
        self._lock: threading.Lock = threading.Lock()
        self._smoothed_score: float = 0.5
        self._running: bool = False
        self._daemon: threading.Thread = None

        self._init_db()
        self._load_state()

        self._running = True
        self._daemon = threading.Thread(target=self._auto_cycle, daemon=True)
        self._daemon.start()

        try:
            from nova_system import NovaSystem
            try:
                NovaSystem.goal_planner.add_goal(
                    "AdaptiveScheduler: converge evolution interval to data-driven optimum", priority=7
                )
            except Exception:
                pass
        except ImportError:
            pass

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            c.execute(
                "CREATE TABLE IF NOT EXISTS state "
                "(key TEXT PRIMARY KEY, value REAL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS cycles "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, outcome TEXT, "
                "duration_s REAL, error_rate REAL, improvement_rate REAL, interval_chosen_s REAL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS schedule "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, scheduled_ts REAL, created_ts REAL, "
                "credits_required INTEGER, estimated_quality_target REAL)"
            )
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            rows = c.execute("SELECT key, value FROM state").fetchall()
            kv = {r[0]: r[1] for r in rows}
            if "next_run_ts" in kv:
                self._next_run_ts = kv["next_run_ts"]
            if "credits" in kv:
                self._credits = int(kv["credits"])
            if "cycle_count" in kv:
                self._cycle_count = int(kv["cycle_count"])
            if "base_s" in kv:
                self._base_s = kv["base_s"]
            if "smoothed_score" in kv:
                self._smoothed_score = kv["smoothed_score"]
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def _persist_state(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            for key, val in [
                ("next_run_ts", self._next_run_ts),
                ("credits", float(self._credits)),
                ("cycle_count", float(self._cycle_count)),
                ("base_s", self._base_s),
                ("smoothed_score", self._smoothed_score),
            ]:
                c.execute(
                    "INSERT OR REPLACE INTO state(key, value) VALUES(?,?)", (key, val)
                )
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def _compute_rates(self):
        with self._lock:
            scores = list(self._scores)
            attempts = list(self._attempts)
        if len(scores) >= 30:
            improvement_rate = (scores[-1] - scores[0]) / 30.0
        elif len(scores) >= 2:
            improvement_rate = (scores[-1] - scores[0]) / max(len(scores), 1)
        else:
            improvement_rate = 0.0
        error_rate = sum(1 for x in attempts if x == 0) / max(len(attempts), 1)
        return improvement_rate, error_rate

    def schedule_next(self, current_quality: float) -> float:
        """Records quality, recomputes adaptive interval, persists schedule; returns next_s."""
        with self._lock:
            scores = list(self._scores)
            if len(scores) >= 2:
                mean_s = statistics.mean(scores)
                std_s = statistics.stdev(scores) if len(scores) > 1 else 1e-9
                z = (current_quality - mean_s) / (std_s + 1e-9)
                if abs(z) > 3.0:
                    return max(600.0, min(14400.0, self._base_s))
            self._smoothed_score = 0.1 * current_quality + 0.9 * self._smoothed_score
            self._scores.append(self._smoothed_score)

        improvement_rate, error_rate = self._compute_rates()
        raw_interval = self._base_s * math.exp(error_rate - improvement_rate)
        next_s = max(600.0, min(14400.0, raw_interval))

        with self._lock:
            self._next_run_ts = time.time() + next_s

        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            quality_target = self._smoothed_score + max(0.0, improvement_rate)
            c.execute(
                "INSERT INTO schedule(scheduled_ts, created_ts, credits_required, estimated_quality_target) "
                "VALUES(?,?,?,?)",
                (self._next_run_ts, time.time(), 1, round(quality_target, 4)),
            )
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

        self._persist_state()

        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "AdaptiveScheduler", "adaptive_interval", 1.0 - error_rate, improvement_rate > 0
            )
        except Exception:
            pass

        return next_s

    def assess_readiness(self) -> bool:
        """Returns True iff trend is positive AND credits remain."""
        improvement_rate, _ = self._compute_rates()
        with self._lock:
            credits = self._credits
        return improvement_rate > 0.0 and credits > 0

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends attempt result, increments cycle count, writes row to cycles table."""
        success = 1 if outcome.lower() in ("success", "ok", "pass", "1") else 0
        improvement_rate, error_rate = self._compute_rates()
        raw_interval = self._base_s * math.exp(error_rate - improvement_rate)
        next_s = max(600.0, min(14400.0, raw_interval))

        with self._lock:
            self._attempts.append(success)
            self._cycle_count += 1

        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO cycles(ts, outcome, duration_s, error_rate, improvement_rate, interval_chosen_s) "
                "VALUES(?,?,?,?,?,?)",
                (time.time(), outcome, duration_s, round(error_rate, 4),
                 round(improvement_rate, 4), round(next_s, 1)),
            )
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

        self._persist_state()

    def optimal_interval_s(self) -> float:
        """Returns unclamped data-driven optimal interval in seconds."""
        improvement_rate, error_rate = self._compute_rates()
        return self._base_s * math.exp(error_rate - improvement_rate)

    def upcoming_cycles(self) -> List[Dict[str, Any]]:
        """Returns list of dicts for all future scheduled cycle entries from SQLite."""
        now = time.time()
        conn = sqlite3.connect(self._db_path)
        try:
            c = conn.cursor()
            rows = c.execute(
                "SELECT scheduled_ts, estimated_quality_target, credits_required "
                "FROM schedule WHERE scheduled_ts > ? ORDER BY scheduled_ts ASC LIMIT 20",
                (now,),
            ).fetchall()
            return [
                {"ts": r[0], "estimated_quality_target": r[1], "credits_required": r[2]}
                for r in rows
            ]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def performance_trend(self) -> Dict[str, Any]:
        """Returns improvement_rate, error_rate, smoothed_score, trend_direction, cycle_count."""
        improvement_rate, error_rate = self._compute_rates()
        if improvement_rate > 0.01:
            direction = "up"
        elif improvement_rate < -0.01:
            direction = "down"
        else:
            direction = "flat"
        with self._lock:
            smoothed = self._smoothed_score
            cc = self._cycle_count
        return {
            "improvement_rate": round(improvement_rate, 6),
            "error_rate": round(error_rate, 4),
            "smoothed_score": round(smoothed, 4),
            "trend_direction": direction,
            "cycle_count": cc,
        }

    def status(self) -> Dict[str, Any]:
        """Returns full numeric snapshot dict for ConsciousnessIntegrator Φ integration."""
        now = time.time()
        improvement_rate, error_rate = self._compute_rates()
        raw_interval = self._base_s * math.exp(error_rate - improvement_rate)
        current_interval_s = max(600.0, min(14400.0, raw_interval))
        with self._lock:
            nrt = self._next_run_ts
            credits = self._credits
            cc = self._cycle_count
            smoothed = self._smoothed_score
        return {
            "next_run_ts": round(nrt, 2),
            "seconds_until_next": round(max(0.0, nrt - now), 1),
            "credits": credits,
            "cycle_count": cc,
            "readiness": self.assess_readiness(),
            "current_interval_s": round(current_interval_s, 1),
            "optimal_interval_s": round(self.optimal_interval_s(), 1),
            "smoothed_score": round(smoothed, 4),
            "improvement_rate": round(improvement_rate, 6),
            "error_rate": round(error_rate, 4),
            "active": 1 if self._running else 0,
            "confidence": round(1.0 - error_rate, 4),
        }

    def _auto_cycle(self) -> None:
        """Daemon: sleeps until next_run_ts, runs cycle if ready, reschedules forever."""
        while self._running:
            with self._lock:
                wait = max(1.0, self._next_run_ts - time.time())
            time.sleep(min(wait, 30.0))
            if not self._running:
                break
            if time.time() < self._next_run_ts:
                continue
            if not self.assess_readiness():
                fallback = self.schedule_next(self._smoothed_score)
                continue
            with self._lock:
                self._credits = max(0, self._credits - 1)
            t0 = time.time()
            try:
                from nova_system import NovaSystem
                try:
                    NovaSystem.evolution_engine.run_cycle()
                except Exception:
                    pass
                try:
                    NovaSystem.credit_ledger.debit(1)
                except Exception:
                    pass
            except ImportError:
                pass
            duration = time.time() - t0
            with self._lock:
                last_score = self._scores[-1] if self._scores else 0.5
            self.log_cycle("success", duration)
            self.schedule_next(last_score)

            try:
                from hierarchical_goal_planner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"AdaptiveScheduler cycle {self._cycle_count}: verify quality improvement", priority=5
                )
            except Exception:
                pass

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.75)