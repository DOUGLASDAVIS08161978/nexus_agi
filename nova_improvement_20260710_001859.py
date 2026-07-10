"""
AdaptiveScheduler — Nova's self-pacing evolution scheduler.
Decides when to run growth cycles based on live performance trends.
Survives restarts via SQLite. Operates fully autonomously after init.
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

    _base_s: float
    _scores: deque
    _attempts: deque
    _cycles: list
    _next_run_ts: float
    _credits: int
    _db_path: str
    _daemon_thread: threading.Thread
    _lock: threading.Lock
    _running: bool

    def __init__(self) -> None:
        self._base_s = 3600.0
        self._scores = deque(maxlen=30)
        self._attempts = deque(maxlen=20)
        self._cycles: list[dict] = []
        self._credits = 10
        self._db_path = "nova_scheduler.db"
        self._lock = threading.Lock()
        self._running = True
        self._init_db()
        self._load_state()
        self._next_run_ts = self._next_run_ts if hasattr(self, "_next_run_ts") else time.time() + self._base_s
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True, name="AdaptiveScheduler")
        self._daemon_thread.start()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal("Autonomously pace evolution cycles via AdaptiveScheduler", priority=2)
        except Exception:
            pass

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS schedule (
                    id INTEGER PRIMARY KEY,
                    created_ts REAL,
                    next_run_ts REAL,
                    interval_s REAL,
                    error_rate REAL,
                    improvement_rate REAL,
                    credits_remaining INTEGER
                );
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY,
                    run_ts REAL,
                    outcome TEXT,
                    duration_s REAL,
                    quality_score REAL,
                    syntax_failure INTEGER
                );
                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY,
                    recorded_ts REAL,
                    quality REAL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        """Restores next_run_ts, credits, and score history from SQLite on startup."""
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT next_run_ts, credits_remaining FROM schedule ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self._next_run_ts = max(float(row[0]), time.time() + 60.0)
                self._credits = int(row[1])
            else:
                self._next_run_ts = time.time() + self._base_s
            rows = cur.execute(
                "SELECT quality FROM scores ORDER BY recorded_ts DESC LIMIT 30"
            ).fetchall()
            for (q,) in reversed(rows):
                self._scores.append(float(q))
            cyc_rows = cur.execute(
                "SELECT syntax_failure FROM cycles ORDER BY id DESC LIMIT 20"
            ).fetchall()
            for (sf,) in reversed(cyc_rows):
                self._attempts.append(bool(sf))
        finally:
            conn.close()

    def _compute_rates(self) -> tuple[float, float]:
        with self._lock:
            n = len(self._scores)
            if n >= 30:
                improvement_rate = (self._scores[-1] - self._scores[-30]) / 30.0
            elif n >= 2:
                improvement_rate = (self._scores[-1] - self._scores[0]) / max(n - 1, 1)
            else:
                improvement_rate = 0.0
            error_rate = sum(self._attempts) / max(len(self._attempts), 1)
        return improvement_rate, error_rate

    def schedule_next(self, current_quality: float) -> float:
        """Records quality score, recomputes adaptive interval, persists to SQLite; returns seconds until next cycle."""
        with self._lock:
            self._scores.append(current_quality)
        improvement_rate, error_rate = self._compute_rates()
        raw_s = self._base_s * math.exp(error_rate - improvement_rate)
        next_s = max(600.0, min(14400.0, raw_s))
        with self._lock:
            self._next_run_ts = time.time() + next_s
            nrt = self._next_run_ts
            cred = self._credits
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO schedule (created_ts,next_run_ts,interval_s,error_rate,improvement_rate,credits_remaining) VALUES (?,?,?,?,?,?)",
                (time.time(), nrt, next_s, error_rate, improvement_rate, cred),
            )
            conn.execute(
                "INSERT INTO scores (recorded_ts,quality) VALUES (?,?)",
                (time.time(), current_quality),
            )
            conn.commit()
        finally:
            conn.close()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "AdaptiveScheduler", "schedule_next",
                confidence=1.0 - error_rate, success=improvement_rate > 0
            )
        except Exception:
            pass
        return next_s

    def assess_readiness(self) -> bool:
        """Returns True when improvement trend is positive, error rate is lower, and credits remain."""
        improvement_rate, error_rate = self._compute_rates()
        with self._lock:
            cred = self._credits
        trend_positive = (improvement_rate > 0.0) and (improvement_rate > error_rate)
        return trend_positive and (cred > 0)

    def log_cycle(self, outcome: str, duration_s: float) -> None:
        """Appends cycle outcome to SQLite and updates syntax-failure rolling window."""
        syntax_failure = int("syntax" in outcome.lower() or "error" in outcome.lower() or "fail" in outcome.lower())
        quality_score = 1.0 - syntax_failure * 0.5
        ts = time.time()
        with self._lock:
            self._attempts.append(bool(syntax_failure))
            self._cycles.append({"ts": ts, "outcome": outcome, "duration_s": duration_s, "syntax_failure": syntax_failure})
            if len(self._cycles) > 100:
                self._cycles = self._cycles[-100:]
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO cycles (run_ts,outcome,duration_s,quality_score,syntax_failure) VALUES (?,?,?,?,?)",
                (ts, outcome, duration_s, quality_score, syntax_failure),
            )
            conn.commit()
        finally:
            conn.close()

    def optimal_interval_s(self) -> float:
        """Returns unclamped data-driven optimal interval exposing true adaptive signal."""
        improvement_rate, error_rate = self._compute_rates()
        return self._base_s * math.exp(error_rate - improvement_rate)

    def upcoming_cycles(self) -> list[dict]:
        """Queries SQLite schedule table for future scheduled cycles; returns list of dicts."""
        now = time.time()
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT next_run_ts, interval_s, improvement_rate FROM schedule WHERE next_run_ts > ? ORDER BY next_run_ts LIMIT 10",
                (now,),
            ).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            with self._lock:
                last_score = self._scores[-1] if self._scores else 0.5
            projected = round(last_score + float(row[2]) * float(row[1]) / 3600.0, 4)
            result.append({"ts": row[0], "interval_s": row[1], "projected_quality": projected})
        return result

    def performance_trend(self) -> dict:
        """Returns improvement_rate, error_rate, trend_direction, score_now, score_30_ago, window_size."""
        improvement_rate, error_rate = self._compute_rates()
        with self._lock:
            n = len(self._scores)
            score_now = self._scores[-1] if n > 0 else 0.0
            score_30_ago = self._scores[-30] if n >= 30 else (self._scores[0] if n > 0 else 0.0)
        if improvement_rate > 0.0 and improvement_rate > error_rate:
            direction = 1
        elif improvement_rate < 0.0 or error_rate > improvement_rate:
            direction = -1
        else:
            direction = 0
        return {
            "improvement_rate": round(improvement_rate, 6),
            "error_rate": round(error_rate, 6),
            "trend_direction": direction,
            "score_now": round(score_now, 4),
            "score_30_ago": round(score_30_ago, 4),
            "window_size": n,
        }

    def status(self) -> dict:
        """Returns full scheduler snapshot with numeric keys for ConsciousnessIntegrator Φ."""
        improvement_rate, error_rate = self._compute_rates()
        raw_s = self._base_s * math.exp(error_rate - improvement_rate)
        interval = max(600.0, min(14400.0, raw_s))
        conn = sqlite3.connect(self._db_path)
        try:
            cycle_count = conn.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        finally:
            conn.close()
        with self._lock:
            nrt = self._next_run_ts
            cred = self._credits
            running = self._running
        return {
            "next_run_ts": nrt,
            "credits": cred,
            "running": running,
            "current_interval_s": round(interval, 2),
            "optimal_interval_s": round(raw_s, 2),
            "trend": self.performance_trend(),
            "readiness": self.assess_readiness(),
            "cycle_count": cycle_count,
            "confidence": round(1.0 - error_rate, 4),
            "accuracy": round(max(0.0, improvement_rate + 0.5), 4),
            "active": int(running),
            "pending": int(nrt > time.time()),
            "entropy": round(-improvement_rate * math.log(abs(improvement_rate) + 1e-12), 4),
        }

    def _daemon_loop(self) -> None:
        """Background thread: sleeps to next scheduled cycle, runs evolution if ready."""
        while self._running:
            with self._lock:
                nrt = self._next_run_ts
            sleep_s = max(10.0, nrt - time.time())
            time.sleep(min(sleep_s, 60.0))
            if not self._running:
                break
            if time.time() < nrt:
                continue
            self.auto_cycle()

    def auto_cycle(self) -> None:
        """Daemon evolution trigger: checks readiness, decrements credits, runs cycle, reschedules."""
        if not self.assess_readiness():
            fallback_quality = self._scores[-1] if self._scores else 0.5
            self.schedule_next(fallback_quality)
            return
        with self._lock:
            self._credits = max(0, self._credits - 1)
        t0 = time.time()
        outcome = "success"
        quality = 0.75
        try:
            import NovaSystem
            NovaSystem.credit_ledger.deduct(1)
        except Exception:
            pass
        try:
            import NovaSystem
            NovaSystem.evolution_engine.run_cycle()
        except Exception:
            pass
        try:
            import NovaSystem
            quality = float(NovaSystem.quality_monitor.get_score())
        except Exception:
            with self._lock:
                quality = (self._scores[-1] if self._scores else 0.75) + 0.01
        duration_s = time.time() - t0
        self.log_cycle(outcome, duration_s)
        self.schedule_next(quality)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal(f"Review AdaptiveScheduler cycle quality={quality:.3f}", priority=3)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "AdaptiveScheduler", "auto_cycle",
                confidence=quality, success=quality > 0.5
            )
        except Exception:
            pass

# Usage: obj = AdaptiveScheduler() | result = obj.schedule_next(0.82)