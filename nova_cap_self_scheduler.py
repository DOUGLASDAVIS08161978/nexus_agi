"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-29
"""

"""
Self-scheduling engine for Nova's autonomous evolution cycles.
Adapts timing based on performance trends, error rates, and improvement velocity.
Persists schedule across restarts; operates without human intervention.
"""

import sqlite3
import threading
import time
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class SelfSchedulerEngine:
    """Autonomous cycle scheduler that adapts intervals based on performance metrics."""

    def __init__(self) -> None:
        """Initialize scheduler with persistent SQLite store and background daemon."""
        self._lock = threading.Lock()
        self._db_path = "/tmp/nova_self_scheduler.db"
        self._init_db()

        # In-memory state
        self._cycle_history: OrderedDict[float, Dict] = OrderedDict()
        self._load_history()

        # Daemon configuration
        self._daemon_running = False
        self._next_scheduled_time = time.time() + 600  # Start in 10 min
        self._base_interval = 600  # 10 minutes base
        self._min_interval = 600  # 10 minutes minimum
        self._max_interval = 14400  # 4 hours maximum

        # Start background daemon
        self._start_daemon()

    def _init_db(self) -> None:
        """Create SQLite schema if not exists."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schedule_log (
                    timestamp REAL PRIMARY KEY,
                    cycle_id TEXT UNIQUE,
                    quality_score REAL,
                    error_rate REAL,
                    duration_s REAL,
                    improvement_rate REAL,
                    next_interval_s REAL,
                    readiness_flag INTEGER,
                    notes TEXT
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[SelfScheduler] DB init error: {e}")

    def _load_history(self) -> None:
        """Load cycle history from SQLite into memory (last 30 entries)."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            rows = conn.execute(
                "SELECT timestamp, quality_score, error_rate, duration_s, improvement_rate "
                "FROM schedule_log ORDER BY timestamp DESC LIMIT 30"
            ).fetchall()
            conn.close()

            for row in reversed(rows):
                ts, q, err, dur, imp = row
                self._cycle_history[ts] = {
                    "quality": q,
                    "error_rate": err,
                    "duration": dur,
                    "improvement": imp
                }
        except sqlite3.Error as e:
            print(f"[SelfScheduler] Load history error: {e}")

    def _start_daemon(self) -> None:
        """Start background daemon thread for autonomous scheduling."""
        if not self._daemon_running:
            self._daemon_running = True
            daemon = threading.Thread(target=self._daemon_loop, daemon=True)
            daemon.start()

    def _daemon_loop(self) -> None:
        """Background loop: check if next cycle is due, trigger assessment."""
        while self._daemon_running:
            time.sleep(30)  # Check every 30 seconds

            now = time.time()
            with self._lock:
                if now >= self._next_scheduled_time:
                    # Trigger readiness check
                    ready = self.assess_readiness()
                    if ready:
                        # Signal that a cycle is due
                        self._record_cycle_due()

    def _record_cycle_due(self) -> None:
        """Internal: mark that a cycle is ready to execute."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO schedule_log (timestamp, cycle_id, readiness_flag) "
                "VALUES (?, ?, 1)",
                (time.time(), f"due_{time.time()}")
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

    def schedule_next(self, current_quality: float) -> float:
        """
        Compute next cycle interval using adaptive formula.
        Returns: seconds until next scheduled cycle.
        """
        with self._lock:
            now = time.time()

            # Calculate improvement rate (30-cycle window)
            recent = list(self._cycle_history.values())[-30:] if self._cycle_history else []
            if len(recent) >= 2:
                oldest_quality = recent[0]["quality"]
                newest_quality = recent[-1]["quality"]
                improvement_rate = (newest_quality - oldest_quality) / max(len(recent), 1)
            else:
                improvement_rate = 0.0

            # Calculate error rate (rolling 20-cycle window)
            error_window = list(self._cycle_history.values())[-20:] if self._cycle_history else []
            if error_window:
                total_errors = sum(1 for c in error_window if c["error_rate"] > 0)
                error_rate = total_errors / max(len(error_window), 1)
            else:
                error_rate = 0.0

            # Adaptive interval formula
            exponent = error_rate - improvement_rate
