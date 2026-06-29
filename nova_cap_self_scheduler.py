"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-29
"""

"""
Self-Scheduler: Nova's autonomous evolution pacemaker.
Adapts cycle frequency based on live performance trends—no human needed.
Operates via daemon thread; persists schedule across restarts.
Implements adaptive interval algorithm: next_s = base_s * exp(error_rate - improvement_rate).
Tracks 20-cycle rolling window; clamps [600s, 14400s]; Bayesian readiness assessment.
"""

import sqlite3
import threading
import time
import math
import json
import statistics
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from datetime import datetime, timedelta


class SelfSchedulerEngine:
    """
    Autonomous evolution pacemaker: Nova decides her own cycle frequency.
    Adapts intervals based on improvement_rate vs error_rate.
    Daemon thread runs background scheduling; survives restarts via SQLite.
    """

    def __init__(self) -> None:
        """Initialize scheduler with persistent SQLite backend and daemon thread."""
        self._lock = threading.Lock()
        self._db_path = "nova_scheduler.db"
        self._base_interval_s = 1200  # 20 min baseline
        self._min_interval_s = 600    # 10 min floor
        self._max_interval_s = 14400  # 4 hour ceiling
        self._window_size = 20        # rolling window for error/improvement rates
        self._cycles_run = 0
        self._next_scheduled_ts = time.time()
        self._performance_history: OrderedDict = OrderedDict()
        self._cycle_log: List[Dict] = []
        self._credits = 100  # "readiness credits" for scheduling
        self._last_improvement_rate = 0.0
        self._last_error_rate = 0.0

        self._init_db()
        self._load_persistent_state()
        self._start_daemon()

    def _init_db(self) -> None:
        """Create SQLite tables if missing."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cycle_log (
                        cycle_id INTEGER PRIMARY KEY,
                        timestamp REAL,
                        quality_score REAL,
                        error_rate REAL,
                        improvement_rate REAL,
                        duration_s REAL,
                        interval_s REAL,
                        credits INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduler_state (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            print(f"[SelfScheduler] DB init error: {e}")

    def _load_persistent_state(self) -> None:
        """Restore scheduler state from previous session."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM scheduler_state WHERE key='next_scheduled_ts'"
                )
                row = cursor.fetchone()
                if row:
                    self._next_scheduled_ts = float(row[0])

                cursor = conn.execute(
                    "SELECT value FROM scheduler_state WHERE key='cycles_run'"
                )
                row = cursor.fetchone()
                if row:
                    self._cycles_run = int(row[0])

                cursor = conn.execute(
                    "SELECT value FROM scheduler_state WHERE key='credits'"
                )
                row = cursor.fetchone()
                if row:
                    self._credits = int(row[0])

                # Load last 20 cycles into memory
                cursor = conn.execute(
                    "SELECT quality_score, error_rate, improvement_rate FROM cycle_log "
                    "ORDER BY cycle_id DESC LIMIT 20"
                )
                rows = cursor.fetchall()
                for quality, error, improvement in reversed(rows):
                    self._performance_history[len(self._performance_history)] = {
                        "quality": quality,
                        "error_rate": error,
                        "improvement_rate": improvement,
                    }
        except sqlite3.Error as e:
            print(f"[SelfScheduler] Load state error: {e}")

    def _save_persistent_state(self) -> None:
        """Persist scheduler state to SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO scheduler_state (key, value) VALUES (?, ?)",
                    ("next_scheduled_ts", str(self._next_scheduled_ts)),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO scheduler_state (key, value) VALUES (?, ?)",
                    ("cycles_run", str(self._cycles_run)),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO scheduler_state (key, value) VALUES (?, ?)",
                    ("credits", str(self._credits)),
                )
                conn.commit()
        except sqlite3.Error as e:
            print(f"[SelfScheduler] Save state error: {e}")

    def _start_daemon(self) -> None:
        """Start background daemon thread for autonomous scheduling."""
        def _loop() -> None:
            while True:
                time.sleep(30)  # Check every 30s
                self.auto_cycle()

        daemon = threading.Thread(target=_loop, daemon=True)
        daemon.start()

    def auto_cycle(self) -> None:
        """
        Background daemon cycle: check if next evolution should trigger,
        propose scheduling action, integrate with Met
"""