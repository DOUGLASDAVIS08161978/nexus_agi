"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-28
"""

"""
Self-Scheduler: Adaptive evolution cycle orchestrator for Nova's autonomous growth.
Nova decides when to run her own evolution cycles based on live performance trends.
Implements adaptive interval scheduling with Bayesian error tracking, performance trending,
and autonomous daemon operation. Persists all state to SQLite for restart resilience.
Integrates with HierarchicalGoalPlanner, MetacognitiveMonitor, and WorkingMemory.
"""

import sqlite3
import threading
import time
import math
import statistics
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


class SelfScheduler:
    """Adaptive self-scheduling engine: Nova autonomously decides when to evolve."""

    def __init__(self) -> None:
        """Initialize scheduler with SQLite persistence, daemon thread, and state."""
        self._lock = threading.Lock()
        self._db_path = "nova_self_scheduler.db"
        self._init_db()
        
        # In-memory state (synced to DB)
        self._cycle_log: OrderedDict[float, Dict[str, Any]] = OrderedDict()
        self._performance_window: List[float] = []  # Last 30 scores
        self._error_window: List[int] = []  # Last 20 error flags (0/1)
        self._credits: float = 100.0  # Cognitive budget for cycles
        self._last_schedule_time: float = time.time()
        self._next_cycle_time: float = time.time() + 600  # Start in 10 min
        self._base_interval_s: float = 600.0
        self._daemon_running: bool = False
        
        self._load_from_db()
        self._start_daemon()

    def _init_db(self) -> None:
        """Create SQLite schema if not exists."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    outcome_score REAL,
                    duration_s REAL,
                    error_flag INTEGER,
                    quality_delta REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schedule_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        """Restore cycle history and schedule state from SQLite."""
        with sqlite3.connect(self._db_path) as conn:
            # Load cycle log
            rows = conn.execute(
                "SELECT timestamp, outcome_score, duration_s, error_flag, quality_delta "
                "FROM cycles ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            
            for ts, score, dur, err, delta in reversed(rows):
                self._cycle_log[ts] = {
                    "score": score,
                    "duration_s": dur,
                    "error": err,
                    "quality_delta": delta
                }
                if len(self._performance_window) < 30:
                    self._performance_window.append(score)
                if len(self._error_window) < 20:
                    self._error_window.append(err)
            
            # Load schedule state
            state_rows = conn.execute(
                "SELECT key, value FROM schedule_state"
            ).fetchall()
            for key, val in state_rows:
                if key == "credits":
                    self._credits = float(val)
                elif key == "next_cycle_time":
                    self._next_cycle_time = float(val)
                elif key == "base_interval_s":
                    self._base_interval_s = float(val)

    def _save_to_db(self) -> None:
        """Persist cycle log and schedule state to SQLite."""
        with sqlite3.connect(self._db_path) as conn:
            # Prune old cycles (keep last 100)
            all_ts = list(self._cycle_log.keys())
            if len(all_ts) > 100:
                for old_ts in all_ts[:-100]:
                    conn.execute("DELETE FROM cycles WHERE timestamp = ?", (old_ts,))
            
            # Save schedule state
            conn.execute("DELETE FROM schedule_state")
            conn.execute(
                "INSERT INTO schedule_state (key, value) VALUES (?, ?)",
                ("credits", str(self._credits))
            )
            conn.execute(
                "INSERT INTO schedule_state (key, value) VALUES (?, ?)",
                ("next_cycle_time", str(self._next_cycle_time))
            )
            conn.execute(
                "INSERT INTO schedule_state (key, value) VALUES (?, ?)",
                ("base_interval_s", str(self._base_interval_s))
            )
            conn.commit()

    def _start_daemon(self) -> None:
        """Start background daemon thread for autonomous scheduling."""
        if self._daemon_running:
            return
        self._daemon_running = True
        daemon = threading.Thread(target=self._daemon_loop, daemon=True)
        daemon.start()

    def _daemon_loop(self) -> None:
        """Background loop: check readiness, propose cycles, update schedule."""
        while self._daemon_running:
            try:
                time.sleep(30)  # Check every 30 seconds
                now = time.time()
                
                if now >= self._next_cycle_time and self.assess_readiness():
                    # Propose evolution cycle to goal planner
                    self._propose_cycle()
            except Exception:
                pass