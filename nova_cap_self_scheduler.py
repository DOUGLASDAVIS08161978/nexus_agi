"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-28
"""

"""
Self Scheduler — Nova's autonomous evolution pacing engine.
Adapts cycle intervals based on live performance trends without human intervention.
Implements Bayesian error tracking, rolling improvement windows, and persistent scheduling.
Satisfies pillars: ② causal modeling, ③ online learning, ④ calibration, ⑥ self-monitoring,
⑦ goal-directed, ⑧ feedback loop, ⑪ mathematical rigor, ⑫ uncertainty quantification, ⑬ autonomous.
"""

import sqlite3
import threading
import time
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class SelfScheduler:
    """
    Adaptive self-scheduling engine: Nova autonomously decides when to run evolution cycles
    based on live performance trends. No human intervention required after initialization.
    """

    def __init__(self) -> None:
        """Initialize scheduler with persistent SQLite store and daemon thread."""
        self._lock = threading.Lock()
        self._db_path = "nova_self_scheduler.db"
        self._init_db()
        
        # In-memory rolling windows (20-cycle window for error/improvement tracking)
        self._error_window: OrderedDict[int, float] = OrderedDict()
        self._score_window: OrderedDict[int, float] = OrderedDict()
        self._cycle_count = 0
        self._last_scheduled_time = time.time()
        
        # Adaptive interval parameters
        self._base_interval_s = 600.0  # 10 minutes baseline
        self._min_interval_s = 600.0   # 10 minutes floor
        self._max_interval_s = 14400.0 # 4 hours ceiling
        self._current_interval_s = self._base_interval_s
        
        # Daemon thread for autonomous cycling
        self._running = True
        self._daemon = threading.Thread(target=self._auto_cycle_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        """Initialize SQLite schema for persistent scheduling state."""
        try:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    outcome_score REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    interval_s REAL NOT NULL,
                    readiness_flag INTEGER NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS schedule (
                    key TEXT PRIMARY KEY,
                    next_cycle_time REAL NOT NULL,
                    current_interval_s REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[SelfScheduler] DB init error: {e}")

    def _auto_cycle_loop(self) -> None:
        """
        Daemon thread: autonomously checks readiness and schedules next cycle.
        Runs every 30 seconds, proposes evolution when conditions align.
        """
        while self._running:
            try:
                time.sleep(30)
                with self._lock:
                    if self.assess_readiness():
                        # Propose next evolution cycle to HierarchicalGoalPlanner
                        self._propose_evolution_goal()
            except Exception as e:
                print(f"[SelfScheduler] daemon error: {e}")

    def _propose_evolution_goal(self) -> None:
        """
        Internal: propose an evolution goal to HierarchicalGoalPlanner.
        Integrates with Nova's goal system to drive autonomous self-improvement.
        """
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            planner = HierarchicalGoalPlanner()
            interval = self.optimal_interval_s()
            priority = 0.8 if self.performance_trend() > 0.05 else 0.5
            planner.add_goal(
                f"Run evolution cycle (interval={interval:.0f}s, trend={self.performance_trend():.3f})",
                priority
            )
        except ImportError:
            pass  # HierarchicalGoalPlanner not available in this context
        except Exception as e:
            print(f"[SelfScheduler] goal proposal error: {e}")

    def schedule_next(self, current_quality: float) -> float:
        """
        Compute next cycle interval based on current quality and historical trends.
        Returns: next interval in seconds (clamped to [min, max]).
        
        Algorithm: next_s = base_s * exp(error_rate - improvement_rate)
        improvement_rate = (score_now - score_30_ago) / 30
        error_rate = syntax_failures / max(total_attempts, 1)
        """
        with self._lock:
            # Record current score
            self._score_window[self._cycle_count] = current_quality
            if len(self._score_window) > 20:
                oldest_key = next(iter(self._score_window))
                del self._score_window[oldest_key]
            
            # Compute improvement_rate over 30-cycle window (or available history)
            if len(self._score_window) >= 2:
                scores = list(self._score_window.values())
                improvement_rate = (scores[-1] - scores[0]) / max