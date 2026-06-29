"""
AdaptiveMetaStrategistCausalReasoner — Thompson-sampled strategy selector with multiplicative causal chain confidence propagation.
Learns which reasoning approach (analytical, intuitive, analogical, causal) works best per problem domain.
Maintains causal fact graph with forward-chaining inference and confidence decay.
Auto-cycles to update strategy beliefs, detect contradictions, and propose next reasoning challenge.
"""

import sqlite3
import json
import time
import threading
import math
import statistics
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta


class AdaptiveMetaStrategistCausalReasoner:
    """Thompson-sampled meta-strategist with causal inference engine."""

    def __init__(self) -> None:
        """Initialize strategy belief state, causal graph, and background auto-cycle."""
        self._lock = threading.Lock()

        # Strategy Thompson sampling: domain → {strategy → (successes, trials, last_reward)}
        self._strategy_beliefs: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        self._strategies = ["analytical", "intuitive", "analogical", "causal"]

        # Causal graph: (premise, conclusion) → (weight, timestamp, decay_rate)
        self._causal_facts: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

        # Contradiction detection: stores conflicting chains
        self._contradictions: List[Tuple[str, str, float]] = []

        # Performance tracking: domain → [reward, reward, ...]
        self._reward_history: Dict[str, List[float]] = {}

        # Inference cache: hypothesis → (chain_path, confidence, timestamp)
        self._inference_cache: Dict[str, Tuple[List[str], float, float]] = {}

        # Metrics
        self._cycles_run = 0
        self._inferences_made = 0
        self._contradictions_detected = 0
        self._strategy_switches = 0

        # SQLite persistence
        self._init_db()
        self._load_state()

        # Background auto-cycle daemon
        self._running = True
        self._daemon = threading.Thread(target=self._auto_cycle_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        """Initialize SQLite schema for causal facts and strategy beliefs."""
        try:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_facts (
                    premise TEXT, conclusion TEXT, weight REAL, timestamp REAL, decay_rate REAL,
                    PRIMARY KEY (premise, conclusion)
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_beliefs (
                    domain TEXT, strategy TEXT, successes REAL, trials REAL, last_reward REAL,
                    PRIMARY KEY (domain, strategy)
                )
            """)
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"DB init error: {e}")

    def _load_state(self) -> None:
        """Load causal facts and strategy beliefs from SQLite."""
        try:
            with self._lock:
                rows = self._conn.execute("SELECT premise, conclusion, weight, timestamp, decay_rate FROM causal_facts").fetchall()
                for p, c, w, ts, dr in rows:
                    self._causal_facts[(p, c)] = (w, ts, dr)

                rows = self._conn.execute("SELECT domain, strategy, successes, trials, last_reward FROM strategy_beliefs").fetchall()
                for d, s, succ, tri, lr in rows:
                    if d not in self._strategy_beliefs:
                        self._strategy_beliefs[d] = {}
                    self._strategy_beliefs[d][s] = (succ, tri, lr)
        except sqlite3.Error as e:
            print(f"Load state error: {e}")

    def add_fact(self, claim: str, conclusion: str, confidence: float) -> None:
        """Add causal edge: claim→conclusion with confidence weight. Stores in DB."""
        if not (0 <= confidence <= 1):
            raise ValueError("Confidence must be in [0, 1]")

        with self._lock:
            now = time.time()
            decay_rate = 0.001  # Slow decay: ~half-life at 700 hours
            self._causal_facts[(claim, conclusion)] = (confidence, now, decay_rate)

            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO causal_facts (premise, conclusion, weight, timestamp, decay_rate) VALUES (?, ?, ?, ?, ?)",
                    (claim, conclusion, confidence, now, decay_rate)
                )
                self._conn.commit()
            except sqlite3.Error as e:
                print(f"Add fact error: {e}")


    def _auto_cycle_loop(self):
        """Auto-stub: referenced but not generated."""
        pass
    def infer(self, hypothesis: str, max_chain_length: int = 5) -> Tuple[List[str], float]:
        """Forward-chain inference: find all paths from facts to hypothesis. Return (path, confidence)."""
        with self._lock:
            # Check cache
            if hypothesis in self._inference_cache:
                path, conf, ts = self._inference_cache[hypothesis]
                if time.time() - ts < 300:  # 5-min cache TTL
                    return (path, conf)

            # Forward chaining: explore all edges, accumulate confidence multiplic
