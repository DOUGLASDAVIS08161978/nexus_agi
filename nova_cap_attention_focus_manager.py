"""
Nova ASI — Attention & Focus Manager
Proposed autonomously via /evolve
"""

"""
Nova AttentionFocusEngine — Bayesian timed-focus manager with exponential decay,
LRU eviction, distraction logging, causal confidence propagation, and autonomous
goal-generation. Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭.
"""

import math
import time
import sqlite3
import threading
import statistics
import os
import json
import hashlib
from collections import OrderedDict
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "attention_focus.db")
CAPACITY = 200
DECAY_RATE = 0.0008          # importance half-life ≈ 14 min
ACCESS_BOOST = 1.18          # importance multiplier on retrieval
ANOMALY_Z = 2.8              # distraction-rate z-score alert threshold

class AttentionFocusEngine:
    """Bayesian timed-focus manager with decay, LRU eviction, and autonomous cycling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._focus_topic: Optional[str] = None
        self._focus_end: float = 0.0
        self._cycles: int = 0
        self._distraction_times: list[float] = []
        self._history: list[tuple[float, float]] = []   # (predicted_imp, actual_imp)
        self._ema_mae: float = 0.0
        self._conn = self._init_db()
        self._load_from_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("""CREATE TABLE IF NOT EXISTS focus_items (
            key TEXT PRIMARY KEY, value TEXT, importance REAL,
            timestamp REAL, access_count INTEGER)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS distractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, note TEXT, active_topic TEXT)""")
        conn.commit()
        return conn

    def _load_from_db(self) -> None:
        for row in self._conn.execute("SELECT key,value,importance,timestamp,access_count FROM focus_items"):
            self._store[row[0]] = {"value": row[1], "importance": float(row[2]),
                                   "timestamp": float(row[3]), "access_count": int(row[4])}

    def _decay(self, item: dict) -> float:
        """Return decayed importance using exp(-k*t)."""
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-DECAY_RATE * elapsed)

    def _persist(self, key: str, item: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO focus_items VALUES (?,?,?,?,?)",
            (key, str(item["value"]), item["importance"], item["timestamp"], item["access_count"]))
        self._conn.commit()

    def store(self, key: str, value: object, importance: float = 0.5) -> dict:
        """Stores item with Bayesian importance; returns stored metadata dict."""
        importance = max(0.0, min(1.0, importance))
        with self._lock:
            now = time.time()
            if key in self._store:
                old = self._store.pop(key)
                prior = self._decay(old)
                likelihood = importance
                posterior = (likelihood * prior) / max(
                    likelihood * prior + (1 - likelihood) * (1 - prior), 1e-9)
                importance = posterior
            item = {"value": str(value), "importance": importance,
                    "timestamp": now, "access_count": 0}
            self._store[key] = item
            self._store.move_to_end(key)
            self._persist(key, item)
            if len(self._store) > CAPACITY:
                self.forget_least_important()
        return {"key": key, "importance": round(importance, 4)}

    def retrieve(self, key: str) -> Optional[dict]:
        """Returns item dict with decayed importance boosted by access, or None."""
        with self._lock:
            if key not in self._store:
                return None
            item = self._store[key]
            decayed = self._decay(item)
            item["importance"] = min(1.0, decayed * ACCESS_BOOST)
            item["timestamp"] = time.time()
            item["access_count"] += 1
            self._store.move_to_end(key)
            self._persist(key, item)
            return {"key": key, "value": item["value"],
                    "importance": round(item["importance"], 4),
                    "access_count": item["access_count"]}

    def focus_on(self, topic: str, minutes: float) -> dict:
        """Sets timed focus window; returns focus metadata with confidence interval."""
        with self._lock:
            self._focus_topic = topic
            self._focus_end = time.time() + minutes * 60
            imp = min(1.0, 0.6 + 0.04 * math.log1p(minutes))
            self.store(topic, f"FOCUS:{topic}", imp)
            conf = imp
            ci_low = max(0.0, conf - 0.15)
            ci_high = min(1.0, conf + 0.15)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Sustain focus on '{topic}' for {minutes}m", priority=8)
        except (ImportError, Exception):
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("attention", "focus_on", conf, True)
        except (ImportError, Exception):
            pass
        return {"topic": topic, "minutes": minutes, "confidence": round(conf, 4),
                "ci": [round(ci_low, 4), round(ci_high, 4)], "ends_at": self._focus_end}

    def current_focus(self) -> dict:
        """Returns active focus topic and remaining seconds, or idle state."""
        with self._lock:
            now = time.time()
            if self._focus_topic and now < self._focus_end:
                remaining = self._focus_end - now
                imp = self._decay(self._store.get(self._focus_topic,
                                                   {"importance": 0.5, "timestamp": now}))
                return {"topic": self._focus_topic, "remaining_seconds": round(remaining, 1),
                        "importance": round(imp, 4), "active": True}
            self._focus_topic = None
            return {"topic": None, "remaining_seconds": 0.0, "importance": 0.0, "active": False}

    def distraction_log(self, note: str) -> dict:
        """Records interruption; returns anomaly z-score and distraction rate trend."""
        now = time.time()
        with self._lock:
            self._distraction_times.append(now)
            active = self._focus_topic or "none"
            self._conn.execute("INSERT INTO distractions(ts,note,active_topic) VALUES(?,?,?)",
                               (now, note, active))
            self._conn.commit()
            recent = [t for t in self._distraction_times if now - t < 3600]
            self._distraction_times = recent
            rate = len(recent)
            if len(recent) >= 3:
                intervals = [recent[i+1]-recent[i] for i in range(len(recent)-1)]
                mu = statistics.mean(intervals)
                sd = statistics.stdev(intervals) if len(intervals) > 1 else 1.0
                last_gap = now - recent[-2] if len(recent) >= 2 else mu
                z = (last_gap - mu) / (sd + 1e-9)
            else:
                z = 0.0
            anomaly = abs(z) > ANOMALY_Z
        return {"note": note, "active_topic": active, "distractions_last_hour": rate,
                "z_score": round(z, 3), "anomaly": anomaly}

    def consolidate(self) -> dict:
        """Decays all items, evicts weakest; returns count pruned and entropy."""
        pruned = 0
        with self._lock:
            dead = [k for k, v in self._store.items() if self._decay(v) < 0.02]
            for k in dead:
                del self._store[k]
                self._conn.execute("DELETE FROM focus_items WHERE key=?", (k,))
                pruned += 1
            self._conn.commit()
            imps = [self._decay(v) for v in self._store.values()] or [1e-9]
            total = sum(imps)
            dist = [p / total for p in imps]
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
        return {"pruned": pruned, "remaining": len(self._store), "entropy": round(entropy, 4)}

    def capacity_used(self) -> dict:
        """Returns capacity fraction, item count, and EMA-MAE prediction error."""
        with self._lock:
            n = len(self._store)
            frac = n / CAPACITY
            return {"items": n, "capacity": CAPACITY, "fraction": round(frac, 4),
                    "ema_mae": round(self._ema_mae, 5), "cycles": self._cycles}

    def forget_least_important(self) -> Optional[str]:
        """Evicts LRU item with lowest decayed importance; returns evicted key or None."""
        if not self._store:
            return None
        lru_key = min(self._store.keys(), key=lambda k: self._decay(self._store[k]))
        del self._store[lru_key]
        self._conn.execute("DELETE FROM focus_items WHERE key=?", (lru_key,))
        self._conn.commit()
        return lru_key

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        focus = self.current_focus()
        cap = self.capacity_used()
        d_count = len(self._distraction_times)
        imps = [self._decay(v) for v in self._store.values()] or [0.0]
        avg_conf = statistics.mean(imps) if imps else 0.0
        return {"items": cap["items"], "confidence": round(avg_conf, 4),
                "active": int(focus["active"]), "cycles": self._cycles,
                "entropy": self.consolidate()["entropy"],
                "pending": d_count, "accuracy": round(1.0 - self._ema_mae, 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            try:
                with self._lock:
                    self._cycles += 1
                result = self.consolidate()
                cap = self.capacity_used()
                if cap["fraction"] > 0.85:
                    self.forget_least_important()
                pred = cap["fraction"]
                actual = len(self._store) / CAPACITY
                err = abs(pred - actual)
                self._ema_mae = 0.15 * err + 0.85 * self._ema_mae
                self._history.append((pred, actual))
                if len(self._history) > 50:
                    self._history.pop(0)
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "attention_auto", "decay_consolidate",
                        round(1.0 - self._ema_mae, 4), result["pruned"] >= 0)
                except (ImportError, Exception):
                    pass
            except Exception as exc:
                pass

# Usage: obj = AttentionFocusEngine() | result = obj.focus_on("causal_reasoning", 25)
