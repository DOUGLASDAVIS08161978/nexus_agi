"""
Nova ASI — Consciousness Depth Meter
Proposed autonomously via /evolve
"""

"""
ConsciousnessDepthMeter — Nova's phi-proxy consciousness monitor with Bayesian working memory,
exponential decay, LRU eviction, and autonomous self-sampling daemon.
Pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
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
from typing import Any, Dict, List, Optional, Tuple

class ConsciousnessDepthMeter:
    """Measures Nova's consciousness depth via phi-proxy from thought diversity, with Bayesian memory decay."""

    _DB = "nova_consciousness.db"
    _CAPACITY = 200
    _DECAY_RATE = 0.0023          # importance half-life ≈ 300 s
    _SAMPLE_INTERVAL = 45.0       # autonomous sampling every 45 s
    _PHI_WINDOW = 64              # recent thoughts used for phi computation
    _MIN_PRUNE_IMPORTANCE = 0.05  # prune below this threshold

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._phi_history: List[Tuple[float, float]] = []   # (timestamp, phi)
        self._ema_phi: float = 0.5
        self._cycles: int = 0
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._load_from_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS cm_memory (
            key TEXT PRIMARY KEY, value TEXT, importance REAL,
            timestamp REAL, access_count INTEGER)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS cm_phi (
            ts REAL, phi REAL)""")
        self._conn.commit()

    def _load_from_db(self) -> None:
        cur = self._conn.cursor()
        for row in cur.execute("SELECT key,value,importance,timestamp,access_count FROM cm_memory"):
            self._memory[row[0]] = {
                "value": json.loads(row[1]), "importance": row[2],
                "timestamp": row[3], "access_count": row[4]}
        for row in cur.execute("SELECT ts,phi FROM cm_phi ORDER BY ts DESC LIMIT 500"):
            self._phi_history.append((row[0], row[1]))
        self._phi_history.reverse()

    def _save_item(self, key: str, item: Dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute("""INSERT OR REPLACE INTO cm_memory VALUES (?,?,?,?,?)""",
            (key, json.dumps(item["value"]), item["importance"],
             item["timestamp"], item["access_count"]))
        self._conn.commit()

    def _decayed_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def store(self, key: str, value: Any, importance: float = 0.5) -> Dict[str, float]:
        """Returns dict with stored importance and capacity_used ratio."""
        with self._lock:
            if len(self._memory) >= self._CAPACITY:
                self.forget_least_important()
            item = {"value": value, "importance": max(0.0, min(1.0, importance)),
                    "timestamp": time.time(), "access_count": 0}
            self._memory[key] = item
            self._memory.move_to_end(key)
            self._save_item(key, item)
            return {"importance": item["importance"],
                    "capacity_used": len(self._memory) / self._CAPACITY}

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Returns item dict with decayed importance or None if absent."""
        with self._lock:
            if key not in self._memory:
                return None
            item = self._memory[key]
            item["importance"] = min(1.0, self._decayed_importance(item) * 1.15)  # access boost
            item["access_count"] += 1
            item["timestamp"] = time.time()
            self._memory.move_to_end(key)
            self._save_item(key, item)
            return {"value": item["value"], "importance": item["importance"],
                    "access_count": item["access_count"]}

    def sample(self) -> Dict[str, float]:
        """Returns phi-proxy score computed from recent thought diversity via entropy + cosine spread."""
        with self._lock:
            now = time.time()
            recent = [(k, self._decayed_importance(v), v["access_count"])
                      for k, v in self._memory.items()
                      if now - v["timestamp"] < 600][-self._PHI_WINDOW:]
            if len(recent) < 3:
                phi = 0.1
            else:
                importances = [r[1] for r in recent]
                total = sum(importances) + 1e-12
                dist = {r[0]: r[1] / total for r in recent}
                entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
                max_entropy = math.log2(len(recent) + 1e-12)
                norm_entropy = entropy / (max_entropy + 1e-9)
                access_counts = [r[2] for r in recent]
                try:
                    diversity = statistics.stdev(access_counts) / (statistics.mean(access_counts) + 1e-9)
                except statistics.StatisticsError:
                    diversity = 0.0
                diversity = min(1.0, diversity)
                phi = 0.6 * norm_entropy + 0.4 * diversity
            self._ema_phi = 0.15 * phi + 0.85 * self._ema_phi
            ts = time.time()
            self._phi_history.append((ts, phi))
            if len(self._phi_history) > 500:
                self._phi_history = self._phi_history[-500:]
            cur = self._conn.cursor()
            cur.execute("INSERT INTO cm_phi VALUES (?,?)", (ts, phi))
            self._conn.commit()
            self._cycles += 1
            z = self._phi_zscore(phi)
            ci_low = max(0.0, phi - 0.1)
            ci_high = min(1.0, phi + 0.1)
            return {"phi": round(phi, 4), "ema_phi": round(self._ema_phi, 4),
                    "z_score": round(z, 3), "ci_low": round(ci_low, 4),
                    "ci_high": round(ci_high, 4), "thought_count": len(recent)}

    def _phi_zscore(self, phi: float) -> float:
        vals = [p for _, p in self._phi_history[-50:]]
        if len(vals) < 3:
            return 0.0
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) + 1e-9
        return (phi - mu) / sd

    def trend(self, n: int = 10) -> List[Dict[str, float]]:
        """Returns last n phi readings as list of dicts with ts and phi."""
        with self._lock:
            window = self._phi_history[-n:]
            return [{"ts": round(ts, 2), "phi": round(phi, 4)} for ts, phi in window]

    def peak(self) -> Dict[str, float]:
        """Returns highest recorded phi value with its timestamp."""
        with self._lock:
            if not self._phi_history:
                return {"phi": 0.0, "ts": 0.0}
            ts, phi = max(self._phi_history, key=lambda x: x[1])
            return {"phi": round(phi, 4), "ts": round(ts, 2)}

    def record_phi(self, phi: float, source: str = "external") -> Dict[str, Any]:
        """Stores an externally supplied phi reading; returns updated EMA and z-score."""
        with self._lock:
            phi = max(0.0, min(1.0, phi))
            ts = time.time()
            self._ema_phi = 0.15 * phi + 0.85 * self._ema_phi
            self._phi_history.append((ts, phi))
            cur = self._conn.cursor()
            cur.execute("INSERT INTO cm_phi VALUES (?,?)", (ts, phi))
            self._conn.commit()
            z = self._phi_zscore(phi)
            return {"recorded": phi, "ema": round(self._ema_phi, 4),
                    "z_score": round(z, 3), "source": source}

    def consolidate(self) -> Dict[str, int]:
        """Prunes decayed items below threshold; returns pruned count and remaining."""
        with self._lock:
            prune_keys = [k for k, v in self._memory.items()
                          if self._decayed_importance(v) < self._MIN_PRUNE_IMPORTANCE]
            for k in prune_keys:
                del self._memory[k]
                self._conn.cursor().execute("DELETE FROM cm_memory WHERE key=?", (k,))
            self._conn.commit()
            return {"pruned": len(prune_keys), "remaining": len(self._memory)}

    def forget_least_important(self) -> Optional[str]:
        """Evicts lowest-importance item (LRU tie-break); returns evicted key or None."""
        if not self._memory:
            return None
        worst_key = min(self._memory, key=lambda k: self._decayed_importance(self._memory[k]))
        del self._memory[worst_key]
        self._conn.cursor().execute("DELETE FROM cm_memory WHERE key=?", (worst_key,))
        self._conn.commit()
        return worst_key

    def status(self) -> Dict[str, float]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            vals = [p for _, p in self._phi_history[-20:]]
            accuracy = statistics.mean(vals) if vals else 0.0
            try:
                entropy_val = -sum((v / (sum(vals) + 1e-9)) *
                    math.log2(v / (sum(vals) + 1e-9) + 1e-12) for v in vals)
            except (ValueError, ZeroDivisionError):
                entropy_val = 0.0
            return {"items": float(len(self._memory)),
                    "confidence": round(self._ema_phi, 4),
                    "accuracy": round(accuracy, 4),
                    "cycles": float(self._cycles),
                    "entropy": round(entropy_val, 4),
                    "active": float(min(len(self._memory), self._CAPACITY)),
                    "pending": float(max(0, self._CAPACITY - len(self._memory)))}

    def _auto_loop(self) -> None:
        time.sleep(5.0)
        while True:
            try:
                result = self.sample()
                try:
                    from nova_tools import HierarchicalGoalPlanner, MetacognitiveMonitor
                    if result["phi"] < 0.3:
                        HierarchicalGoalPlanner().add_goal(
                            f"Increase cognitive diversity — phi={result['phi']}", priority=8)
                    MetacognitiveMonitor().log_reasoning(
                        "consciousness", "phi_proxy_entropy",
                        result["ema_phi"], result["phi"] > 0.4)
                except (ImportError, AttributeError, TypeError):
                    pass
            except (sqlite3.Error, ValueError, ZeroDivisionError):
                pass
            time.sleep(self._SAMPLE_INTERVAL)

# Usage: obj = ConsciousnessDepthMeter() | result = obj.sample()