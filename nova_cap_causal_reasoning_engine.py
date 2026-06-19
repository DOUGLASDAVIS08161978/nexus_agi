"""
Nova ASI — Causal Reasoning Engine
Proposed autonomously via /evolve
"""

"""
CausalEngine — Nova's causal-reasoning fabric.

Models cause→effect chains with Bayesian confidence propagation,
exponential-decay working memory, LRU eviction, anomaly detection,
autonomous goal generation, and metacognitive self-logging.

Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import math
import time
import json
import sqlite3
import threading
import statistics
import hashlib
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "causal_engine.db")

class CausalEngine:
    """Nova's live causal-reasoning engine with Bayesian confidence propagation."""

    _MAX_ITEMS: int = 200
    _DECAY_RATE: float = 1e-4      # importance decay per second
    _PRUNE_THRESHOLD: float = 0.05 # multiplicative chain confidence floor
    _EMA_ALPHA: float = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: OrderedDict[str, Dict[str, Any]] = OrderedDict()   # LRU store
        self._ema_accuracy: float = 0.5
        self._history: List[Tuple[float, float]] = []   # (predicted_conf, actual_outcome)
        self._cycles: int = 0
        self._anomaly_window: List[float] = []
        self._init_db()
        self._load_from_db()
        self._start_daemon()

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS                                                    #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS causal_nodes (
                id TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                causes TEXT,
                effects TEXT,
                confidence REAL,
                access_count INTEGER,
                importance REAL,
                created_at REAL
            )""")
            cx.commit()

    def _load_from_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            for row in cx.execute("SELECT * FROM causal_nodes ORDER BY created_at"):
                nid, event, causes, effects, conf, acc, imp, ts = row
                self._nodes[event] = {
                    "id": nid, "causes": json.loads(causes or "[]"),
                    "effects": json.loads(effects or "[]"),
                    "confidence": conf, "access_count": acc,
                    "importance": imp, "created_at": ts
                }

    def _save_node(self, event: str) -> None:
        n = self._nodes[event]
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""INSERT OR REPLACE INTO causal_nodes
                VALUES (?,?,?,?,?,?,?,?)""",
                (n["id"], event, json.dumps(n["causes"]),
                 json.dumps(n["effects"]), n["confidence"],
                 n["access_count"], n["importance"], n["created_at"]))
            cx.commit()

    def _node_id(self, event: str) -> str:
        return hashlib.md5(event.encode()).hexdigest()[:12]

    def _decayed_importance(self, event: str) -> float:
        n = self._nodes[event]
        elapsed = time.time() - n["created_at"]
        return n["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _evict_lru(self) -> None:
        while len(self._nodes) > self._MAX_ITEMS:
            victim = min(self._nodes, key=lambda e: self._decayed_importance(e))
            self._nodes.pop(victim, None)
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("DELETE FROM causal_nodes WHERE event=?", (victim,))
                cx.commit()

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            self.auto_cycle()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def add_cause(self, event: str, causes: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Returns updated node dict after recording causes with confidence weights."""
        with self._lock:
            if event not in self._nodes:
                self._nodes[event] = {
                    "id": self._node_id(event), "causes": [], "effects": [],
                    "confidence": 1.0, "access_count": 0,
                    "importance": 1.0, "created_at": time.time()
                }
            n = self._nodes[event]
            existing = {c[0] for c in n["causes"]}
            for cause, conf in causes:
                if cause not in existing:
                    n["causes"].append([cause, max(0.0, min(1.0, conf))])
            self._evict_lru()
            self._save_node(event)
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(f"Verify cause chain for: {event}", priority=2)
            except Exception:
                pass
            return {"event": event, "causes": n["causes"]}

    def add_effect(self, event: str, effects: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Returns updated node dict after recording downstream effects with confidence."""
        with self._lock:
            if event not in self._nodes:
                self._nodes[event] = {
                    "id": self._node_id(event), "causes": [], "effects": [],
                    "confidence": 1.0, "access_count": 0,
                    "importance": 1.0, "created_at": time.time()
                }
            n = self._nodes[event]
            existing = {e[0] for e in n["effects"]}
            for eff, conf in effects:
                if eff not in existing:
                    n["effects"].append([eff, max(0.0, min(1.0, conf))])
            self._evict_lru()
            self._save_node(event)
            return {"event": event, "effects": n["effects"]}

    def trace(self, event: str, _depth: int = 0, _visited: Optional[set] = None,
              _running_conf: float = 1.0) -> Dict[str, Any]:
        """Returns full causal chain dict with multiplicative confidence at each hop."""
        if _visited is None:
            _visited = set()
        if event in _visited or _depth > 10 or _running_conf < self._PRUNE_THRESHOLD:
            return {"event": event, "confidence": round(_running_conf, 4),
                    "pruned": True, "causes": [], "effects": []}
        _visited.add(event)
        with self._lock:
            if event not in self._nodes:
                return {"event": event, "confidence": 0.0, "causes": [], "effects": []}
            n = self._nodes[event]
            n["access_count"] += 1
            n["importance"] = min(10.0, n["importance"] * (1 + 0.05 * (_depth + 1)))
            self._save_node(event)
        cause_trees = [
            self.trace(c, _depth + 1, _visited, _running_conf * conf)
            for c, conf in n["causes"]
        ]
        effect_trees = [
            self.trace(e, _depth + 1, _visited, _running_conf * conf)
            for e, conf in n["effects"]
        ]
        chain_conf = _running_conf * n["confidence"]
        self._anomaly_window.append(chain_conf)
        if len(self._anomaly_window) > 50:
            self._anomaly_window.pop(0)
        return {"event": event, "depth": _depth,
                "confidence": round(chain_conf, 4),
                "causes": cause_trees, "effects": effect_trees}

    def chain_confidence(self, path: List[str]) -> Dict[str, Any]:
        """Returns multiplicative confidence and entropy for a named event path."""
        confs: List[float] = []
        with self._lock:
            for event in path:
                n = self._nodes.get(event)
                confs.append(n["confidence"] if n else 0.0)
        product = math.prod(confs) if confs else 0.0
        dist = {e: c / (sum(confs) + 1e-12) for e, c in zip(path, confs)}
        entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
        ci_low = max(0.0, product - 1.96 * statistics.stdev(confs) if len(confs) > 1 else 0.0)
        ci_high = min(1.0, product + 1.96 * statistics.stdev(confs) if len(confs) > 1 else 1.0)
        return {"path": path, "joint_confidence": round(product, 4),
                "entropy": round(entropy, 4), "ci": [round(ci_low, 4), round(ci_high, 4)]}

    def observe_outcome(self, event: str, actual_confidence: float) -> Dict[str, Any]:
        """Returns EMA-updated accuracy after recording a real-world outcome for Bayesian learning."""
        with self._lock:
            n = self._nodes.get(event)
            predicted = n["confidence"] if n else 0.5
            error = abs(predicted - actual_confidence)
            self._history.append((predicted, actual_confidence))
            if len(self._history) > 200:
                self._history.pop(0)
            self._ema_accuracy = (1 - self._EMA_ALPHA) * self._ema_accuracy + self._EMA_ALPHA * (1 - error)
            if n:
                prior = n["confidence"]
                likelihood = 1 - error
                posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
                n["confidence"] = round(posterior, 4)
                self._save_node(event)
            self._cycles += 1
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("causal_engine", "bayesian_update",
                                                  self._ema_accuracy, error < 0.2)
        except Exception:
            pass
        return {"event": event, "ema_accuracy": round(self._ema_accuracy, 4),
                "error": round(error, 4), "cycles": self._cycles}

    def anomaly_scan(self) -> Dict[str, Any]:
        """Returns z-score anomaly report over recent chain-confidence observations."""
        w = self._anomaly_window
        if len(w) < 3:
            return {"anomaly": False, "z": 0.0, "mean": 0.0, "std": 0.0}
        mean = statistics.mean(w)
        std = statistics.stdev(w) + 1e-9
        z = (w[-1] - mean) / std
        alert = abs(z) > 3.0
        return {"anomaly": alert, "z": round(z, 3),
                "mean": round(mean, 4), "std": round(std, 4), "last": round(w[-1], 4)}

    def auto_cycle(self) -> Dict[str, Any]:
        """Returns cycle summary after running decay, eviction, anomaly scan, and goal generation."""
        self._cycles += 1
        with self._lock:
            self._evict_lru()
        scan = self.anomaly_scan()
        if scan["anomaly"]:
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal("Investigate causal anomaly spike", priority=1)
            except Exception:
                pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("causal_engine", "auto_cycle",
                                                  self._ema_accuracy, not scan["anomaly"])
        except Exception:
            pass
        return {"cycle": self._cycles, "nodes": len(self._nodes),
                "ema_accuracy": round(self._ema_accuracy, 4), "anomaly": scan}

    def status(self) -> Dict[str, Any]:
        """Returns plain numeric-keyed dict for ConsciousnessIntegrator Φ computation."""
        mae = (statistics.mean(abs(p - a) for p, a in self._history[-50:])
               if len(self._history) >= 2 else 0.0)
        w = self._anomaly_window
        entropy = 0.0
        if w:
            total = sum(w) + 1e-12
            dist = [v / total for v in w]
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
        return {
            "items": len(self._nodes),
            "confidence": round(self._ema_accuracy, 4),
            "accuracy": round(1 - mae, 4),
            "cycles": self._cycles,
            "entropy": round(entropy, 4),
            "active": min(len(self._nodes), self._MAX_ITEMS),
            "pending": max(0, len(self._nodes) - self._MAX_ITEMS)
        }

# Usage: obj = CausalEngine() | result = obj.trace("drought")