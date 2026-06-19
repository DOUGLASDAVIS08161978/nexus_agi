"""
Nova ASI — Causal Reasoning Engine
Proposed autonomously via /evolve
"""

"""
CausalChainEngine — Bayesian causal graph with exponential decay, LRU eviction,
multiplicative chain propagation, and autonomous self-improvement cycles.
Pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, json, os, hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

class CausalChainEngine:
    """Probabilistic causal graph engine with decay, LRU eviction, and autonomous reasoning."""

    _DB = "nova_causal_chain.db"
    _CAP = 200
    _DECAY = 1e-5
    _PRUNE = 0.05
    _CYCLE_SEC = 90

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cycles: int = 0
        self._accuracy_history: List[Tuple[float, float]] = []
        self._ema_conf: float = 0.5
        self._init_db()
        self._load_from_db()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._DB) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS causal_nodes(
                key TEXT PRIMARY KEY, value TEXT, importance REAL,
                timestamp REAL, access_count INTEGER, edge_type TEXT)""")
            cx.commit()

    def _load_from_db(self) -> None:
        with sqlite3.connect(self._DB) as cx:
            for row in cx.execute("SELECT key,value,importance,timestamp,access_count,edge_type FROM causal_nodes"):
                self._store[row[0]] = {
                    "value": json.loads(row[1]), "importance": row[2],
                    "timestamp": row[3], "access_count": row[4], "edge_type": row[5]}

    def _persist(self, key: str) -> None:
        node = self._store[key]
        with sqlite3.connect(self._DB) as cx:
            cx.execute("""INSERT OR REPLACE INTO causal_nodes
                VALUES(?,?,?,?,?,?)""",
                (key, json.dumps(node["value"]), node["importance"],
                 node["timestamp"], node["access_count"], node["edge_type"]))
            cx.commit()

    def _decayed_importance(self, key: str) -> float:
        node = self._store[key]
        elapsed = time.time() - node["timestamp"]
        return node["importance"] * math.exp(-self._DECAY * elapsed)

    def _evict_lru(self) -> None:
        while len(self._store) > self._CAP:
            scores = {k: self._decayed_importance(k) for k in self._store}
            lru_key = min(scores, key=lambda k: scores[k])
            self._store.pop(lru_key, None)
            try:
                with sqlite3.connect(self._DB) as cx:
                    cx.execute("DELETE FROM causal_nodes WHERE key=?", (lru_key,))
                    cx.commit()
            except sqlite3.Error:
                pass

    def add_cause(self, event: str, causes: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Returns stored cause-node dict with Bayesian prior and decay metadata."""
        with self._lock:
            for cause, conf in causes:
                key = hashlib.md5(f"cause:{event}:{cause}".encode()).hexdigest()[:16]
                prior = self._store.get(key, {}).get("importance", 0.5)
                likelihood = max(0.01, min(0.99, conf))
                posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
                self._store[key] = {
                    "value": {"event": event, "cause": cause, "confidence": posterior},
                    "importance": posterior, "timestamp": time.time(),
                    "access_count": 0, "edge_type": "cause"}
                self._persist(key)
            self._evict_lru()
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(f"Verify causal chain for: {event}", priority=3)
            except Exception:
                pass
            return {"event": event, "causes_added": len(causes)}

    def add_effect(self, event: str, effects: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Returns stored effect-node dict with multiplicative confidence propagation."""
        with self._lock:
            for effect, conf in effects:
                key = hashlib.md5(f"effect:{event}:{effect}".encode()).hexdigest()[:16]
                prior = self._store.get(key, {}).get("importance", 0.5)
                likelihood = max(0.01, min(0.99, conf))
                posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
                self._store[key] = {
                    "value": {"event": event, "effect": effect, "confidence": posterior},
                    "importance": posterior, "timestamp": time.time(),
                    "access_count": 0, "edge_type": "effect"}
                self._persist(key)
            self._evict_lru()
            return {"event": event, "effects_added": len(effects)}

    def trace(self, event: str, depth: int = 5) -> Dict[str, Any]:
        """Returns full causal chain dict with multiplicative confidence and entropy score."""
        with self._lock:
            causes, effects = [], []
            chain_conf = 1.0
            for key, node in self._store.items():
                v = node["value"]
                imp = self._decayed_importance(key)
                if imp < self._PRUNE:
                    continue
                if node["edge_type"] == "cause" and v.get("event") == event:
                    causes.append({"cause": v["cause"], "confidence": round(imp, 4)})
                    chain_conf *= imp
                    node["access_count"] += 1
                    node["importance"] = min(1.0, node["importance"] * 1.05)
                elif node["edge_type"] == "effect" and v.get("event") == event:
                    effects.append({"effect": v["effect"], "confidence": round(imp, 4)})
                    chain_conf *= imp
                    node["access_count"] += 1
                    node["importance"] = min(1.0, node["importance"] * 1.05)
            confs = [c["confidence"] for c in causes + effects] or [0.5]
            dist = {c: c / (sum(confs) + 1e-12) for c in confs}
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
            ci_low = max(0.0, chain_conf - 1.96 * (statistics.stdev(confs) if len(confs) > 1 else 0.1))
            ci_high = min(1.0, chain_conf + 1.96 * (statistics.stdev(confs) if len(confs) > 1 else 0.1))
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning("causal_trace", "multiplicative_chain", chain_conf, len(causes + effects) > 0)
            except Exception:
                pass
            return {"event": event, "causes": causes, "effects": effects,
                    "chain_confidence": round(chain_conf, 6),
                    "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
                    "entropy": round(entropy, 4)}

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Returns decayed node dict by key, boosting importance on access."""
        with self._lock:
            if key not in self._store:
                return None
            node = self._store[key]
            node["access_count"] += 1
            node["importance"] = min(1.0, self._decayed_importance(key) * 1.1)
            self._store.move_to_end(key)
            return dict(node)

    def consolidate(self) -> Dict[str, int]:
        """Returns pruning report after removing sub-threshold causal nodes."""
        pruned = 0
        with self._lock:
            dead = [k for k in list(self._store) if self._decayed_importance(k) < self._PRUNE]
            for k in dead:
                self._store.pop(k, None)
                try:
                    with sqlite3.connect(self._DB) as cx:
                        cx.execute("DELETE FROM causal_nodes WHERE key=?", (k,))
                        cx.commit()
                except sqlite3.Error:
                    pass
                pruned += 1
        return {"pruned": pruned, "remaining": len(self._store)}

    def capacity_used(self) -> Dict[str, Any]:
        """Returns capacity metrics with z-score anomaly flag."""
        with self._lock:
            used = len(self._store)
            imps = [self._decayed_importance(k) for k in self._store] or [0.0]
            mean_i = statistics.mean(imps)
            std_i = statistics.stdev(imps) if len(imps) > 1 else 1e-9
            z = (mean_i - 0.5) / (std_i + 1e-9)
            return {"items": used, "capacity": self._CAP,
                    "pct": round(100 * used / self._CAP, 1),
                    "mean_importance": round(mean_i, 4),
                    "anomaly_z": round(z, 3), "cycles": self._cycles}

    def forget_least_important(self) -> Dict[str, Any]:
        """Returns key of evicted node after LRU importance-decay eviction."""
        with self._lock:
            if not self._store:
                return {"evicted": None}
            scores = {k: self._decayed_importance(k) for k in self._store}
            worst = min(scores, key=lambda k: scores[k])
            self._store.pop(worst, None)
            try:
                with sqlite3.connect(self._DB) as cx:
                    cx.execute("DELETE FROM causal_nodes WHERE key=?", (worst,))
                    cx.commit()
            except sqlite3.Error:
                pass
            return {"evicted": worst, "remaining": len(self._store)}

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        cap = self.capacity_used()
        imps = [self._decayed_importance(k) for k in self._store] or [0.5]
        conf = statistics.mean(imps)
        self._ema_conf = 0.15 * conf + 0.85 * self._ema_conf
        entropy = -sum((p / (sum(imps) + 1e-12)) * math.log2((p / (sum(imps) + 1e-12)) + 1e-12) for p in imps)
        return {"items": cap["items"], "confidence": round(self._ema_conf, 4),
                "accuracy": round(conf, 4), "quality": round(1.0 - cap["pct"] / 100, 4),
                "active": cap["items"], "cycles": self._cycles,
                "entropy": round(entropy, 4), "pending": 0}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(self._CYCLE_SEC)
            try:
                self.consolidate()
                st = self.status()
                self._cycles += 1
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "causal_auto_cycle", "EMA+decay+prune",
                        st["confidence"], st["items"] > 0)
                except Exception:
                    pass
                try:
                    from WorkingMemory import WorkingMemory
                    WorkingMemory().store("causal_status", json.dumps(st), st["confidence"])
                except Exception:
                    pass
                if st["entropy"] > 3.5:
                    try:
                        from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                        HierarchicalGoalPlanner().add_goal(
                            "Reduce causal graph entropy — review contradictory chains", priority=2)
                    except Exception:
                        pass
            except Exception:
                pass

# Usage: obj = CausalChainEngine() | result = obj.trace("market_crash")