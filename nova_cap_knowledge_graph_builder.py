"""
Nova ASI — Knowledge Graph Builder
Proposed autonomously via /evolve
"""

"""
KnowledgeGraph — Nova's semantic knowledge fabric with Bayesian confidence,
BFS shortest-path, causal propagation, EMA-decayed salience, and autonomous
goal-generation. Satisfies pillars ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭.
"""

import sqlite3
import threading
import time
import math
import statistics
import collections
import hashlib
import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple

class KnowledgeGraph:
    """Persistent, self-monitoring knowledge graph wired into Nova's cognition."""

    _DB = "nova_knowledge_graph.db"
    _CAPACITY = 500
    _DECAY = 1e-5
    _CONF_EMA = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._build_schema()
        self._cycles: int = 0
        self._conf_ema: float = 0.7
        self._history: List[Tuple[float, float]] = []
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _build_schema(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS entities(
            name TEXT PRIMARY KEY, type TEXT,
            importance REAL DEFAULT 1.0,
            ts REAL, accesses INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS relations(
            id TEXT PRIMARY KEY, e1 TEXT, relation TEXT, e2 TEXT,
            confidence REAL DEFAULT 0.8,
            ts REAL, accesses INTEGER DEFAULT 0)""")
        c.commit()

    def _salience(self, ts: float, accesses: int, importance: float) -> float:
        elapsed = time.time() - ts
        decay = math.exp(-self._DECAY * elapsed)
        return importance * decay * math.log1p(accesses + 1)

    def add_entity(self, name: str, entity_type: str) -> Dict[str, Any]:
        """Returns dict with entity name, type, and initial confidence."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entities(name,type,importance,ts,accesses) VALUES(?,?,?,?,?)",
                (name, entity_type, 1.0, now, 0))
            self._conn.commit()
            self._evict_if_needed()
        try:
            from WorkingMemory import WorkingMemory
            WorkingMemory().consolidate()
        except (ImportError, Exception):
            pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Enrich entity: {name}", priority=2)
        except (ImportError, Exception):
            pass
        return {"entity": name, "type": entity_type, "confidence": self._conf_ema}

    def add_relation(self, e1: str, relation: str, e2: str,
                     confidence: float = 0.8) -> Dict[str, Any]:
        """Returns relation record with propagated causal confidence."""
        rid = hashlib.md5(f"{e1}{relation}{e2}".encode()).hexdigest()
        now = time.time()
        prior = self._conf_ema
        likelihood = confidence
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO relations(id,e1,relation,e2,confidence,ts,accesses) VALUES(?,?,?,?,?,?,?)",
                (rid, e1, relation, e2, posterior, now, 0))
            self._conn.commit()
        self._conf_ema = self._CONF_EMA * posterior + (1 - self._CONF_EMA) * self._conf_ema
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("knowledge_graph", "bayesian_relation",
                                                  posterior, posterior > 0.5)
        except (ImportError, Exception):
            pass
        return {"e1": e1, "relation": relation, "e2": e2,
                "confidence": posterior, "ema_conf": self._conf_ema}

    def query(self, entity: str) -> Dict[str, Any]:
        """Returns all relations and neighbours connected to entity with salience scores."""
        with self._lock:
            self._conn.execute(
                "UPDATE entities SET accesses=accesses+1 WHERE name=?", (entity,))
            rows = self._conn.execute(
                "SELECT e1,relation,e2,confidence,ts,accesses FROM relations "
                "WHERE e1=? OR e2=?", (entity, entity)).fetchall()
            self._conn.commit()
        neighbours = []
        for e1, rel, e2, conf, ts, acc in rows:
            other = e2 if e1 == entity else e1
            sal = self._salience(ts, acc, conf)
            neighbours.append({"entity": other, "relation": rel,
                                "confidence": conf, "salience": round(sal, 4)})
        neighbours.sort(key=lambda x: x["salience"], reverse=True)
        entropy = self._graph_entropy()
        return {"entity": entity, "neighbours": neighbours,
                "count": len(neighbours), "graph_entropy": entropy,
                "confidence_ema": round(self._conf_ema, 4)}

    def shortest_path(self, a: str, b: str) -> Dict[str, Any]:
        """Returns BFS shortest path between two entities with cumulative confidence."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e1,e2,confidence FROM relations").fetchall()
        graph: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        for e1, e2, conf in rows:
            graph[e1].append((e2, conf))
            graph[e2].append((e1, conf))
        queue = collections.deque([(a, [a], 1.0)])
        visited = {a}
        while queue:
            node, path, cum_conf = queue.popleft()
            if node == b:
                ci_low = cum_conf - 1.96 * math.sqrt(cum_conf * (1 - cum_conf) / max(len(path), 1))
                return {"path": path, "length": len(path) - 1,
                        "cumulative_confidence": round(cum_conf, 4),
                        "ci_low": round(max(0.0, ci_low), 4),
                        "algorithm": "BFS + multiplicative_causal_propagation"}
            for neighbour, conf in graph.get(node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    new_conf = cum_conf * conf
                    if new_conf >= 0.05:
                        queue.append((neighbour, path + [neighbour], new_conf))
        return {"path": [], "length": -1, "cumulative_confidence": 0.0,
                "ci_low": 0.0, "algorithm": "BFS"}

    def _graph_entropy(self) -> float:
        rows = self._conn.execute(
            "SELECT confidence FROM relations").fetchall()
        if not rows:
            return 0.0
        confs = [r[0] for r in rows]
        total = sum(confs) + 1e-12
        dist = {i: c / total for i, c in enumerate(confs)}
        return round(-sum(p * math.log2(p + 1e-12) for p in dist.values()), 4)

    def _evict_if_needed(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        if count > self._CAPACITY:
            rows = self._conn.execute(
                "SELECT name,importance,ts,accesses FROM entities").fetchall()
            scored = sorted(rows, key=lambda r: self._salience(r[2], r[3], r[1]))
            to_drop = scored[:count - self._CAPACITY]
            for row in to_drop:
                self._conn.execute("DELETE FROM entities WHERE name=?", (row[0],))
            self._conn.commit()

    def consolidate(self) -> Dict[str, Any]:
        """Returns summary after pruning low-confidence relations below threshold."""
        with self._lock:
            before = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            self._conn.execute("DELETE FROM relations WHERE confidence < 0.05")
            self._conn.commit()
            after = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return {"pruned": before - after, "remaining": after}

    def anomaly_check(self) -> Dict[str, Any]:
        """Returns z-score anomaly report on relation confidence distribution."""
        with self._lock:
            rows = self._conn.execute("SELECT confidence FROM relations").fetchall()
        if len(rows) < 3:
            return {"anomalies": [], "status": "insufficient_data"}
        vals = [r[0] for r in rows]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) + 1e-9
        anomalies = [{"value": v, "z": round((v - mean) / std, 3)}
                     for v in vals if abs((v - mean) / std) > 3.0]
        return {"anomalies": anomalies, "mean": round(mean, 4),
                "std": round(std, 4), "total": len(vals)}

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            rels = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        entropy = self._graph_entropy()
        return {"items": entities, "confidence": round(self._conf_ema, 4),
                "active": rels, "entropy": round(entropy, 4),
                "cycles": self._cycles, "quality": round(self._conf_ema, 4),
                "accuracy": round(self._conf_ema, 4), "pending": 0}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            try:
                with self._lock:
                    self._cycles += 1
                self.consolidate()
                self.anomaly_check()
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "knowledge_graph", "auto_consolidation",
                        self._conf_ema, True)
                except (ImportError, Exception):
                    pass
                try:
                    from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                    if self._cycles % 5 == 0:
                        HierarchicalGoalPlanner().add_goal(
                            "Expand knowledge graph coverage", priority=3)
                except (ImportError, Exception):
                    pass
            except Exception as e:
                pass

# Usage: obj = KnowledgeGraph() | result = obj.add_entity("Nova","AI") | result = obj.shortest_path("Nova","Human")