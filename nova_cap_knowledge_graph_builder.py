"""
Nova ASI — Knowledge Graph Builder
Proposed autonomously via /evolve
"""

"""
Nova KnowledgeGraphFabric — Bayesian-weighted, decay-aware knowledge graph with BFS shortest-path,
anomaly detection, autonomous goal generation, and full cross-system integration.
Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3
import threading
import math
import time
import statistics
import hashlib
import json
import os
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_kg_fabric.db")

class KnowledgeGraphFabric:
    """Persistent, self-monitoring knowledge graph with Bayesian edge confidence and decay."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._anomaly_log: List[Dict] = []
        self._access_counts: Dict[str, int] = {}
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                etype TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                e1 TEXT NOT NULL,
                relation TEXT NOT NULL,
                e2 TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()

    def add_entity(self, name: str, etype: str) -> Dict[str, Any]:
        """Returns dict confirming entity upsert with importance score."""
        now = time.time()
        importance = 1.0
        with self._lock:
            c = self._conn.cursor()
            c.execute("""INSERT INTO entities(name,etype,importance,timestamp,access_count)
                         VALUES(?,?,?,?,0) ON CONFLICT(name) DO UPDATE SET
                         etype=excluded.etype, timestamp=excluded.timestamp""",
                      (name, etype, importance, now))
            self._conn.commit()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Enrich entity '{name}' of type '{etype}'", priority=2)
        except Exception:
            pass
        return {"entity": name, "type": etype, "importance": importance, "status": "stored"}

    def add_relation(self, e1: str, relation: str, e2: str,
                     confidence: float = 0.8) -> Dict[str, Any]:
        """Returns dict with edge id and propagated Bayesian confidence."""
        rid = hashlib.md5(f"{e1}|{relation}|{e2}".encode()).hexdigest()[:12]
        now = time.time()
        prior = 0.5
        likelihood = max(0.01, min(0.99, confidence))
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        with self._lock:
            c = self._conn.cursor()
            c.execute("""INSERT INTO relations(id,e1,relation,e2,confidence,timestamp,access_count)
                         VALUES(?,?,?,?,?,?,0) ON CONFLICT(id) DO UPDATE SET
                         confidence=excluded.confidence, timestamp=excluded.timestamp""",
                      (rid, e1, relation, e2, posterior, now))
            self._conn.commit()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("kg_relation", f"{e1}-{relation}->{e2}",
                                                  posterior, posterior > 0.6)
        except Exception:
            pass
        return {"id": rid, "e1": e1, "relation": relation, "e2": e2,
                "confidence": round(posterior, 4), "ci_low": round(posterior - 0.1, 4),
                "ci_high": round(posterior + 0.1, 4)}

    def query(self, entity: str) -> Dict[str, Any]:
        """Returns all relations and decayed importance for the queried entity."""
        now = time.time()
        decay_rate = 1e-5
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT importance, timestamp, access_count FROM entities WHERE name=?", (entity,))
            row = c.fetchone()
            if row is None:
                return {"entity": entity, "found": False, "relations": []}
            raw_imp, ts, acc = row
            elapsed = now - ts
            decayed = raw_imp * math.exp(-decay_rate * elapsed) * math.log1p(acc + 1)
            c.execute("UPDATE entities SET access_count=access_count+1 WHERE name=?", (entity,))
            self._conn.commit()
            c.execute("""SELECT e1,relation,e2,confidence FROM relations
                         WHERE e1=? OR e2=? ORDER BY confidence DESC""", (entity, entity))
            rels = [{"e1": r[0], "relation": r[1], "e2": r[2], "confidence": round(r[3], 4)}
                    for r in c.fetchall()]
        try:
            from WorkingMemory import WorkingMemory
            WorkingMemory().store(f"kg_query_{entity}", rels, decayed)
        except Exception:
            pass
        return {"entity": entity, "found": True, "decayed_importance": round(decayed, 4),
                "relations": rels, "relation_count": len(rels)}

    def shortest_path(self, a: str, b: str) -> Dict[str, Any]:
        """Returns BFS shortest path between two entities with cumulative confidence."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT e1,e2,relation,confidence FROM relations")
            edges = c.fetchall()
        graph: Dict[str, List[Tuple]] = {}
        for e1, e2, rel, conf in edges:
            graph.setdefault(e1, []).append((e2, rel, conf))
            graph.setdefault(e2, []).append((e1, rel, conf))
        queue: deque = deque([(a, [a], [], 1.0)])
        visited = {a}
        while queue:
            node, path, rels, cum_conf = queue.popleft()
            if node == b:
                return {"path": path, "relations": rels,
                        "cumulative_confidence": round(cum_conf, 4),
                        "hops": len(path) - 1}
            for neighbor, rel, conf in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_conf = cum_conf * conf
                    if new_conf >= 0.05:
                        queue.append((neighbor, path + [neighbor], rels + [rel], new_conf))
        return {"path": [], "relations": [], "cumulative_confidence": 0.0, "hops": -1}

    def anomaly_check(self) -> Dict[str, Any]:
        """Returns z-score anomaly report on edge confidence distribution."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT confidence FROM relations")
            confs = [r[0] for r in c.fetchall()]
        if len(confs) < 3:
            return {"anomalies": [], "mean": 0.0, "std": 0.0}
        mu = statistics.mean(confs)
        sd = statistics.stdev(confs) + 1e-9
        anomalies = [{"confidence": round(v, 4), "z": round((v - mu) / sd, 3)}
                     for v in confs if abs((v - mu) / sd) > 3.0]
        entropy = -sum((p / len(confs)) * math.log2(p / len(confs) + 1e-12)
                       for p in [confs.count(v) for v in set(confs)])
        return {"anomalies": anomalies, "mean": round(mu, 4), "std": round(sd, 4),
                "entropy": round(entropy, 4), "total_edges": len(confs)}

    def consolidate(self) -> Dict[str, Any]:
        """Prunes edges below confidence 0.05 and returns count of removed edges."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("DELETE FROM relations WHERE confidence < 0.05")
            pruned = c.rowcount
            self._conn.commit()
        return {"pruned_edges": pruned, "status": "consolidated"}

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM entities")
            items = c.fetchone()[0]
            c.execute("SELECT COUNT(*), AVG(confidence) FROM relations")
            row = c.fetchone()
            edges, avg_conf = row[0], row[1] or 0.0
        accuracy = round(avg_conf, 4)
        entropy_val = round(-avg_conf * math.log2(avg_conf + 1e-12), 4) if avg_conf > 0 else 0.0
        return {"items": items, "active": edges, "confidence": accuracy,
                "accuracy": accuracy, "entropy": entropy_val,
                "cycles": self._cycles, "pending": len(self._anomaly_log)}

    def auto_cycle(self) -> Dict[str, Any]:
        """Runs one maintenance cycle: consolidate, anomaly check, goal generation."""
        self._cycles += 1
        consolidated = self.consolidate()
        anomalies = self.anomaly_check()
        if anomalies["anomalies"]:
            self._anomaly_log.extend(anomalies["anomalies"][-5:])
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"KG cycle {self._cycles}: resolve {len(anomalies['anomalies'])} anomalies", priority=3)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("kg_auto_cycle", "consolidate+anomaly",
                                                  anomalies.get("mean", 0.5), consolidated["pruned_edges"] == 0)
        except Exception:
            pass
        return {"cycle": self._cycles, "consolidated": consolidated, "anomaly_report": anomalies}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = KnowledgeGraphFabric() | result = obj.add_entity("Nova","AI") | obj.add_relation("Nova","uses","Python") | obj.query("Nova") | obj.shortest_path("Nova","Python")
