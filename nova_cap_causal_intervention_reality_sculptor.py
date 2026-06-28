"""
nova_cap_causal_intervention_reality_sculptor.py
Nova invented this autonomously — Causal Intervention Reality Sculptor
Generated via /evolve · v29 pipeline · 2026-06-28
"""

"""
InterventionPoint — Causal Intervention Reality Sculptor for Nova ASI.
"""
import math
import time
import threading
import statistics
import sqlite3
import os
from collections import OrderedDict

class InterventionPoint:
    """Identifies minimal high-leverage causal intervention points and sculpts reality by acting on them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._nodes: OrderedDict = OrderedDict()
        self._edges: list = []
        self._history: list = []
        self._cycles: int = 0
        self._db = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "intervention.db")
        self._init_db()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        """Initialises SQLite schema for persistent intervention records."""
        try:
            with sqlite3.connect(self._db) as c:
                c.execute("CREATE TABLE IF NOT EXISTS interventions (ts REAL, node TEXT, leverage REAL, outcome TEXT)")
        except Exception as e:
            pass

    def add_node(self, name: str, prior: float = 0.5) -> dict:
        """Adds a causal node with a prior probability; returns node record."""
        with self._lock:
            self._nodes[name] = {"prior": max(0.01, min(0.99, prior)), "accesses": 0, "ts": time.time()}
        return self._nodes[name]

    def add_edge(self, cause: str, effect: str, strength: float) -> dict:
        """Adds a directed causal edge; returns edge dict with multiplicative chain confidence."""
        edge = {"cause": cause, "effect": effect, "strength": max(0.05, min(0.99, strength))}
        with self._lock:
            self._edges.append(edge)
        return edge

    def leverage_score(self, node: str) -> float:
        """Computes intervention leverage via causal fan-out and entropy; returns float score."""
        with self._lock:
            if node not in self._nodes:
                return 0.0
            downstream = [e["strength"] for e in self._edges if e["cause"] == node]
            if not downstream:
                return 0.0
            chain = math.prod(downstream)
            p = self._nodes[node]["prior"]
            entropy = -(p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12))
            return round(chain * entropy * len(downstream), 4)

    def top_interventions(self, k: int = 3) -> list:
        """Returns top-k nodes ranked by leverage score as list of (node, score) tuples."""
        with self._lock:
            nodes = list(self._nodes.keys())
        scored = [(n, self.leverage_score(n)) for n in nodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def intervene(self, node: str, new_prior: float) -> dict:
        """Applies a do-calculus intervention on a node; returns updated belief propagation dict."""
        result = {"node": node, "old_prior": 0.0, "new_prior": new_prior, "propagated": []}
        with self._lock:
            if node not in self._nodes:
                return result
            result["old_prior"] = self._nodes[node]["prior"]
            self._nodes[node]["prior"] = max(0.01, min(0.99, new_prior))
            self._nodes[node]["accesses"] += 1
            for e in self._edges:
                if e["cause"] == node and e["effect"] in self._nodes:
                    child = e["effect"]
                    old = self._nodes[child]["prior"]
                    updated = (e["strength"] * new_prior) / ((e["strength"] * new_prior) + ((1 - e["strength"]) * (1 - new_prior)))
                    self._nodes[child]["prior"] = round(updated, 4)
                    result["propagated"].append({"node": child, "old": old, "new": updated})
            self._history.append({"ts": time.time(), "node": node, "leverage": self.leverage_score(node)})
        try:
            with sqlite3.connect(self._db) as c:
                c.execute("INSERT INTO interventions VALUES (?,?,?,?)", (time.time(), node, self.leverage_score(node), "intervened"))
        except Exception as e:
            pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Monitor downstream effects of intervention on {node}", priority=7)
        except Exception as e:
            pass
        return result

    def anomaly_scan(self) -> dict:
        """Detects nodes with z-score leverage anomalies; returns dict of flagged nodes."""
        with self._lock:
            scores = [self.leverage_score(n) for n in self._nodes]
        if len(scores) < 3:
            return {"flagged": [], "mean": 0.0, "std": 0.0}
        mu = statistics.mean(scores)
        sd = statistics.stdev(scores) + 1e-9
        flagged = [n for n in self._nodes if abs((self.leverage_score(n) - mu) / sd) > 2.0]
        return {"flagged": flagged, "mean": round(mu, 4), "std": round(sd, 4)}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            scores = [self.leverage_score(n) for n in self._nodes]
            conf = round(statistics.mean(scores), 4) if scores else 0.0
            return {"items": len(self._nodes), "cycles": self._cycles, "confidence": conf, "active": len(self._edges), "entropy": round(sum(scores), 4)}

    def _auto_loop(self) -> None:
        """Background daemon: runs anomaly scan and logs reasoning every 60 seconds."""
        while True:
            time.sleep(60)
            try:
                with self._lock:
                    self._cycles += 1
                self.anomaly_scan()
                top = self.top_interventions(1)
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning("causal_intervention", "leverage_entropy", top[0][1] if top else 0.0, True)
                except Exception as e:
                    pass
            except Exception as e:
                pass

# Usage: obj = InterventionPoint()
