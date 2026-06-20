"""
Nova ASI — an
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
NovaAdaptiveReasoningCore — a self-improving probabilistic reasoning engine that
integrates Bayesian belief updates, causal chain propagation, EMA-based online
learning, TF-IDF salience scoring, z-score anomaly detection, and autonomous
goal generation. Permanently wired into Nova's cognition as her meta-reasoning
backbone, running 24/7 without human prompting.
"""

import math
import time
import sqlite3
import threading
import statistics
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_adaptive_reasoning.db")

class NovaAdaptiveReasoningCore:
    """Autonomous probabilistic reasoning backbone with Bayesian, causal, and EMA learning."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_pred: Dict[str, float] = {}
        self._ema_history: Dict[str, List[Tuple[float, float]]] = {}
        self._cycles: int = 0
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_state()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS beliefs(
            domain TEXT PRIMARY KEY, prior REAL, posterior REAL,
            confidence REAL, entropy REAL, updated REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS causal_chains(
            chain_id TEXT PRIMARY KEY, nodes TEXT, confidences TEXT,
            joint_conf REAL, created REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS reasoning_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT,
            insight TEXT, z_score REAL, ts REAL)""")
        self._conn.commit()

    def _load_state(self) -> None:
        c = self._conn.cursor()
        c.execute("SELECT domain, posterior, confidence FROM beliefs")
        for row in c.fetchall():
            domain, posterior, conf = row
            self._ema_pred[domain] = posterior
            self._ema_history[domain] = []

    def update_belief(self, domain: str, evidence: float, likelihood: float) -> Dict[str, float]:
        """Returns updated Bayesian posterior, entropy, and confidence for a domain."""
        with self._lock:
            prior = self._ema_pred.get(domain, 0.5)
            lp = likelihood * prior
            ln = (1.0 - likelihood) * (1.0 - prior)
            denom = lp + ln + 1e-12
            posterior = lp / denom
            alt = ln / denom
            entropy = -sum(p * math.log2(p + 1e-12) for p in [posterior, alt])
            confidence = 1.0 - (entropy / 1.0)
            self._ema_pred[domain] = 0.15 * posterior + 0.85 * prior
            c = self._conn.cursor()
            c.execute("""INSERT OR REPLACE INTO beliefs VALUES(?,?,?,?,?,?)""",
                      (domain, prior, self._ema_pred[domain], confidence, entropy, time.time()))
            self._conn.commit()
            return {"domain": domain, "posterior": round(self._ema_pred[domain], 4),
                    "entropy": round(entropy, 4), "confidence": round(confidence, 4)}

    def observe_outcome(self, domain: str, predicted: float, actual: float) -> Dict[str, float]:
        """Returns EMA-updated prediction error and MAE for a domain after observing an outcome."""
        with self._lock:
            prev = self._ema_pred.get(domain, predicted)
            updated = 0.15 * actual + 0.85 * prev
            self._ema_pred[domain] = updated
            hist = self._ema_history.setdefault(domain, [])
            hist.append((predicted, actual))
            if len(hist) > 100:
                hist.pop(0)
            mae = statistics.mean(abs(p - a) for p, a in hist[-50:]) if len(hist) >= 2 else abs(predicted - actual)
            return {"domain": domain, "ema_pred": round(updated, 4),
                    "mae": round(mae, 4), "samples": len(hist)}

    def propagate_causal_chain(self, nodes: List[str], confidences: List[float]) -> Dict[str, Any]:
        """Returns joint causal confidence for a chain A→B→C, pruning links below 0.05."""
        if len(nodes) != len(confidences) + 1:
            return {"error": "nodes must be len(confidences)+1"}
        pruned_nodes, pruned_confs = [nodes[0]], []
        joint = 1.0
        for i, conf in enumerate(confidences):
            if conf < 0.05:
                break
            joint *= conf
            pruned_nodes.append(nodes[i + 1])
            pruned_confs.append(conf)
        chain_id = hashlib.md5("->".join(pruned_nodes).encode()).hexdigest()[:10]
        with self._lock:
            c = self._conn.cursor()
            c.execute("""INSERT OR REPLACE INTO causal_chains VALUES(?,?,?,?,?)""",
                      (chain_id, json.dumps(pruned_nodes), json.dumps(pruned_confs),
                       round(joint, 6), time.time()))
            self._conn.commit()
        return {"chain_id": chain_id, "nodes": pruned_nodes,
                "joint_confidence": round(joint, 6), "steps": len(pruned_confs)}

    def tfidf_salience(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Returns documents ranked by TF-IDF salience score against the query."""
        tokens = re.findall(r'\\w+', query.lower())
        N = len(documents) + 1
        results = []
        for doc in documents:
            words = re.findall(r'\\w+', doc.lower())
            if not words:
                continue
            score = 0.0
            for t in tokens:
                tf = words.count(t) / len(words)
                df = sum(1 for d in documents if t in re.findall(r'\\w+', d.lower()))
                idf = math.log(N / (df + 1))
                score += tf * idf
            results.append((doc[:60], round(score, 4)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def anomaly_check(self, domain: str, value: float) -> Dict[str, Any]:
        """Returns z-score anomaly assessment and alert flag for a new observation."""
        with self._lock:
            hist = self._ema_history.get(domain, [])
            actuals = [a for _, a in hist]
            if len(actuals) < 3:
                return {"domain": domain, "z_score": 0.0, "alert": False, "reason": "insufficient_data"}
            mu = statistics.mean(actuals)
            sigma = statistics.stdev(actuals) + 1e-9
            z = (value - mu) / sigma
            alert = abs(z) > 3.0
            if alert:
                c = self._conn.cursor()
                c.execute("INSERT INTO reasoning_log(domain,insight,z_score,ts) VALUES(?,?,?,?)",
                          (domain, f"Anomaly detected: value={value:.3f}", round(z, 3), time.time()))
                self._conn.commit()
            return {"domain": domain, "z_score": round(z, 3), "alert": alert,
                    "mean": round(mu, 4), "std": round(sigma, 4)}

    def generate_goal(self, context: str) -> str:
        """Returns a self-generated sub-goal string and registers it with HierarchicalGoalPlanner."""
        entropy_vals = []
        c = self._conn.cursor()
        c.execute("SELECT domain, entropy FROM beliefs ORDER BY entropy DESC LIMIT 5")
        rows = c.fetchall()
        top_domain = rows[0][0] if rows else "general_reasoning"
        entropy_vals = [r[1] for r in rows]
        avg_entropy = statistics.mean(entropy_vals) if entropy_vals else 0.5
        goal = f"Reduce epistemic uncertainty in '{top_domain}' (entropy={avg_entropy:.3f}) via targeted evidence gathering"
        try:
            from tools import HierarchicalGoalPlanner
            planner = HierarchicalGoalPlanner()
            planner.add_goal(goal, priority=max(1, int(avg_entropy * 10)))
        except (ImportError, AttributeError, Exception):
            pass
        return goal

    def status(self) -> Dict[str, Any]:
        """Returns a numeric-keyed status dict compatible with ConsciousnessIntegrator Φ."""
        c = self._conn.cursor()
        c.execute("SELECT COUNT(*) FROM beliefs")
        items = c.fetchone()[0]
        c.execute("SELECT AVG(confidence) FROM beliefs")
        row = c.fetchone()
        avg_conf = row[0] if row[0] else 0.0
        c.execute("SELECT AVG(entropy) FROM beliefs")
        row2 = c.fetchone()
        avg_entropy = row2[0] if row2[0] else 0.0
        maes = [statistics.mean(abs(p - a) for p, a in h[-20:])
                for h in self._ema_history.values() if len(h) >= 2]
        accuracy = round(1.0 - (statistics.mean(maes) if maes else 0.5), 4)
        return {"items": items, "confidence": round(avg_conf, 4),
                "accuracy": accuracy, "entropy": round(avg_entropy, 4),
                "cycles": self._cycles, "active": 1, "pending": max(0, items - self._cycles)}

    def auto_cycle(self) -> Dict[str, Any]:
        """Returns a summary of one autonomous reasoning cycle with metacognitive logging."""
        with self._lock:
            self._cycles += 1
        goal = self.generate_goal("autonomous_cycle")
        st = self.status()
        try:
            from tools import MetacognitiveMonitor
            mon = MetacognitiveMonitor()
            mon.log_reasoning("adaptive_reasoning", "bayesian+ema+causal",
                              st["confidence"], st["accuracy"] > 0.6)
        except (ImportError, AttributeError, Exception):
            pass
        return {"cycle": self._cycles, "goal": goal, "status": st}

    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(60)
                self.auto_cycle()
            except Exception:
                time.sleep(5)

# Usage: obj = NovaAdaptiveReasoningCore() | result = obj.update_belief("physics", 0.9, 0.85)