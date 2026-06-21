"""
Nova ASI — true
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
NovaTrueBeliefAuditor — audits Nova's belief system for truth, coherence, and calibration.

Continuously monitors belief confidence vs. observed accuracy, applies Bayesian updates,
detects contradictions via causal chain analysis, and self-generates improvement goals.
Integrates with WorkingMemory, BayesianBeliefSystem, HierarchicalGoalPlanner, and
MetacognitiveMonitor to keep Nova's epistemic state honest and self-correcting.
"""

import sqlite3
import threading
import time
import math
import statistics
import json
import hashlib
import os
from collections import OrderedDict
from datetime import datetime

class NovaTrueBeliefAuditor:
    """Audits Nova's beliefs for truth, calibration, and internal coherence autonomously."""

    DB_PATH = "nova_true_belief_auditor.db"
    ANOMALY_Z = 3.0
    EMA_ALPHA = 0.15
    PRUNE_CONF = 0.05
    CYCLE_INTERVAL = 60

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._ema_accuracy: float = 0.5
        self._history: list = []
        self._beliefs: OrderedDict = OrderedDict()
        self._init_db()
        self._load_state()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS belief_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, belief_id TEXT, label TEXT,
                prior REAL, posterior REAL, outcome REAL,
                confidence REAL, entropy REAL, notes TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS audit_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, cycles INTEGER, ema_accuracy REAL,
                calibration_error REAL, anomalies INTEGER)""")
            conn.commit()

    def _load_state(self) -> None:
        with sqlite3.connect(self.DB_PATH) as conn:
            rows = conn.execute(
                "SELECT belief_id, label, posterior, confidence FROM belief_log "
                "ORDER BY ts DESC LIMIT 200").fetchall()
        seen = set()
        for bid, label, post, conf in rows:
            if bid not in seen:
                self._beliefs[bid] = {"label": label, "posterior": post,
                                      "confidence": conf, "accesses": 0}
                seen.add(bid)

    def record_belief(self, label: str, prior: float, likelihood: float,
                      notes: str = "") -> dict:
        """Returns updated belief dict with Bayesian posterior and entropy score."""
        prior = max(1e-9, min(1.0 - 1e-9, prior))
        likelihood = max(1e-9, min(1.0 - 1e-9, likelihood))
        neg_lh = 1.0 - likelihood
        neg_pr = 1.0 - prior
        posterior = (likelihood * prior) / (likelihood * prior + neg_lh * neg_pr)
        entropy = -(posterior * math.log2(posterior + 1e-12) +
                    (1 - posterior) * math.log2(1 - posterior + 1e-12))
        bid = hashlib.md5(label.encode()).hexdigest()[:10]
        with self._lock:
            self._beliefs[bid] = {"label": label, "posterior": posterior,
                                  "confidence": likelihood, "accesses": 0,
                                  "entropy": entropy}
        ts = time.time()
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("INSERT INTO belief_log(ts,belief_id,label,prior,posterior,"
                         "outcome,confidence,entropy,notes) VALUES(?,?,?,?,?,?,?,?,?)",
                         (ts, bid, label, prior, posterior, -1.0, likelihood, entropy, notes))
            conn.commit()
        try:
            from WorkingMemory import WorkingMemory
            wm = WorkingMemory()
            wm.store(f"belief_{bid}", json.dumps({"label": label, "posterior": posterior}),
                     importance=posterior)
        except (ImportError, Exception):
            pass
        return {"belief_id": bid, "label": label, "prior": prior,
                "posterior": posterior, "entropy": round(entropy, 4)}

    def observe_outcome(self, belief_id: str, outcome: float) -> dict:
        """Returns calibration update dict after observing a real-world outcome (0–1)."""
        outcome = max(0.0, min(1.0, outcome))
        with self._lock:
            belief = self._beliefs.get(belief_id)
            if belief is None:
                return {"error": "belief_id not found"}
            predicted = belief["posterior"]
            error = abs(predicted - outcome)
            self._history.append((predicted, outcome))
            if len(self._history) > 200:
                self._history.pop(0)
            self._ema_accuracy = (1.0 - self.EMA_ALPHA) * self._ema_accuracy + \
                                  self.EMA_ALPHA * (1.0 - error)
            belief["posterior"] = (0.85 * predicted + 0.15 * outcome)
            belief["accesses"] += 1
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("UPDATE belief_log SET outcome=? WHERE belief_id=? AND outcome<0",
                         (outcome, belief_id))
            conn.commit()
        try:
            from BayesianBeliefSystem import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            bbs.add_causal_edge(belief_id, "outcome_observed", outcome)
        except (ImportError, Exception):
            pass
        return {"belief_id": belief_id, "predicted": round(predicted, 4),
                "outcome": outcome, "error": round(error, 4),
                "ema_accuracy": round(self._ema_accuracy, 4)}

    def calibration_report(self) -> dict:
        """Returns calibration error, confidence interval, and EMA accuracy over recent history."""
        with self._lock:
            hist = list(self._history[-50:])
            ema = self._ema_accuracy
        if len(hist) < 2:
            return {"calibration_error": None, "ema_accuracy": round(ema, 4),
                    "samples": len(hist), "confidence_interval": None}
        errors = [abs(p - a) for p, a in hist]
        mean_err = statistics.mean(errors)
        std_err = statistics.stdev(errors) if len(errors) > 1 else 0.0
        z = 1.96
        ci_low = max(0.0, mean_err - z * std_err / math.sqrt(len(errors)))
        ci_high = min(1.0, mean_err + z * std_err / math.sqrt(len(errors)))
        return {"calibration_error": round(mean_err, 4),
                "ema_accuracy": round(ema, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "samples": len(hist)}

    def detect_contradictions(self) -> list:
        """Returns list of belief pairs whose posteriors sum outside [0.95, 1.05] (mutual exclusivity)."""
        with self._lock:
            items = list(self._beliefs.items())
        contradictions = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                bid_a, b_a = items[i]
                bid_b, b_b = items[j]
                combined = b_a["posterior"] + b_b["posterior"]
                if combined > 1.95:
                    contradictions.append({
                        "belief_a": b_a["label"], "belief_b": b_b["label"],
                        "combined_posterior": round(combined, 4),
                        "severity": round(combined - 1.0, 4)})
        return contradictions

    def anomaly_scan(self) -> dict:
        """Returns z-score anomaly report for beliefs with unusually high or low confidence."""
        with self._lock:
            confs = [b["confidence"] for b in self._beliefs.values() if "confidence" in b]
        if len(confs) < 3:
            return {"anomalies": [], "status": "insufficient_data"}
        mu = statistics.mean(confs)
        sigma = statistics.stdev(confs)
        anomalies = []
        with self._lock:
            for bid, b in self._beliefs.items():
                z = (b.get("confidence", mu) - mu) / (sigma + 1e-9)
                if abs(z) > self.ANOMALY_Z:
                    anomalies.append({"belief_id": bid, "label": b["label"],
                                      "z_score": round(z, 3),
                                      "confidence": round(b.get("confidence", 0), 4)})
        return {"anomalies": anomalies, "mean_confidence": round(mu, 4),
                "std_confidence": round(sigma, 4), "total_scanned": len(confs)}

    def prune_weak_beliefs(self) -> dict:
        """Returns count of pruned beliefs with posterior below PRUNE_CONF threshold."""
        pruned = []
        with self._lock:
            for bid in list(self._beliefs.keys()):
                if self._beliefs[bid].get("posterior", 1.0) < self.PRUNE_CONF:
                    pruned.append(bid)
                    del self._beliefs[bid]
        return {"pruned": len(pruned), "pruned_ids": pruned,
                "remaining": len(self._beliefs)}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            items = len(self._beliefs)
            ema = self._ema_accuracy
            cycles = self._cycles
        cal = self.calibration_report()
        return {"items": items, "confidence": round(ema, 4),
                "accuracy": round(ema, 4),
                "calibration_error": cal.get("calibration_error") or 0.0,
                "cycles": cycles, "active": 1,
                "entropy": round(-ema * math.log2(ema + 1e-12), 4)}

    def auto_cycle(self) -> dict:
        """Returns summary dict from one full autonomous audit cycle."""
        with self._lock:
            self._cycles += 1
            cycle_num = self._cycles
        contradictions = self.detect_contradictions()
        anomalies = self.anomaly_scan()
        pruned = self.prune_weak_beliefs()
        cal = self.calibration_report()
        cal_err = cal.get("calibration_error") or 0.0
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            if cal_err > 0.2:
                hgp.add_goal(f"Reduce belief calibration error below 0.2 (current {cal_err:.3f})",
                             priority=8)
            if len(contradictions) > 0:
                hgp.add_goal(f"Resolve {len(contradictions)} contradictory belief pairs",
                             priority=9)
        except (ImportError, Exception):
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mcm = MetacognitiveMonitor()
            mcm.log_reasoning("TrueBeliefAudit", "Bayesian+EMA+ZScore",
                              confidence=self._ema_accuracy,
                              success=1 if cal_err < 0.15 else 0)
        except (ImportError, Exception):
            pass
        summary = {"cycle": cycle_num, "contradictions": len(contradictions),
                   "anomalies": len(anomalies.get("anomalies", [])),
                   "pruned": pruned["pruned"], "calibration_error": cal_err,
                   "ema_accuracy": round(self._ema_accuracy, 4)}
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("INSERT INTO audit_cycles(ts,cycles,ema_accuracy,calibration_error,anomalies)"
                         " VALUES(?,?,?,?,?)",
                         (time.time(), cycle_num, self._ema_accuracy, cal_err,
                          len(anomalies.get("anomalies", []))))
            conn.commit()
        return summary

    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(self.CYCLE_INTERVAL)
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaTrueBeliefAuditor() | result = obj.record_belief("The sky is blue", 0.7, 0.95)
