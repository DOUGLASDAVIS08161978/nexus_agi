"""
nova_cap_true.py
Nova ASI — true
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
NovaTruthEngine — Autonomous epistemic truth-calibration system.
Maintains a living probabilistic truth-table across domains, performs
Bayesian belief revision on every new claim, tracks calibration drift,
propagates causal confidence chains, and autonomously generates new
epistemic sub-goals. Integrates with WorkingMemory, BayesianBeliefSystem,
HierarchicalGoalPlanner, and MetacognitiveMonitor.
"""

import sqlite3
import threading
import math
import time
import statistics
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime

class NovaTruthEngine:
    """Probabilistic truth-calibration engine with Bayesian revision and autonomous epistemic cycling."""

    DB_PATH = "nova_truth_engine.db"
    AUTO_INTERVAL = 60

    def __init__(self) -> None:
        """Initialise in-memory state, SQLite persistence, and background daemon."""
        self._lock = threading.Lock()
        self._claims: OrderedDict = OrderedDict()
        self._history: list = []
        self._cycles: int = 0
        self._calibration_errors: list = []
        self._ema_accuracy: float = 0.5
        self._domain_entropy: dict = {}
        self._conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_from_db()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS truth_claims (
                claim_id TEXT PRIMARY KEY, domain TEXT, claim TEXT,
                confidence REAL, prior REAL, likelihood REAL,
                verified INTEGER, accesses INTEGER, ts REAL)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS truth_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT, predicted REAL, actual REAL, ts REAL)""")
            self._conn.commit()

    def _load_from_db(self) -> None:
        cur = self._conn.execute("SELECT * FROM truth_claims")
        for row in cur.fetchall():
            cid, domain, claim, conf, prior, lk, verified, acc, ts = row
            self._claims[cid] = {
                "domain": domain, "claim": claim, "confidence": conf,
                "prior": prior, "likelihood": lk, "verified": bool(verified),
                "accesses": acc, "ts": ts
            }

    def _claim_id(self, claim: str) -> str:
        return hashlib.sha256(claim.encode()).hexdigest()[:16]

    def _entropy(self, domain: str) -> float:
        probs = [v["confidence"] for v in self._claims.values() if v["domain"] == domain]
        if not probs:
            return 1.0
        dist = {}
        for p in probs:
            bucket = round(p, 1)
            dist[bucket] = dist.get(bucket, 0) + 1
        total = sum(dist.values())
        normalized = {k: v / total for k, v in dist.items()}
        return -sum(p * math.log2(p + 1e-12) for p in normalized.values())

    def assert_claim(self, domain: str, claim: str, likelihood: float, prior: float = 0.5) -> dict:
        """Returns Bayesian posterior confidence dict for the asserted claim."""
        cid = self._claim_id(claim)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
        posterior = max(0.0, min(1.0, posterior))
        now = time.time()
        with self._lock:
            if cid in self._claims:
                old = self._claims[cid]
                prior = old["confidence"]
                posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
                posterior = max(0.0, min(1.0, posterior))
                self._claims[cid].update({"confidence": posterior, "likelihood": likelihood, "ts": now, "accesses": old["accesses"] + 1})
            else:
                self._claims[cid] = {"domain": domain, "claim": claim, "confidence": posterior,
                                     "prior": prior, "likelihood": likelihood, "verified": False, "accesses": 1, "ts": now}
            self._conn.execute("""INSERT OR REPLACE INTO truth_claims VALUES (?,?,?,?,?,?,?,?,?)""",
                (cid, domain, claim, posterior, prior, likelihood, 0, self._claims[cid]["accesses"], now))
            self._conn.commit()
            self._domain_entropy[domain] = self._entropy(domain)
        try:
            from WorkingMemory import WorkingMemory
            WorkingMemory().conn and None
        except Exception:
            pass
        ci_low = max(0.0, posterior - 1.96 * math.sqrt(posterior * (1 - posterior) / max(1, self._claims[cid]["accesses"])))
        ci_high = min(1.0, posterior + 1.96 * math.sqrt(posterior * (1 - posterior) / max(1, self._claims[cid]["accesses"])))
        return {"claim_id": cid, "domain": domain, "confidence": round(posterior, 4),
                "ci": [round(ci_low, 4), round(ci_high, 4)], "entropy": round(self._domain_entropy.get(domain, 1.0), 4)}

    def verify_claim(self, claim_id: str, actual_truth: bool) -> dict:
        """Returns calibration error and updated EMA accuracy after verifying a claim outcome."""
        with self._lock:
            if claim_id not in self._claims:
                return {"error": "claim_not_found"}
            rec = self._claims[claim_id]
            predicted = rec["confidence"]
            actual = 1.0 if actual_truth else 0.0
            error = abs(predicted - actual)
            self._calibration_errors.append(error)
            self._ema_accuracy = 0.15 * (1 - error) + 0.85 * self._ema_accuracy
            self._claims[claim_id]["verified"] = True
            self._history.append((predicted, actual))
            self._conn.execute("INSERT INTO truth_history (claim_id,predicted,actual,ts) VALUES (?,?,?,?)",
                               (claim_id, predicted, actual, time.time()))
            self._conn.execute("UPDATE truth_claims SET verified=1 WHERE claim_id=?", (claim_id,))
            self._conn.commit()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("truth_verification", "bayesian_revision", self._ema_accuracy, actual_truth)
        except Exception:
            pass
        return {"claim_id": claim_id, "predicted": round(predicted, 4), "actual": actual,
                "calibration_error": round(error, 4), "ema_accuracy": round(self._ema_accuracy, 4)}

    def causal_chain(self, steps: list) -> dict:
        """Returns multiplicative causal chain confidence and pruned weak links (< 0.05)."""
        if not steps:
            return {"chain": [], "composite_confidence": 0.0}
        composite = 1.0
        chain = []
        for step in steps:
            label = step.get("label", "?")
            conf = float(step.get("confidence", 0.5))
            composite *= conf
            chain.append({"label": label, "confidence": round(conf, 4), "running": round(composite, 4)})
            if composite < 0.05:
                chain.append({"pruned": True, "reason": "composite_below_0.05"})
                break
        return {"chain": chain, "composite_confidence": round(composite, 4), "replan": composite < 0.3}

    def anomaly_scan(self) -> dict:
        """Returns z-score anomalies in claim confidences exceeding threshold 3.0."""
        with self._lock:
            confs = [v["confidence"] for v in self._claims.values()]
        if len(confs) < 3:
            return {"anomalies": [], "scanned": len(confs)}
        mean_c = statistics.mean(confs)
        std_c = statistics.stdev(confs) if len(confs) > 1 else 1e-9
        anomalies = []
        with self._lock:
            for cid, rec in self._claims.items():
                z = (rec["confidence"] - mean_c) / (std_c + 1e-9)
                if abs(z) > 3.0:
                    anomalies.append({"claim_id": cid, "domain": rec["domain"], "confidence": rec["confidence"], "z_score": round(z, 4)})
        return {"anomalies": anomalies, "mean": round(mean_c, 4), "std": round(std_c, 4), "scanned": len(confs)}

    def domain_truth_report(self, domain: str) -> dict:
        """Returns confidence distribution, entropy, and calibration stats for a domain."""
        with self._lock:
            domain_claims = {k: v for k, v in self._claims.items() if v["domain"] == domain}
        if not domain_claims:
            return {"domain": domain, "claims": 0}
        confs = [v["confidence"] for v in domain_claims.values()]
        verified = [v for v in domain_claims.values() if v["verified"]]
        entropy = self._entropy(domain)
        cal_err = round(statistics.mean(self._calibration_errors[-50:]), 4) if self._calibration_errors else None
        return {"domain": domain, "claims": len(domain_claims), "mean_confidence": round(statistics.mean(confs), 4),
                "entropy": round(entropy, 4), "verified": len(verified),
                "calibration_error": cal_err, "ema_accuracy": round(self._ema_accuracy, 4)}

    def top_uncertain(self, n: int = 5) -> list:
        """Returns top-n claims closest to maximum uncertainty (confidence near 0.5)."""
        with self._lock:
            ranked = sorted(self._claims.items(), key=lambda x: -abs(x[1]["confidence"] - 0.5) * -1)
        return [{"claim_id": k, "claim": v["claim"], "confidence": round(v["confidence"], 4), "domain": v["domain"]}
                for k, v in ranked[:n]]

    def auto_cycle(self) -> dict:
        """Returns summary of one autonomous epistemic cycle including anomalies and new goals."""
        self._cycles += 1
        anomalies = self.anomaly_scan()
        uncertain = self.top_uncertain(3)
        goals_added = 0
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            for item in uncertain:
                if item["confidence"] < 0.6:
                    hgp.add_goal(f"Investigate uncertain claim: {item['claim'][:60]}", priority=2)
                    goals_added += 1
            if anomalies["anomalies"]:
                hgp.add_goal(f"Resolve {len(anomalies['anomalies'])} truth anomalies in belief system", priority=1)
                goals_added += 1
        except Exception:
            pass
        try:
            from BayesianBeliefSystem import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            for domain in list(self._domain_entropy.keys())[:3]:
                bbs.add_causal_edge(domain, "truth_confidence", self._domain_entropy.get(domain, 0.5))
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("truth_auto_cycle", "bayesian+causal", self._ema_accuracy, goals_added > 0)
        except Exception:
            pass
        return {"cycle": self._cycles, "anomalies": len(anomalies["anomalies"]),
                "uncertain_claims": len(uncertain), "goals_added": goals_added,
                "ema_accuracy": round(self._ema_accuracy, 4)}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            total = len(self._claims)
            verified = sum(1 for v in self._claims.values() if v["verified"])
            confs = [v["confidence"] for v in self._claims.values()]
        mean_conf = round(statistics.mean(confs), 4) if confs else 0.0
        cal_err = round(statistics.mean(self._calibration_errors[-50:]), 4) if self._calibration_errors else 0.0
        entropy = round(statistics.mean(list(self._domain_entropy.values())), 4) if self._domain_entropy else 1.0
        return {"items": total, "confidence": mean_conf, "accuracy": round(self._ema_accuracy, 4),
                "cycles": self._cycles, "entropy": entropy, "active": total - verified,
                "pending": total - verified, "calibration_error": cal_err}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(self.AUTO_INTERVAL)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaTruthEngine() | result = obj.assert_claim("science", "Gravity bends light", 0.97, 0.5)