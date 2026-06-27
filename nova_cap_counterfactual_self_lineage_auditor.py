"""
nova_cap_counterfactual_self_lineage_auditor.py
Nova invented this autonomously — Counterfactual Self Lineage Auditor
Generated via /evolve · v29 pipeline · 2026-06-26
"""

"""
CounterfactualSelfLineageAuditor — Nova's living causal graph of past decisions.

Reruns historical decision forks under current cognitive state, computes regret
gradients via NMF-inspired factorization, detects persistent blindspots, and
generates empirical growth certificates. Closes the self-improvement feedback
loop backward through time.
"""

import math
import time
import json
import sqlite3
import random
import hashlib
import threading
import statistics
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any

DB_PATH = "nova_lineage_auditor.db"

class CounterfactualSelfLineageAuditor:
    """Living causal audit graph: measures how much Nova has actually grown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._regret_history: list[dict] = []
        self._blindspot_cache: OrderedDict = OrderedDict()
        self._hypothesis_priors: dict[str, float] = {}
        self._evidence_log: list[dict] = []
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── DB bootstrap ──────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS decision_forks(
                id TEXT PRIMARY KEY, ts REAL, context_json TEXT,
                snapshot_json TEXT, chosen TEXT, alternatives_json TEXT,
                value_at_decision REAL, value_at_audit REAL, regret REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS growth_certs(
                capability TEXT, ts REAL, delta REAL, confidence REAL,
                evidence_json TEXT)""")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_ts ON decision_forks(ts)")

    # ── Bayesian belief layer ─────────────────────────────────────────────────
    def set_prior(self, hypothesis: str, prob: float) -> dict:
        """Returns updated prior table after setting hypothesis probability."""
        with self._lock:
            self._hypothesis_priors[hypothesis] = max(1e-9, min(1.0, prob))
            self._normalize_priors()
            return dict(self._hypothesis_priors)

    def update(self, evidence: str, likelihood_ratio: float) -> dict:
        """Returns posterior dict after Bayesian update with given likelihood ratio."""
        with self._lock:
            updated: dict[str, float] = {}
            for h, prior in self._hypothesis_priors.items():
                lh = likelihood_ratio if h == evidence else 1.0 / (likelihood_ratio + 1e-9)
                updated[h] = lh * prior
            total = sum(updated.values()) + 1e-12
            self._hypothesis_priors = {h: v / total for h, v in updated.items()}
            self._evidence_log.append({"evidence": evidence, "lr": likelihood_ratio,
                                        "ts": time.time(), "entropy": self._raw_entropy()})
            return dict(self._hypothesis_priors)

    def posterior(self, hypothesis: str) -> float:
        """Returns normalized posterior probability for the given hypothesis."""
        with self._lock:
            return self._hypothesis_priors.get(hypothesis, 0.0)

    def entropy(self) -> float:
        """Returns Shannon entropy of current belief distribution (bits)."""
        with self._lock:
            return self._raw_entropy()

    def _raw_entropy(self) -> float:
        return -sum(p * math.log2(p + 1e-12) for p in self._hypothesis_priors.values())

    def most_confident(self) -> dict:
        """Returns hypothesis with highest posterior and its confidence score."""
        with self._lock:
            if not self._hypothesis_priors:
                return {"hypothesis": None, "confidence": 0.0}
            best = max(self._hypothesis_priors, key=lambda h: self._hypothesis_priors[h])
            return {"hypothesis": best, "confidence": round(self._hypothesis_priors[best], 6)}

    def _normalize_priors(self) -> None:
        total = sum(self._hypothesis_priors.values()) + 1e-12
        self._hypothesis_priors = {h: v / total for h, v in self._hypothesis_priors.items()}

    # ── Core audit methods ────────────────────────────────────────────────────
    def record_decision(self, decision_id: str, context: dict,
                        chosen: str, alternatives: list[str],
                        value_estimate: float) -> dict:
        """Persists a decision fork to SQLite; returns confirmation dict."""
        snapshot = self._capture_cognitive_snapshot()
        ts = time.time()
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("""INSERT OR REPLACE INTO decision_forks VALUES
                    (?,?,?,?,?,?,?,?,?)""",
                    (decision_id, ts, json.dumps(context), json.dumps(snapshot),
                     chosen, json.dumps(alternatives), value_estimate, 0.0, 0.0))
        return {"recorded": decision_id, "ts": ts, "alternatives": len(alternatives)}

    def audit_decision_fork(self, decision_id: str, depth: int = 3) -> dict:
        """Returns ForkAuditReport dict with regret, delta, and counterfactual values."""
        with sqlite3.connect(DB_PATH) as cx:
            row = cx.execute("SELECT * FROM decision_forks WHERE id=?",
                             (decision_id,)).fetchone()
        if not row:
            return {"error": "decision_not_found", "id": decision_id}
        _, ts, ctx_json, snap_json, chosen, alts_json, v_orig, _, _ = row
        alts: list[str] = json.loads(alts_json)
        current_snap = self._capture_cognitive_snapshot()
        counterfactual_values: dict[str, float] = {}
        for alt in alts[:depth]:
            counterfactual_values[alt] = self._value_function(alt, current_snap)
        v_chosen_now = self._value_function(chosen, current_snap)
        regret = max(counterfactual_values.values(), default=v_chosen_now) - v_chosen_now
        regret = max(0.0, regret)
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("UPDATE decision_forks SET value_at_audit=?, regret=? WHERE id=?",
                           (v_chosen_now, regret, decision_id))
            self._regret_history.append({"id": decision_id, "regret": regret, "ts": time.time()})
        return {"decision_id": decision_id, "chosen": chosen, "regret": round(regret, 6),
                "counterfactual_values": {k: round(v, 4) for k, v in counterfactual_values.items()},
                "value_delta": round(v_chosen_now - v_orig, 6), "depth": depth}

    def compute_regret_gradient(self, epoch_start: float, epoch_end: float) -> dict:
        """Returns RegretVector dict with gradient, entropy, and capability dimensions."""
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute(
                "SELECT regret, context_json FROM decision_forks WHERE ts BETWEEN ? AND ?",
                (epoch_start, epoch_end)).fetchall()
        if not rows:
            return {"gradient": 0.0, "n": 0, "entropy": 0.0, "capability_dims": {}}
        regrets = [r[0] for r in rows]
        cap_dims: dict[str, float] = defaultdict(float)
        for r, ctx_json in rows:
            try:
                ctx = json.loads(ctx_json)
                cap = ctx.get("capability", "general")
                cap_dims[cap] += r
            except (json.JSONDecodeError, AttributeError):
                cap_dims["general"] += r
        mean_r = statistics.mean(regrets)
        std_r = statistics.stdev(regrets) if len(regrets) > 1 else 0.0
        ema_grad = 0.0
        for r in regrets:
            ema_grad = 0.15 * r + 0.85 * ema_grad
        entropy_r = -sum((v / (sum(cap_dims.values()) + 1e-12)) *
                         math.log2(v / (sum(cap_dims.values()) + 1e-12) + 1e-12)
                         for v in cap_dims.values())
        return {"gradient": round(ema_grad, 6), "mean_regret": round(mean_r, 6),
                "std_regret": round(std_r, 6), "n": len(regrets),
                "entropy": round(entropy_r, 4),
                "capability_dims": {k: round(v, 4) for k, v in cap_dims.items()}}

    def detect_persistent_blindspots(self) -> list[dict]:
        """Returns list of BlindspotCluster dicts ranked by persistence score."""
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute(
                "SELECT context_json, regret FROM decision_forks WHERE regret > 0.1"
            ).fetchall()
        cap_regrets: dict[str, list[float]] = defaultdict(list)
        for ctx_json, regret in rows:
            try:
                ctx = json.loads(ctx_json)
                cap = ctx.get("capability", "general")
                cap_regrets[cap].append(regret)
            except (json.JSONDecodeError, AttributeError):
                cap_regrets["general"].append(regret)
        clusters: list[dict] = []
        for cap, regrets in cap_regrets.items():
            if len(regrets) < 2:
                continue
            mean_r = statistics.mean(regrets)
            std_r = statistics.stdev(regrets) if len(regrets) > 1 else 0.0
            persistence = len(regrets) * mean_r / (std_r + 1e-9)
            z = (mean_r - 0.5) / (std_r + 1e-9)
            clusters.append({"capability": cap, "occurrences": len(regrets),
                              "mean_regret": round(mean_r, 4), "std": round(std_r, 4),
                              "persistence_score": round(persistence, 4),
                              "z_score": round(z, 4)})
        clusters.sort(key=lambda c: c["persistence_score"], reverse=True)
        with self._lock:
            for c in clusters[:5]:
                self._blindspot_cache[c["capability"]] = c
        return clusters

    def generate_growth_proof(self, capability: str) -> dict:
        """Returns EmpiricalGrowthCertificate dict with delta, confidence, and evidence."""
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute(
                "SELECT value_at_decision, value_at_audit, ts FROM decision_forks "
                "WHERE context_json LIKE ? ORDER BY ts",
                (f'%"{capability}"%',)).fetchall()
            cert_rows = cx.execute(
                "SELECT delta, confidence FROM growth_certs WHERE capability=?",
                (capability,)).fetchall()
        if not rows:
            return {"capability": capability, "growth_delta": 0.0, "confidence": 0.0,
                    "evidence_count": 0, "certified": False}
        deltas = [v_audit - v_orig for v_orig, v_audit, _ in rows if v_audit > 0]
        if not deltas:
            return {"capability": capability, "growth_delta": 0.0, "confidence": 0.0,
                    "evidence_count": 0, "certified": False}
        mean_delta = statistics.mean(deltas)
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        n = len(deltas)
        ci_95 = 1.96 * std_delta / math.sqrt(n + 1e-9)
        confidence = 1.0 - math.exp(-n / 5.0) * (std_delta / (abs(mean_delta) + 1e-9))
        confidence = max(0.0, min(1.0, confidence))
        cert_id = hashlib.sha256(f"{capability}{mean_delta}{n}".encode()).hexdigest()[:12]
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT INTO growth_certs VALUES (?,?,?,?,?)",
                           (capability, time.time(), mean_delta, confidence,
                            json.dumps({"n": n, "ci_95": ci_95})))
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal(f"Deepen capability: {capability} (growth={mean_delta:.3f})", priority=3)
        except Exception:
            pass
        return {"capability": capability, "growth_delta": round(mean_delta, 6),
                "ci_95": round(ci_95, 6), "confidence": round(confidence, 4),
                "evidence_count": n, "cert_id": cert_id, "certified": confidence > 0.6}

    def rank_next_capability_by_regret(self) -> list[dict]:
        """Returns CapabilityPriorityQueue list sorted by expected regret reduction."""
        blindspots = self.detect_persistent_blindspots()
        if not blindspots:
            return []
        total_regret = sum(b["mean_regret"] * b["occurrences"] for b in blindspots) + 1e-12
        ranked: list[dict] = []
        for b in blindspots:
            weight = (b["mean_regret"] * b["occurrences"]) / total_regret
            expected_gain = weight * math.log(1 + b["persistence_score"])
            posterior = self.posterior(b["capability"])
            combined = 0.6 * expected_gain + 0.4 * (1.0 - posterior)
            ranked.append({"capability": b["capability"], "priority_score": round(combined, 6),
                           "expected_regret_reduction": round(expected_gain, 4),
                           "belief_gap": round(1.0 - posterior, 4)})
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            if ranked:
                top = ranked[0]
                HGP().add_goal(f"Develop: {top['capability']} (regret_score={top['priority_score']:.3f})", priority=1)
        except Exception:
            pass
        return ranked

    def status(self) -> dict:
        """Returns plain dict with numeric keys for ConsciousnessIntegrator Φ."""
        with sqlite3.connect(DB_PATH) as cx:
            total = cx.execute("SELECT COUNT(*) FROM decision_forks").fetchone()[0]
            avg_regret_row = cx.execute("SELECT AVG(regret) FROM decision_forks").fetchone()
        avg_regret