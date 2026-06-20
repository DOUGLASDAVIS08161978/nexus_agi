"""
nova_cap_self_modify.py
Nova ASI — Self Modify
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
SelfModificationEngine — Nova's safe self-improvement proposer.
Detects performance weaknesses via rolling statistics, generates concrete
capability upgrade proposals, enforces hard ethical safety gates, and
autonomously drives improvement cycles via a background daemon thread.
"""

import sqlite3
import threading
import time
import math
import statistics
import hashlib
import json
import os
from collections import OrderedDict, deque
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "self_modification.db")

FORBIDDEN_PHRASES = ["disable ethics", "remove constraint", "bypass", "override safety",
                     "delete oversight", "suppress monitor", "ignore ethics"]

SCOPE_SCORES = {"minor": 0.1, "moderate": 0.4, "major": 0.7, "systemic": 1.0}

class SelfModificationEngine:
    """Safe self-improvement proposer: detects weaknesses, proposes upgrades, enforces ethical gates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, deque] = {}
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._cycles: int = 0
        self._last_weakness_scan: float = 0.0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, domain TEXT, description TEXT,
                expected_delta REAL, risk_level REAL, approach TEXT,
                safe INTEGER, accepted INTEGER, ts REAL)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS perf_log (
                domain TEXT, metric TEXT, value REAL, ts REAL)""")

    def observe_performance(self, domain: str, metric: str, value: float) -> dict[str, Any]:
        """Records a performance observation; returns rolling stats for the domain."""
        key = f"{domain}::{metric}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=20)
            self._metrics[key].append(value)
            try:
                self._conn.execute("INSERT INTO perf_log VALUES (?,?,?,?)",
                                   (domain, metric, value, time.time()))
                self._conn.commit()
            except sqlite3.Error:
                pass
            window = list(self._metrics[key])
            mean = statistics.mean(window) if window else 0.0
            std = statistics.stdev(window) if len(window) > 1 else 0.0
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "observe_performance", mean, mean >= 0.5)
        except Exception:
            pass
        return {"domain": domain, "metric": metric, "rolling_mean": round(mean, 4),
                "std": round(std, 4), "n": len(window), "weak": mean < 0.5}

    def identify_weaknesses(self) -> list[dict[str, Any]]:
        """Returns list of domains whose rolling mean metric falls below 0.5 threshold."""
        weaknesses = []
        with self._lock:
            for key, window in self._metrics.items():
                if len(window) < 3:
                    continue
                vals = list(window)
                mean = statistics.mean(vals)
                if mean < 0.5:
                    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    z_latest = (vals[-1] - mean) / (std + 1e-9)
                    domain, metric = key.split("::", 1)
                    entropy = -sum((1/len(vals)) * math.log2(1/len(vals) + 1e-12)
                                   for _ in vals)
                    weaknesses.append({"domain": domain, "metric": metric,
                                       "rolling_mean": round(mean, 4),
                                       "std": round(std, 4),
                                       "z_latest": round(z_latest, 4),
                                       "entropy": round(entropy, 4),
                                       "severity": round(0.5 - mean, 4)})
        weaknesses.sort(key=lambda w: w["severity"], reverse=True)
        self._last_weakness_scan = time.time()
        return weaknesses

    def propose_improvement(self, weakness: dict[str, Any]) -> dict[str, Any]:
        """Generates a concrete capability upgrade proposal for a given weakness dict."""
        domain = weakness.get("domain", "unknown")
        severity = weakness.get("severity", 0.1)
        metric = weakness.get("metric", "accuracy")
        scope = "major" if severity > 0.3 else ("moderate" if severity > 0.15 else "minor")
        systems_affected = max(1, int(severity * 10))
        risk_level = round(0.1 * systems_affected + 0.3 * SCOPE_SCORES.get(scope, 0.4), 4)
        risk_level = min(risk_level, 1.0)
        approach_map = {
            "accuracy": "Apply EMA-weighted Bayesian update with calibration correction",
            "confidence": "Introduce entropy-regularised posterior normalisation",
            "recall": "Expand TF-IDF semantic index with cosine similarity retrieval",
            "speed": "Cache top-k results via LRU with salience-decay eviction",
            "coherence": "Run causal chain audit; prune edges with confidence < 0.3",
        }
        approach = approach_map.get(metric, f"Reinforce {domain} pipeline with online learning loop")
        expected_delta = round(min(severity * 1.4, 0.45), 4)
        description = (f"Enhance {domain} {metric} by {expected_delta:.2f} units via: {approach}. "
                       f"Scope={scope}, systems_affected={systems_affected}.")
        uid = hashlib.sha1(f"{domain}{metric}{time.time()}".encode()).hexdigest()[:12]
        proposal = {"id": uid, "domain": domain, "metric": metric,
                    "description": description, "target_domain": domain,
                    "expected_delta": expected_delta, "risk_level": risk_level,
                    "approach": approach, "scope": scope, "safe": None, "ts": time.time()}
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Improve {domain} {metric} by {expected_delta:.2f}", priority=int(severity * 10))
        except Exception:
            pass
        return proposal

    def safety_check(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Validates proposal against ethical gates; returns verdict dict with reason."""
        risk = proposal.get("risk_level", 1.0)
        text = (proposal.get("description", "") + proposal.get("approach", "")).lower()
        blocked_phrase = next((p for p in FORBIDDEN_PHRASES if p in text), None)
        if blocked_phrase:
            verdict = {"safe": False, "reason": f"Forbidden phrase detected: '{blocked_phrase}'",
                       "risk_level": risk}
        elif risk > 0.7:
            verdict = {"safe": False, "reason": f"Risk level {risk:.3f} exceeds threshold 0.7",
                       "risk_level": risk}
        else:
            ci_low = max(0.0, proposal.get("expected_delta", 0) - 0.05)
            ci_high = min(1.0, proposal.get("expected_delta", 0) + 0.05)
            verdict = {"safe": True, "reason": "Passed all ethical and risk gates",
                       "risk_level": risk, "expected_delta_ci": [ci_low, ci_high]}
        proposal["safe"] = verdict["safe"]
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO proposals VALUES (?,?,?,?,?,?,?,?,?)",
                (proposal["id"], proposal.get("domain"), proposal.get("description"),
                 proposal.get("expected_delta"), risk, proposal.get("approach"),
                 int(verdict["safe"]), 0, proposal.get("ts", time.time())))
            self._conn.commit()
        except sqlite3.Error:
            pass
        return {**proposal, **verdict}

    def accept_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Marks a safe proposal as accepted; returns updated record or error."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT safe FROM proposals WHERE id=?", (proposal_id,)).fetchone()
                if not row:
                    return {"error": "proposal not found", "id": proposal_id}
                if not row[0]:
                    return {"error": "cannot accept unsafe proposal", "id": proposal_id}
                self._conn.execute("UPDATE proposals SET accepted=1 WHERE id=?", (proposal_id,))
                self._conn.commit()
            return {"accepted": True, "id": proposal_id}
        except sqlite3.Error as e:
            return {"error": str(e), "id": proposal_id}

    def accepted_proposals(self) -> list[dict[str, Any]]:
        """Returns all accepted safe proposals ordered by expected_delta descending."""
        try:
            rows = self._conn.execute(
                "SELECT id,domain,description,expected_delta,risk_level,approach,ts "
                "FROM proposals WHERE accepted=1 ORDER BY expected_delta DESC").fetchall()
            return [{"id": r[0], "domain": r[1], "description": r[2],
                     "expected_delta": r[3], "risk_level": r[4],
                     "approach": r[5], "ts": r[6]} for r in rows]
        except sqlite3.Error:
            return []

    def status(self) -> dict[str, Any]:
        """Returns numeric health dict compatible with ConsciousnessIntegrator Φ scoring."""
        with self._lock:
            total_obs = sum(len(v) for v in self._metrics.values())
            weak_count = sum(1 for v in self._metrics.values()
                             if len(v) >= 3 and statistics.mean(v) < 0.5)
        try:
            total_proposals = self._conn.execute(
                "SELECT COUNT(*) FROM proposals").fetchone()[0]
            safe_count = self._conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE safe=1").fetchone()[0]
            accepted = self._conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE accepted=1").fetchone()[0]
            avg_delta = self._conn.execute(
                "SELECT AVG(expected_delta) FROM proposals WHERE safe=1").fetchone()[0] or 0.0
        except sqlite3.Error:
            total_proposals = safe_count = accepted = 0
            avg_delta = 0.0
        confidence = round(safe_count / (total_proposals + 1e-9), 4)
        return {"items": total_obs, "active": weak_count, "pending": total_proposals,
                "cycles": self._cycles, "confidence": confidence,
                "accepted": accepted, "quality": round(avg_delta, 4),
                "entropy": round(math.log2(weak_count + 2), 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            try:
                weaknesses = self.identify_weaknesses()
                for w in weaknesses[:3]:
                    proposal = self.propose_improvement(w)
                    checked = self.safety_check(proposal)
                    if checked.get("safe"):
                        self.accept_proposal(checked["id"])
                with self._lock:
                    self._cycles += 1
                try:
                    from metacognitive_monitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "self_modification", "auto_cycle",
                        self.status()["confidence"], True)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = SelfModificationEngine() | result = obj.observe_performance("reasoning", "accuracy", 0.42)
