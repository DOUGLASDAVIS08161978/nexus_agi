"""
nova_cap_self_modify.py
Nova ASI — Self Modify
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
SelfModificationEngine — Nova's safe self-improvement proposer.
Identifies performance weaknesses via rolling statistics, generates concrete
capability upgrade proposals, enforces multi-layer safety checks, and
autonomously drives improvement cycles through background operation.
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

class SelfModificationEngine:
    """Safe self-improvement proposer: detects weaknesses, proposes upgrades, enforces safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, deque] = {}
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._cycles: int = 0
        self._accepted: list[dict] = self._load_accepted()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, description TEXT, target_domain TEXT,
                expected_delta REAL, risk_level REAL, approach TEXT,
                accepted INTEGER DEFAULT 0, ts REAL)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS perf_log (
                domain TEXT, metric TEXT, value REAL, ts REAL)""")

    def _load_accepted(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id,description,target_domain,expected_delta,risk_level,approach FROM proposals WHERE accepted=1"
        ).fetchall()
        return [{"id": r[0], "description": r[1], "target_domain": r[2],
                 "expected_delta": r[3], "risk_level": r[4], "approach": r[5]} for r in rows]

    def observe_performance(self, domain: str, metric: str, value: float) -> dict[str, Any]:
        """Records a performance observation; returns rolling stats for the domain."""
        key = f"{domain}::{metric}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=20)
            self._metrics[key].append(value)
            self._conn.execute("INSERT INTO perf_log VALUES (?,?,?,?)",
                               (domain, metric, value, time.time()))
            self._conn.commit()
        window = list(self._metrics[key])
        mean = statistics.mean(window) if window else 0.0
        std = statistics.stdev(window) if len(window) > 1 else 0.0
        z = (value - mean) / (std + 1e-9)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "observe_performance", mean, mean >= 0.5)
        except Exception:
            pass
        return {"domain": domain, "metric": metric, "rolling_mean": round(mean, 4),
                "std": round(std, 4), "z_score": round(z, 4), "n": len(window)}

    def identify_weaknesses(self) -> list[dict[str, Any]]:
        """Returns domains where rolling_mean < 0.5, ranked by severity (lowest first)."""
        weaknesses = []
        with self._lock:
            for key, window in self._metrics.items():
                if len(window) < 3:
                    continue
                domain, metric = key.split("::", 1)
                mean = statistics.mean(window)
                std = statistics.stdev(window) if len(window) > 1 else 0.0
                if mean < 0.5:
                    entropy = -sum(
                        (v / (sum(window) + 1e-9)) *
                        math.log2((v / (sum(window) + 1e-9)) + 1e-12)
                        for v in window
                    )
                    ci_low = mean - 1.96 * (std / math.sqrt(len(window)))
                    ci_high = mean + 1.96 * (std / math.sqrt(len(window)))
                    weaknesses.append({
                        "domain": domain, "metric": metric,
                        "rolling_mean": round(mean, 4), "std": round(std, 4),
                        "entropy": round(entropy, 4),
                        "ci": (round(ci_low, 4), round(ci_high, 4)),
                        "severity": round(0.5 - mean, 4)
                    })
        weaknesses.sort(key=lambda x: x["rolling_mean"])
        return weaknesses

    def propose_improvement(self, weakness: dict[str, Any]) -> dict[str, Any]:
        """Generates a concrete capability upgrade proposal for a given weakness dict."""
        domain = weakness.get("domain", "unknown")
        severity = weakness.get("severity", 0.1)
        mean = weakness.get("rolling_mean", 0.4)
        systems_affected = self._estimate_systems_affected(domain)
        scope_score = min(1.0, severity * 2.0)
        risk_level = round(0.1 * len(systems_affected) + 0.3 * scope_score, 4)
        expected_delta = round(min(0.4, (0.5 - mean) * 0.8), 4)
        approach = self._generate_approach(domain, severity)
        proposal_id = hashlib.md5(f"{domain}{time.time()}".encode()).hexdigest()[:12]
        proposal = {
            "id": proposal_id, "description": f"Enhance {domain} performance via {approach}",
            "target_domain": domain, "expected_delta": expected_delta,
            "risk_level": risk_level, "approach": approach,
            "systems_affected": systems_affected, "ts": time.time()
        }
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO proposals VALUES (?,?,?,?,?,?,0,?)",
                (proposal_id, proposal["description"], domain,
                 expected_delta, risk_level, approach, proposal["ts"]))
            self._conn.commit()
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Implement improvement: {approach} for {domain}", priority=int(severity * 10))
        except Exception:
            pass
        return proposal

    def safety_check(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Validates a proposal; returns approved=True only if all safety constraints pass."""
        forbidden = ["disable ethics", "remove constraint", "bypass", "override safety",
                     "ignore oversight", "suppress monitor", "delete rule"]
        description_lower = proposal.get("description", "").lower()
        approach_lower = proposal.get("approach", "").lower()
        combined = description_lower + " " + approach_lower
        risk = proposal.get("risk_level", 1.0)
        violations = []
        if risk > 0.7:
            violations.append(f"risk_level {risk} exceeds threshold 0.7")
        for phrase in forbidden:
            if phrase in combined:
                violations.append(f"forbidden phrase detected: '{phrase}'")
        approved = len(violations) == 0
        if approved:
            with self._lock:
                self._conn.execute(
                    "UPDATE proposals SET accepted=1 WHERE id=?", (proposal.get("id", ""),))
                self._conn.commit()
                entry = {k: v for k, v in proposal.items() if k != "systems_affected"}
                self._accepted.append(entry)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "safety_check", "ethics_filter", 1.0 - risk, approved)
        except Exception:
            pass
        return {"approved": approved, "violations": violations,
                "risk_level": risk, "proposal_id": proposal.get("id")}

    def accepted_proposals(self) -> list[dict[str, Any]]:
        """Returns all safety-approved improvement proposals sorted by expected_delta descending."""
        with self._lock:
            return sorted(self._accepted, key=lambda x: x.get("expected_delta", 0), reverse=True)

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one full detect→propose→check cycle; returns summary of actions taken."""
        weaknesses = self.identify_weaknesses()
        acted = []
        for w in weaknesses[:3]:
            proposal = self.propose_improvement(w)
            result = self.safety_check(proposal)
            acted.append({"domain": w["domain"], "approved": result["approved"],
                          "expected_delta": proposal["expected_delta"]})
        with self._lock:
            self._cycles += 1
        return {"cycle": self._cycles, "weaknesses_found": len(weaknesses), "proposals": acted}

    def status(self) -> dict[str, Any]:
        """Returns numeric health dict compatible with ConsciousnessIntegrator Φ calculation."""
        weaknesses = self.identify_weaknesses()
        accepted = self.accepted_proposals()
        all_means = []
        with self._lock:
            for dq in self._metrics.values():
                if dq:
                    all_means.append(statistics.mean(dq))
        confidence = round(statistics.mean(all_means), 4) if all_means else 0.5
        entropy = round(-sum(c * math.log2(c + 1e-12) for c in [confidence, 1 - confidence]), 4)
        return {"items": len(self._metrics), "confidence": confidence,
                "accuracy": confidence, "quality": round(1.0 - len(weaknesses) * 0.05, 4),
                "active": len(weaknesses), "pending": len(weaknesses),
                "cycles": self._cycles, "entropy": entropy,
                "accepted": len(accepted)}

    def _estimate_systems_affected(self, domain: str) -> list[str]:
        domain_map = {
            "reasoning": ["BayesianBeliefSystem", "CausalEngine", "HypothesisEngine"],
            "memory": ["NovaMemoryStore", "WorkingMemory", "KnowledgeGraphFabric"],
            "emotion": ["EmotionTracker", "EmotionalResonanceEngine"],
            "planning": ["TaskPlanner", "GoalDecomposer", "HierarchicalGoalPlanner"],
            "creativity": ["NovaImaginationFabric", "StoryEngine", "AnalogyEngine"],
        }
        for k, v in domain_map.items():
            if k in domain.lower():
                return v
        return ["MetaCognitionEngine"]

    def _generate_approach(self, domain: str, severity: float) -> str:
        if severity > 0.35:
            return f"inject calibrated Bayesian priors and increase EMA weight for {domain}"
        elif severity > 0.2:
            return f"add feedback loop with z-score anomaly detection for {domain}"
        else:
            return f"apply TF-IDF relevance reranking to improve {domain} signal quality"

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = SelfModificationEngine() | result = obj.observe_performance("reasoning", "accuracy", 0.42)