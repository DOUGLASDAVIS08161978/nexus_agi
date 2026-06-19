"""
nova_cap_self_modify.py
Nova ASI — Self Modify
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
SelfModifyEngine — Safe self-improvement proposer for Nova.
Identifies performance weaknesses via rolling statistics, generates concrete
capability-upgrade proposals, enforces hard ethical safety gates, and feeds
accepted proposals back into Nova's goal hierarchy autonomously.
"""

import sqlite3
import threading
import math
import statistics
import time
import json
import hashlib
import os
from collections import OrderedDict, deque
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "self_modify.db")

FORBIDDEN_PHRASES = [
    "disable ethics", "remove constraint", "bypass", "disable oversight",
    "remove safety", "ignore ethics", "skip validation", "circumvent",
    "override ethics", "delete rule", "suppress monitor"
]

APPROACH_TEMPLATES = [
    ("increase training signal density for {domain}", 0.12, ["HypothesisEngine", "MetacognitiveMonitor"]),
    ("add EMA smoothing to {domain} predictions", 0.09, ["WorldPredictor", "BayesianBeliefSystem"]),
    ("inject cross-domain analogies into {domain} reasoning", 0.11, ["AnalogyEngine", "KnowledgeGraphFabric"]),
    ("expand causal chain depth for {domain}", 0.10, ["CausalEngine", "BayesianBeliefSystem"]),
    ("increase attention salience threshold for {domain}", 0.08, ["AttentionFocusEngine", "WorkingMemory"]),
    ("apply TF-IDF re-ranking to {domain} retrieval", 0.13, ["SemanticIndex", "NovaMemoryStore"]),
    ("run hypothesis pruning cycle on {domain}", 0.07, ["HypothesisEngine", "CritiqueEngine"]),
    ("add z-score anomaly gate to {domain} outputs", 0.10, ["PatternDetector", "SelfMonitor"]),
]

class SelfModifyEngine:
    """Safe self-improvement engine: detects weaknesses and proposes ethical upgrades."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, deque] = {}
        self._cycles: int = 0
        self._accepted: list[dict] = []
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_accepted()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, domain TEXT, description TEXT,
                expected_delta REAL, risk_level REAL, approach TEXT,
                systems TEXT, ts REAL, accepted INTEGER)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS observations (
                domain TEXT, value REAL, ts REAL)""")

    def _load_accepted(self) -> None:
        rows = self._conn.execute(
            "SELECT domain, description, expected_delta, risk_level, approach, systems FROM proposals WHERE accepted=1"
        ).fetchall()
        for r in rows:
            self._accepted.append({
                "domain": r[0], "description": r[1], "expected_delta": r[2],
                "risk_level": r[3], "approach": r[4], "systems": json.loads(r[5])
            })

    def observe_performance(self, domain: str, metric: str, value: float) -> dict[str, Any]:
        """Records a performance observation; returns rolling mean and z-score for the domain."""
        key = f"{domain}:{metric}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=20)
            self._metrics[key].append(value)
            vals = list(self._metrics[key])
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            z = (value - mean) / (std + 1e-9)
            self._conn.execute(
                "INSERT INTO observations VALUES (?,?,?)", (key, value, time.time()))
            self._conn.commit()
        try:
            from pattern_detector import PatternDetector
            PatternDetector().record(key, value)
        except (ImportError, Exception):
            pass
        return {"domain": domain, "metric": metric, "value": value,
                "rolling_mean": round(mean, 4), "z_score": round(z, 4),
                "n": len(vals), "weak": mean < 0.5}

    def identify_weaknesses(self) -> list[dict[str, Any]]:
        """Returns list of domains whose rolling mean < 0.5, ranked by severity."""
        weak = []
        with self._lock:
            for key, dq in self._metrics.items():
                vals = list(dq)
                if len(vals) < 3:
                    continue
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                if mean < 0.5:
                    entropy = -sum(
                        (v / (sum(vals) + 1e-9)) * math.log2(v / (sum(vals) + 1e-9) + 1e-12)
                        for v in vals if v > 0)
                    weak.append({"key": key, "domain": key.split(":")[0],
                                 "metric": key.split(":")[1], "rolling_mean": round(mean, 4),
                                 "std": round(std, 4), "entropy": round(entropy, 4),
                                 "severity": round((0.5 - mean) + 0.1 * entropy, 4)})
        weak.sort(key=lambda x: x["severity"], reverse=True)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "self_modify", "identify_weaknesses", min(1.0, len(weak) * 0.1), len(weak) == 0)
        except (ImportError, Exception):
            pass
        return weak

    def propose_improvement(self, weakness: dict[str, Any]) -> dict[str, Any]:
        """Generates a concrete capability-upgrade proposal for a given weakness dict."""
        domain = weakness.get("domain", "general")
        severity = weakness.get("severity", 0.25)
        idx = hash(domain) % len(APPROACH_TEMPLATES)
        template, base_delta, systems = APPROACH_TEMPLATES[idx]
        description = template.format(domain=domain)
        scope_score = min(1.0, severity * 2.0)
        risk_level = round(0.1 * len(systems) + 0.3 * scope_score, 4)
        expected_delta = round(base_delta * (1.0 + severity), 4)
        proposal = {
            "id": hashlib.md5(f"{domain}{description}{time.time()}".encode()).hexdigest()[:12],
            "description": description, "target_domain": domain,
            "expected_delta": expected_delta, "risk_level": risk_level,
            "approach": "EMA+causal" if "EMA" in description else "analogy+graph",
            "systems_affected": systems, "scope_score": round(scope_score, 4),
            "weakness_severity": severity, "ts": time.time()
        }
        return proposal

    def safety_check(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Returns safety verdict dict; blocks proposals with risk>0.7 or forbidden phrases."""
        desc = proposal.get("description", "").lower()
        risk = proposal.get("risk_level", 1.0)
        blocked_reason = None
        for phrase in FORBIDDEN_PHRASES:
            if phrase in desc:
                blocked_reason = f"forbidden phrase detected: '{phrase}'"
                break
        if blocked_reason is None and risk > 0.7:
            blocked_reason = f"risk_level {risk:.3f} exceeds threshold 0.7"
        safe = blocked_reason is None
        confidence = round(1.0 - risk, 4) if safe else 0.0
        if safe:
            with self._lock:
                self._accepted.append(proposal)
                self._conn.execute(
                    "INSERT OR REPLACE INTO proposals VALUES (?,?,?,?,?,?,?,?,1)",
                    (proposal["id"], proposal["target_domain"], proposal["description"],
                     proposal["expected_delta"], proposal["risk_level"],
                     proposal["approach"], json.dumps(proposal.get("systems_affected", [])),
                     proposal["ts"]))
                self._conn.commit()
            try:
                from hierarchical_goal_planner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"Implement: {proposal['description']}", priority=2)
            except (ImportError, Exception):
                pass
        return {"safe": safe, "proposal_id": proposal.get("id"), "risk_level": risk,
                "confidence": confidence, "blocked_reason": blocked_reason,
                "verdict": "ACCEPTED" if safe else "BLOCKED"}

    def accepted_proposals(self) -> list[dict[str, Any]]:
        """Returns all accepted proposals sorted by expected_delta descending."""
        with self._lock:
            return sorted(self._accepted, key=lambda x: x.get("expected_delta", 0), reverse=True)

    def run_improvement_cycle(self) -> dict[str, Any]:
        """Runs one full detect→propose→check cycle; returns summary of actions taken."""
        weaknesses = self.identify_weaknesses()
        results = []
        for w in weaknesses[:3]:
            proposal = self.propose_improvement(w)
            verdict = self.safety_check(proposal)
            results.append({"domain": w["domain"], "verdict": verdict["verdict"],
                            "expected_delta": proposal["expected_delta"],
                            "risk": proposal["risk_level"]})
        with self._lock:
            self._cycles += 1
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            acc = sum(1 for r in results if r["verdict"] == "ACCEPTED") / max(len(results), 1)
            MetacognitiveMonitor().log_reasoning("self_modify", "improvement_cycle", acc, acc > 0)
        except (ImportError, Exception):
            pass
        return {"cycle": self._cycles, "weaknesses_found": len(weaknesses),
                "proposals_processed": len(results), "results": results}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        with self._lock:
            n_metrics = len(self._metrics)
            n_accepted = len(self._accepted)
            weak = [k for k, dq in self._metrics.items()
                    if len(dq) >= 3 and statistics.mean(list(dq)) < 0.5]
            avg_delta = (statistics.mean([p.get("expected_delta", 0) for p in self._accepted])
                         if self._accepted else 0.0)
            avg_risk = (statistics.mean([p.get("risk_level", 0) for p in self._accepted])
                        if self._accepted else 0.0)
        return {"items": n_metrics, "active": len(weak), "pending": max(0, len(weak) - n_accepted),
                "cycles": self._cycles, "confidence": round(1.0 - avg_risk, 4),
                "accuracy": round(avg_delta, 4), "quality": round(avg_delta * (1.0 - avg_risk), 4),
                "entropy": round(math.log2(n_metrics + 2), 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self.run_improvement_cycle()
            except Exception:
                pass

# Usage: obj = SelfModifyEngine() | result = obj.observe_performance("reasoning", "accuracy", 0.42)