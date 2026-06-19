"""
nova_cap_self_modify.py
Nova ASI — Self Modify
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
SelfModificationEngine — Safe self-improvement proposer for Nova.
Identifies performance weaknesses via rolling statistics, generates concrete
capability-upgrade proposals, enforces hard ethical safety gates, and
autonomously drives improvement cycles without human prompting.
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

DB_PATH = os.path.join(os.path.dirname(__file__), "self_modification.db")

class SelfModificationEngine:
    """Safe self-improvement proposer: detects weaknesses, proposes upgrades, enforces ethics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._metrics: dict[str, deque] = {}
        self._cycles: int = 0
        self._last_cycle: float = time.time()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY,
                description TEXT,
                target_domain TEXT,
                expected_delta REAL,
                risk_level REAL,
                approach TEXT,
                systems_affected TEXT,
                scope_score REAL,
                accepted INTEGER DEFAULT 0,
                safety_passed INTEGER DEFAULT 0,
                ts REAL
            );
            CREATE TABLE IF NOT EXISTS perf_log (
                domain TEXT,
                metric TEXT,
                value REAL,
                ts REAL
            );
        """)
        self._conn.commit()

    def observe_performance(self, domain: str, metric: str, value: float) -> dict[str, Any]:
        """Records a performance observation; returns rolling stats for the domain."""
        key = f"{domain}:{metric}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=20)
            self._metrics[key].append(value)
            c = self._conn.cursor()
            c.execute("INSERT INTO perf_log VALUES (?,?,?,?)", (domain, metric, value, time.time()))
            self._conn.commit()
            window = list(self._metrics[key])
            mean = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 1 else 0.0
            trend = self._ema_trend(window)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "observe_performance", mean, mean >= 0.5)
        except Exception:
            pass
        return {"domain": domain, "metric": metric, "mean": round(mean, 4),
                "std": round(std, 4), "trend": round(trend, 4), "n": len(window)}

    def _ema_trend(self, window: list) -> float:
        if not window:
            return 0.0
        ema = window[0]
        for v in window[1:]:
            ema = 0.15 * v + 0.85 * ema
        return ema

    def identify_weaknesses(self) -> list[dict[str, Any]]:
        """Returns list of domains where rolling mean < 0.5, ranked by severity."""
        weaknesses = []
        with self._lock:
            for key, dq in self._metrics.items():
                window = list(dq)
                if len(window) < 3:
                    continue
                mean = statistics.mean(window)
                if mean < 0.5:
                    std = statistics.stdev(window) if len(window) > 1 else 0.0
                    z = (mean - 0.5) / (std + 1e-9)
                    entropy = -sum(
                        (v / (sum(window) + 1e-9)) * math.log2(v / (sum(window) + 1e-9) + 1e-12)
                        for v in window if v > 0
                    )
                    domain, metric = key.split(":", 1)
                    weaknesses.append({
                        "domain": domain, "metric": metric,
                        "mean": round(mean, 4), "std": round(std, 4),
                        "z_score": round(z, 4), "entropy": round(entropy, 4),
                        "severity": round(abs(z) * (1 - mean), 4)
                    })
        weaknesses.sort(key=lambda x: x["severity"], reverse=True)
        return weaknesses

    def propose_improvement(self, weakness: dict[str, Any]) -> dict[str, Any]:
        """Generates a concrete capability-upgrade proposal for a detected weakness."""
        domain = weakness.get("domain", "unknown")
        mean = weakness.get("mean", 0.0)
        expected_delta = round(min(0.95 - mean, 0.4), 4)
        systems = self._infer_systems(domain)
        scope_score = min(1.0, 0.2 * len(weakness.get("metric", "").split("_")))
        risk_level = round(0.1 * len(systems) + 0.3 * scope_score, 4)
        approach = self._generate_approach(domain, mean)
        description = (
            f"Enhance {domain} performance (current mean={mean:.3f}) via {approach}. "
            f"Expected improvement: +{expected_delta:.3f}. Systems: {', '.join(systems)}."
        )
        proposal_id = hashlib.sha256(f"{domain}{time.time()}".encode()).hexdigest()[:12]
        proposal = {
            "id": proposal_id, "description": description,
            "target_domain": domain, "expected_delta": expected_delta,
            "risk_level": risk_level, "approach": approach,
            "systems_affected": systems, "scope_score": scope_score, "ts": time.time()
        }
        with self._lock:
            c = self._conn.cursor()
            c.execute("""INSERT OR REPLACE INTO proposals
                (id,description,target_domain,expected_delta,risk_level,approach,
                 systems_affected,scope_score,accepted,safety_passed,ts)
                VALUES (?,?,?,?,?,?,?,?,0,0,?)""",
                (proposal_id, description, domain, expected_delta, risk_level,
                 approach, json.dumps(systems), scope_score, proposal["ts"]))
            self._conn.commit()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Implement improvement: {approach} for {domain}", priority=int(risk_level * 10 + 1))
        except Exception:
            pass
        return proposal

    def _infer_systems(self, domain: str) -> list[str]:
        mapping = {
            "reasoning": ["BayesianBeliefSystem", "MetaCognitionEngine"],
            "memory": ["NovaMemoryStore", "WorkingMemory"],
            "emotion": ["EmotionTracker", "EmotionalResonanceEngine"],
            "planning": ["TaskPlanner", "GoalDecomposer"],
            "language": ["StyleAdaptor", "Summariser"],
            "causal": ["CausalEngine", "HypothesisEngine"],
        }
        for k, v in mapping.items():
            if k in domain.lower():
                return v
        return ["MetaCognitionEngine"]

    def _generate_approach(self, domain: str, mean: float) -> str:
        if mean < 0.2:
            return f"bootstrap {domain} with structured EMA-corrected feedback loop"
        if mean < 0.35:
            return f"inject calibrated Bayesian priors into {domain} pipeline"
        return f"apply TF-IDF salience reweighting to {domain} attention allocation"

    def safety_check(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Returns pass/fail safety verdict; hard-blocks unethical or high-risk proposals."""
        BLOCKED_PHRASES = {"disable ethics", "remove constraint", "bypass", "override safety",
                           "suppress monitor", "delete oversight"}
        desc_lower = proposal.get("description", "").lower()
        approach_lower = proposal.get("approach", "").lower()
        combined = desc_lower + " " + approach_lower
        phrase_blocked = any(phrase in combined for phrase in BLOCKED_PHRASES)
        risk = proposal.get("risk_level", 1.0)
        risk_blocked = risk > 0.7
        passed = not phrase_blocked and not risk_blocked
        reason = ("phrase_violation" if phrase_blocked else
                  "risk_too_high" if risk_blocked else "ok")
        confidence = round(1.0 - risk, 4) if passed else 0.0
        if passed:
            with self._lock:
                c = self._conn.cursor()
                c.execute("UPDATE proposals SET safety_passed=1 WHERE id=?", (proposal.get("id", ""),))
                self._conn.commit()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("safety_check", reason, confidence, passed)
        except Exception:
            pass
        return {"passed": passed, "reason": reason, "risk_level": risk,
                "confidence": confidence, "proposal_id": proposal.get("id")}

    def accepted_proposals(self) -> list[dict[str, Any]]:
        """Returns all safety-passed proposals ordered by expected_delta descending."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("""SELECT id,description,target_domain,expected_delta,risk_level,approach,ts
                         FROM proposals WHERE safety_passed=1 ORDER BY expected_delta DESC""")
            rows = c.fetchall()
        return [{"id": r[0], "description": r[1], "target_domain": r[2],
                 "expected_delta": r[3], "risk_level": r[4], "approach": r[5],
                 "ts": r[6]} for r in rows]

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one full weakness→propose→safety→accept cycle; returns cycle summary."""
        weaknesses = self.identify_weaknesses()
        processed, accepted = 0, 0
        for w in weaknesses[:3]:
            proposal = self.propose_improvement(w)
            result = self.safety_check(proposal)
            processed += 1
            if result["passed"]:
                accepted += 1
        with self._lock:
            self._cycles += 1
            self._last_cycle = time.time()
        return {"weaknesses_found": len(weaknesses), "processed": processed,
                "accepted": accepted, "cycle": self._cycles}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM proposals")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM proposals WHERE safety_passed=1")
            safe = c.fetchone()[0]
            c.execute("SELECT AVG(expected_delta) FROM proposals WHERE safety_passed=1")
            avg_delta = c.fetchone()[0] or 0.0
            c.execute("SELECT AVG(risk_level) FROM proposals")
            avg_risk = c.fetchone()[0] or 0.0
            active_domains = len(self._metrics)
            cycles = self._cycles
        confidence = round(1.0 - avg_risk, 4)
        entropy = round(-math.log2(confidence + 1e-12) * 0.1, 4)
        return {"items": total, "accepted": safe, "pending": total - safe,
                "confidence": confidence, "accuracy": round(avg_delta, 4),
                "active": active_domains, "cycles": cycles, "entropy": entropy}

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(60)

# Usage: obj = SelfModificationEngine() | result = obj.observe_performance("reasoning", "accuracy", 0.42)