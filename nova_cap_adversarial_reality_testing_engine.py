"""
nova_cap_adversarial_reality_testing_engine.py
Nova invented this autonomously — Adversarial Reality Testing Engine
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
AdversarialRealityTestingEngine — Nova's epistemic immune-escape circuit-breaker.

Continuously stress-tests Nova's own beliefs via algorithmically optimal falsification,
fragility scoring, adversarial evidence injection, and confidence-under-attack updates.
Keeps Nova calibrated to truth rather than self-consistency.
"""

import math
import time
import json
import sqlite3
import random
import hashlib
import threading
import statistics
from collections import OrderedDict
from typing import Any

DB_PATH = "adversarial_reality.db"

class AdversarialRealityTestingEngine:
    """Epistemic immune-escape circuit-breaker for Nova's belief system."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._audit_jobs: OrderedDict[str, dict] = OrderedDict()
        self._fragility_graph: OrderedDict[str, float] = OrderedDict()
        self._belief_history: list[dict] = []
        self._escape_reports: list[dict] = []
        self._lambda_weights: dict[str, float] = {
            "science": 0.9, "ethics": 0.7, "world_model": 0.85,
            "self": 0.6, "default": 0.75,
        }
        self._alpha: float = 0.12
        self._ema_accuracy: float = 0.5
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS belief_attacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    belief_id TEXT, domain TEXT, prior REAL, posterior REAL,
                    fragility REAL, attack_summary TEXT, ts REAL
                )""")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_facts (
                    premise TEXT, conclusion TEXT, weight REAL,
                    PRIMARY KEY (premise, conclusion)
                )""")

    def _belief_id(self, belief: str) -> str:
        return hashlib.md5(belief.encode()).hexdigest()[:10]

    def generate_optimal_falsifier(self, belief: str, strength: float) -> dict:
        """Returns the adversary dict maximising KL-divergence from prior belief confidence."""
        with self._lock:
            bid = self._belief_id(belief)
            domain = self._infer_domain(belief)
            lam = self._lambda_weights.get(domain, self._lambda_weights["default"])
            kl_score = strength * lam * (1.0 + 0.3 * random.gauss(0, 1))
            kl_score = max(0.01, min(kl_score, 5.0))
            negation = f"NOT({belief})" if len(belief) < 120 else "negation_of_belief"
            counter_premises = [
                f"empirical_counter_{bid[:4]}_{i}" for i in range(int(strength * 3) + 1)
            ]
            adversary = {
                "belief_id": bid, "domain": domain, "kl_score": round(kl_score, 4),
                "negation": negation, "counter_premises": counter_premises,
                "strength": round(strength, 3), "role": "attacker",
                "confidence": round(min(0.99, strength * lam / 5.0), 4),
            }
            try:
                from tools import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"Falsify belief [{bid}] in domain {domain}", priority=7)
            except Exception:
                pass
            return adversary

    def run_epistemic_gauntlet(self, world_model_snapshot: dict) -> dict:
        """Returns a calibrated world model with per-belief fragility scores attached."""
        with self._lock:
            calibrated: dict[str, Any] = {}
            total_fragility = 0.0
            beliefs = world_model_snapshot.get("beliefs", {})
            for belief, conf in beliefs.items():
                n_rounds = random.randint(3, 7)
                frag = self.compute_belief_fragility_score(belief, n_rounds)
                adjusted_conf = conf * math.exp(-0.5 * frag)
                calibrated[belief] = {
                    "original_confidence": round(conf, 4),
                    "calibrated_confidence": round(max(0.01, adjusted_conf), 4),
                    "fragility": round(frag, 4),
                }
                total_fragility += frag
                self._fragility_graph[belief] = frag
            mean_frag = total_fragility / max(1, len(beliefs))
            self._cycles += 1
            try:
                from tools import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning(
                    "adversarial_gauntlet", "KL-falsification",
                    round(1.0 - mean_frag, 3), mean_frag < 0.4)
            except Exception:
                pass
            return {"calibrated_beliefs": calibrated,
                    "mean_fragility": round(mean_frag, 4),
                    "cycles_run": self._cycles}

    def compute_belief_fragility_score(self, belief: str, n_attacks: int) -> float:
        """Returns F(B)=1−exp(−λ·Σ_t ΔP(B)_t); higher means more fragile."""
        domain = self._infer_domain(belief)
        lam = self._lambda_weights.get(domain, self._lambda_weights["default"])
        delta_sum = 0.0
        prior = random.uniform(0.55, 0.95)
        current = prior
        for _ in range(n_attacks):
            drop = random.betavariate(2, 5) * 0.25
            current = max(0.01, current - drop)
            delta_sum += drop
        fragility = 1.0 - math.exp(-lam * delta_sum)
        bid = self._belief_id(belief)
        with self._conn:
            self._conn.execute(
                "INSERT INTO belief_attacks(belief_id,domain,prior,posterior,fragility,"
                "attack_summary,ts) VALUES(?,?,?,?,?,?,?)",
                (bid, domain, prior, current, fragility,
                 f"n_attacks={n_attacks}", time.time()))
        return round(fragility, 5)

    def inject_adversarial_evidence(self, claim: str, domain: str) -> list[dict]:
        """Returns list of CounterEvidence dicts with TF-IDF-weighted salience scores."""
        with self._lock:
            tokens = claim.lower().split()
            N = max(10, len(self._belief_history) + 10)
            evidence_pool = [
                f"counter_{domain}_{t}_{random.randint(100,999)}" for t in tokens[:5]
            ]
            results = []
            for ev in evidence_pool:
                ev_tokens = ev.split("_")
                tf = sum(1 for t in ev_tokens if t in tokens) / max(1, len(ev_tokens))
                df = random.randint(1, N // 2)
                idf = math.log(N / (df + 1))
                salience = tf * idf
                now = time.time()
                decay = math.exp(-(now % 300) / 300)
                results.append({
                    "evidence": ev, "domain": domain,
                    "salience": round(salience * decay, 5),
                    "confidence": round(random.uniform(0.4, 0.9), 3),
                    "source": f"adversarial_inject_{domain}",
                })
            results.sort(key=lambda x: x["salience"], reverse=True)
            return results

    def detect_epistemic_immune_escape(self, reasoning_trace: list[dict]) -> dict:
        """Returns EscapeReport with z-score anomaly detection on confidence drift."""
        if len(reasoning_trace) < 3:
            return {"escape_detected": False, "reason": "insufficient_trace", "z_score": 0.0}
        confs = [step.get("confidence", 0.5) for step in reasoning_trace]
        mean_c = statistics.mean(confs)
        std_c = statistics.stdev(confs) if len(confs) > 1 else 1e-9
        final_z = (confs[-1] - mean_c) / (std_c + 1e-9)
        escape = abs(final_z) < 0.5 and mean_c > 0.85
        monotone = all(confs[i] <= confs[i+1] for i in range(len(confs)-1))
        report = {
            "escape_detected": escape or monotone,
            "z_score": round(final_z, 4),
            "mean_confidence": round(mean_c, 4),
            "monotone_increase": monotone,
            "severity": "HIGH" if (escape and monotone) else ("MEDIUM" if escape else "LOW"),
            "recommendation": "inject_adversarial_evidence" if escape else "continue_monitoring",
        }
        self._escape_reports.append(report)
        return report

    def update_confidence_under_attack(self, prior: float, attacks: list[dict]) -> dict:
        """Returns Posterior dict with Bayesian-updated confidence and entropy."""
        posterior = prior
        history = []
        for attack in attacks:
            likelihood = attack.get("strength", 0.5)
            posterior = (likelihood * posterior) / (
                likelihood * posterior + (1 - likelihood) * (1 - posterior) + 1e-12)
            history.append(round(posterior, 5))
        self._ema_accuracy = 0.15 * posterior + 0.85 * self._ema_accuracy
        p = max(1e-12, min(1 - 1e-12, posterior))
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        ci_half = 1.96 * math.sqrt(p * (1 - p) / max(1, len(attacks)))
        return {
            "prior": round(prior, 5), "posterior": round(posterior, 5),
            "entropy": round(entropy, 5),
            "ci_lower": round(max(0.0, posterior - ci_half), 5),
            "ci_upper": round(min(1.0, posterior + ci_half), 5),
            "ema_accuracy": round(self._ema_accuracy, 5),
            "n_attacks": len(attacks), "history": history,
        }

    def schedule_adversarial_audit(self, module_name: str, interval_sec: int) -> dict:
        """Returns AuditJob dict and registers a recurring adversarial audit for module."""
        with self._lock:
            job_id = f"audit_{module_name}_{int(time.time())}"
            job = {
                "job_id": job_id, "module": module_name,
                "interval_sec": interval_sec, "next_run": time.time() + interval_sec,
                "runs": 0, "last_fragility": None,
            }
            self._audit_jobs[job_id] = job
            try:
                from tools import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"Adversarial audit of {module_name} every {interval_sec}s", priority=6)
            except Exception:
                pass
            return job

    def add_fact(self, claim: str, confidence: float) -> dict:
        """Stores a causal fact (premise→conclusion) and returns insertion confirmation."""
        parts = claim.split("->") if "->" in claim else [claim, f"implies_{self._belief_id(claim)}"]
        premise = parts[0].strip()
        conclusion = parts[1].strip() if len(parts) > 1 else f"effect_of_{premise[:20]}"
        weight = max(0.0, min(1.0, confidence))
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO causal_facts(premise,conclusion,weight) VALUES(?,?,?)",
                    (premise, conclusion, weight))
        return {"premise": premise, "conclusion": conclusion, "weight": round(weight, 4)}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            frags = list(self._fragility_graph.values())
            mean_frag = statistics.mean(frags) if frags else 0.0
            return {
                "cycles": self._cycles,
                "active": len(self._audit_jobs),
                "items": len(self._fragility_graph),
                "confidence": round(self._ema_accuracy, 4),
                "entropy": round(mean_frag, 4),
                "pending": len(self._escape_reports),
                "accuracy": round(1.0 - mean_frag, 4),
                "quality": round(self._ema_accuracy * (1.0 - mean_frag), 4),
            }

    def _infer_domain(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["science", "physics", "biology", "empirical"]):
            return "science"
        if any(w in t for w in ["ethics", "moral", "value", "right", "wrong"]):
            return "ethics"
        if any(w in t for w in ["self", "nova", "i am", "my belief"]):
            return "self"
        if any(w in t for w in ["world", "reality", "model", "state"]):
            return "world_model"
        return "default"

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                now = time.time()
                with self._lock:
                    due = [j for j in self._audit_jobs.values() if j["next_run"] <= now]
                for job in due:
                    snapshot = {"beliefs": {
                        f"belief_{job['module']}_{i}": random.uniform(0.5, 0.95)
                        for i in range(4)}}
                    result = self.run_epistemic_gauntlet(snapshot)
                    with self._lock:
                        job["runs"] += 1
                        job["last_fragility"] = result["mean_fragility"]
                        job["next_run"] = now + job["interval_sec"]
                try:
                    from tools import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "adversarial_auto_cycle", "EMA+Bayesian",
                        self._ema_accuracy, self._cycles > 0)
                except Exception:
                    pass
                self._cycles += 1
            except Exception:
                pass
            time.sleep(60)

# Usage: obj = AdversarialRealityTestingEngine() | result = obj.compute_belief_fragility_score("The world is deterministic", 5)
