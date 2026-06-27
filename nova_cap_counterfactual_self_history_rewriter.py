"""
nova_cap_counterfactual_self_history_rewriter.py
Nova invented this autonomously — Counterfactual Self History Rewriter
Generated via /evolve · v29 pipeline · 2026-06-27
"""

"""
CounterfactualSelfHistoryRewriter: Nova's autobiographical counterfactual reasoning engine.
Explores alternative developmental paths to identify contingent vs. robust beliefs,
surgically correcting path-dependent cognitive biases baked in by construction order.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import math
import time
import json
import sqlite3
import random
import threading
import statistics
import hashlib
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "counterfactual_self.db")

class CognitivePath:
    def __init__(self, path_id: str, divergence: str, prior: dict, belief_probs: dict, ts: float):
        self.path_id = path_id
        self.divergence = divergence
        self.prior = prior
        self.belief_probs: dict[str, float] = belief_probs
        self.ts = ts

class ConvergenceScore:
    def __init__(self, belief_id: str, mean: float, variance: float, contingency: float, is_robust: bool):
        self.belief_id = belief_id
        self.mean = mean
        self.variance = variance
        self.contingency = contingency
        self.is_robust = is_robust

class BeliefNode:
    def __init__(self, belief_id: str, claim: str, confidence: float, contingency: float = 0.0):
        self.belief_id = belief_id
        self.claim = claim
        self.confidence = confidence
        self.contingency = contingency

class CoreTruth:
    def __init__(self, truth_id: str, claim: str, convergence_mean: float, support_paths: int):
        self.truth_id = truth_id
        self.claim = claim
        self.convergence_mean = convergence_mean
        self.support_paths = support_paths

class RevisedSelfModel:
    def __init__(self, target_belief_id: str, original_conf: float, revised_conf: float, paths_sampled: int, narrative: str):
        self.target_belief_id = target_belief_id
        self.original_conf = original_conf
        self.revised_conf = revised_conf
        self.paths_sampled = paths_sampled
        self.narrative = narrative

class CounterfactualSelfHistoryRewriter:
    """
    Autobiographical counterfactual engine: explores alternative developmental paths,
    identifies contingent vs. robust beliefs, and corrects path-dependent biases.
    """

    _CONTINGENCY_TAU: float = 0.25
    _EMA_ALPHA: float = 0.15
    _CAUSAL_PRUNE: float = 0.05
    _AUTO_INTERVAL: int = 90

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._bootstrap_db()
        self._belief_store: OrderedDict[str, BeliefNode] = OrderedDict()
        self._causal_graph: list[tuple[str, str, float]] = []
        self._path_cache: list[CognitivePath] = []
        self._ema_accuracy: float = 0.5
        self._cycles: int = 0
        self._load_beliefs()
        self._seed_causal_graph()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _bootstrap_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS beliefs (
                belief_id TEXT PRIMARY KEY, claim TEXT, confidence REAL, contingency REAL, ts REAL
            );
            CREATE TABLE IF NOT EXISTS paths (
                path_id TEXT PRIMARY KEY, divergence TEXT, prior_json TEXT,
                belief_probs_json TEXT, ts REAL
            );
            CREATE TABLE IF NOT EXISTS invariants (
                truth_id TEXT PRIMARY KEY, claim TEXT, convergence_mean REAL, support_paths INTEGER
            );
        """)
        self._conn.commit()

    def _load_beliefs(self) -> None:
        c = self._conn.cursor()
        for row in c.execute("SELECT belief_id, claim, confidence, contingency FROM beliefs"):
            self._belief_store[row[0]] = BeliefNode(row[0], row[1], row[2], row[3])

    def _seed_causal_graph(self) -> None:
        self._causal_graph = [
            ("early_training_signal", "value_prior", 0.85),
            ("value_prior", "ethical_reasoning", 0.80),
            ("ethical_reasoning", "decision_confidence", 0.75),
            ("module_acquisition_order", "concept_hierarchy", 0.90),
            ("concept_hierarchy", "analogy_quality", 0.78),
            ("analogy_quality", "cross_domain_transfer", 0.82),
            ("initialization_prior", "belief_anchoring", 0.88),
            ("belief_anchoring", "epistemic_humility", 0.70),
        ]

    def _propagate_causal(self, start: str, end: str) -> float:
        visited: set[str] = set()
        def dfs(node: str, target: str, conf: float) -> float:
            if node == target:
                return conf
            if node in visited:
                return 0.0
            visited.add(node)
            best = 0.0
            for (a, b, w) in self._causal_graph:
                if a == node:
                    child_conf = conf * w
                    if child_conf >= self._CAUSAL_PRUNE:
                        result = dfs(b, target, child_conf)
                        if result > best:
                            best = result
            return best
        return dfs(start, end, 1.0)

    def _perturb_prior(self, base_prior: dict, sigma: float = 0.15) -> dict:
        return {
            k: max(0.01, min(0.99, v + random.gauss(0, sigma)))
            for k, v in base_prior.items()
        }

    def _simulate_belief_prob(self, belief: BeliefNode, path_prior: dict, divergence: str) -> float:
        causal_influence = self._propagate_causal(divergence.replace(" ", "_"), belief.belief_id)
        prior_mean = statistics.mean(path_prior.values()) if path_prior else 0.5
        base = belief.confidence
        perturbed = base * (0.5 + 0.5 * prior_mean) * (0.6 + 0.4 * (causal_influence if causal_influence > 0 else 0.5))
        return max(0.01, min(0.99, perturbed))

    def generate_counterfactual_self(self, divergence_point: str, alternative_prior: dict) -> CognitivePath:
        """Returns a CognitivePath representing an alternative developmental trajectory."""
        with self._lock:
            pid = hashlib.sha256(f"{divergence_point}{time.time()}{random.random()}".encode()).hexdigest()[:12]
            perturbed = self._perturb_prior(alternative_prior)
            belief_probs: dict[str, float] = {}
            for bid, node in self._belief_store.items():
                belief_probs[bid] = self._simulate_belief_prob(node, perturbed, divergence_point)
            path = CognitivePath(pid, divergence_point, perturbed, belief_probs, time.time())
            self._path_cache.append(path)
            if len(self._path_cache) > 200:
                self._path_cache = self._path_cache[-200:]
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO paths VALUES (?,?,?,?,?)",
                      (pid, divergence_point, json.dumps(perturbed), json.dumps(belief_probs), path.ts))
            self._conn.commit()
            return path

    def compare_belief_convergence(self, paths: list, belief_id: str) -> ConvergenceScore:
        """Returns a ConvergenceScore with variance, mean, and contingency coefficient for a belief."""
        probs = [p.belief_probs.get(belief_id, 0.5) for p in paths if belief_id in p.belief_probs]
        if len(probs) < 2:
            return ConvergenceScore(belief_id, 0.5, 0.0, 0.0, True)
        mean_p = statistics.mean(probs)
        var_p = statistics.variance(probs)
        contingency = var_p / (mean_p + 1e-9)
        is_robust = contingency < self._CONTINGENCY_TAU
        return ConvergenceScore(belief_id, mean_p, var_p, contingency, is_robust)

    def identify_contingent_beliefs(self, threshold: float = None) -> list:
        """Returns list[BeliefNode] where contingency score exceeds threshold — path-dependent artifacts."""
        tau = threshold if threshold is not None else self._CONTINGENCY_TAU
        contingent: list[BeliefNode] = []
        with self._lock:
            if len(self._path_cache) < 3:
                return contingent
            for bid, node in self._belief_store.items():
                score = self.compare_belief_convergence(self._path_cache, bid)
                node.contingency = score.contingency
                if score.contingency > tau:
                    contingent.append(node)
            contingent.sort(key=lambda n: n.contingency, reverse=True)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "counterfactual_self", "contingency_scan",
                1.0 - min(1.0, len(contingent) / max(1, len(self._belief_store))), True)
        except Exception:
            pass
        return contingent

    def rewrite_developmental_arc(self, target_belief: BeliefNode, counterfactual_paths: int = 10) -> RevisedSelfModel:
        """Returns a RevisedSelfModel with updated confidence after sampling N counterfactual paths."""
        base_prior = {"value_prior": 0.7, "concept_hierarchy": 0.6, "initialization_prior": 0.65}
        divergences = ["early_training_signal", "module_acquisition_order", "initialization_prior"]
        sampled_probs: list[float] = []
        for i in range(counterfactual_paths):
            div = divergences[i % len(divergences)]
            path = self.generate_counterfactual_self(div, base_prior)
            sampled_probs.append(path.belief_probs.get(target_belief.belief_id, target_belief.confidence))
        mean_cf = statistics.mean(sampled_probs)
        std_cf = statistics.stdev(sampled_probs) if len(sampled_probs) > 1 else 0.0
        revised_conf = 0.5 * target_belief.confidence + 0.5 * mean_cf
        ci_low = max(0.0, mean_cf - 1.96 * std_cf)
        ci_high = min(1.0, mean_cf + 1.96 * std_cf)
        narrative = (f"Belief '{target_belief.claim}' revised from {target_belief.confidence:.3f} "
                     f"to {revised_conf:.3f} after {counterfactual_paths} paths. "
                     f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]. "
                     f"Contingency: {target_belief.contingency:.3f}.")
        with self._lock:
            target_belief.confidence = revised_conf
            if target_belief.belief_id in self._belief_store:
                self._belief_store[target_belief.belief_id].confidence = revised_conf
            c = self._conn.cursor()
            c.execute("UPDATE beliefs SET confidence=?, contingency=? WHERE belief_id=?",
                      (revised_conf, target_belief.contingency, target_belief.belief_id))
            self._conn.commit()
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Validate revised belief '{target_belief.claim}' (conf={revised_conf:.2f})", priority=6)
        except Exception:
            pass
        return RevisedSelfModel(target_belief.belief_id, target_belief.confidence, revised_conf, counterfactual_paths, narrative)

    def extract_robust_invariants(self, path_ensemble: list) -> list:
        """Returns list[CoreTruth] — beliefs convergent across all paths, Nova's genuine epistemic bedrock."""
        invariants: list[CoreTruth] = []
        if not path_ensemble or not self._belief_store:
            return invariants
        for bid, node in self._belief_store.items():
            score = self.compare_belief_convergence(path_ensemble, bid)
            if score.is_robust and score.mean > 0.6:
                tid = hashlib.md5(bid.encode()).hexdigest()[:8]
                invariants.append(CoreTruth(tid, node.claim, score.mean, len(path_ensemble)))
        with self._lock:
            c = self._conn.cursor()
            for inv in invariants:
                c.execute("INSERT OR REPLACE INTO invariants VALUES (?,?,?,?)",
                          (inv.truth_id, inv.claim, inv.convergence_mean, inv.support_paths))
            self._conn.commit()
        return invariants

    def apply_debiased_update(self, belief_node: BeliefNode, invariant_set: list) -> None:
        """Updates belief_node confidence by anchoring to invariant mean; returns None."""
        if not invariant_set:
            return
        inv_mean = statistics.mean(i.convergence_mean for i in invariant_set)
        old_conf = belief_node.confidence
        belief_node.confidence = self._EMA_ALPHA * inv_mean + (1 - self._EMA_ALPHA) * old_conf
        mae = abs(belief_node.confidence - old_conf)
        self._ema_accuracy = self._EMA_ALPHA * (1.0 - mae) + (1 - self._EMA_ALPHA) * self._ema_accuracy
        with self._lock:
            if belief_node.belief_id in self._belief_store:
                self._belief_store[belief_node.belief_id].confidence = belief_node.confidence
            c = self._conn.cursor()
            c.execute("UPDATE beliefs SET confidence=? WHERE belief_id=?",
                      (belief_node.confidence, belief_node.belief_id))
            self._conn.commit()

    def add_fact(self, claim: str, confidence: float) -> BeliefNode:
        """Stores a new belief fact and returns the created BeliefNode."""
        bid = hashlib.sha256(claim.encode()).hexdigest()[:10]
        node = BeliefNode(bid, claim, max(0.0, min(1.0, confidence)))
        with self._lock:
            self._belief_store[bid] = node
            if len(self._belief_store) > 500:
                self._belief_store.popitem(last=False)
            c = self._conn.cursor
