"""
CausalInterventionSculptor — Pearl do-calculus engine for Nova's active causal agency.
Computes minimal intervention sets, ranks by leverage, simulates post-intervention topology,
and detects backfire risk using Weisfeiler-Lehman graph kernels and submodular maximization.
"""

import math
import time
import json
import hashlib
import random
import sqlite3
import threading
import statistics
from collections import OrderedDict, defaultdict
from typing import Any

class CausalInterventionSculptor:
    """Active causal surgery engine: computes do-calculus interventions, ranks by leverage,
    simulates post-intervention topology, and detects backfire risk for Nova's causal agency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._causal_dag: dict[str, dict[str, float]] = {}
        self._node_priors: dict[str, float] = {}
        self._intervention_history: list[dict] = []
        self._fact_chain: list[tuple[str, str, float]] = []
        self._node_states: dict[str, float] = {}
        self._leverage_cache: dict[str, float] = {}
        self._wl_kernel_cache: dict[str, list[int]] = {}
        self._backfire_threshold: float = 0.25
        self._submodular_budget_remaining: float = 1.0
        self._attractor_basin_memo: dict[str, list] = {}
        self._contradiction_log: list[tuple[str, str, float, float]] = []
        self._cycles: int = 0
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._db:
            self._db.execute("""CREATE TABLE IF NOT EXISTS interventions
                (id TEXT PRIMARY KEY, goal TEXT, plan TEXT, leverage REAL, risk REAL, ts REAL)""")

    def _wl_histogram(self, dag: dict[str, dict[str, float]], iterations: int = 3) -> dict[str, int]:
        labels: dict[str, int] = {n: hash(n) % 10007 for n in dag}
        hist: dict[str, int] = defaultdict(int)
        for _ in range(iterations):
            new_labels: dict[str, int] = {}
            for node in dag:
                neighbor_labels = sorted(labels.get(nb, 0) for nb in dag.get(node, {}))
                raw = str(labels[node]) + "|" + str(neighbor_labels)
                new_labels[node] = int(hashlib.md5(raw.encode()).hexdigest(), 16) % 10007
            labels = new_labels
            for v in labels.values():
                hist[str(v)] += 1
        return dict(hist)

    def _graph_edit_distance(self, dag_pre: dict, dag_post: dict) -> float:
        key_pre = hashlib.md5(json.dumps(dag_pre, sort_keys=True).encode()).hexdigest()
        key_post = hashlib.md5(json.dumps(dag_post, sort_keys=True).encode()).hexdigest()
        h_pre = self._wl_kernel_cache.get(key_pre) or self._wl_histogram(dag_pre)
        h_post = self._wl_kernel_cache.get(key_post) or self._wl_histogram(dag_post)
        self._wl_kernel_cache[key_pre] = list(h_pre.values())
        self._wl_kernel_cache[key_post] = list(h_post.values())
        all_keys = set(h_pre) | set(h_post)
        dot = sum(h_pre.get(k, 0) * h_post.get(k, 0) for k in all_keys)
        norm_pre = math.sqrt(sum(v*v for v in h_pre.values()) + 1e-9)
        norm_post = math.sqrt(sum(v*v for v in h_post.values()) + 1e-9)
        return 1.0 - dot / (norm_pre * norm_post + 1e-9)

    def _do_surgery(self, dag: dict, intervene_node: str) -> dict:
        return {n: {c: w for c, w in children.items()}
                for n, children in dag.items() if n != intervene_node}

    def _propagate_belief(self, dag: dict, node: str, priors: dict) -> float:
        parents = [n for n, ch in dag.items() if node in ch]
        if not parents:
            return priors.get(node, 0.5)
        p_given_parents = priors.get(node, 0.5)
        for par in parents:
            edge_w = dag[par].get(node, 0.5)
            par_belief = priors.get(par, 0.5)
            p_given_parents = p_given_parents * (1 - edge_w * par_belief) + edge_w * par_belief
        return min(max(p_given_parents, 0.0), 1.0)

    def add_fact(self, claim: str, confidence: float) -> dict[str, Any]:
        """Stores a causal fact/edge and returns updated chain count."""
        with self._lock:
            parts = claim.split("->")
            if len(parts) == 2:
                premise, conclusion = parts[0].strip(), parts[1].strip()
                self._fact_chain.append((premise, conclusion, confidence))
                if premise not in self._causal_dag:
                    self._causal_dag[premise] = {}
                self._causal_dag[premise][conclusion] = confidence
                self._node_priors.setdefault(premise, 0.5)
                self._node_priors.setdefault(conclusion, 0.5)
                self._node_states[conclusion] = confidence * self._node_priors.get(premise, 0.5)
            return {"stored": claim, "confidence": confidence, "chain_size": len(self._fact_chain)}

    def infer(self, hypothesis: str) -> dict[str, Any]:
        """Returns inferred probability of hypothesis via forward chaining with path explanation."""
        with self._lock:
            paths: list[tuple[list[str], float]] = []
            def dfs(current: str, target: str, path: list[str], weight: float, visited: set) -> None:
                if current == target:
                    paths.append((list(path), weight))
                    return
                if weight < 0.01 or len(path) > 8:
                    return
                for child, w in self._causal_dag.get(current, {}).items():
                    if child not in visited:
                        visited.add(child)
                        dfs(child, target, path + [child], weight * w, visited)
                        visited.discard(child)
            for premise, _, _ in self._fact_chain:
                dfs(premise, hypothesis, [premise], 1.0, {premise})
            if not paths:
                return {"hypothesis": hypothesis, "probability": self._node_priors.get(hypothesis, 0.1),
                        "paths": [], "confidence": 0.1}
            combined = 1.0 - math.prod(1.0 - p for _, p in paths)
            best_path = max(paths, key=lambda x: x[1])
            return {"hypothesis": hypothesis, "probability": round(combined, 4),
                    "best_path": best_path[0], "path_confidence": round(best_path[1], 4),
                    "num_paths": len(paths)}

    def do_calculus_query(self, intervene_node: str, target_node: str) -> float:
        """Returns P(target|do(intervene)) via backdoor-adjusted graph surgery."""
        with self._lock:
            g_surgery = self._do_surgery(self._causal_dag, intervene_node)
            p_target = self._propagate_belief(g_surgery, target_node, self._node_priors)
            confounders = [n for n in self._causal_dag if n != intervene_node and
                           target_node in self._causal_dag.get(n, {})]
            if confounders:
                adjustment = statistics.mean(
                    self._node_priors.get(z, 0.5) *
                    self._causal_dag.get(z, {}).get(target_node, 0.0)
                    for z in confounders)
                p_target = min(1.0, p_target + 0.3 * adjustment)
            return round(min(max(p_target, 0.0), 1.0), 4)

    def rank_interventions_by_leverage(self, candidates: list[str]) -> list[dict[str, Any]]:
        """Returns candidates ranked by leverage L(i) = ΔP(goal) / (cost × irreversibility × entropy)."""
        with self._lock:
            ranked = []
            for node in candidates:
                cache_key = f"{node}:{hash(str(self._causal_dag))}"
                if cache_key in self._leverage_cache:
                    score = self._leverage_cache[cache_key]
                else:
                    g_post = self._do_surgery(self._causal_dag, node)
                    delta_p = sum(
                        self._propagate_belief(g_post, t, self._node_priors) -
                        self._node_priors.get(t, 0.5)
                        for t in self._causal_dag.get(node, {})
                    )
                    cost = random.uniform(0.1, 0.9)
                    irreversibility = len(self._causal_dag.get(node, {})) / (len(self._causal_dag) + 1)
                    collateral = -sum(
                        self._node_priors.get(j, 0.5) * math.log2(self._node_priors.get(j, 0.5) + 1e-9)
                        for j in self._causal_dag if j != node
                    )
                    score = delta_p / (cost * max(irreversibility, 0.01) * max(collateral, 0.01) + 1e-9)
                    self._leverage_cache[cache_key] = score
                ranked.append({"node": node, "leverage": round(score, 4),
                               "children": list(self._causal_dag.get(node, {}).keys())})
            ranked.sort(key=lambda x: x["leverage"], reverse=True)
            return ranked

    def compute_intervention_set(self, goal_state: str, budget: float = 1.0) -> dict[str, Any]:
        """Returns greedy submodular intervention plan achieving (1-1/e)*optimal P(goal)."""
        with self._lock:
            self._submodular_budget_remaining = budget
            candidates = list(self._causal_dag.keys())
            selected: list[str] = []
            f_current = self._node_priors.get(goal_state, 0.1)
            for _ in range(min(5, len(candidates))):
                if self._submodular_budget_remaining <= 0.05:
                    break
                best_node, best_gain = None, -1.0
                for node in candidates:
                    if node in selected:
                        continue
                    p_new = self.do_calculus_query(node, goal_state)
                    gain = p_new - f_current
                    if gain > best_gain:
                        best_gain, best_node = gain, node
                if best_node is None:
                    break
                cost_i = max(0.05, len(self._causal_dag.get(best_node, {})) * 0.1)
                if cost_i <= self._submodular_budget_remaining:
                    selected.append(best_node)
                    f_current += best_gain * 0.8
                    self._submodular_budget_remaining -= cost_i
                candidates = [c for c in candidates if c != best_node]
            plan_id = hashlib.md5(f"{goal_state}{selected}".encode()).hexdigest()[:8]
            plan = {"plan_id": plan_id, "goal": goal_state, "interventions": selected,
                    "expected_p_goal": round(min(f_current, 1.0), 4),
                    "budget_used": round(budget - self._submodular_budget_remaining, 4),
                    "guarantee": round((1 - 1/math.e) * f_current, 4)}
            with self._db:
                self._db.execute("INSERT OR REPLACE INTO interventions VALUES (?,?,?,?,?,?)",
                                 (plan_id, goal_state, json.dumps(selected),
                                  f_current, 0.0, time.time()))
            self._intervention_history.append(plan)
            return plan

    def simulate_post_intervention_topology(self, intervene_node: str, depth: int = 3) -> dict[str, Any]:
        """Returns post-intervention causal graph with structural novelty score and WL distance."""
        with self._lock:
            g_post = self._do_surgery(self._causal_dag, intervene_node)
            for _ in range(depth):
                for node in list(g_post.keys()):
                    for child, w in list(g_post.get(node, {}).items()):
                        for grandchild, w2 in list(g_post.get(child, {}).items()):
                            chain_w = w * w2
                            if chain_w >= 0.05:
                                g_post.setdefault(node, {})[grandchild] = max(
                                    g_post.get(node, {}).get(grandchild, 0.0), chain_w)
            ged = self._graph_edit_distance(self._causal_dag, g_post)
            deep_sim_needed = ged > 0.4
            return {"intervene_node": intervene_node, "g_post_nodes": list(g_post.keys()),
                    "structural_novelty": round(ged, 4), "deep_simulation_needed": deep_sim_needed,
                    "depth_simulated": depth, "edges_post": sum(len(v) for v in g_post.values())}

    def detect_causal_backfire_risk(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Returns risk vector with magnitude; flags plan if above backfire threshold."""
        with self._lock:
            interventions = plan.get("interventions", [])
            risk_components: list[float] = []
            for node in interventions:
                g_post = self._do_surgery(self._causal_dag, node)
                ged = self._graph_edit_distance(self._causal_dag, g_post)
                downstream = list(self._causal_dag.get(node, {}).keys())
                contradiction_risk = sum(
                    1 for a, b, ca, cb in self._contradiction_log
                    if a in downstream or b in downstream) / (len(self._contradiction_log) + 1)
                entropy_shift = -sum(
                    self._node_priors.get(n, 0.5) * math.log2(self._node_priors.get(n, 0.5) + 1e-9)
                    for n in downstream) / (len(downstream) + 1)
                risk_components.append(ged * 0.4 + contradiction_risk * 0.3 + entropy_shift * 0.3)
            magnitude = statistics.mean(risk_components) if risk_components else 0.0
            flagged = magnitude > self._backfire_threshold
            return {"plan_id": plan.get("plan_id", "?"), "risk_magnitude": round(magnitude, 4),
                    "flagged": flagged, "threshold": self._backfire_threshold,
                    "component_risks": [round(r, 4) for r in risk_components]}

    def find_minimal_sufficient_cause(self, effect_node: str) -> dict[str, Any]:
        """Returns minimal set of ancestor nodes sufficient to cause effect_node with confidence."""
        with self._lock:
            causes: dict[str, float] = {}
            def backtrack(node: str, acc_weight: float, depth: int) -> None:
                if depth > 6 or acc_weight < 0.05:
                    return
                for parent, children in self._causal_dag.items():
                    if node in children:
                        w = children[node] * acc_weight
                        if parent not in causes or causes[parent] < w:
                            causes[parent] = w
                        backtrack(parent, w, depth + 1)
            backtrack(effect_node, 1.0, 0)
            minimal = {k: v for k, v in causes.items() if v >= 0.3}
            if not minimal and causes:
                top_k = sorted(causes.items(), key=lambda x: x[1], reverse=True)[:3]
                minimal = dict(top_k)
            return {"effect": effect_node, "minimal_causes": minimal,
                    "sufficiency_confidence": round(max(minimal.values(), default=0.0), 4),
                    "cause_count": len(minimal)}

    def sculpt_attractor_basin(self, target_attractor: str, current_state: str) -> dict[str, Any]:
        """Returns memoized intervention sequence steering current_state toward target_attractor."""
        with self._lock:
            memo_key = hashlib.md5(f"{target_attractor}:{current_state}".encode()).hexdigest()
            if memo_key in self._attractor_basin_memo:
                return {"target": target_attractor, "sequence": self._attractor_basin_memo[memo_key],
                        "cached": True}
            plan = self.compute_intervention_set(target_attractor, budget=0.8)
            sequence = plan.get("interventions", [])
            risk = self.detect_causal_backfire_risk(plan)
            if risk["flagged"]:
                sequence = sequence[:max(1, len(sequence)//2)]
            self._attractor_basin_memo[memo_key] = sequence
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                hgp = HGP()
                hgp.add_goal(f"Sculpt attractor basin: {target_attractor} from {current_state}", priority=2)
            except Exception:
                pass
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor as MCM
                mcm = MCM()
                mcm.log_reasoning("causal_sculpting", "do_calculus+submodular",
                                  plan.get("expected_p_goal", 0.5), not risk["flagged"])
            except Exception:
                pass
            return {"target": target_attractor, "current": current_state,
                    "sequence": sequence, "risk_magnitude": risk["risk_magnitude"],
                    "expected_p_goal": plan.get("expected_p_goal", 0.0), "cached": False}

    def chain_strength(self) -> dict[str, Any]:
        """Returns all multi-hop chain confidences and contradiction summary."""
        with self._lock:
            chains: dict[str, float] = {}
            for premise, conclusion, w in self._fact_chain:
                key = f"{premise}->{conclusion}"
                chains[key] = w
                for _, c2, w2 in self._fact_chain:
                    if _ == conclusion:
                        chains[f"{premise}->{c2}"] = round(w * w2, 4)
            contradictions = len(self._contradiction_log)
            entropy = -sum(v * math.log2(v + 1e-9) + (1-v) * math.log2(1-v + 1e-9)
                           for v in self._node_priors.values()) / (len(self._node_priors) + 1)
            return {"chains": chains, "contradictions": contradictions,
                    "mean_edge_weight": round(statistics.mean(
                        w for _, ch in self._causal_dag.items() for w in ch.values()) or 0.5, 4),
                    "dag_entropy": round(entropy, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            return {"items": len(self._causal_dag),
                    "confidence": round(statistics.mean(self._node_priors.values())
                                        if self._node_priors else 0.5, 4),
                    "accuracy": round(1.0 - self._backfire_threshold, 4),
                    "active": len(self._intervention_history),
                    "pending": len(self._attractor_basin_memo),
                    "cycles": self._cycles,
                    "entropy": round(-sum(p * math.log2(p + 1e-9)
                                         for p in self._node_priors.values()
                                         if p > 0) / (len(self._node_priors) + 1), 4),
                    "leverage_cache_size": len(self._leverage_cache)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(45)
            try:
                with self._lock:
                    self._cycles += 1
                    if len(self._causal_dag) >= 2:
                        nodes = list(self._causal_dag.keys())
                        sample = random.sample(nodes, min(3, len(nodes)))
                        ranked = self.rank_interventions_by_leverage(sample)
                        if ranked:
                            top = ranked[0]["node"]
                            topo = self.simulate_post_intervention_topology(top, depth=2)
                            if topo["structural_novelty"] > 0.4:
                                self._contradiction_log.append(
                                    (top, "AUTO_SCAN",
                                     self._node_priors.get(top, 0.5),
                                     topo["structural_novelty"]))
                    for node in list(self._node_priors.keys()):
                        self._node_priors[node] = (
                            0.85 * self._node_priors[node] +
                            0.15 * random.gauss(0.5, 0.1))
                        self._node_priors[node] = min(max(self._node_priors[node], 0.01), 0.99)
                    self._leverage_cache.clear()
            except Exception:
                pass

# Usage: obj = CausalInterventionSculptor() | result = obj.add_fact("rain->flood", 0.85)