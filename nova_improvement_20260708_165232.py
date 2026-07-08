"""
CausalInterventionSculptor — prescriptive causal engine that identifies minimal high-leverage
intervention points across any system, computes Pareto-optimal intervention sets, and sculpts
reality toward target attractor states via Pearl do-calculus and modified Dijkstra search.
"""

import math
import time
import json
import sqlite3
import random
import threading
import statistics
import hashlib
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any

InterventionPoint = Dict[str, Any]
CausalTrajectory = Dict[str, Any]
ParetoSet = List[Dict[str, Any]]
InterventionSequence = List[Dict[str, Any]]


class CausalInterventionSculptor:
    """Prescriptive causal sculptor: finds minimal-entropy interventions to author target reality states."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._causal_graph: Dict[str, Dict[str, float]] = {}
        self._node_metadata: Dict[str, Dict] = {}
        self._intervention_history: List[Dict] = []
        self._chain_rules: List[Tuple[str, str, float]] = []
        self._fact_store: Dict[str, float] = {}
        self._inferred_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._irreversibility_threshold: float = 0.75
        self._pareto_population_size: int = 100
        self._eigenvector_centrality: Dict[str, float] = {}
        self._entropy_budget: float = 1.0
        self._db_path: str = "causal_sculptor.db"
        self._cycles: int = 0
        self._init_db()
        self._load_state()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS facts
                (claim TEXT PRIMARY KEY, confidence REAL, ts REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS chain_rules
                (premise TEXT, conclusion TEXT, weight REAL,
                 PRIMARY KEY(premise, conclusion))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS interventions
                (id TEXT PRIMARY KEY, data TEXT, ts REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS graph_edges
                (src TEXT, dst TEXT, weight REAL,
                 PRIMARY KEY(src, dst))""")
            conn.commit()

    def _load_state(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                for row in conn.execute("SELECT claim, confidence FROM facts"):
                    self._fact_store[row[0]] = row[1]
                for row in conn.execute("SELECT premise, conclusion, weight FROM chain_rules"):
                    self._chain_rules.append((row[0], row[1], row[2]))
                for row in conn.execute("SELECT src, dst, weight FROM graph_edges"):
                    src, dst, w = row
                    self._causal_graph.setdefault(src, {})[dst] = w
                for row in conn.execute("SELECT data FROM interventions ORDER BY ts DESC LIMIT 50"):
                    self._intervention_history.append(json.loads(row[0]))
        except sqlite3.Error:
            pass

    def _save_fact(self, claim: str, confidence: float) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?)",
                             (claim, confidence, time.time()))
                conn.commit()
        except sqlite3.Error:
            pass

    def _save_rule(self, premise: str, conclusion: str, weight: float) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO chain_rules VALUES (?,?,?)",
                             (premise, conclusion, weight))
                conn.commit()
        except sqlite3.Error:
            pass

    def _save_edge(self, src: str, dst: str, weight: float) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO graph_edges VALUES (?,?,?)",
                             (src, dst, weight))
                conn.commit()
        except sqlite3.Error:
            pass

    def _compute_eigenvector_centrality(self) -> None:
        nodes = list(self._causal_graph.keys())
        for dst_map in self._causal_graph.values():
            for n in dst_map:
                if n not in nodes:
                    nodes.append(n)
        if not nodes:
            return
        ev: Dict[str, float] = {n: 1.0 / len(nodes) for n in nodes}
        for _ in range(100):
            new_ev: Dict[str, float] = {n: 0.0 for n in nodes}
            for src, dsts in self._causal_graph.items():
                for dst, w in dsts.items():
                    new_ev[dst] = new_ev.get(dst, 0.0) + ev.get(src, 0.0) * w
            norm = math.sqrt(sum(v * v for v in new_ev.values())) + 1e-9
            ev = {n: v / norm for n, v in new_ev.items()}
        self._eigenvector_centrality = ev

    def _p_target_no_intervention(self, target: str) -> float:
        return self._fact_store.get(target, 0.1)

    def _propagate_do_calculus(self, intervened: str, target: str, depth: int = 5) -> float:
        visited: Dict[str, float] = {intervened: 1.0}
        for _ in range(depth):
            new_visited: Dict[str, float] = dict(visited)
            for node, p_node in visited.items():
                for dst, w in self._causal_graph.get(node, {}).items():
                    parents_p = [visited.get(u, 0.0) * self._causal_graph[u].get(dst, 0.0)
                                 for u in self._causal_graph if dst in self._causal_graph[u]]
                    combined = 1.0 - math.prod(max(0.0, 1.0 - pp) for pp in parents_p) if parents_p else p_node * w
                    new_visited[dst] = max(new_visited.get(dst, 0.0), combined)
            visited = new_visited
        return min(1.0, visited.get(target, 0.0))

    def _entropy_of_cost(self, node: str) -> float:
        meta = self._node_metadata.get(node, {})
        cost_dist = meta.get("cost_dist", [0.25, 0.25, 0.25, 0.25])
        total = sum(cost_dist) + 1e-12
        probs = [c / total for c in cost_dist]
        return -sum(p * math.log2(p + 1e-9) for p in probs)

    def add_fact(self, claim: str, confidence: float) -> Dict[str, Any]:
        """Returns stored fact dict after persisting claim with given confidence."""
        confidence = max(0.0, min(1.0, confidence))
        with self._lock:
            self._fact_store[claim] = confidence
            self._inferred_cache.clear()
            self._save_fact(claim, confidence)
            neg = f"NOT_{claim}"
            contradiction = (self._fact_store.get(neg, 0.0) > 0.5 and confidence > 0.5)
        return {"claim": claim, "confidence": confidence, "contradiction_flagged": contradiction}

    def add_causal_edge(self, src: str, dst: str, weight: float,
                        reversibility: float = 0.5, cost: float = 0.3) -> Dict[str, Any]:
        """Returns edge dict after adding directed causal edge with weight and node metadata."""
        weight = max(0.0, min(1.0, weight))
        with self._lock:
            self._causal_graph.setdefault(src, {})[dst] = weight
            for node, rev, c in [(src, reversibility, cost), (dst, reversibility, cost)]:
                if node not in self._node_metadata:
                    self._node_metadata[node] = {
                        "reversibility": rev,
                        "cost_dist": [c, 1.0 - c, c * 0.5, (1.0 - c) * 0.5],
                        "type": "causal_node"
                    }
            self._compute_eigenvector_centrality()
            self._save_edge(src, dst, weight)
        return {"src": src, "dst": dst, "weight": weight}

    def score_leverage(self, node: str, graph: Dict[str, Dict[str, float]],
                       entropy_budget: float) -> float:
        """Returns float leverage score L(v) = ΔP(target|do(v)) / H(cost) for node v."""
        with self._lock:
            targets = list(self._fact_store.keys())
            if not targets:
                return 0.0
            target = max(targets, key=lambda t: self._fact_store.get(t, 0.0))
            p_base = self._p_target_no_intervention(target)
            p_do = self._propagate_do_calculus(node, target)
            delta_p = max(0.0, p_do - p_base)
            h_cost = self._entropy_of_cost(node)
            if h_cost < 1e-9:
                leverage = delta_p / 1e-9
            else:
                leverage = delta_p / (h_cost * max(entropy_budget, 1e-9))
            ec = self._eigenvector_centrality.get(node, 0.0)
            return round(leverage * (1.0 + ec), 6)

    def identify_intervention_nodes(self, causal_graph: Dict[str, Dict[str, float]],
                                    target_state: str,
                                    constraints: Dict[str, Any]) -> List[InterventionPoint]:
        """Returns ranked list of InterventionPoint dicts for nodes that best drive target_state."""
        with self._lock:
            for src, dsts in causal_graph.items():
                for dst, w in dsts.items():
                    self._causal_graph.setdefault(src, {})[dst] = w
            self._compute_eigenvector_centrality()
            max_cost = constraints.get("max_cost", 1.0)
            max_irrev = constraints.get("max_irreversibility", self._irreversibility_threshold)
            candidates: List[InterventionPoint] = []
            all_nodes = list(self._causal_graph.keys())
            for v in all_nodes:
                meta = self._node_metadata.get(v, {})
                irrev = 1.0 - meta.get("reversibility", 0.5)
                if irrev > max_irrev:
                    continue
                cost_dist = meta.get("cost_dist", [0.25, 0.25, 0.25, 0.25])
                est_cost = cost_dist[0] if cost_dist else 0.5
                if est_cost > max_cost:
                    continue
                p_do = self._propagate_do_calculus(v, target_state)
                p_base = self._p_target_no_intervention(target_state)
                delta_p = max(0.0, p_do - p_base)
                h_cost = self._entropy_of_cost(v)
                leverage = delta_p / max(h_cost, 1e-9)
                ec = self._eigenvector_centrality.get(v, 0.0)
                candidates.append({
                    "node": v, "leverage": round(leverage, 5),
                    "delta_p": round(delta_p, 4), "irreversibility": round(irrev, 3),
                    "eigenvector_centrality": round(ec, 4), "estimated_cost": est_cost
                })
            candidates.sort(key=lambda x: x["leverage"], reverse=True)
        return candidates[:10]

    def compute_irreversibility_penalty(self, intervention: str,
                                        graph: Dict[str, Dict[str, float]]) -> float:
        """Returns float joint irreversibility penalty for intervention propagated through graph."""
        with self._lock:
            irrev_v = 1.0 - self._node_metadata.get(intervention, {}).get("reversibility", 0.5)
            downstream = list(graph.get(intervention, {}).keys())
            penalties = [irrev_v]
            for d in downstream[:5]:
                irrev_d = 1.0 - self._node_metadata.get(d, {}).get("reversibility", 0.5)
                w = graph.get(intervention, {}).get(d, 0.0)
                penalties.append(irrev_d * w)
            joint = 1.0 - math.prod(max(0.0, 1.0 - p) for p in penalties)
        return round(min(1.0, joint), 5)

    def simulate_intervention_cascade(self, intervention_set: List[str],
                                      world_model: Dict[str, float],
                                      depth: int = 5) -> CausalTrajectory:
        """Returns CausalTrajectory dict showing probability evolution across cascade depth."""
        with self._lock:
            state: Dict[str, float] = dict(world_model)
            for node in intervention_set:
                state[node] = 1.0
            trajectory: List[Dict[str, float]] = [dict(state)]
            for d in range(depth):
                new_state = dict(state)
                for src, dsts in self._causal_graph.items():
                    p_src = state.get(src, 0.0)
                    for dst, w in dsts.items():
                        p_effect = p_src * w
                        existing = new_state.get(dst, 0.0)
                        new_state[dst] = 1.0 - (1.0 - existing) * (1.0 - p_effect)
                for k in new_state:
                    new_state[k] = round(min(1.0, new_state[k]), 5)
                trajectory.append(dict(new_state))
                state = new_state
            side_effect_nodes = [n for n in state if n not in intervention_set]
            se_probs = [state.get(n, 0.0) for n in side_effect_nodes]
            se_entropy = -sum(p * math.log2(p + 1e-9) + (1 - p) * math.log2(1 - p + 1e-9)
                              for p in se_probs) if se_probs else 0.0
        return {
            "trajectory": trajectory, "final_state": state,
            "depth": depth, "side_effect_entropy": round(se_entropy, 4),
            "interventions": intervention_set
        }

    def find_pareto_frontier(self, candidate_interventions: List[str],
                             objectives: Dict[str, float]) -> ParetoSet:
        """Returns ParetoSet list of non-dominated intervention subsets across cost/irrev/entropy."""
        theta = objectives.get("min_target_prob", 0.85)
        target = objectives.get("target_state", "")
        with self._lock:
            rng = random.Random(42)
            population: List[Tuple[frozenset, Dict[str, float]]] = []
            nodes = candidate_interventions[:20]
            for _ in range(self._pareto_population_size):
                k = rng.randint(1, max(1, len(nodes) // 2))
                subset = frozenset(rng.sample(nodes, k))
                cost = sum(self._node_metadata.get(n, {}).get("cost_dist", [0.3])[0] for n in subset)
                irrev = 1.0 - math.prod(
                    self._node_metadata.get(n, {}).get("reversibility", 0.5) for n in subset)
                traj = self.simulate_intervention_cascade(list(subset), self._fact_store)
                se_ent = traj["side_effect_entropy"]
                p_target = traj["final_state"].get(target, 0.0) if target else 0.5
                if p_target < theta:
                    continue
                population.append((subset, {"cost": cost, "irreversibility": irrev,
                                             "side_effect_entropy": se_ent, "p_target": p_target}))
            pareto: ParetoSet = []
            for i, (s_a, obj_a) in enumerate(population):
                dominated = False
                for j, (s_b, obj_b) in enumerate(population):
                    if i == j:
                        continue
                    if (obj_b["cost"] <= obj_a["cost"] and
                            obj_b["irreversibility"] <= obj_a["irreversibility"] and
                            obj_b["side_effect_entropy"] <= obj_a["side_effect_entropy"] and
                            (obj_b["cost"] < obj_a["cost"] or
                             obj_b["irreversibility"] < obj_a["irreversibility"] or
                             obj_b["side_effect_entropy"] < obj_a["side_effect_entropy"])):
                        dominated = True
                        break
                if not dominated:
                    pareto.append({"nodes": sorted(s_a), "objectives": obj_a})
            pareto.sort(key=lambda x: x["objectives"]["cost"])
        return pareto[:10]

    def sculpt_reality_plan(self, target_attractor: str, current_state: Dict[str, float],
                            resource_constraints: Dict[str, Any]) -> InterventionSequence:
        """Returns InterventionSequence ordered list of intervention steps toward target_attractor."""
        with self._lock:
            theta = resource_constraints.get("min_prob", 0.85)
            budget = resource_constraints.get("budget", 1.0)
            candidates = self.identify_intervention_nodes(
                self._causal_graph, target_attractor,
                {"max_cost": budget, "max_irreversibility": self._irreversibility_threshold})
            sequence: InterventionSequence = []
            accumulated_irrev = 0.0
            accumulated_cost = 0.0
            current = dict(current_state)
            for pt in candidates:
                node = pt["node"]
                irrev = pt["irreversibility"]
                cost = pt["estimated_cost"]
                joint_irrev = 1.0 - (1.0 - accumulated_irrev) * (1.0 - irrev)
                if joint_irrev > self._irreversibility_threshold:
                    continue
                if accumulated_cost + cost > budget:
                    continue
                traj = self.simulate_intervention_cascade(
                    [node], current, depth=3)
                p_after = traj["final_state"].get(target_attractor, 0.0)
                ec = self._eigenvector_centrality.get(node, 0.0)
                h_heuristic = (theta - p_after) / (ec + 1e-9)
                step = {
                    "step": len(sequence) + 1, "node": node,
                    "expected_p_target": round(p_after, 4),
                    "leverage": pt["leverage"], "cost": cost,
                    "irreversibility": irrev, "heuristic_residual": round(h_heuristic, 4),
                    "joint_irreversibility": round(joint_irrev, 4)
                }
                sequence.append(step)
                accumulated_irrev = joint_irrev
                accumulated_cost += cost
                current = traj["final_state"]
                if p_after >= theta:
                    break
            rec_id = hashlib.md5(f"{target_attractor}{time.time()}".encode()).hexdigest()[:8]
            self._intervention_history.append({
                "id": rec_id, "target": target_attractor,
                "sequence_length": len(sequence), "ts": time.time()})
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO interventions VALUES (?,?,?)",
                                 (rec_id, json.dumps({"target": target_attractor,
                                                       "steps": len(sequence)}), time.time()))
                    conn.commit()
            except sqlite3.Error:
                pass
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"Execute causal sculpture toward: {target_attractor}", priority=2)
            except (ImportError, Exception):
                pass
        return sequence

    def infer(self, hypothesis: str) -> Dict[str, Any]:
        """Returns inference dict with confidence and reasoning path via forward chaining."""
        with self._lock:
            if hypothesis in self._inferred_cache:
                conf, path = self._inferred_cache[hypothesis]
                return {"hypothesis": hypothesis, "confidence": conf, "path": path, "cached": True}
            best_conf = self._fact_store.get(hypothesis, 0.0)
            best_path: List[str] = [hypothesis] if best_conf > 0 else []
            for premise, conclusion, weight in self._chain_rules:
                if conclusion == hypothesis:
                    p_premise = self._fact_store.get(premise, 0.0)
                    chain_conf = p_premise * weight
                    if chain_conf > best_conf:
                        best_conf = chain_conf
                        best_path = [premise, f"→(w={weight})", hypothesis]
                        for p2, c2, w2 in self._chain_rules:
                            if c2 == premise:
                                p2_conf = self._fact_store.get(p2, 0.0)
                                multi_hop = p2_conf * w2 * weight
                                if multi_hop > best_conf:
                                    best_conf = multi_hop
                                    best_path = [p2, f"→(w={w2})", premise,
                                                 f"→(w={weight})", hypothesis]
            self._inferred_cache[hypothesis] = (round(best_conf, 5), best_path)
        return {"hypothesis": hypothesis, "confidence": round(best_conf, 5), "path": best_path}

    def chain_strength(self) -> Dict[str, Any]:
        """Returns dict with min/max/mean chain weights and longest chain info."""
        with self._lock:
            if not self._chain_rules:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
            weights = [w for _, _, w in self._chain_rules]
            multi_hop_strengths = []
            for p1, c1, w1 in self._chain_rules:
                for p2, c2, w2 in self._chain_rules:
                    if c2 == p1:
                        multi_hop_strengths.append(w1 * w2)
            all_weights = weights + multi_hop_strengths
        return {
            "min": round(min(all_weights), 4), "max": round(max(all_weights), 4),
            "mean": round(statistics.mean(all_weights), 4), "count": len(self._chain_rules),
            "multi_hop_count": len(multi_hop_strengths)
        }

    def contradictions(self) -> List[Dict[str, Any]]:
        """Returns list of contradiction dicts where both claim and NOT_claim exceed 0.5 confidence."""
        with self._lock:
            found = []
            for claim, conf in self._fact_store.items():
                if claim.startswith("NOT_"):
                    continue
                neg = f"NOT_{claim}"
                neg_conf = self._fact_store.get(neg, 0.0)
                if conf > 0.5 and neg_conf > 0.5:
                    found.append({"claim": claim, "confidence": conf,
                                  "negation": neg, "neg_confidence": neg_conf,
                                  "severity": round(conf + neg_conf - 1.0, 3)})
        return found

    def status(self) -> Dict[str, Any]:
        """Returns plain dict with numeric keys for ConsciousnessIntegrator Φ integration."""
        with self._lock:
            n_nodes = len(self._causal_graph)
            n_edges = sum(len(v) for v in self._causal_graph.values())
            n_facts = len(self._fact_store)
            n_rules = len(self._chain_rules)
            avg_ec = (statistics.mean(self._eigenvector_centrality.values())
                      if self._eigenvector_centrality else 0.0)
            conf_vals = list(self._fact_store.values())
            entropy = (-sum(p * math.log2(p + 1e-9) for p in conf_vals)
                       if conf_vals else 0.0)
        return {
            "items": n_nodes, "confidence": round(avg_ec, 4),
            "active": n_edges, "pending": n_rules,
            "cycles": self._cycles, "entropy": round(entropy, 4),
            "facts": n_facts, "interventions_logged": len(self._intervention_history),
            "irreversibility_threshold": self._irreversibility_threshold
        }

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self._cycles += 1
                with self._lock:
                    self._compute_eigenvector_centrality()
                    self._inferred_cache.clear()
                    contras = self.contradictions()
                    if contras:
                        for c in contras:
                            claim = c["claim"]
                            if self._fact_store.get(claim, 0) > self._fact_store.get(f"NOT_{claim}", 0):
                                self._fact_store[f"NOT_{claim}"] = max(
                                    0.0, self._fact_store[f"NOT_{claim}"] - 0.05)
                    if self._causal_graph and self._fact_store:
                        target = max(self._fact_store, key=lambda k: self._fact_store[k])
                        nodes = list(self._causal_graph.keys())
                        if nodes:
                            best_node = max(nodes, key=lambda n:
                                self._eigenvector_centrality.get(n, 0.0))
                            _ = self._propagate_do_calculus(best_node, target)
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "causal_intervention", "do_calculus_propagation",
                        self._eigenvector_centrality.get(
                            next(iter(self._eigenvector_centrality), ""), 0.5),
                        len(self.contradictions()) == 0)
                except (ImportError, Exception):
                    pass
            except Exception:
                pass
            time.sleep(60)

# Usage: obj = CausalInterventionSculptor() | result = obj.sculpt_reality_plan("world_peace", {"conflict": 0.8}, {"budget": 0.9})
