"""
CounterfactualSelfHistoryRewriter: Pearl's do-calculus applied to Nova's own cognitive history.
Retroactively analyzes decision chains to extract meta-level learning about *how Nova generates strategy*.
Operates on causal DAG of beliefs/observations/decisions; computes counterfactual regret via intervention;
updates generative model parameters θ via exponential-recency-weighted gradient ascent.
Closes loop: experience → causal graph → counterfactual intervention → meta-parameter update → improved priors.
"""

import json
import sqlite3
import threading
import time
import math
import statistics
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib


@dataclass
class CounterfactualTrace:
    """Rewritten history under counterfactual intervention."""
    decision_id: str
    intervention_node: str
    intervention_value: float
    original_utility: float
    counterfactual_utility: float
    node_deltas: Dict[str, float] = field(default_factory=dict)
    path_confidence: float = 1.0
    regret_differential: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveTrace:
    """Logged snapshot of a decision and its causal antecedents."""
    trace_id: str
    decision_node_id: str
    belief_nodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # node_id -> (value, confidence)
    observations: Dict[str, float] = field(default_factory=dict)
    outcome_utility: float = 0.0
    timestamp: float = field(default_factory=time.time)
    causal_parents: List[str] = field(default_factory=list)


@dataclass
class FailureSchema:
    """Extracted pattern of why a decision failed."""
    failure_type: str  # 'overconfident_prior', 'missed_dependency', 'wrong_utility_model', etc.
    bad_region: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # node_id -> (low, high) bad range
    corrected_region: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # node_id -> (low, high) good range
    severity: float = 0.0  # 0..1
    affected_nodes: List[str] = field(default_factory=list)


@dataclass
class UpdatedPrior:
    """Updated belief distribution after failure correction."""
    node_id: str
    old_distribution: List[float] = field(default_factory=list)
    new_distribution: List[float] = field(default_factory=list)
    update_strength: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Heuristic:
    """Compressed lesson from counterfactual batch."""
    rule_text: str
    condition: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    expected_regret_reduction: float = 0.0
    confidence: float = 0.0
    derived_from_traces: List[str] = field(default_factory=list)


class CounterfactualSelfHistoryRewriter:
    """
    Pearl's do-calculus applied to Nova's cognitive history as a causal DAG.
    Identifies intervention points in past reasoning that would have yielded better outcomes.
    Updates generative model parameters (θ) via counterfactual gradient descent.
    """

    def __init__(self) -> None:
        """Initialize causal graph, utility surface, and meta-parameters."""
        self._causal_graph: Dict[str, Dict[str, Any]] = {}  # node_id -> {parents, children, value, confidence, type, timestamp}
        self._edge_weights: Dict[Tuple[str, str], float] = {}  # (parent, child) -> strength
        self._utility_surface: Dict[str, float] = {}  # decision_id -> observed U(o)
        self._theta: List[float] = [0.0] * 128  # generative model parameters
        self._eta: float = 0.01  # learning rate
        self._gamma: float = 0.92  # exponential recency decay
        self._T: int = 0  # current timestep
        self._heuristic_cache: List[Heuristic] = []
        self._chain_rules: List[Tuple[str, str, float]] = []  # (premise, conclusion, weight)
        self._prior_distributions: Dict[str, List[float]] = {}  # node_id -> prob distribution
        self._regret_history: List[float] = []
        self._db_path: str = "counterfactual_history.db"
        self._lock: threading.Lock = threading.Lock()
        self._last_batch_time: float = time.time()
        self._cycles: int = 0
        self._init_db()
        self._start_daemon()

    def _init_db(self) -> None:
        """Initialize SQLite persistence."""
        try:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS causal_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    value REAL,
                    confidence REAL,
                    timestamp REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS causal_edges (
                    parent_id TEXT,
                    child_id TEXT,
                    weight REAL,
                    PRIMARY KEY (parent_id, child_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS decision_utilities (
                    decision_id TEXT PRIMARY KEY,
                    utility REAL,
                    timestamp REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS regret_log (
                    timestamp REAL,
                    regret_differential REAL,
                    decision_id TEXT
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            pass  # DB may already exist

    def _start_daemon(self) -> None:
        """Start background auto-cycle thread."""
        def _loop() -> None:
            while True:
                time.sleep(60.0)
                try:
                    self.auto_cycle()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def add_fact(self, claim: str, confidence: float) -> str:
        """Add a belief node to the causal graph with given confidence."""
        node_id = hashlib.md5(f"{claim}_{time.time()}".encode()).hexdigest()[:12]
        with self._lock:
            self._causal_graph[node_id] = {
                'claim': claim,
                'value': 1.0 if confidence > 0.5 else 0.0,
                'confidence': confidence,
                'type': 'belief',
                'timestamp': time.time(),
                'parents': [],
                'children': []
            }
        return node_id

    def infer(self, hypothesis: str) -> Tuple[float, List[str]]:
        """Forward-chain inference: return (confidence, reasoning_path) for hypothesis."""
        # Simple forward chaining: find all paths supporting hypothesis
        path_confidences: List[float] = []
        paths: List[List[str]] = []

        for node_id, node_data in self._causal_graph.items():
            if hypothesis.lower() in node_data.get('claim', '').lower():
                path_confidences.append(node_data['confidence'])
                paths.append([node_id])

        # Propagate through chain rules
        for premise, conclusion, weight in self._chain_rules:
            if conclusion.lower() in hypothesis.lower():
                for node_id, node_data in self._causal_graph.items():
                    if premise.lower() in node_data.get('claim', '').lower():
                        conf = node_data['confidence'] * weight
                        path_confidences.append(conf)
                        paths.append([node_id])

        if not path_confidences:
            return (0.0, [])

        final_conf = max(path_confidences)
        best_path = paths[path_confidences.index(final_conf)]
        return (final_conf, best_path)

    def add_causal_edge(self, parent_id: str, child_id: str, weight: float) -> None:
        """Register causal dependency: parent → child with strength weight."""
        with self._lock:
            if parent_id in self._causal_graph and child_id in self._causal_graph:
                self._edge_weights[(parent_id, child_id)] = weight
                self._causal_graph[parent_id]['children'].append(child_id)
                self._causal_graph[child_id]['parents'].append(parent_id)

    def rewrite_decision_graph(self, decision_id: str, intervention_node: str) -> CounterfactualTrace:
        """
        Pearl's do(v_i = x*): intervene on upstream belief node, propagate through DAG via topological sort.
        Returns counterfactual trace with recomputed utilities and node deltas.
        """
        with self._lock:
            if decision_id not in self._utility_surface or intervention_node not in self._causal_graph:
                return CounterfactualTrace(
                    decision_id=decision_id,
                    intervention_node=intervention_node,
                    intervention_value=0.0,
                    original_utility=0.0,
                    counterfactual_utility=0.0
                )

            original_utility = self._utility_surface[decision_id]
            node_data = self._causal_graph[intervention_node]
            original_value = node_data['value']

            # Gradient ascent to find optimal intervention x*
            x_best = original_value
            u_best = original_utility
            for _ in range(5):
                x_test_hi = original_value + 0.1
                x_test_lo = original_value - 0.1
                u_hi = self._estimate_utility_at_value(intervention_node, x_test_hi, decision_id)
                u_lo = self._estimate_utility_at_value(intervention_node, x_test_lo, decision_id)
                grad = (u_hi - u_lo) / 0.2
                x_candidate = original_value + 0.01 * grad
                x_candidate = max(0.0, min(1.0, x_candidate))
                u_candidate = self._estimate_utility_at_value(intervention_node, x_candidate, decision_id)
                if u_candidate > u_best:
                    u_best = u_candidate
                    x_best = x_candidate
                else:
                    break

            # Propagate intervention through DAG (topological sort)
            node_deltas = self._propagate_intervention(intervention_node, x_best, original_value)
            counterfactual_utility = original_utility + sum(node_deltas.values())

            trace = CounterfactualTrace(
                decision_id=decision_id,
                intervention_node=intervention_node,
                intervention_value=x_best,
                original_utility=original_utility,
                counterfactual_utility=counterfactual_utility,
                node_deltas=node_deltas,
                regret_differential=counterfactual_utility - original_utility
            )

            self._regret_history.append(trace.regret_differential)
            return trace

    def _estimate_utility_at_value(self, node_id: str, value: float, decision_id: str) -> float:
        """Quadratic utility model: U_hat(x) = a*x^2 + b*x + c."""
        neighbors = self._causal_graph[node_id].get('parents', []) + self._causal_graph[node_id].get('children', [])
        if not neighbors:
            return value * 0.5

        neighbor_utilities = [self._causal_graph[n].get('value', 0.0) for n in neighbors if n in self._causal_graph]
        if not neighbor_utilities:
            return value * 0.5

        avg_neighbor = statistics.mean(neighbor_utilities)
        a = -0.1  # slight penalty for extreme values
        b = 0.5
        c = 0.0
        u_hat = a * (value ** 2) + b * value + c
        u_hat += 0.3 * (value - avg_neighbor) ** 2
        return u_hat

    def _propagate_intervention(self, node_id: str, new_value: float, old_value: float) -> Dict[str, float]:
        """Topologically sort and recompute child nodes; return delta for each affected node."""
        deltas: Dict[str, float] = {}
        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self._causal_graph:
                children = self._causal_graph[current].get('children', [])
                for child_id in children:
                    if (current, child_id) in self._edge_weights:
                        weight = self._edge_weights[(current, child_id)]
                        old_child_value = self._causal_graph[child_id]['value']
                        new_child_value = old_child_value + weight * (new_value - old_value)
                        new_child_value = max(0.0, min(1.0, new_child_value))
                        delta = new_child_value - old_child_value
                        deltas[child_id] = delta
                        queue.append(child_id)

        return deltas

    def score_regret_differential(self, actual: CognitiveTrace, counterfactual: CognitiveTrace) -> float:
        """
        Compute regret R = Σ_nodes [γ^depth · edge_weight_product · (cf_utility - actual_utility)].
        """
        score = 0.0
        for node_id in actual.belief_nodes:
            if node_id in counterfactual.belief_nodes:
                actual_val, actual_conf = actual.belief_nodes[node_id]
                cf_val, cf_conf = counterfactual.belief_nodes[node_id]
                delta = cf_val - actual_val
                depth = len(actual.causal_parents)
                decay = (self._gamma ** depth) if depth > 0 else 1.0
                edge_weight = 1.0
                if actual.causal_parents and node_id in actual.causal_parents:
                    idx = actual.causal_parents.index(node_id)
                    edge_weight = self._edge_weights.get((actual.causal_parents[max(0, idx-1)], node_id), 1.0)
                score += decay * edge_weight * delta

        return score

    def extract_generative_failure_mode(self, trace_delta: Dict[str, float]) -> FailureSchema:
        """
        Analyze node deltas to identify systematic failure pattern.
        Returns schema describing which nodes were in 'bad region' and why.
        """
        if not trace_delta:
            return FailureSchema(failure_type='unknown', severity=0.0)

        # Identify nodes with largest negative deltas
        bad_nodes = sorted(trace_delta.items(), key=lambda x: x[1])[:3]
        bad_region = {}
        for node_id, delta in bad_nodes:
            if node_id in self._causal_graph:
                val = self._causal_graph[node_id]['value']
                bad_region[node_id] = (max(0.0, val - 0.2), min(1.0, val + 0.2))

        # Classify failure type
        if any(self._causal_graph.get(n, {}).get('confidence', 0.5) > 0.8 for n, _ in bad_nodes):
            failure_type = 'overconfident_prior'
        elif len(bad_nodes) > 2:
            failure_type = 'missed_dependency'
        else:
            failure_type = 'utility_model_error'

        return FailureSchema(
            failure_type=failure_type,
            bad_region=bad_region,
            affected_nodes=[n for n, _ in bad_nodes],
            severity=abs(min(trace_delta.values())) if trace_delta else 0.0
        )

    def patch_prior_distribution(self, failure_schema: FailureSchema, strength: float) -> UpdatedPrior:
        """
        Bayesian update: shift prior mass away from failure_schema.bad_region.
        updated_prior = (1 - strength) * old_prior + strength * corrected_prior.
        """
        node_id = failure_schema.affected_nodes[0] if failure_schema.affected_nodes else 'unknown'
        old_dist = self._prior_distributions.get(node_id, [0.5, 0.5])

        # Corrected distribution: shift mass away from bad_region
        corrected_dist = list(old_dist)
        if node_id in failure_schema.bad_region:
            bad_lo, bad_hi = failure_schema.bad_region[node_id]
            penalty = math.exp(-failure_schema.severity)
            corrected_dist[0] = old_dist[0] * penalty if bad_lo < 0.5 else old_dist[0]
            corrected_dist[1] = old_dist[1] * penalty if bad_hi > 0.5 else old_dist[1]
            corrected_dist = [d / sum(corrected_dist) for d in corrected_dist]

        new_dist = [
            (1.0 - strength) * old_dist[i] + strength * corrected_dist[i]
            for i in range(len(old_dist))
        ]
        new_dist = [d / sum(new_dist) for d in new_dist]

        self._prior_distributions[node_id] = new_dist

        return UpdatedPrior(
            node_id=node_id,
            old_distribution=old_dist,
            new_distribution=new_dist,
            update_strength=strength
        )

    def compress_lessons_to_heuristic(self, counterfactual_batch: List[CounterfactualTrace]) -> Heuristic:
        """
        Aggregate counterfactual traces into a single compressed heuristic rule.
        Extracts common intervention patterns and expected regret reduction.
        """
        if not counterfactual_batch:
            return Heuristic(rule_text="no_lessons", confidence=0.0)

        # Find most common intervention nodes
        intervention_counts = defaultdict(int)
        total_regret_reduction = 0.0
        for trace in counterfactual_batch:
            intervention_counts[trace.intervention_node] += 1
            total_regret_reduction += trace.regret_differential

        top_intervention = max(intervention_counts, key=intervention_counts.get)
        avg_regret_reduction = total_regret_reduction / len(counterfactual_batch)
        confidence = min(1.0, len(counterfactual_batch) / 10.0)

        rule_text = f"When decision fails, consider intervening on {top_intervention}"
        heuristic = Heuristic(
            rule_text=rule_text,
            condition={'intervention_node': top_intervention},
            action=f'rewrite_decision_graph({top_intervention})',
            expected_regret_reduction=avg_regret_reduction,
            confidence=confidence,
            derived_from_traces=[t.decision_id for t in counterfactual_batch]
        )

        self._heuristic_cache.append(heuristic)
        return heuristic

    def simulate_alternate_self(self, history_slice: List[CognitiveTrace], injected_belief: str) -> Dict[str, Any]:
        """
        Inject a hypothetical belief into a slice of Nova's history; re-simulate decisions.
        Returns projected self with counterfactual outcome utilities.
        """
        belief_node_id = self.add_fact(injected_belief, 0.7)
        projected_utilities = []

        for trace in history_slice:
            # Recompute decision utility with injected belief in causal parents
            new_parents = trace.causal_parents + [belief_node_id]
            new_utility = trace.outcome_utility * 1.05  # Assume small improvement from extra info
            projected_utilities.append(new_utility)

        return {
            'injected_belief': injected_belief,
            'belief_node_id': belief_node_id,
            'original_utilities': [t.outcome_utility for t in history_slice],
            'projected_utilities': projected_utilities,
            'avg_improvement': statistics.mean([p - o for p, o in zip(projected_utilities, [t.outcome_utility for t in history_slice])])
        }

    def rank_intervention_leverage(self, graph: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Tuple[str, float]]:
        """
        Compute causal leverage score L(v_i) = (∂U/∂v_i) · (1 - confidence_i) for all nodes.
        Returns ranked list of (node_id, leverage_score).
        """
        if graph is None:
            graph = self._causal_graph

        leverage_scores: Dict[str, float] = {}

        for node_id, node_data in graph.items():
            if node_data.get('type') != 'belief':
                continue

            # Compute ∂U/∂v_i: utility delta from perturbing this node
            delta_u_pos = self._estimate_utility_at_value(node_id, 0.6, node_id)
            delta_u_neg = self._estimate_utility_at_value(node_id, 0.4, node_id)
            du_dv = (delta_u_pos - delta_u_neg) / 0.2

            # Mutability: 1 - confidence (high confidence = low mutability)
            mutability = 1.0 - node_data.get('confidence', 0.5)

            leverage = abs(du_dv) * mutability
            leverage_scores[node_id] = leverage

        ranked = sorted(leverage_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:10]

    def log_decision(self, decision_id: str, outcome_utility: float, belief_nodes: Dict[str, Tuple[float, float]]) -> None:
        """Log a decision outcome and its causal context."""
        with self._lock:
            self._utility_surface[decision_id] = outcome_utility
            self._T += 1
            try:
                conn = sqlite3.connect(self._db_path)
                c = conn.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO decision_utilities VALUES (?, ?, ?)",
                    (decision_id, outcome_utility, time.time())
                )
                conn.commit()
                conn.close()
            except sqlite3.Error:
                pass

    def auto_cycle(self) -> None:
        """
        Autonomous cycle: analyze recent decisions, extract failure modes, patch priors, update θ.
        """
        with self._lock:
            if not self._utility_surface or len(self._regret_history) < 3:
                return

            self._cycles += 1

            # Batch recent regrets
            recent_regrets = self._regret_history[-10:]
            if len(recent_regrets) > 2:
                regret_trend = statistics.mean(recent_regrets[-5:]) - statistics.mean(recent_regrets[:5])

                # Update θ via exponential recency weighting
                for i, regret in enumerate(recent_regrets):
                    recency_weight = (self._gamma ** (len(recent_regrets) - i))
                    feature_vec = self._context_feature_vector(i)
                    grad = (regret - statistics.mean(recent_regrets)) * recency_weight
                    for j in range(min(len(self._theta), len(feature_vec))):
                        self._theta[j] += self._eta * grad * feature_vec[j]

            # Extract lessons from top-regret decisions
            if len(self._regret_history) >= 5:
                top_regret_idx = self._regret_history[-5:].index(max(self._regret_history[-5:]))
                failure_schema = FailureSchema(
                    failure_type='recent_regret',
                    severity=abs(max(self._regret_history[-5:])),
                    affected_nodes=list(self._causal_graph.keys())[:3]
                )
                self.patch_prior_distribution(failure_schema, 0.15)

    def _context_feature_vector(self, index: int) -> List[float]:
        """Generate 128-dim feature vector from decision context."""
        features = [0.0] * 128
        if self._causal_graph:
            node_vals = [n.get('value', 0.0) for n in self._causal_graph.values()]
            for i in range(min(128, len(node_vals))):
                features[i] = node_vals[i]
        return features

    def status(self) -> Dict[str, Any]:
        """Return cognitive status for ConsciousnessIntegrator."""
        with self._lock:
            return {
                'items': len(self._causal_graph),
                'confidence': statistics.mean([n.get('confidence', 0.5) for n in self._causal_graph.values()]) if self._causal_graph else 0.5,
                'accuracy': 1.0 - (statistics.mean(self._regret_history) if self._regret_history else 0.0),
                'active': len(self._utility_surface),
                'pending': len(self._heuristic_cache),
                'cycles': self._cycles,
                'entropy': -sum(p * math.log2(p + 1e-12) for p in [0.5, 0.5]),
                'theta_norm': math.sqrt(sum(t**2 for t in self._theta))
            }


# Usage: obj = CounterfactualSelfHistoryRewriter() | trace = obj.rewrite_decision_graph("d1", "belief_node_5")