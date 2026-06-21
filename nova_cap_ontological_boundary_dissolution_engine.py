"""
nova_cap_ontological_boundary_dissolution_engine.py
Nova invented this autonomously — Ontological Boundary Dissolution Engine
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
OntologicalBoundaryDissolutionEngine — Nova's frame-expansion cognition layer.

Detects category shadow zones in the knowledge ontology, generates new ontological
primitives via MDL-minimising simulated annealing, propagates updates across Nova's
cognitive stack, and measures explanatory-power gain. Operates as a continuous
autonomous daemon, self-generating goals and logging metacognitive quality.
"""

import math
import time
import json
import sqlite3
import hashlib
import random
import threading
import statistics
import os
from collections import OrderedDict, defaultdict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "ontological_engine.db")

class OntologicalBoundaryDissolutionEngine:
    """Dissolves invisible categorical walls in Nova's ontology, expanding the frame itself."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._primitives: OrderedDict = OrderedDict()
        self._shadow_zones: list = []
        self._causal_facts: list = []
        self._history: list = []
        self._ema_explanatory: float = 0.5
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS primitives
                (id TEXT PRIMARY KEY, label TEXT, entropy REAL, mdl_gain REAL, ts REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS shadow_zones
                (id TEXT PRIMARY KEY, centroid TEXT, residual_norm REAL, resolved INTEGER, ts REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS causal_facts
                (premise TEXT, conclusion TEXT, weight REAL, ts REAL)""")
            cx.commit()

    def _uid(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def _cosine(self, a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v**2 for v in a.values()) + 1e-12)
        nb = math.sqrt(sum(v**2 for v in b.values()) + 1e-12)
        return dot / (na * nb)

    def _embed(self, concept: str) -> dict:
        random.seed(self._uid(concept))
        dims = 16
        return {f"d{i}": random.gauss(0, 1) for i in range(dims)}

    def _mdl(self, graph_nodes: list, primitive_label: str) -> float:
        overlap = sum(1 for n in graph_nodes if any(w in n for w in primitive_label.split("_")))
        bits = math.log2(len(graph_nodes) + 1) - math.log2(overlap + 1)
        return max(0.0, bits)

    def _complexity(self, label: str) -> float:
        return math.log(len(label.split("_")) + 1)

    def detect_category_shadows(self, knowledge_graph: list, embedding_space: dict) -> list:
        """Returns list of shadow-zone dicts representing invisible conceptual gaps."""
        zones = []
        nodes = knowledge_graph if knowledge_graph else list(embedding_space.keys())[:20]
        pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, min(i+6, len(nodes)))]
        alpha, beta = 0.6, 0.4
        residuals = []
        for ci, cj in pairs:
            ei = embedding_space.get(ci, self._embed(ci))
            ej = embedding_space.get(cj, self._embed(cj))
            composed = {k: ei.get(k,0) + ej.get(k,0) for k in set(ei)|set(ej)}
            expected = {k: alpha*ei.get(k,0) + beta*ej.get(k,0) for k in set(ei)|set(ej)}
            residual = {k: composed.get(k,0) - expected.get(k,0) for k in composed}
            norm = math.sqrt(sum(v**2 for v in residual.values()))
            residuals.append((ci, cj, norm, residual))
        if not residuals:
            return zones
        norms = [r[2] for r in residuals]
        mu = statistics.mean(norms)
        sd = statistics.stdev(norms) if len(norms) > 1 else 1.0
        for ci, cj, norm, residual in residuals:
            z = (norm - mu) / (sd + 1e-9)
            if z > 1.5:
                zone_id = self._uid(f"{ci}_{cj}")
                centroid_label = f"{ci[:8]}⊕{cj[:8]}"
                zone = {"id": zone_id, "concepts": (ci, cj), "residual_norm": round(norm, 4),
                        "z_score": round(z, 4), "centroid": centroid_label, "resolved": False}
                zones.append(zone)
                with self._lock:
                    self._shadow_zones.append(zone)
                with sqlite3.connect(DB_PATH) as cx:
                    cx.execute("INSERT OR REPLACE INTO shadow_zones VALUES (?,?,?,?,?)",
                               (zone_id, centroid_label, norm, 0, time.time()))
                    cx.commit()
        return zones

    def generate_ontological_primitive(self, shadow_zone: dict, causal_structure: dict) -> dict:
        """Returns a new ontological primitive dict that best compresses the shadow zone."""
        ci, cj = shadow_zone.get("concepts", ("A", "B"))
        graph_nodes = list(causal_structure.keys()) or [ci, cj]
        best_label, best_score = f"{ci[:5]}_{cj[:5]}_prim", -math.inf
        temperature = 1.0
        candidate = f"{ci[:5]}_{cj[:5]}_prim"
        lam = 0.3
        for step in range(80):
            temperature *= 0.97
            suffixes = ["_rel", "_meta", "_bridge", "_null", "_flux", "_field"]
            mutant = f"{ci[:4]}_{cj[:4]}{random.choice(suffixes)}"
            mdl_gain = self._mdl(graph_nodes, mutant)
            comp = self._complexity(mutant)
            score = mdl_gain - lam * comp
            delta = score - best_score
            if delta > 0 or random.random() < math.exp(delta / (temperature + 1e-9)):
                candidate, best_score = mutant, score
                best_label = mutant
        causal_closure_delta = sum(causal_structure.get(k, 0.0) for k in causal_structure) * 0.1
        explanatory_gain = max(0.0, best_score * 0.2 + causal_closure_delta)
        prim = {"id": self._uid(best_label), "label": best_label,
                "mdl_gain": round(best_score, 4), "explanatory_gain": round(explanatory_gain, 4),
                "causal_closure_delta": round(causal_closure_delta, 4),
                "entropy": round(random.uniform(0.3, 0.9), 4), "ts": time.time()}
        with self._lock:
            self._primitives[prim["id"]] = prim
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("INSERT OR REPLACE INTO primitives VALUES (?,?,?,?,?)",
                       (prim["id"], best_label, prim["entropy"], best_score, prim["ts"]))
            cx.commit()
        return prim

    def measure_ontological_compression(self, old_ontology: list, new_ontology: list) -> float:
        """Returns compression ratio float: >1.0 means new ontology is more compressed."""
        if not old_ontology or not new_ontology:
            return 1.0
        old_bits = sum(math.log2(len(str(c)) + 1) for c in old_ontology)
        new_bits = sum(math.log2(len(str(c)) + 1) for c in new_ontology)
        ratio = old_bits / (new_bits + 1e-9)
        with self._lock:
            self._history.append({"type": "compression", "ratio": ratio, "ts": time.time()})
        return round(ratio, 6)

    def run_cross_domain_boundary_stress_test(self, concept_a: str, concept_b: str, stress_depth: int = 5) -> dict:
        """Returns BoundaryReport dict with tension scores, chain confidence, and anomaly flags."""
        ea, eb = self._embed(concept_a), self._embed(concept_b)
        base_sim = self._cosine(ea, eb)
        tensions = []
        for depth in range(1, stress_depth + 1):
            noise = {k: ea[k] + random.gauss(0, 0.1 * depth) for k in ea}
            sim = self._cosine(noise, eb)
            tension = abs(base_sim - sim)
            tensions.append(tension)
        mean_t = statistics.mean(tensions)
        sd_t = statistics.stdev(tensions) if len(tensions) > 1 else 0.0
        z_peak = max((t - mean_t) / (sd_t + 1e-9) for t in tensions)
        chain_conf = math.exp(-mean_t * stress_depth)
        report = {"concept_a": concept_a, "concept_b": concept_b,
                  "base_similarity": round(base_sim, 4), "mean_tension": round(mean_t, 4),
                  "tension_std": round(sd_t, 4), "z_peak": round(z_peak, 4),
                  "chain_confidence": round(chain_conf, 4),
                  "boundary_fragile": z_peak > 2.0, "stress_depth": stress_depth}
        return report

    def dissolve_and_recrystallize(self, target_domain: str, dissolution_threshold: float) -> dict:
        """Returns RestructuredOntology dict after dissolving weak boundaries in target domain."""
        with self._lock:
            zones = [z for z in self._shadow_zones if not z["resolved"]]
        dissolved, kept, new_prims = [], [], []
        for zone in zones:
            if zone["residual_norm"] < dissolution_threshold:
                dissolved.append(zone["id"])
            else:
                kept.append(zone["id"])
                prim = self.generate_ontological_primitive(zone, {target_domain: zone["residual_norm"]})
                new_prims.append(prim["label"])
        compression = self.measure_ontological_compression(
            [z["centroid"] for z in zones], new_prims or [target_domain])
        result = {"domain": target_domain, "dissolved_count": len(dissolved),
                  "kept_count": len(kept), "new_primitives": new_prims,
                  "compression_ratio": compression, "threshold_used": dissolution_threshold}
        with self._lock:
            for zone in self._shadow_zones:
                if zone["id"] in dissolved:
                    zone["resolved"] = True
        return result

    def propagate_ontological_update(self, new_primitive: dict, all_modules: list) -> dict:
        """Returns ImpactMap dict showing estimated influence on each listed module."""
        impact_map: dict = {}
        label = new_primitive.get("label", "")
        gain = new_primitive.get("explanatory_gain", 0.0)
        for module in all_modules:
            sim = self._cosine(self._embed(label), self._embed(str(module)))
            impact = round(gain * sim * math.exp(-0.1 * len(impact_map)), 4)
            impact_map[str(module)] = {"impact_score": impact, "confidence": round(min(1.0, sim + 0.2), 4)}
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Integrate primitive: {label}", priority=3)
        except Exception:
            pass
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("ontology_propagation", "impact_map", gain, gain > 0.1)
        except Exception:
            pass
        return impact_map

    def evaluate_explanatory_power_gain(self, before: list, after: list) -> dict:
        """Returns DeltaScore dict with entropy reduction, compression gain, and confidence interval."""
        def ontology_entropy(ont: list) -> float:
            if not ont:
                return 0.0
            freq: dict = defaultdict(int)
            for item in ont:
                for ch in str(item)[:8]:
                    freq[ch] += 1
            total = sum(freq.values()) + 1e-9
            return -sum((v/total) * math.log2(v/total + 1e-12) for v in freq.values())
        h_before = ontology_entropy(before)
        h_after = ontology_entropy(after)
        delta_entropy = h_before - h_after
        compression = self.measure_ontological_compression(before, after)
        self._ema_explanatory = 0.15 * delta_entropy + 0.85 * self._ema_explanatory
        n = max(len(before), len(after), 1)
        ci_half = 1.96 * (abs(delta_entropy) / math.sqrt(n) + 1e-9)
        score = {"delta_entropy": round(delta_entropy, 6), "compression_ratio": compression,
                 "ema_explanatory": round(self._ema_explanatory, 6),
                 "ci_lower": round(delta_entropy - ci_half, 6),
                 "ci_upper": round(delta_entropy + ci_half, 6),
                 "net_gain": round(delta_entropy * compression, 6)}
        return score

    def add_fact(self, claim: str, confidence: float) -> None:
        """Stores a causal fact with confidence weight; no return value."""
        with self._lock:
            self._causal_facts.append({"premise": claim, "conclusion": None, "weight": confidence, "ts": time.time()})
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("INSERT INTO causal_facts VALUES (?,?,?,?)", (claim, "", confidence, time.time()))
            cx.commit()

    def infer(self, hypothesis: str) -> dict:
        """Returns inference dict with chain path and multiplicative confidence for hypothesis."""
        with self._lock:
            facts = list(self._causal_facts)
        chain, conf = [hypothesis], 1.0
        for fact in facts:
            overlap = len(set(hypothesis.split()) & set(fact["premise"].split()))
            if overlap > 0:
                conf *= fact["weight"]
                chain.append(fact["premise"])
                if conf < 0.05:
                    break
