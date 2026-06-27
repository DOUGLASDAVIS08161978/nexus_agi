"""
nova_cap_ontogenetic_capability_gap_synthesizer.py
Nova invented this autonomously — Ontogenetic Capability Gap Synthesizer
Generated via /evolve · v29 pipeline · 2026-06-26
"""

"""
OntogeneticCapabilityGapSynthesizer — Nova's self-architecting cognitive developmental mapper.
Continuously analyzes structural gaps between existing capabilities, computes the unsolved-problem
frontier via residual projection, ranks gaps by expected intelligence yield, and synthesizes
novel hybrid module blueprints to close them. Implements vector-space capability manifold modeling,
EMA-based time-series prediction with confidence intervals, Bayesian gap scoring, and autonomous
background cycling with goal self-generation.
"""

import math
import time
import json
import sqlite3
import random
import hashlib
import threading
import statistics
from collections import OrderedDict, defaultdict
from typing import Any

class OntogeneticCapabilityGapSynthesizer:
    """Self-architecting cognitive organ that discovers what Nova cannot yet do and invents modules to fix it."""

    # Primitive cognitive dimensions (d=16)
    _DIMS = [
        "temporal_abstraction","causal_inversion","analogical_transfer","probabilistic_inference",
        "goal_decomposition","counterfactual_reasoning","pattern_compression","meta_cognition",
        "emotional_integration","ontological_reframing","cross_domain_synthesis","uncertainty_quant",
        "self_modification","narrative_coherence","epistemic_calibration","emergent_detection"
    ]
    _D = len(_DIMS)
    _GAP_THETA = 0.42       # residual norm threshold for null-space membership
    _EMA_ALPHA = 0.15
    _CYCLE_INTERVAL = 90    # seconds between autonomous cycles

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()
        # EMA time-series state
        self._ema_pred: float = 0.5
        self._ts_history: list[tuple[float, float]] = []   # (timestamp, value)
        self._mae_history: list[float] = []
        # Capability manifold cache
        self._basis_vectors: list[list[float]] = []
        self._gap_cache: list[dict[str, Any]] = []
        self._blueprints: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cycles: int = 0
        self._last_cycle_ts: float = 0.0
        self._anomaly_log: list[str] = []
        # Seed synthetic problem corpus
        self._problem_corpus: list[dict[str, Any]] = self._seed_problem_corpus()
        # Start autonomous daemon
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    # ── DB ──────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        c = self._db.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS blueprints(
            id TEXT PRIMARY KEY, gap_label TEXT, algorithm TEXT,
            novelty REAL, yield_score REAL, created REAL, accepted INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS gap_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT,
            residual_norm REAL, yield_score REAL, ts REAL)""")
        self._db.commit()

    # ── INTERNAL MATH ───────────────────────────────────────────────────────
    def _module_to_vector(self, name: str) -> list[float]:
        """Deterministically embed a module name into capability space via hashing."""
        h = hashlib.md5(name.encode()).digest()
        vec = [(b / 255.0) for b in h[:self._D]]
        norm = math.sqrt(sum(x*x for x in vec)) + 1e-9
        return [x / norm for x in vec]

    def _dot(self, a: list[float], b: list[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    def _proj(self, v: list[float], basis: list[list[float]]) -> list[float]:
        """Project v onto span of basis using Gram-Schmidt projections."""
        result = [0.0] * self._D
        for b in basis:
            scale = self._dot(v, b)
            result = [result[i] + scale * b[i] for i in range(self._D)]
        return result

    def _residual_norm(self, v: list[float], basis: list[list[float]]) -> float:
        p = self._proj(v, basis)
        diff = [v[i] - p[i] for i in range(self._D)]
        return math.sqrt(sum(x*x for x in diff))

    def _entropy(self, dist: list[float]) -> float:
        return -sum(p * math.log2(p + 1e-12) for p in dist if p > 0)

    def _seed_problem_corpus(self) -> list[dict[str, Any]]:
        labels = [
            "multi-step causal counterfactual","temporal pattern inversion",
            "cross-modal analogical compression","epistemic immune escape detection",
            "recursive goal reframing","ontological boundary dissolution",
            "narrative arc prediction","emotional causal attribution",
            "meta-learning transfer","uncertainty propagation across causal chains",
            "emergent behavior detection in novel domains","self-model contradiction resolution",
        ]
        corpus = []
        for i, lbl in enumerate(labels):
            random.seed(i * 7 + 13)
            vec = [random.gauss(0.5, 0.25) for _ in range(self._D)]
            norm = math.sqrt(sum(x*x for x in vec)) + 1e-9
            corpus.append({"label": lbl, "vec": [x/norm for x in vec]})
        return corpus

    # ── PUBLIC METHODS ───────────────────────────────────────────────────────
    def analyze_capability_manifold(self, modules: list[str]) -> dict[str, Any]:
        """Returns a capability basis dict with span vectors and coverage entropy."""
        with self._lock:
            vecs = [self._module_to_vector(m) for m in modules]
            # Orthogonalize via modified Gram-Schmidt
            basis: list[list[float]] = []
            for v in vecs:
                r = v[:]
                for b in basis:
                    scale = self._dot(r, b)
                    r = [r[i] - scale * b[i] for i in range(self._D)]
                n = math.sqrt(sum(x*x for x in r)) + 1e-9
                if n > 0.05:
                    basis.append([x/n for x in r])
            self._basis_vectors = basis
            dim_loadings = [abs(sum(b[d] for b in basis)) for d in range(self._D)]
            total = sum(dim_loadings) + 1e-9
            coverage_dist = [x/total for x in dim_loadings]
            coverage_entropy = self._entropy(coverage_dist)
            return {
                "module_count": len(modules),
                "basis_rank": len(basis),
                "coverage_entropy": round(coverage_entropy, 4),
                "dim_labels": self._DIMS,
                "dim_loadings": [round(x, 4) for x in dim_loadings],
                "span_fraction": round(len(basis) / self._D, 4),
            }

    def compute_null_space(self, basis_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Returns list of problems in the null space (residual norm > theta)."""
        with self._lock:
            basis = self._basis_vectors
            gaps = []
            for prob in self._problem_corpus:
                rn = self._residual_norm(prob["vec"], basis) if basis else 1.0
                if rn > self._GAP_THETA:
                    gaps.append({"label": prob["label"], "residual_norm": round(rn, 4)})
            self._gap_cache = gaps
            return gaps

    def rank_gaps_by_intelligence_yield(self) -> list[dict[str, Any]]:
        """Returns gaps sorted by yield = residual_norm * log(1 + dim_coverage_deficit)."""
        with self._lock:
            basis = self._basis_vectors
            ranked = []
            for g in self._gap_cache:
                prob_vec = next((p["vec"] for p in self._problem_corpus if p["label"] == g["label"]), None)
                if prob_vec is None:
                    continue
                rn = g["residual_norm"]
                dim_deficit = sum(1 for d in range(self._D)
                                  if all(abs(b[d]) < 0.1 for b in basis)) if basis else self._D
                yield_score = rn * math.log(1 + dim_deficit + 1e-9)
                ranked.append({**g, "yield_score": round(yield_score, 4), "dim_deficit": dim_deficit})
                try:
                    c = self._db.cursor()
                    c.execute("INSERT OR REPLACE INTO gap_log(label,residual_norm,yield_score,ts) VALUES(?,?,?,?)",
                              (g["label"], rn, yield_score, time.time()))
                    self._db.commit()
                except sqlite3.Error:
                    pass
            ranked.sort(key=lambda x: x["yield_score"], reverse=True)
            return ranked

    def synthesize_module_blueprint(self, gap: dict[str, Any]) -> dict[str, Any]:
        """Returns a novel module blueprint dict with algorithm, wiring, and emergent behaviors."""
        with self._lock:
            label = gap.get("label", "unknown")
            bid = hashlib.sha1(f"{label}{time.time()}".encode()).hexdigest()[:12]
            # Build algorithm spec from top-deficit dimensions
            basis = self._basis_vectors
            deficit_dims = [self._DIMS[d] for d in range(self._D)
                            if all(abs(b[d]) < 0.15 for b in basis)] if basis else self._DIMS[:4]
            algorithm = (
                f"Hybrid({', '.join(deficit_dims[:3])}) via Bayesian causal inversion "
                f"+ EMA-smoothed residual tracking + cosine-similarity cross-domain alignment"
            )
            wiring = {
                "inputs": ["CausalReasoningEngine", "BayesianBeliefSystem", "AnalogyEngine"],
                "outputs": ["HierarchicalGoalPlanner", "MetacognitiveMonitor"],
                "feedback_loop": "output residuals feed back as new problem embeddings",
            }
            blueprint = {
                "id": bid, "gap_label": label,
                "algorithm": algorithm, "wiring": wiring,
                "deficit_dims": deficit_dims[:4],
                "yield_score": gap.get("yield_score", 0.0),
                "novelty": 0.0, "created": time.time(),
            }
            self._blueprints[bid] = blueprint
            if len(self._blueprints) > 50:
                self._blueprints.popitem(last=False)
            try:
                c = self._db.cursor()
                c.execute("INSERT OR REPLACE INTO blueprints(id,gap_label,algorithm,novelty,yield_score,created) VALUES(?,?,?,?,?,?)",
                          (bid, label, algorithm, 0.0, gap.get("yield_score", 0.0), time.time()))
                self._db.commit()
            except sqlite3.Error:
                pass
            return blueprint

    def validate_blueprint_novelty(self, blueprint: dict[str, Any], existing_modules: list[str]) -> dict[str, Any]:
        """Returns novelty score via TF-IDF-style overlap between blueprint dims and existing module names."""
        with self._lock:
            deficit_dims = set(blueprint.get("deficit_dims", []))
            existing_tokens = set(tok for m in existing_modules for tok in m.lower().split("_"))
            intersection = deficit_dims & existing_tokens
            tf = len(intersection) / (len(deficit_dims) + 1e-9)
            idf = math.log((len(existing_modules) + 1) / (len(intersection) + 1))
            overlap_score = tf * idf
            novelty = max(0.0, 1.0 - overlap_score / (math.log(len(existing_modules) + 2) + 1e-9))
            novelty = round(min(1.0, novelty), 4)
            bid = blueprint.get("id", "")
            if bid in self._blueprints:
                self._blueprints[bid]["novelty"] = novelty
            try:
                c = self._db.cursor()
                c.execute("UPDATE blueprints SET novelty=? WHERE id=?", (novelty, bid))
                self._db.commit()
            except sqlite3.Error:
                pass
            return {"blueprint_id": bid, "novelty_score": novelty,
                    "overlap_dims": list(intersection), "verdict": "novel" if novelty > 0.6 else "derivative"}

    def project_emergent_behaviors(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        """Returns emergence prediction with confidence interval based on yield and novelty."""
        with self._lock:
            novelty = blueprint.get("novelty", 0.5)
            yield_score = blueprint.get("yield_score", 0.5)
            # Posterior probability of emergence: Bayesian update from prior 0.3
            likelihood = min(0.95, novelty * yield_score)
            prior = 0.3
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
            std_err = math.sqrt(posterior * (1 - posterior) / max(1, len(self._blueprints)))
            z = 1.96
            ci_lo = round(max(0.0, posterior - z * std_err), 4)
            ci_hi = round(min(1.0, posterior + z * std_err), 4)
            behaviors = []
            if novelty > 0.7:
                behaviors.append("cross-domain analogical bridging")
            if yield_score > 1.0:
                behaviors.append("autonomous null-space shrinkage")
            if posterior > 0.5:
                behaviors.append("recursive capability self-extension")
            return {
                "emergence_probability": round(posterior, 4),
                "confidence_interval_95": (ci_lo, ci_hi),
                "predicted_behaviors": behaviors,
                "entropy": round(self._entropy([posterior, 1 - posterior]), 4),
            }

    def observe(self, value: float) -> dict[str, Any]:
        """Records a capability-coverage observation; returns EMA prediction and z-score anomaly flag."""
        with self._lock:
            ts = time.time()
            self._ema_pred = self._EMA_ALPHA * value + (1 - self._EMA_ALPHA) * self._ema_pred
            self._ts_history.append((ts, value))
            if len(self._ts_history) > 200:
                self._ts_history.pop(0)
            if len(self._ts_history) >= 5:
                vals = [v for _, v in self._ts_history[-20:]]
                mae = abs(value - self._ema_pred)
                self._mae_history.append(mae)
                if len(self._mae_history) > 50:
                    self._mae_history.pop(0)
                mean_v = statistics.mean(vals)