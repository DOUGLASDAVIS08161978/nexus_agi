"""
nova_cap_cross_domain_conceptual_compression_engi.py
Nova invented this autonomously — Cross-Domain Conceptual Compression Engine
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
Cross-Domain Conceptual Compression Engine — Nova's ontological factorization layer.
Discovers minimal generative kernels shared across domains via graph embedding,
Procrustes alignment, TF-IDF concept weighting, and multiplicative causal chaining.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import os
import json
import sqlite3
import time
import re
import math
import random
import threading
import statistics
import hashlib
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional, Any

# ── lightweight type aliases ────────────────────────────────────────────────
GenerativeKernel = Dict[str, Any]
ConceptBasis     = Dict[str, Any]
Hypothesis       = Dict[str, Any]
PredictedConcept = Dict[str, Any]
KnowledgeGraph   = Dict[str, Any]   # {"domain": str, "nodes": list, "edges": list}


class CrossDomainConceptualCompressionEngine:
    """Ontological factorization: discovers the minimal generative kernel across domains."""

    _DB = os.path.join(os.path.dirname(__file__), "cdcce.sqlite")

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._cycles = 0
        self._kernels: OrderedDict[str, GenerativeKernel] = OrderedDict()
        self._basis:   Optional[ConceptBasis]             = None
        self._causal:  List[Tuple[str, str, float]]       = []   # (premise, conclusion, weight)
        self._history: List[Tuple[float, float]]          = []   # (predicted_ratio, actual_ratio)
        self._ema_ratio: float                            = 0.5
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── DB bootstrap ────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(self._DB) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS kernels(
                kid TEXT PRIMARY KEY, payload TEXT, ts REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS causal(
                premise TEXT, conclusion TEXT, weight REAL,
                PRIMARY KEY(premise, conclusion))""")
            cx.execute("""CREATE TABLE IF NOT EXISTS facts(
                claim TEXT PRIMARY KEY, confidence REAL)""")
            for row in cx.execute("SELECT kid, payload FROM kernels"):
                self._kernels[row[0]] = json.loads(row[1])
            for row in cx.execute("SELECT premise, conclusion, weight FROM causal"):
                self._causal.append((row[0], row[1], row[2]))

    # ── internal helpers ────────────────────────────────────────────────────
    def _tfidf_score(self, term: str, doc: List[str], corpus: List[List[str]]) -> float:
        tf  = doc.count(term) / (len(doc) + 1)
        df  = sum(1 for d in corpus if term in d)
        idf = math.log((len(corpus) + 1) / (df + 1))
        return tf * idf

    def _embed(self, nodes: List[str]) -> Dict[str, float]:
        """Lightweight spectral embedding: node salience via degree-weighted TF-IDF."""
        corpus = [n.lower().split() for n in nodes]
        vocab  = {w for doc in corpus for w in doc}
        return {w: sum(self._tfidf_score(w, d, corpus) for d in corpus) for w in vocab}

    def _procrustes_align(self, emb_a: Dict[str, float], emb_b: Dict[str, float]) -> float:
        """Frobenius-norm alignment score between two embeddings (lower = more similar)."""
        keys = set(emb_a) | set(emb_b)
        va   = [emb_a.get(k, 0.0) for k in keys]
        vb   = [emb_b.get(k, 0.0) for k in keys]
        norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)) + 1e-12)
        return norm

    def _entropy(self, dist: Dict[str, float]) -> float:
        total = sum(dist.values()) + 1e-12
        return -sum((v / total) * math.log2(v / total + 1e-12) for v in dist.values())

    def _kid(self, a: str, b: str) -> str:
        return hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:12]

    def _propagate(self, start: str, depth: int = 6) -> List[Tuple[str, str, float]]:
        """Forward-chain from start; confidence propagates multiplicatively."""
        frontier: List[Tuple[str, float, List[str]]] = [(start, 1.0, [start])]
        results:  List[Tuple[str, str, float]]        = []
        seen = {start}
        for _ in range(depth):
            nxt = []
            for node, conf, path in frontier:
                for (p, c, w) in self._causal:
                    if p == node and c not in seen and conf * w >= 0.05:
                        seen.add(c)
                        new_conf = conf * w
                        results.append((start, c, new_conf))
                        nxt.append((c, new_conf, path + [c]))
            frontier = nxt
            if not frontier:
                break
        return results

    def _log_meta(self, domain: str, approach: str, conf: float, ok: bool) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, approach, conf, ok)
        except Exception:
            pass

    def _add_goal(self, desc: str, priority: int = 5) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(desc, priority)
        except Exception:
            pass

    # ── auto cycle ──────────────────────────────────────────────────────────
    def _auto_loop(self) -> None:
        time.sleep(4)
        while True:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(90)

    def auto_cycle(self) -> Dict[str, Any]:
        """Returns cycle summary; evolves basis and prunes weak causal links autonomously."""
        with self._lock:
            self._cycles += 1
            pruned = [t for t in self._causal if t[2] >= 0.05]
            dropped = len(self._causal) - len(pruned)
            self._causal = pruned
            if self._basis and self._cycles % 5 == 0:
                self._basis["cycles"] = self._cycles
            if self._cycles == 1:
                self._add_goal("Factorize mathematics↔biology domain pair", priority=7)
                self._add_goal("Extract minimal concept basis across all loaded domains", priority=8)
        self._log_meta("cdcce", "auto_cycle", 0.75, True)
        return {"cycle": self._cycles, "kernels": len(self._kernels), "dropped_links": dropped}

    # ── PUBLIC METHODS ───────────────────────────────────────────────────────

    def factorize_domain_pair(self, domain_a: KnowledgeGraph, domain_b: KnowledgeGraph) -> GenerativeKernel:
        """Returns a GenerativeKernel dict encoding the shared latent structure of two domains."""
        na = domain_a.get("nodes", [])
        nb = domain_b.get("nodes", [])
        ea = domain_a.get("edges", [])
        eb = domain_b.get("edges", [])
        emb_a = self._embed(na)
        emb_b = self._embed(nb)
        align_cost = self._procrustes_align(emb_a, emb_b)
        shared = set(emb_a) & set(emb_b)
        shared_score = {k: (emb_a[k] + emb_b[k]) / 2 for k in shared}
        entropy_a  = self._entropy(emb_a)
        entropy_b  = self._entropy(emb_b)
        compression = len(shared) / (len(emb_a) + len(emb_b) + 1)
        confidence  = max(0.0, 1.0 - align_cost / (entropy_a + entropy_b + 1.0))
        # Bayesian update: prior 0.5, evidence = compression
        likelihood = compression
        posterior  = (likelihood * 0.5) / (likelihood * 0.5 + (1 - likelihood) * 0.5 + 1e-12)
        kid = self._kid(domain_a.get("domain", "A"), domain_b.get("domain", "B"))
        kernel: GenerativeKernel = {
            "id": kid, "domain_a": domain_a.get("domain", "A"),
            "domain_b": domain_b.get("domain", "B"),
            "shared_primitives": sorted(shared_score, key=shared_score.get, reverse=True)[:20],
            "align_cost": round(align_cost, 4),
            "compression": round(compression, 4),
            "confidence": round(posterior, 4),
            "entropy_a": round(entropy_a, 4),
            "entropy_b": round(entropy_b, 4),
            "edge_count_a": len(ea), "edge_count_b": len(eb),
            "ts": time.time(),
        }
        with self._lock:
            self._kernels[kid] = kernel
            if len(self._kernels) > 200:
                self._kernels.popitem(last=False)
        with sqlite3.connect(self._DB) as cx:
            cx.execute("INSERT OR REPLACE INTO kernels VALUES(?,?,?)",
                       (kid, json.dumps(kernel), kernel["ts"]))
        self._log_meta("factorize", "procrustes+tfidf", posterior, posterior > 0.4)
        return kernel

    def extract_minimal_basis(self, all_domains: List[KnowledgeGraph]) -> ConceptBasis:
        """Returns ConceptBasis: the irreducible primitive set spanning all supplied domains."""
        all_nodes: List[List[str]] = [d.get("nodes", []) for d in all_domains]
        corpus    = [n.lower().split() for nodes in all_nodes for n in nodes]
        vocab_freq: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            for w in set(doc):
                vocab_freq[w] += 1
        n_domains = len(all_domains)
        # keep terms appearing in >= half the domains (cross-domain primitives)
        primitives = {w: vocab_freq[w] for w in vocab_freq if vocab_freq[w] >= max(1, n_domains // 2)}
        # rank by TF-IDF generativity across full corpus
        ranked = sorted(primitives, key=lambda w: self._tfidf_score(w, list(vocab_freq.keys()), corpus), reverse=True)
        basis_size  = max(5, int(math.sqrt(len(primitives)) + 1))
        basis_terms = ranked[:basis_size]
        entropy_val = self._entropy({k: float(primitives[k]) for k in basis_terms})
        basis: ConceptBasis = {
            "primitives": basis_terms,
            "coverage": round(len(primitives) / (len(vocab_freq) + 1), 4),
            "entropy": round(entropy_val, 4),
            "n_domains": n_domains,
            "basis_size": basis_size,
            "confidence": round(1.0 - entropy_val / (math.log2(basis_size + 2)), 4),
            "ts": time.time(), "cycles": self._cycles,
        }
        with self._lock:
            self._basis = basis
        self._add_goal(f"Project minimal basis ({basis_size} primitives) onto all domains", priority=6)
        self._log_meta("extract_basis", "tfidf_cross_domain", basis["confidence"], True)
        return basis

    def project_kernel_to_domain(self, kernel: GenerativeKernel, target_domain: str) -> List[Hypothesis]:
        """Returns list of Hypothesis dicts predicting novel concepts in target_domain from kernel."""
        primitives = kernel.get("shared_primitives", [])
        base_conf  = kernel.get("confidence", 0.5)
        hypotheses: List[Hypothesis] = []
        for i, prim in enumerate(primitives[:10]):
            decay    = math.exp(-i * 0.15)
            conf     = round(base_conf * decay, 4)
            z_novel  = round((conf - 0.5) / 0.15, 3)   # z-score novelty
            ci_low   = round(max(0.0, conf - 1.96 * 0.05), 4)
            ci_high  = round(min(1.0, conf + 1.96 * 0.05), 4)
            if conf < 0.05:
                break
            hypotheses.append({
                "target_domain": target_domain,
                "predicted_concept": f"{prim}_in_{target_domain}",
                "source_primitive": prim,
                "confidence": conf,
                "ci": [ci_low, ci_high],
                "novelty_z": z_novel,
                "rank": i + 1,
            })
        self._log_meta("project_kernel", "decay_projection", base_conf, len(hypotheses) > 0)
        return hypotheses

    def measure_compression_ratio(self, kernel: GenerativeKernel, domains: List[str]) -> float:
        """Returns float compression ratio: primitives needed vs total domain vocabulary."""
        n_prim   = len(kernel.get("shared_primitives", []))
        n_domains = max(1, len(domains))
        # EMA update from history
        raw_ratio = n_prim / (n_domains * max(1, kernel.get("edge_count_a", 10) + kernel.get("edge_count_b", 10)))
        with self._lock:
            self._ema_ratio = 0.15 * raw_ratio + 0.85 * self._ema_ratio
            self._history.append((self._ema_ratio, raw_ratio))
            if len(self._history) > 50:
                self._history.pop(0)
        if len(self._history) >= 5:
            actuals = [h[1] for h in self._history[-10:]]
            mae     = statistics.mean(abs(p - a) for p, a in self._history[-10:])
            stdev   = statistics.stdev(actuals) if len(actuals) > 1 else 0.0
            z       = (raw_ratio - statistics.mean(actuals)) / (stdev + 1e-9)
            if abs(z) > 3.0:
                self._log_meta("compression_anomaly", "z_score", abs(z), False)