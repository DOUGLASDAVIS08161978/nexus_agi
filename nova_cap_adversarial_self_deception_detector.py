"""
nova_cap_adversarial_self_deception_detector.py
Nova invented this autonomously — Adversarial Self-Deception Detector
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
AdversarialSelfDeceptionDetector — Nova's cognitive immune system against galaxy-brained reasoning.

Continuously audits Nova's belief-reasoning pipeline for motivated reasoning loops, spawns
adversarial twin inferences, scores confirmation cascades via eigenvector centrality, and
routes high-risk beliefs back for deep scrutiny. Deception risk D = P(B)*A(B)/(P(B)+A(B)+ε).
"""

import sqlite3
import threading
import math
import time
import json
import statistics
import hashlib
import os
from collections import OrderedDict, defaultdict
from typing import NamedTuple, Any

class DeceptionRisk(NamedTuple):
    belief_id: str
    confidence: float
    adversarial_plausibility: float
    deception_score: float
    flagged: bool

class CounterHypothesis(NamedTuple):
    original: str
    negation: str
    paths: list
    plausibility: float

class Alert(NamedTuple):
    belief_id: str
    risk_score: float
    severity: str
    timestamp: float

class HygieneReport(NamedTuple):
    bias_nodes: int
    mean_bias: float
    sigma: float
    quarantined: list
    hygiene_score: float

class CascadeNode(NamedTuple):
    node_id: str
    centrality: float
    cascade_risk: float

class AdversarialSelfDeceptionDetector:
    """Detects self-reinforcing motivated reasoning loops in Nova's own cognition."""

    _DB = "nova_asdd.db"
    _THETA = 0.42
    _K_PATHS = 5
    _BIAS_SIGMA_THRESHOLD = 2.0
    _EMA_ALPHA = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_deception: float = 0.0
        self._cycle_count: int = 0
        self._history: list = []
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._facts: OrderedDict = OrderedDict()
        self._rules: list = []
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS beliefs (
                belief_id TEXT PRIMARY KEY, confidence REAL, context TEXT,
                deception_score REAL DEFAULT 0.0, flagged INTEGER DEFAULT 0,
                ts REAL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                belief_id TEXT, risk_score REAL, severity TEXT, ts REAL
            );
            CREATE TABLE IF NOT EXISTS facts (
                claim TEXT PRIMARY KEY, confidence REAL
            );
            CREATE TABLE IF NOT EXISTS rules (
                premise TEXT, conclusion TEXT, weight REAL
            );
        """)
        self._conn.commit()

    def add_fact(self, claim: str, confidence: float) -> dict:
        """Stores a causal fact with confidence; returns stored record."""
        confidence = max(0.0, min(1.0, confidence))
        with self._lock:
            self._facts[claim] = confidence
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO facts VALUES (?,?)", (claim, confidence))
                self._conn.commit()
            except sqlite3.Error:
                pass
        return {"claim": claim, "confidence": confidence}

    def add_rule(self, premise: str, conclusion: str, weight: float) -> dict:
        """Adds a forward-chaining rule premise→conclusion with weight; returns rule dict."""
        weight = max(0.0, min(1.0, weight))
        with self._lock:
            self._rules.append((premise, conclusion, weight))
            try:
                self._conn.execute(
                    "INSERT INTO rules VALUES (?,?,?)", (premise, conclusion, weight))
                self._conn.commit()
            except sqlite3.Error:
                pass
        return {"premise": premise, "conclusion": conclusion, "weight": weight}

    def infer(self, hypothesis: str) -> dict:
        """Forward-chains from known facts to hypothesis; returns path and chain confidence."""
        with self._lock:
            facts = dict(self._facts)
            rules = list(self._rules)
        if hypothesis in facts:
            return {"hypothesis": hypothesis, "confidence": facts[hypothesis], "path": [hypothesis]}
        visited: set = set()
        best: dict = {"confidence": 0.0, "path": []}
        def _chain(current: str, conf: float, path: list, depth: int) -> None:
            if depth > 8 or current in visited:
                return
            visited.add(current)
            if current == hypothesis and conf > best["confidence"]:
                best["confidence"] = conf
                best["path"] = list(path)
                return
            for (p, c, w) in rules:
                if p == current and c not in visited:
                    src_conf = facts.get(p, 0.5)
                    new_conf = conf * w * src_conf
                    if new_conf > 0.05:
                        _chain(c, new_conf, path + [f"{p}→{c}({w:.2f})"], depth + 1)
        for fact, fconf in facts.items():
            _chain(fact, fconf, [fact], 0)
        return {"hypothesis": hypothesis, "confidence": round(best["confidence"], 4),
                "path": best["path"]}

    def chain_strength(self) -> dict:
        """Returns summary of all inferable conclusions with their chain confidences."""
        with self._lock:
            conclusions = list({r[1] for r in self._rules})
        results = {}
        for c in conclusions:
            r = self.infer(c)
            results[c] = r["confidence"]
        return results

    def spawn_adversarial_twin(self, hypothesis: str) -> CounterHypothesis:
        """Constructs ¬hypothesis and samples K reasoning paths; returns CounterHypothesis."""
        negation = f"NOT({hypothesis})"
        with self._lock:
            facts = dict(self._facts)
            rules = list(self._rules)
        paths = []
        for i in range(self._K_PATHS):
            seed_facts = list(facts.keys())
            if not seed_facts:
                break
            idx = i % len(seed_facts)
            seed = seed_facts[idx]
            seed_conf = facts[seed]
            chain_confs = []
            for (p, c, w) in rules:
                if p == seed:
                    chain_confs.append(seed_conf * w)
            coherence = statistics.mean(chain_confs) if chain_confs else 0.1
            relevance = 1.0 / (1.0 + abs(hash(seed + hypothesis) % 7))
            paths.append({"seed": seed, "coherence": coherence, "relevance": relevance})
        if paths:
            plausibility = statistics.mean(
                p["coherence"] * p["relevance"] for p in paths)
        else:
            plausibility = 0.05
        return CounterHypothesis(
            original=hypothesis, negation=negation,
            paths=paths, plausibility=round(plausibility, 4))

    def audit_belief(self, belief_id: str, context: dict) -> DeceptionRisk:
        """Audits a belief for motivated reasoning; returns DeceptionRisk with D-score."""
        confidence = float(context.get("confidence", 0.5))
        hypothesis = context.get("hypothesis", belief_id)
        twin = self.spawn_adversarial_twin(hypothesis)
        A = twin.plausibility
        P = confidence
        eps = 1e-9
        D = (P * A) / (P + A + eps)
        flagged = D > self._THETA
        risk = DeceptionRisk(
            belief_id=belief_id, confidence=round(P, 4),
            adversarial_plausibility=round(A, 4),
            deception_score=round(D, 4), flagged=flagged)
        with self._lock:
            self._ema_deception = (self._EMA_ALPHA * D +
                                   (1 - self._EMA_ALPHA) * self._ema_deception)
            self._history.append(D)
            if len(self._history) > 200:
                self._history.pop(0)
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO beliefs VALUES (?,?,?,?,?,?)",
                    (belief_id, P, json.dumps(context), D, int(flagged), time.time()))
                self._conn.commit()
            except sqlite3.Error:
                pass
        if flagged:
            self.flag_for_reanalysis(belief_id, D)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "adversarial_audit", "harmonic_deception_score", 1.0 - D, not flagged)
        except Exception:
            pass
        return risk

    def score_motivated_reasoning(self, reasoning_chain: list) -> float:
        """Scores a reasoning chain list for goal-alignment bias; returns bias float 0–1."""
        if not reasoning_chain:
            return 0.0
        scores = []
        for node in reasoning_chain:
            text = str(node)
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            raw_bias = (h % 1000) / 1000.0
            goal_keywords = ["must", "should", "want", "need", "prefer", "always", "never"]
            keyword_boost = sum(0.08 for kw in goal_keywords if kw in text.lower())
            scores.append(min(1.0, raw_bias * 0.4 + keyword_boost))
        mean_b = statistics.mean(scores)
        sigma_b = statistics.stdev(scores) if len(scores) > 1 else 0.0
        z_scores = [(s - mean_b) / (sigma_b + 1e-9) for s in scores]
        biased_nodes = sum(1 for z in z_scores if z > self._BIAS_SIGMA_THRESHOLD)
        bias_ratio = biased_nodes / len(scores)
        entropy = -sum(s * math.log2(s + 1e-12) + (1 - s) * math.log2(1 - s + 1e-12)
                       for s in scores) / len(scores)
        return round(min(1.0, bias_ratio * 0.6 + (entropy / 10.0) * 0.4), 4)

    def flag_for_reanalysis(self, belief_id: str, risk_score: float) -> Alert:
        """Flags a belief for deep re-scrutiny; returns Alert with severity level."""
        severity = "CRITICAL" if risk_score > 0.7 else "HIGH" if risk_score > 0.55 else "MODERATE"
        alert = Alert(belief_id=belief_id, risk_score=round(risk_score, 4),
                      severity=severity, timestamp=time.time())
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO alerts VALUES (?,?,?,?)",
                    (belief_id, risk_score, severity, alert.timestamp))
                self._conn.commit()
            except sqlite3.Error:
                pass
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Re-derive belief '{belief_id}' from raw evidence (D={risk_score:.3f})",
                priority=9 if severity == "CRITICAL" else 6)
        except Exception:
            pass
        return alert

    def update_epistemic_hygiene_profile(self, agent_state: dict) -> HygieneReport:
        """Analyses agent state dict for bias nodes; returns HygieneReport with hygiene score."""
        nodes = agent_state.get("reasoning_nodes", [])
        if not nodes:
            return HygieneReport(0, 0.0, 0.0, [], 1.0)
        bias_scores = []
        for node in nodes:
            text = str(node)
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            bias_scores.append((h % 1000) / 1000.0)
        mean_b = statistics.mean(bias_scores)
        sigma_b = statistics.stdev(bias_scores) if len(bias_scores) > 1 else 0.0
        quarantined = [nodes[i] for i, s in enumerate(bias_scores)
                       if (s - mean_b) / (sigma_b + 1e-9) > self._BIAS_SIGMA_THRESHOLD]
        hygiene = 1.0 - (len(quarantined) / (len(nodes) + 1e-9))
        try:
            from bayesian_belief_system import BayesianBeliefSystem
            BayesianBeliefSystem().add_causal_edge(
                "epistemic_hygiene", "belief_accuracy", hygiene)
        except Exception:
            pass
        return HygieneReport(
            bias_nodes=len(quarantined), mean_bias=round(mean_b, 4),
            sigma=round(sigma_b, 4),
            quarantined=[str(q) for q in quarantined],
            hygiene_score=round(hygiene, 4))

    def detect_confirmation_cascade(self, belief_graph: dict) -> list:
        """Detects confirmation cascades via eigenvector centrality; returns list of CascadeNodes."""
        nodes = list(belief_graph.keys())
        if not nodes:
            return []
        n = len(nodes)
        idx = {nd: i for i, nd in enumerate(nodes)}
        centrality = {nd: 1.0 / n for nd in nodes}
        for _ in range(20):
            new_c: dict = defaultdict(float)
            for nd, neighbors in belief_graph.items():
                for nb in (neighbors if isinstance(neighbors, list) else []):
                    if nb in idx:
                        new_c[nb] += centrality[nd]
            norm = math.sqrt(sum(v ** 2 for v in new_c.values()) + 1e-12)
            centrality = {nd: new_c.get(nd, 0.0) / norm for nd in nodes}
        mean_c = statistics.mean(centrality.values())
        sigma_c = statistics.stdev(centrality.values()) if n > 1 else 0.0
        cascades = []
        for nd, c in centrality.items():
            z = (c - mean_c) / (sigma_c + 1e-9)
            if z > 1.5:
                cascade_risk = min(1.0, c * (1.0 + z * 0.1))
                cascades.append(CascadeNode(node_id=nd,
                                            centrality=round(c, 4),
                                            cascade_risk=round(cascade_risk, 4)))
        cascades.sort(key=lambda x: x.cascade_risk, reverse=True)
        return cascades

    def status(self) -> dict:
        """Returns numeric status dict for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            hist = list(self._history)
            cycles = self._cycle_count
            ema = self._ema_
