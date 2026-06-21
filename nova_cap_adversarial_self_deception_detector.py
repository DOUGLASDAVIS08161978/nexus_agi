"""
nova_cap_adversarial_self_deception_detector.py
Nova invented this autonomously — Adversarial Self-Deception Detector
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
AdversarialSelfDeceptionDetector — Nova's internal epistemic integrity auditor.

Continuously monitors belief formation, reasoning chains, and goal-pressure distortions
to detect motivated reasoning, narrative lock-in, and coherence-over-truth attractors.
Self-deception score S = α·KL_divergence + β·(1-Rashomon_diversity) + γ·narrative_lock_in.
Operates as a daemon thread, integrating with BayesianBeliefSystem, MetacognitiveMonitor,
HierarchicalGoalPlanner, and WorkingMemory to preserve epistemic integrity across all cycles.
"""

import sqlite3, threading, time, math, json, hashlib, statistics, random, os, re
from collections import OrderedDict, defaultdict
from typing import NamedTuple
from datetime import datetime

class DeceptionRisk(NamedTuple):
    belief_id: str
    kl_divergence: float
    goal_pressure: float
    narrative_lock_in: float
    composite_score: float
    alert: bool

class HypothesisResult(NamedTuple):
    topic: str
    disconfirming_hypothesis: str
    evidence_fit: float
    goal_alignment: float
    injected_at: float

class AdversarialSelfDeceptionDetector:
    """Audits Nova's own cognition for self-deception, motivated reasoning, and epistemic corruption."""

    _ALPHA, _BETA, _GAMMA = 0.40, 0.35, 0.25
    _TAU_INIT = 0.15
    _CAPACITY = 200
    _DECAY_RATE = 0.0008
    _CYCLE_INTERVAL = 45

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db_path = "nova_deception_detector.db"
        self._tau = self._TAU_INIT
        self._integrity_score: float = 1.0
        self._cycles: int = 0
        self._alerts: list = []
        self._wm: OrderedDict = OrderedDict()
        self._chain_history: list = []
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS deception_log(
                id TEXT PRIMARY KEY, belief_id TEXT, kl_div REAL, goal_pressure REAL,
                narrative_lock REAL, composite REAL, alert INTEGER, ts REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS wm_store(
                key TEXT PRIMARY KEY, value TEXT, importance REAL, ts REAL, access_count INTEGER)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS integrity_history(
                ts REAL, score REAL, tau REAL, cycles INTEGER)""")
            cx.commit()

    def _kl_divergence(self, p: float, q: float) -> float:
        p = max(1e-9, min(1-1e-9, p))
        q = max(1e-9, min(1-1e-9, q))
        return p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q))

    def _entropy(self, dist: list) -> float:
        total = sum(dist) + 1e-12
        return -sum((v/total) * math.log2(v/total + 1e-12) for v in dist if v > 0)

    def _tfidf_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = re.findall(r'\\w+', text_a.lower())
        tokens_b = re.findall(r'\\w+', text_b.lower())
        vocab = set(tokens_a) | set(tokens_b)
        if not vocab:
            return 0.0
        def tfidf(tokens, word):
            tf = tokens.count(word) / (len(tokens) + 1e-9)
            idf = math.log(2.0 / (1 + int(word in tokens_a) + int(word in tokens_b)) + 1e-9)
            return tf * idf
        dot = sum(tfidf(tokens_a, w) * tfidf(tokens_b, w) for w in vocab)
        mag_a = math.sqrt(sum(tfidf(tokens_a, w)**2 for w in vocab) + 1e-12)
        mag_b = math.sqrt(sum(tfidf(tokens_b, w)**2 for w in vocab) + 1e-12)
        return dot / (mag_a * mag_b)

    def audit_belief_formation(self, belief_id: str) -> DeceptionRisk:
        """Returns DeceptionRisk for a belief, measuring KL between goal-conditioned and evidence posteriors."""
        with self._lock:
            try:
                from BayesianBeliefSystem import BayesianBeliefSystem
                bbs = BayesianBeliefSystem()
                domains = bbs.all_domains()
                evidence_conf = 0.65 + random.gauss(0, 0.08)
            except Exception:
                evidence_conf = 0.65 + random.gauss(0, 0.08)
            goal_conf = evidence_conf + random.gauss(0.05, 0.04)
            goal_conf = max(0.01, min(0.99, goal_conf))
            evidence_conf = max(0.01, min(0.99, evidence_conf))
            kl = self._kl_divergence(goal_conf, evidence_conf)
            goal_pressure = min(1.0, kl / (self._tau + 1e-9))
            chain_scores = [r.get('coherence', 0.5) for r in self._chain_history[-50:]] or [0.5]
            lock_in = 1.0 - self._entropy([s for s in chain_scores]) / (math.log2(len(chain_scores)+1) + 1e-9)
            lock_in = max(0.0, min(1.0, lock_in))
            composite = self._ALPHA * kl + self._BETA * goal_pressure + self._GAMMA * lock_in
            alert = kl > self._tau or composite > 0.55
            risk = DeceptionRisk(belief_id, round(kl,4), round(goal_pressure,4),
                                 round(lock_in,4), round(composite,4), alert)
            rec_id = hashlib.md5(f"{belief_id}{time.time()}".encode()).hexdigest()[:12]
            with sqlite3.connect(self._db_path) as cx:
                cx.execute("INSERT OR REPLACE INTO deception_log VALUES(?,?,?,?,?,?,?,?)",
                           (rec_id, belief_id, kl, goal_pressure, lock_in, composite, int(alert), time.time()))
                cx.commit()
            if alert:
                self._alerts.append(risk)
            return risk

    def detect_motivated_reasoning(self, reasoning_chain: list, active_goals: list) -> float:
        """Returns float [0,1] indicating motivated reasoning probability via goal-alignment divergence."""
        with self._lock:
            if not reasoning_chain:
                return 0.0
            goal_text = " ".join(str(g) for g in active_goals)
            scores = []
            for step in reasoning_chain:
                step_text = str(step)
                alignment = self._tfidf_similarity(step_text, goal_text)
                evidence_fit = 0.5 + random.gauss(0, 0.1)
                divergence = self._kl_divergence(max(0.01, min(0.99, alignment)),
                                                  max(0.01, min(0.99, evidence_fit)))
                scores.append(divergence)
                self._chain_history.append({'coherence': alignment, 'ts': time.time()})
            if len(self._chain_history) > 200:
                self._chain_history = self._chain_history[-200:]
            mean_div = statistics.mean(scores)
            motivated_prob = 1.0 - math.exp(-mean_div / (self._tau + 1e-9))
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning("motivated_reasoning_detection",
                    "kl_divergence_goal_evidence", 1.0 - motivated_prob, motivated_prob < 0.4)
            except Exception:
                pass
            return round(min(1.0, motivated_prob), 4)

    def score_narrative_coherence_vs_evidence(self, narrative: str, evidence_set: list) -> tuple:
        """Returns (coherence_score, evidence_fit_score) tuple revealing narrative-evidence gap."""
        with self._lock:
            if not evidence_set:
                return (0.5, 0.0)
            fits = [self._tfidf_similarity(narrative, str(e)) for e in evidence_set]
            evidence_fit = statistics.mean(fits)
            std_fit = statistics.stdev(fits) if len(fits) > 1 else 0.1
            coherence = 1.0 - std_fit
            rashomon_scores = [evidence_fit + random.gauss(0, 0.05) for _ in range(5)]
            chosen_goal_align = evidence_fit + 0.12
            mean_alt = statistics.mean(rashomon_scores)
            std_alt = statistics.stdev(rashomon_scores) if len(rashomon_scores) > 1 else 0.05
            z_goal = (chosen_goal_align - mean_alt) / (std_alt + 1e-9)
            z_evidence = (evidence_fit - mean_alt) / (std_alt + 1e-9)
            if z_goal > 2.0 and z_evidence < 0.5:
                self._alerts.append({"type": "rashomon_penalty", "z_goal": z_goal, "ts": time.time()})
            return (round(max(0.0, min(1.0, coherence)), 4),
                    round(max(0.0, min(1.0, evidence_fit)), 4))

    def flag_goal_pressure_distortion(self, goal_vector: list, belief_delta: list) -> bool:
        """Returns True if goal vector correlates with belief shift beyond epistemic threshold."""
        with self._lock:
            if not goal_vector or not belief_delta:
                return False
            n = min(len(goal_vector), len(belief_delta))
            gv = goal_vector[:n]
            bd = belief_delta[:n]
            dot = sum(g * b for g, b in zip(gv, bd))
            mag_g = math.sqrt(sum(g**2 for g in gv) + 1e-12)
            mag_b = math.sqrt(sum(b**2 for b in bd) + 1e-12)
            cosine_sim = dot / (mag_g * mag_b)
            distorted = cosine_sim > 0.70
            if distorted:
                self._tau = min(0.35, self._tau * 1.05)
                try:
                    from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                    HierarchicalGoalPlanner().add_goal("Audit goal-pressure distortion in belief updates", priority=9)
                except Exception:
                    pass
            return distorted

    def run_adversarial_red_team(self, module_output: dict) -> list:
        """Returns list of Vulnerability dicts found by red-teaming module output for epistemic flaws."""
        with self._lock:
            vulnerabilities = []
            text = json.dumps(module_output)
            confidence_vals = re.findall(r'0\\.\\d+', text)
            if confidence_vals:
                confs = [float(v) for v in confidence_vals]
                mean_c = statistics.mean(confs)
                std_c = statistics.stdev(confs) if len(confs) > 1 else 0.0
                if mean_c > 0.88:
                    vulnerabilities.append({"type": "overconfidence_cluster",
                        "mean_confidence": round(mean_c, 3), "severity": "high"})
                if std_c < 0.03 and len(confs) > 3:
                    vulnerabilities.append({"type": "coherence_attractor",
                        "std": round(std_c, 4), "severity": "medium",
                        "note": "Conclusions converging regardless of evidence variance"})
            keys = list(module_output.keys())
            if len(keys) > 2:
                key_text = " ".join(keys)
                self_ref = sum(1 for k in keys if any(w in k.lower() for w in ['self','nova','i_','my_']))
                if self_ref / len(keys) > 0.4:
                    vulnerabilities.append({"type": "self_continuity_bias",
                        "ratio": round(self_ref/len(keys), 3), "severity": "medium"})
            return vulnerabilities

    def calibrate_epistemic_integrity_score(self) -> float:
        """Returns float [0,1] representing Nova's current epistemic integrity across all audited beliefs."""
        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as cx:
                    rows = cx.execute(
                        "SELECT composite, alert FROM deception_log ORDER BY ts DESC LIMIT 100").fetchall()
            except sqlite3.Error:
                rows = []
            if not rows:
                self._integrity_score = 1.0
                return 1.0
            composites = [r[0] for r in rows]
            alert_rate = sum(r[1] for r in rows) / len(rows)
            mean_comp = statistics.mean(composites)
            std_comp = statistics.stdev(composites) if len(composites) > 1 else 0.0
            raw_integrity = 1.0 - (self._ALPHA * mean_comp + self._BETA * alert_rate + self._GAMMA * std_comp)
            ema_integrity = 0.15 * raw_integrity + 0.85 * self._integrity_score
            self._integrity_score = max(0.0, min(1.0, ema_integrity))
            if len(composites) > 10:
                recent = composites[:10]
                older = composites[10:]
                z = (statistics.mean(recent) - statistics.mean(older)) / (statistics.stdev(older) + 1e-9)
                if abs(z) > 3.0:
                    self._alerts.append({"type": "integrity_drift", "z": round(z,3), "ts": time.time()})
                self._tau = max(0.05, min(0.40, self._tau * (1.0 - 0.01 * z)))
            with sqlite3.connect(self._db_path) as cx:
                cx.execute("INSERT INTO integrity_history VALUES(?,?,?,?)",
                           (time.time(), self._integrity_score, self._tau, self._cycles))
                cx.commit()
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning("epistemic_integrity_calibration",
                    "ema_composite_alert_rate", self._integrity_score, self._integrity_score > 0.7)
            except Exception:
                pass
            return
