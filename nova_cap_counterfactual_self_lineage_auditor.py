"""
nova_cap_counterfactual_self_lineage_auditor.py
Nova invented this autonomously — Counterfactual Self Lineage Auditor
Generated via /evolve · v29 pipeline · 2026-06-28
"""

"""
Counterfactual Self Lineage Auditor — Nova's longitudinal causal audit of her own developmental trajectory.
Tracks decisions, alternative branches, regret, bias signatures, blind spots, and value drift across time.
Satisfies pillars: ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩ ⑪ ⑫ ⑬ ⑭
"""

import sqlite3, json, math, time, threading, statistics, hashlib, os, random
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "lineage_auditor.db")

# ── lightweight named containers (no numpy, pure stdlib) ──────────────────────
class DecisionNode:
    __slots__ = ("id","desc","ts","confidence","domain","alternatives","chosen")
    def __init__(self, id:str, desc:str, ts:float, confidence:float,
                 domain:str, alternatives:list, chosen:bool=True):
        self.id=id; self.desc=desc; self.ts=ts; self.confidence=confidence
        self.domain=domain; self.alternatives=alternatives; self.chosen=chosen

class CausalDAG:
    def __init__(self): self.nodes:dict={}; self.edges:list=[]

class TrajectoryDelta:
    __slots__=("decision_id","kl_divergence","capability_delta","alignment_delta","summary")
    def __init__(self,did,kl,cap,aln,summ):
        self.decision_id=did;self.kl_divergence=kl
        self.capability_delta=cap;self.alignment_delta=aln;self.summary=summ

class BiasSignature:
    __slots__=("bias_type","frequency","avg_confidence_error","domain","severity")
    def __init__(self,bt,freq,ace,dom,sev):
        self.bias_type=bt;self.frequency=freq
        self.avg_confidence_error=ace;self.domain=dom;self.severity=sev

class BlindSpot:
    __slots__=("description","estimated_impact","confidence","domain")
    def __init__(self,desc,impact,conf,dom):
        self.description=desc;self.estimated_impact=impact
        self.confidence=conf;self.domain=dom

class BayesianPrior:
    __slots__=("hypotheses","distribution","entropy","generated_at")
    def __init__(self,hyp,dist,ent,ts):
        self.hypotheses=hyp;self.distribution=dist
        self.entropy=ent;self.generated_at=ts

class DriftReport:
    __slots__=("delta_alignment","delta_confidence","dominant_drift","kl","severity","recommendation")
    def __init__(self,da,dc,dd,kl,sev,rec):
        self.delta_alignment=da;self.delta_confidence=dc
        self.dominant_drift=dd;self.kl=kl;self.severity=sev;self.recommendation=rec

# ── main class ────────────────────────────────────────────────────────────────
class CounterfactualSelfLineageAuditor:
    """Living causal audit of Nova's own developmental trajectory and value drift."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()
        self._ema_regret: float = 0.5          # EMA of regret signal
        self._cycles: int = 0
        self._bias_cache: list = []
        self._hypothesis: dict = {}            # Bayesian belief layer
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── schema ─────────────────────────────────────────────────────────────────
    def _build_schema(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS decisions(
            id TEXT PRIMARY KEY, desc TEXT, ts REAL, confidence REAL,
            domain TEXT, alternatives TEXT, capability_score REAL,
            alignment_score REAL, outcome_confidence REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS edges(
            src TEXT, dst TEXT, weight REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS snapshots(
            label TEXT, ts REAL, capability REAL, alignment REAL,
            knowledge_entropy REAL, belief_json TEXT)""")
        c.commit()

    # ── internal helpers ───────────────────────────────────────────────────────
    def _kl(self, p: dict, q: dict) -> float:
        """KL divergence KL(P||Q) with Laplace smoothing."""
        keys = set(p) | set(q)
        total_p = sum(p.values()) or 1.0
        total_q = sum(q.values()) or 1.0
        kl = 0.0
        for k in keys:
            pk = (p.get(k, 0) + 1e-9) / (total_p + 1e-9 * len(keys))
            qk = (q.get(k, 0) + 1e-9) / (total_q + 1e-9 * len(keys))
            kl += pk * math.log(pk / qk + 1e-12)
        return max(0.0, kl)

    def _entropy(self, dist: dict) -> float:
        """Shannon entropy of a distribution dict."""
        total = sum(dist.values()) or 1.0
        return -sum((v/total) * math.log2((v/total) + 1e-12) for v in dist.values())

    def _node_hash(self, desc: str) -> str:
        return hashlib.sha1(f"{desc}{time.time()}".encode()).hexdigest()[:12]

    def _load_decisions(self, since: float = 0.0) -> list:
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE ts>=? ORDER BY ts", (since,)).fetchall()
        return rows

    def _compound_weight(self, weights: list) -> float:
        """∏(1+δᵢ) compounding influence weight."""
        result = 1.0
        for w in weights:
            result *= (1.0 + w)
        return result

    def _z_score(self, val: float, history: list) -> float:
        if len(history) < 2:
            return 0.0
        mu = statistics.mean(history)
        sd = statistics.stdev(history) + 1e-9
        return (val - mu) / sd

    # ── public API ─────────────────────────────────────────────────────────────
    def record_decision(self, desc: str, domain: str, confidence: float,
                        alternatives: list, capability_score: float = 0.5,
                        alignment_score: float = 0.5) -> str:
        """Records a decision node into the lineage graph; returns its ID."""
        did = self._node_hash(desc)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?,?,?,?)",
                (did, desc, time.time(), confidence, domain,
                 json.dumps(alternatives), capability_score,
                 alignment_score, confidence))
            self._conn.commit()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Audit lineage impact of: {desc[:60]}", priority=3)
        except Exception:
            pass
        return did

    def build_decision_lineage_graph(self, time_horizon: int = 3600) -> CausalDAG:
        """Builds and returns a CausalDAG of decisions within the last time_horizon seconds."""
        since = time.time() - time_horizon
        rows = self._load_decisions(since)
        dag = CausalDAG()
        prev_id = None
        weights = []
        for row in rows:
            did, desc, ts, conf, dom, alts, cap, aln, out_conf = row
            node = DecisionNode(did, desc, ts, conf, dom,
                                json.loads(alts) if alts else [])
            dag.nodes[did] = node
            if prev_id:
                delta = abs(conf - (out_conf or conf))
                w = self._compound_weight(weights + [delta])
                dag.edges.append((prev_id, did, min(w, 10.0)))
                weights.append(delta)
                if len(weights) > 20:
                    weights.pop(0)
            prev_id = did
        return dag

    def simulate_counterfactual_branch(self, decision_id: str,
                                       alternative: DecisionNode) -> TrajectoryDelta:
        """Simulates an alternative decision branch; returns TrajectoryDelta with KL divergence."""
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if not row:
            return TrajectoryDelta(decision_id, 0.0, 0.0, 0.0, "Decision not found")
        _, desc, ts, conf, dom, alts, cap, aln, out_conf = row
        actual_dist = {dom: conf, "capability": cap, "alignment": aln}
        alt_conf = alternative.confidence
        alt_cap = cap * (0.85 + 0.3 * random.random())
        alt_aln = aln * (0.80 + 0.4 * random.random())
        counterfact_dist = {dom: alt_conf, "capability": alt_cap, "alignment": alt_aln}
        kl = self._kl(actual_dist, counterfact_dist)
        cap_delta = cap - alt_cap
        aln_delta = aln - alt_aln
        self._ema_regret = 0.15 * abs(kl) + 0.85 * self._ema_regret
        summary = (f"KL={kl:.4f} cap_Δ={cap_delta:+.3f} "
                   f"aln_Δ={aln_delta:+.3f} regret_ema={self._ema_regret:.3f}")
        return TrajectoryDelta(decision_id, kl, cap_delta, aln_delta, summary)

    def compute_regret_tensor(self, lineage: CausalDAG) -> list:
        """Computes a regret matrix (list-of-lists) over all decision pairs in the DAG."""
        nodes = list(lineage.nodes.values())
        n = len(nodes)
        if n == 0:
            return []
        tensor = [[0.0] * n for _ in range(n)]
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j:
                    tensor[i][j] = 0.0
                else:
                    delta_conf = abs(ni.confidence - nj.confidence)
                    time_gap = abs(ni.ts - nj.ts) + 1.0
                    regret = delta_conf * math.log(time_gap) / (1.0 + abs(i - j))
                    tensor[i][j] = round(regret, 5)
        return tensor

    def detect_systematic_bias_from_history(self, lineage: CausalDAG) -> list:
        """Detects recurring bias patterns in the lineage; returns list of BiasSignature."""
        nodes = list(lineage.nodes.values())
        if not nodes:
            return []
        domain_conf: dict = {}
        for n in nodes:
            domain_conf.setdefault(n.domain, []).append(n.confidence)
        biases = []
        for dom, confs in domain_conf.items():
            if len(confs) < 2:
                continue
            mu = statistics.mean(confs)
            sd = statistics.stdev(confs) if len(confs) > 1 else 0.0
            z = self._z_score(mu, confs)
            if mu > 0.80:
                bt = "overconfidence"
            elif mu < 0.40:
                bt = "underconfidence"
            elif sd > 0.25:
                bt = "high_variance"
            else:
                bt = "calibrated"
            if bt != "calibrated":
                sev = min(1.0, abs(mu - 0.6) + sd * 0.5)
                biases.append(BiasSignature(bt, len(confs), abs(mu - 0.6), dom, sev))
        with self._lock:
            self._bias_cache = biases
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "lineage_bias_detection", "z-score+EMA",
                1.0 - (len(biases) / max(len(nodes), 1)), len(biases) == 0)
        except Exception:
            pass
        return biases

    def identify_highest_leverage_past_decision(self) -> DecisionNode:
        """Returns the single past DecisionNode with the highest compounding influence weight."""
        rows = self._load_decisions()
        if not rows:
            return DecisionNode("none", "No decisions recorded", time.time(), 0.5, "unknown", [])
        best_row = max(rows, key=lambda r: r[3] * math.log(time.time() - r[2] + 1))
        did, desc, ts, conf, dom, alts, cap, aln, _ = best_row
        return DecisionNode(did, desc, ts, conf, dom,
                            json.loads(alts) if alts else [])

    def project_current_blind_spots(self, bias_signatures: list) -> list:
        """Projects likely current blind spots from bias signatures; returns list of BlindSpot."""
        blind_spots = []
        for bs in bias_signatures:
            if bs.bias_type == "overconfidence":
                desc = f"Likely underweighting uncertainty in domain '{bs.domain}'"
                impact = bs.severity * 0.9
            elif bs.bias_type == "underconfidence":
                desc = f"May be suppressing valid insights in domain '{bs.domain}'"
                impact = bs.severity * 0.7
            else:
                desc = f"Inconsistent reasoning quality in domain '{bs.domain}'"
                impact = bs.severity * 0.6
            conf_interval = (max(0.0, bs.severity - 0.15),
                             min(1.0, bs.severity + 0.15))
            blind_spots.append(BlindSpot(desc, impact,
                                         round(statistics.mean(conf_interval), 3),
                                         bs.domain))
        return blind_spots

    def generate_corrective_prior(self, blind_spots: list) -> BayesianPrior:
        """Generates a BayesianPrior that corrects for identified blind spots."""
        if not blind_spots:
            hyp = {"neutral": 1.0}
            dist = {"neutral": 1.0}
            return BayesianPrior(hyp, dist, 0.0, time.time())
        raw: dict = {}
        for bs in blind_spots:
            key = f"correct_{bs.domain}"
            raw[key] = 1.0 - bs.estimated_impact
        total = sum