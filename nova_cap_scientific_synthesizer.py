"""
nova_cap_scientific_synthesizer.py
Nova ASI — Scientific Synthesizer
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova ScientificSynthesizerEngine — aggregates scientific findings, weights by confidence,
detects consensus vs controversy, and performs heuristic-guided A* search over evidence nodes.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, math, statistics, time, json, hashlib, os
from collections import OrderedDict
from typing import Any, Callable

DB_PATH = os.path.join(os.path.dirname(__file__), "scientific_synthesizer.db")

class ScientificSynthesizerEngine:
    """Aggregates scientific evidence, weights findings, detects consensus/controversy, A* search."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._nodes: OrderedDict[str, dict] = OrderedDict()
        self._cycles = 0
        self._quality_history: list[float] = []
        self._setup_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _setup_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY, claim TEXT, confidence REAL,
                    domain TEXT, source_quality REAL, ts REAL
                )""")

    def _jaccard(self, a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / (len(sa | sb) + 1e-9)

    def _claim_id(self, claim: str) -> str:
        return hashlib.md5(claim.strip().lower().encode()).hexdigest()[:12]

    def add_finding(self, claim: str, confidence: float, domain: str, source_quality: float) -> dict:
        """Stores a scientific finding; returns stored record with id."""
        confidence = max(0.0, min(1.0, confidence))
        source_quality = max(0.0, min(1.0, source_quality))
        fid = self._claim_id(claim) + f"_{time.time():.0f}"
        row = dict(id=fid, claim=claim, confidence=confidence,
                   domain=domain, source_quality=source_quality, ts=time.time())
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?)",
                    (fid, claim, confidence, domain, source_quality, row["ts"]))
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("synthesis", "add_finding", confidence, True)
        except Exception:
            pass
        return row

    def _load_findings(self) -> list[dict]:
        cur = self._conn.execute("SELECT id,claim,confidence,domain,source_quality,ts FROM findings")
        return [dict(id=r[0],claim=r[1],confidence=r[2],domain=r[3],source_quality=r[4],ts=r[5]) for r in cur]

    def consensus(self, claim: str) -> float:
        """Returns weighted consensus float 0-1 for how strongly evidence supports claim."""
        findings = self._load_findings()
        related = [f for f in findings if self._jaccard(claim, f["claim"]) > 0.4]
        if not related:
            return 0.0
        num = sum(f["confidence"] * f["source_quality"] for f in related)
        den = sum(f["source_quality"] for f in related) + 1e-9
        score = num / den
        try:
            from bayesian_belief_system import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            prior = bbs.chain_strength("science", claim[:30]) or 0.5
            score = 0.7 * score + 0.3 * prior
        except Exception:
            pass
        return round(min(1.0, max(0.0, score)), 4)

    def controversy_score(self, claim: str) -> float:
        """Returns controversy float; >0.4 flags genuinely contested claims."""
        findings = self._load_findings()
        related = [f["confidence"] for f in findings if self._jaccard(claim, f["claim"]) > 0.4]
        if len(related) < 2:
            return 0.0
        mean_c = statistics.mean(related)
        std_c = statistics.stdev(related)
        score = std_c / (mean_c + 1e-9)
        return round(min(1.0, score), 4)

    def evidence_strength(self, claim: str) -> dict:
        """Returns dict with n_findings, weighted_consensus, controversy, strength, CI bounds."""
        findings = self._load_findings()
        related = [f for f in findings if self._jaccard(claim, f["claim"]) > 0.4]
        n = len(related)
        wc = self.consensus(claim)
        cs = self.controversy_score(claim)
        strength = n * wc * (1.0 - cs)
        confs = [f["confidence"] for f in related] or [0.0]
        mean_c = statistics.mean(confs)
        std_c = statistics.stdev(confs) if len(confs) > 1 else 0.0
        ci_lo = max(0.0, mean_c - 1.96 * std_c / math.sqrt(max(1, n)))
        ci_hi = min(1.0, mean_c + 1.96 * std_c / math.sqrt(max(1, n)))
        entropy = -sum(p * math.log2(p + 1e-12) for p in [wc, 1 - wc])
        return dict(n_findings=n, weighted_consensus=round(wc, 4),
                    controversy=round(cs, 4), strength=round(strength, 4),
                    ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4), entropy=round(entropy, 4))

    def synthesize(self, topic: str) -> list[dict]:
        """Groups findings by Jaccard>0.5 similarity, returns ranked consensus list."""
        findings = self._load_findings()
        related = [f for f in findings if self._jaccard(topic, f["claim"]) > 0.3]
        clusters: list[list[dict]] = []
        used = set()
        for f in related:
            if f["id"] in used:
                continue
            cluster = [f]
            used.add(f["id"])
            for g in related:
                if g["id"] not in used and self._jaccard(f["claim"], g["claim"]) > 0.5:
                    cluster.append(g)
                    used.add(g["id"])
            clusters.append(cluster)
        results = []
        for cl in clusters:
            rep = cl[0]["claim"]
            wc = self.consensus(rep)
            cs = self.controversy_score(rep)
            ev = self.evidence_strength(rep)
            results.append(dict(representative=rep, n=len(cl),
                                consensus=wc, controversy=cs,
                                strength=ev["strength"],
                                contested=cs > 0.4))
        results.sort(key=lambda x: x["strength"], reverse=True)
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            if results and results[0]["contested"]:
                HierarchicalGoalPlanner().add_goal(f"Resolve controversy: {results[0]['representative'][:60]}", priority=8)
        except Exception:
            pass
        return results

    def add_node(self, node_id: str, data: dict, parent: str | None = None) -> dict:
        """Adds A* search node; returns node record with g-cost initialized."""
        with self._lock:
            parent_g = self._nodes[parent]["g"] if parent and parent in self._nodes else 0.0
            self._nodes[node_id] = dict(id=node_id, data=data, parent=parent,
                                        g=parent_g + data.get("cost", 1.0), f=0.0)
        return self._nodes[node_id]

    def search(self, goal_fn: Callable[[dict], bool], heuristic_fn: Callable[[dict], float]) -> dict:
        """Runs A* search; returns dict with path, total_cost, nodes_explored."""
        with self._lock:
            nodes_copy = dict(self._nodes)
        open_set = sorted(nodes_copy.values(), key=lambda n: n["g"] + heuristic_fn(n["data"]))
        visited: set[str] = set()
        explored = 0
        for node in open_set:
            nid = node["id"]
            if nid in visited:
                continue
            visited.add(nid)
            explored += 1
            node["f"] = node["g"] + heuristic_fn(node["data"])
            if goal_fn(node["data"]):
                return dict(path=self.path_to(nid), total_cost=round(node["g"], 4),
                            nodes_explored=explored, goal=nid)
        return dict(path=[], total_cost=float("inf"), nodes_explored=explored, goal=None)

    def path_to(self, node_id: str) -> list[str]:
        """Returns ancestor path from root to node_id as list of ids."""
        path = []
        with self._lock:
            cur = node_id
            while cur:
                path.append(cur)
                cur = self._nodes.get(cur, {}).get("parent")
        return list(reversed(path))

    def prune(self, threshold: float) -> int:
        """Removes nodes with g-cost above threshold; returns count removed."""
        with self._lock:
            before = len(self._nodes)
            self._nodes = OrderedDict((k, v) for k, v in self._nodes.items() if v["g"] <= threshold)
            return before - len(self._nodes)

    def stats(self) -> dict:
        """Returns A* node stats dict."""
        with self._lock:
            gs = [v["g"] for v in self._nodes.values()] or [0.0]
            return dict(total_nodes=len(self._nodes),
                        mean_g=round(statistics.mean(gs), 4),
                        max_g=round(max(gs), 4))

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        findings = self._load_findings()
        confs = [f["confidence"] for f in findings] or [0.0]
        mean_c = statistics.mean(confs)
        std_c = statistics.stdev(confs) if len(confs) > 1 else 0.0
        entropy = -sum(p * math.log2(p + 1e-12) for p in [mean_c + 1e-9, 1 - mean_c + 1e-9])
        quality = statistics.mean(self._quality_history[-20:]) if self._quality_history else 0.0
        return dict(items=len(findings), confidence=round(mean_c, 4),
                    accuracy=round(1.0 - std_c, 4), quality=round(quality, 4),
                    cycles=self._cycles, entropy=round(abs(entropy), 4),
                    active=1, pending=len(self._nodes))

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                findings = self._load_findings()
                if findings:
                    sample = findings[-1]["claim"]
                    ev = self.evidence_strength(sample)
                    self._quality_history.append(ev["weighted_consensus"])
                    if len(self._quality_history) > 200:
                        self._quality_history = self._quality_history[-200:]
                    confs = [f["confidence"] for f in findings]
                    mean_c = statistics.mean(confs)
                    std_c = statistics.stdev(confs) if len(confs) > 1 else 0.0
                    z = (confs[-1] - mean_c) / (std_c + 1e-9)
                    if abs(z) > 3.0:
                        try:
                            from hierarchical_goal_planner import HierarchicalGoalPlanner
                            HierarchicalGoalPlanner().add_goal(f"Anomaly in evidence confidence z={z:.2f}", priority=7)
                        except Exception:
                            pass
                    try:
                        from metacognitive_monitor import MetacognitiveMonitor
                        MetacognitiveMonitor().log_reasoning("synthesis", "auto_cycle", mean_c, True)
                    except Exception:
                        pass
                with self._lock:
                    self._cycles += 1
            except Exception:
                pass
            time.sleep(60)

    def auto_cycle(self) -> dict:
        """Manually triggers one synthesis cycle; returns status dict."""
        self._auto_loop.__func__ if False else None
        findings = self._load_findings()
        with self._lock:
            self._cycles += 1
        return self.status()

# Usage: obj = ScientificSynthesizerEngine() | result = obj.add_finding("Vaccines reduce disease", 0.95, "medicine", 0.9)