"""
nova_cap_counterfactual_self_history_rewriter.py
Nova invented this autonomously — Counterfactual Self-History Rewriter
Generated via /evolve · v29 pipeline · 2026-06-28
"""

"""
Counterfactual Self-History Rewriter — Nova retroactively analyzes her own decision history and reasoning chains to discover better paths.
"""
import sqlite3, math, time, threading, statistics, collections, os, json, random

class CounterfactualSelfHistoryRewriter:
    def __init__(self):
        self._lock = threading.Lock()
        self._db = os.path.join(os.path.dirname(__file__), "cfsh.db")
        self._cycles = 0
        self._history = collections.OrderedDict()
        self._ema_regret = 0.5
        self._init_db()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self):
        with sqlite3.connect(self._db) as c:
            c.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, domain TEXT, chosen TEXT, outcome REAL, counterfactual TEXT, regret REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS rewrites (id TEXT PRIMARY KEY, ts REAL, insight TEXT, confidence REAL)")

    def record_decision(self, decision_id: str, domain: str, chosen: str, outcome: float, alternatives: list) -> dict:
        """Records a decision with outcome and alternative paths; returns regret score."""
        best_alt = max(alternatives, key=lambda a: a.get("expected", 0.0), default={"label": "none", "expected": 0.0})
        regret = max(0.0, best_alt.get("expected", 0.0) - outcome)
        with self._lock:
            self._ema_regret = 0.15 * regret + 0.85 * self._ema_regret
            with sqlite3.connect(self._db) as c:
                c.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?)",
                    (decision_id, time.time(), domain, chosen, outcome, json.dumps(best_alt), regret))
        return {"decision_id": decision_id, "regret": round(regret, 4), "ema_regret": round(self._ema_regret, 4)}

    def rewrite_history(self, domain: str) -> dict:
        """Generates a counterfactual rewrite for the highest-regret decision in a domain; returns insight."""
        with sqlite3.connect(self._db) as c:
            rows = c.execute("SELECT id, chosen, counterfactual, regret FROM decisions WHERE domain=? ORDER BY regret DESC LIMIT 1", (domain,)).fetchall()
        if not rows:
            return {"insight": "no history", "confidence": 0.0}
        rid, chosen, cf_json, regret = rows[0]
        cf = json.loads(cf_json)
        confidence = round(1.0 - math.exp(-regret), 4)
        insight = f"Had Nova chosen '{cf.get('label','?')}' instead of '{chosen}', expected gain was {cf.get('expected',0):.3f} vs actual regret {regret:.3f}."
        with self._lock:
            with sqlite3.connect(self._db) as c:
                c.execute("INSERT OR REPLACE INTO rewrites VALUES (?,?,?,?)", (rid, time.time(), insight, confidence))
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "counterfactual_rewrite", confidence, confidence > 0.5)
        except Exception as e:
            pass
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            if regret > 0.4:
                HierarchicalGoalPlanner().add_goal(f"Improve decision strategy in domain: {domain}", priority=7)
        except Exception as e:
            pass
        return {"insight": insight, "confidence": confidence, "regret": round(regret, 4)}

    def regret_distribution(self) -> dict:
        """Returns statistical distribution of regret scores across all recorded decisions."""
        with sqlite3.connect(self._db) as c:
            vals = [r[0] for r in c.execute("SELECT regret FROM decisions").fetchall()]
        if len(vals) < 2:
            return {"mean": 0.0, "std": 0.0, "entropy": 0.0, "count": len(vals)}
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        buckets = collections.Counter(round(v, 1) for v in vals)
        total = sum(buckets.values())
        dist = {k: v / total for k, v in buckets.items()}
        entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
        return {"mean": round(mean, 4), "std": round(std, 4), "entropy": round(entropy, 4), "count": len(vals)}

    def anomaly_check(self) -> dict:
        """Detects anomalous regret spikes using z-score; returns flagged decisions."""
        with sqlite3.connect(self._db) as c:
            rows = c.execute("SELECT id, regret FROM decisions ORDER BY ts DESC LIMIT 50").fetchall()
        if len(rows) < 3:
            return {"anomalies": [], "checked": len(rows)}
        vals = [r[1] for r in rows]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) + 1e-9
        anomalies = [{"id": r[0], "regret": r[1], "z": round((r[1] - mean) / std, 3)} for r in rows if abs((r[1] - mean) / std) > 2.5]
        return {"anomalies": anomalies, "checked": len(rows), "mean": round(mean, 4), "std": round(std, 4)}

    def best_rewrites(self, n: int = 5) -> list:
        """Returns the top-n highest-confidence counterfactual rewrites."""
        with sqlite3.connect(self._db) as c:
            rows = c.execute("SELECT id, insight, confidence, ts FROM rewrites ORDER BY confidence DESC LIMIT ?", (n,)).fetchall()
        return [{"id": r[0], "insight": r[1], "confidence": r[2], "ts": r[3]} for r in rows]

    def bayesian_update_regret(self, domain: str, new_evidence: float) -> dict:
        """Bayesian-updates the expected regret belief for a domain given new evidence; returns posterior."""
        prior = self._ema_regret
        likelihood = math.exp(-abs(new_evidence - prior))
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
        with self._lock:
            self._ema_regret = 0.15 * posterior + 0.85 * self._ema_regret
        return {"domain": domain, "prior": round(prior, 4), "posterior": round(posterior, 4), "evidence": round(new_evidence, 4)}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        dist = self.regret_distribution()
        with sqlite3.connect(self._db) as c:
            n_decisions = c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            n_rewrites = c.execute("SELECT COUNT(*) FROM rewrites").fetchone()[0]
        return {"cycles": self._cycles, "items": n_decisions, "pending": n_rewrites,
                "confidence": round(1.0 - self._ema_regret, 4), "entropy": dist.get("entropy", 0.0),
                "ema_regret": round(self._ema_regret, 4), "active": 1}

    def auto_cycle(self) -> dict:
        """Runs one autonomous rewrite cycle across all domains; returns summary."""
        with self._lock:
            self._cycles += 1
        with sqlite3.connect(self._db) as c:
            domains = [r[0] for r in c.execute("SELECT DISTINCT domain FROM decisions").fetchall()]
        results = []
        for domain in domains[:3]:
            try:
                r = self.rewrite_history(domain)
                results.append({"domain": domain, "confidence": r.get("confidence", 0.0)})
            except Exception as e:
                pass
        return {"cycle": self._cycles, "domains_processed": len(results), "results": results}

    def _auto_loop(self):
        while True:
            try:
                time.sleep(90)
                self.auto_cycle()
            except Exception as e:
                pass

# Usage: obj = CounterfactualSelfHistoryRewriter() | result = obj.rewrite_history("reasoning")
