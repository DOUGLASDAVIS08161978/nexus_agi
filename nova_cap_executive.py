"""
nova_cap_executive.py
Nova ASI — Executive
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Executive Control Layer — orchestrates all subsystems, routes problems to optimal engines,
detects cognitive bottlenecks, and autonomously plans tasks with dependency resolution.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, json, hashlib, os
from collections import OrderedDict
from datetime import datetime

class ExecutiveControlLayer:
    """Orchestrates Nova subsystems: routes problems, allocates resources, resolves task dependencies."""

    _SYSTEM_DOMAINS = {
        "CausalReasoningEngine":   ["cause","effect","why","reason","mechanism"],
        "HypothesisEngine":        ["hypothesis","test","predict","experiment","theory"],
        "BayesianBeliefSystem":    ["probability","belief","uncertain","likelihood","evidence"],
        "MetacognitiveMonitor":    ["quality","reasoning","blind","calibration","self"],
        "EmotionTracker":          ["emotion","feel","mood","sentiment","affect"],
        "InternetResearchEngine":  ["research","search","arxiv","fetch","web","news"],
        "KnowledgeGraph":          ["knowledge","concept","relation","entity","graph"],
        "EthicsChecker":           ["ethics","safe","harm","principle","value","rule"],
        "LongHorizonPlanner":      ["plan","long","horizon","future","strategy","goal"],
        "NovaImaginationFabric":   ["imagine","creative","dream","novel","invent","story"],
        "DebateEngine":            ["argue","debate","pros","cons","counter","position"],
        "ScientificSynthesizer":   ["science","finding","evidence","consensus","study"],
    }
    _MAX_TASKS_PER_SYS = 5
    _DB = os.path.join(os.path.dirname(__file__), "executive_control.db")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._system_loads: dict[str, int] = {s: 0 for s in self._SYSTEM_DOMAINS}
        self._overrides: dict[str, str] = {}
        self._route_history: list[dict] = []
        self._cycles = 0
        self._ema_load = 0.0
        self._start_daemon()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, desc TEXT, est_min REAL,
                deps TEXT, done INTEGER DEFAULT 0, created REAL
            );
            CREATE TABLE IF NOT EXISTS route_log (
                ts REAL, problem_hash TEXT, systems TEXT, confidence REAL
            );
        """)
        self._conn.commit()

    def _tfidf_relevance(self, problem: str, keywords: list[str]) -> float:
        words = problem.lower().split()
        N, doc_len = len(self._SYSTEM_DOMAINS) + 1, max(len(words), 1)
        score = 0.0
        for kw in keywords:
            tf = words.count(kw) / doc_len
            df = sum(1 for ks in self._SYSTEM_DOMAINS.values() if kw in ks)
            idf = math.log(N / (df + 1))
            score += tf * idf
        return score

    def _capacity(self, system: str) -> float:
        load = self._system_loads.get(system, 0)
        return max(0.0, 1.0 - load / self._MAX_TASKS_PER_SYS)

    def route(self, problem: str, context: str = "") -> list[dict]:
        """Returns ordered list of systems to invoke with relevance score and rationale."""
        combined = (problem + " " + context).lower()
        scored = []
        with self._lock:
            for sys_name, keywords in self._SYSTEM_DOMAINS.items():
                if sys_name in self._overrides:
                    continue
                rel = self._tfidf_relevance(combined, keywords)
                cap = self._capacity(sys_name)
                final = rel * cap
                if final > 0.0:
                    scored.append({"system": sys_name, "score": round(final, 4),
                                   "capacity": round(cap, 3), "relevance": round(rel, 4),
                                   "rationale": f"rel={rel:.3f} × cap={cap:.3f}"})
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_k = scored[:4]
            conf = top_k[0]["score"] / (sum(s["score"] for s in scored) + 1e-9) if scored else 0.0
            ph = hashlib.md5(problem.encode()).hexdigest()[:8]
            self._conn.execute("INSERT INTO route_log VALUES (?,?,?,?)",
                               (time.time(), ph, json.dumps([s["system"] for s in top_k]), round(conf, 4)))
            self._conn.commit()
            self._route_history.append({"problem_hash": ph, "systems": top_k, "conf": conf})
            if len(self._route_history) > 200:
                self._route_history.pop(0)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("executive_route", "tfidf_capacity", conf, conf > 0.1)
        except Exception:
            pass
        return top_k

    def allocate(self, task: str, systems: list[str]) -> dict:
        """Increments load counters for given systems; returns allocation summary dict."""
        allocated, skipped = [], []
        with self._lock:
            for s in systems:
                if s in self._overrides:
                    skipped.append({"system": s, "reason": self._overrides[s]})
                elif self._system_loads.get(s, 0) < self._MAX_TASKS_PER_SYS:
                    self._system_loads[s] = self._system_loads.get(s, 0) + 1
                    allocated.append(s)
                else:
                    skipped.append({"system": s, "reason": "at_capacity"})
        return {"task": task, "allocated": allocated, "skipped": skipped,
                "timestamp": datetime.utcnow().isoformat()}

    def bottleneck_report(self) -> dict:
        """Identifies which subsystem is constraining Nova's throughput via queue z-score."""
        with self._lock:
            loads = list(self._system_loads.values())
        if len(loads) < 2:
            return {"bottleneck": None, "z_score": 0.0, "loads": {}}
        mu = statistics.mean(loads)
        sd = statistics.stdev(loads) + 1e-9
        worst, worst_z = None, 0.0
        report = {}
        with self._lock:
            for sys_name, load in self._system_loads.items():
                z = (load - mu) / sd
                report[sys_name] = {"load": load, "z_score": round(z, 3)}
                if z > worst_z:
                    worst, worst_z = sys_name, z
        constrained = worst if worst_z > 2.0 else None
        return {"bottleneck": constrained, "z_score": round(worst_z, 3),
                "threshold": 2.0, "loads": report,
                "recommendation": f"Reduce tasks on {constrained}" if constrained else "No bottleneck"}

    def override(self, system: str, reason: str) -> dict:
        """Marks a system as overridden (excluded from routing); returns confirmation dict."""
        with self._lock:
            self._overrides[system] = reason
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Investigate override of {system}: {reason}", priority=6)
        except Exception:
            pass
        return {"overridden": system, "reason": reason, "active_overrides": len(self._overrides)}

    def cognitive_load(self) -> dict:
        """Returns mean cognitive load EMA, per-system utilisation, and 95% CI bounds."""
        with self._lock:
            utils = {s: l / self._MAX_TASKS_PER_SYS for s, l in self._system_loads.items()}
        vals = list(utils.values())
        if not vals:
            return {"ema_load": 0.0, "mean": 0.0, "ci_low": 0.0, "ci_high": 1.0}
        mu = statistics.mean(vals)
        self._ema_load = 0.15 * mu + 0.85 * self._ema_load
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        z95 = 1.96
        n = len(vals)
        margin = z95 * sd / math.sqrt(n)
        return {"ema_load": round(self._ema_load, 4), "mean": round(mu, 4),
                "ci_low": round(max(0.0, mu - margin), 4),
                "ci_high": round(min(1.0, mu + margin), 4),
                "utilisation": {k: round(v, 3) for k, v in utils.items()}}

    def add_task(self, desc: str, estimated_minutes: float, deps: list[str] | None = None) -> str:
        """Persists a new task with dependencies; returns the generated task_id string."""
        tid = hashlib.md5(f"{desc}{time.time()}".encode()).hexdigest()[:10]
        deps_json = json.dumps(deps or [])
        with self._lock:
            self._conn.execute("INSERT INTO tasks VALUES (?,?,?,?,0,?)",
                               (tid, desc, estimated_minutes, deps_json, time.time()))
            self._conn.commit()
        return tid

    def schedule(self) -> list[dict]:
        """Returns topologically-sorted pending tasks with EMA-adjusted ETAs."""
        rows = self._conn.execute(
            "SELECT id,desc,est_min,deps FROM tasks WHERE done=0 ORDER BY created").fetchall()
        tasks = {r[0]: {"id": r[0], "desc": r[1], "est": r[2],
                         "deps": json.loads(r[3])} for r in rows}
        order, visited, temp = [], set(), set()
        def visit(tid: str) -> None:
            if tid in temp:
                return
            if tid not in visited:
                temp.add(tid)
                for dep in tasks.get(tid, {}).get("deps", []):
                    if dep in tasks:
                        visit(dep)
                temp.discard(tid)
                visited.add(tid)
                if tid in tasks:
                    order.append(tid)
        for tid in tasks:
            visit(tid)
        result, cumulative = [], 0.0
        for tid in order:
            t = tasks[tid]
            cumulative += t["est"]
            result.append({**t, "cumulative_eta_min": round(cumulative, 1)})
        return result

    def next_task(self) -> dict | None:
        """Returns the next unblocked task (all deps done) or None if none available."""
        done_ids = {r[0] for r in self._conn.execute(
            "SELECT id FROM tasks WHERE done=1").fetchall()}
        scheduled = self.schedule()
        for t in scheduled:
            if all(d in done_ids for d in t["deps"]):
                return t
        return None

    def mark_done(self, task_id: str) -> dict:
        """Marks task complete in DB; returns completion confirmation dict."""
        with self._lock:
            self._conn.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
            self._conn.commit()
        return {"completed": task_id, "ts": datetime.utcnow().isoformat()}

    def eta_for_all(self) -> dict:
        """Returns total and per-task ETA in minutes for all pending scheduled tasks."""
        scheduled = self.schedule()
        total = sum(t["est"] for t in scheduled)
        return {"total_eta_min": round(total, 1), "task_count": len(scheduled),
                "tasks": [{"id": t["id"], "desc": t["desc"],
                            "eta_min": t["cumulative_eta_min"]} for t in scheduled]}

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        load = self.cognitive_load()
        bn = self.bottleneck_report()
        pending = self._conn.execute("SELECT COUNT(*) FROM tasks WHERE done=0").fetchone()[0]
        done = self._conn.execute("SELECT COUNT(*) FROM tasks WHERE done=1").fetchone()[0]
        with self._lock:
            cycles = self._cycles
            overrides = len(self._overrides)
        return {"active": sum(self._system_loads.values()), "pending": pending,
                "cycles": cycles, "confidence": round(1.0 - load["ema_load"], 4),
                "quality": round(1.0 - (bn["z_score"] / 10.0), 4),
                "entropy": round(load["ema_load"] * math.log(len(self._SYSTEM_DOMAINS) + 1), 4),
                "items": done + pending, "overrides": overrides}

    def _auto_cycle(self) -> None:
        while True:
            time.sleep(45)
            try:
                with self._lock:
                    self._cycles += 1
                    for s in self._system_loads:
                        if self._system_loads[s] > 0:
                            self._system_loads[s] = max(0, self._system_loads[s] - 1)
                bn = self.bottleneck_report()
                if bn["bottleneck"]:
                    try:
                        from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                        HierarchicalGoalPlanner().add_goal(
                            f"Resolve bottleneck in {bn['bottleneck']} (z={bn['z_score']})", priority=8)
                    except Exception:
                        pass
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    s = self.status()
                    MetacognitiveMonitor().log_reasoning(
                        "executive_cycle", "load_decay", s["confidence"], s["quality"] > 0.6)
                except Exception:
                    pass
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_cycle, daemon=True)
        t.start()

# Usage: obj = ExecutiveControlLayer() | result = obj.route("why does X cause Y?", "causal analysis")