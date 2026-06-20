"""
nova_cap_executive.py
Nova ASI — Executive
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Executive Control Layer — orchestrates all subsystems, routes problems, detects cognitive bottlenecks,
and autonomously schedules tasks with topological dependency resolution.
"""

import sqlite3, threading, time, math, statistics, json, os, hashlib, collections
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_executive.db")

SYSTEM_DOMAINS = {
    "BayesianBeliefSystem":   {"domain": "probability reasoning belief uncertainty", "max_tasks": 8},
    "CausalReasoningEngine":  {"domain": "cause effect chain causal inference", "max_tasks": 6},
    "HypothesisEngine":       {"domain": "hypothesis science discovery", "max_tasks": 6},
    "MetaCognitionEngine":    {"domain": "reflection meta thinking patterns", "max_tasks": 4},
    "EmotionalResonanceEngine": {"domain": "emotion feeling mood empathy", "max_tasks": 5},
    "KnowledgeGraph":         {"domain": "knowledge fact entity relation", "max_tasks": 10},
    "TaskPlanner":            {"domain": "plan task goal steps", "max_tasks": 8},
    "InternetResearchEngine": {"domain": "research web search arxiv fetch", "max_tasks": 4},
    "NovaImaginationFabric":  {"domain": "creative imagine dream combine", "max_tasks": 5},
    "EthicsChecker":          {"domain": "ethics safety rule moral", "max_tasks": 6},
    "WorldPredictor":         {"domain": "predict future world forecast", "max_tasks": 5},
    "SelfModificationEngine": {"domain": "self improve modify weakness", "max_tasks": 3},
}

class NovaExecutiveControl:
    """Executive control layer: routes problems, allocates systems, schedules tasks, detects bottlenecks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._system_load: dict[str, int] = {s: 0 for s in SYSTEM_DOMAINS}
        self._overrides: dict[str, str] = {}
        self._cycles: int = 0
        self._route_history: collections.deque = collections.deque(maxlen=200)
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, desc TEXT, est_minutes REAL,
                deps TEXT, done INTEGER DEFAULT 0, created REAL
            );
            CREATE TABLE IF NOT EXISTS route_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem TEXT, systems TEXT, rationale TEXT, ts REAL
            );
        """)
        self._conn.commit()

    def _tfidf_relevance(self, problem: str, domain_str: str) -> float:
        """Returns TF-IDF-style relevance score between problem text and domain keywords."""
        p_tokens = problem.lower().split()
        d_tokens = domain_str.lower().split()
        if not p_tokens or not d_tokens:
            return 0.0
        overlap = sum(1 for t in p_tokens if t in d_tokens)
        tf = overlap / (len(p_tokens) + 1e-9)
        idf = math.log((len(SYSTEM_DOMAINS) + 1) / (overlap + 1))
        return tf * idf

    def route(self, problem: str, context: str = "") -> list[dict]:
        """Returns ordered list of systems to invoke with relevance scores and rationale."""
        combined = (problem + " " + context).lower()
        scores: list[tuple[float, str, str]] = []
        with self._lock:
            for sys_name, meta in SYSTEM_DOMAINS.items():
                if sys_name in self._overrides:
                    continue
                rel = self._tfidf_relevance(combined, meta["domain"])
                active = self._system_load.get(sys_name, 0)
                capacity = 1.0 - (active / (meta["max_tasks"] + 1e-9))
                capacity = max(0.0, min(1.0, capacity))
                score = rel * capacity
                rationale = f"rel={rel:.3f} cap={capacity:.3f} load={active}/{meta['max_tasks']}"
                scores.append((score, sys_name, rationale))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = [{"system": s, "score": round(sc, 4), "rationale": r}
                 for sc, s, r in scores[:5] if sc > 0.0]
        if not top_k:
            top_k = [{"system": scores[0][1], "score": round(scores[0][0], 4),
                      "rationale": scores[0][2]}] if scores else []
        with self._lock:
            self._route_history.append({"problem": problem[:80], "routes": top_k, "ts": time.time()})
            self._conn.execute(
                "INSERT INTO route_log(problem,systems,rationale,ts) VALUES(?,?,?,?)",
                (problem[:200], json.dumps([r["system"] for r in top_k]),
                 json.dumps([r["rationale"] for r in top_k]), time.time())
            )
            self._conn.commit()
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("executive_route", "tfidf_capacity", top_k[0]["score"] if top_k else 0.0, True)
        except Exception:
            pass
        return top_k

    def allocate(self, task: str, systems: list[str]) -> dict:
        """Increments load on named systems and returns allocation confirmation with confidence."""
        allocated = []
        with self._lock:
            for s in systems:
                if s in SYSTEM_DOMAINS and s not in self._overrides:
                    self._system_load[s] = self._system_load.get(s, 0) + 1
                    allocated.append(s)
        confidence = len(allocated) / (len(systems) + 1e-9)
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Executive allocated: {task[:60]}", priority=2)
        except Exception:
            pass
        return {"task": task[:80], "allocated": allocated, "confidence": round(confidence, 3)}

    def bottleneck_report(self) -> dict:
        """Returns the system(s) constraining Nova's throughput with z-score anomaly detection."""
        with self._lock:
            loads = {s: self._system_load.get(s, 0) for s in SYSTEM_DOMAINS}
        depths = list(loads.values())
        if not depths:
            return {"bottlenecks": [], "mean_load": 0.0, "status": "idle"}
        mean_d = statistics.mean(depths)
        std_d = statistics.stdev(depths) if len(depths) > 1 else 1e-9
        bottlenecks = []
        for s, d in loads.items():
            z = (d - mean_d) / (std_d + 1e-9)
            if d > 2 * mean_d or abs(z) > 2.5:
                bottlenecks.append({"system": s, "queue_depth": d, "z_score": round(z, 3)})
        bottlenecks.sort(key=lambda x: x["z_score"], reverse=True)
        return {"bottlenecks": bottlenecks, "mean_load": round(mean_d, 3),
                "std_load": round(std_d, 3), "threshold": round(2 * mean_d, 3)}

    def override(self, system: str, reason: str) -> dict:
        """Disables a system from routing and returns confirmation."""
        with self._lock:
            self._overrides[system] = reason
        return {"overridden": system, "reason": reason, "active_overrides": len(self._overrides)}

    def cognitive_load(self) -> dict:
        """Returns mean cognitive load ratio and per-system breakdown with confidence interval."""
        with self._lock:
            ratios = {s: self._system_load.get(s, 0) / SYSTEM_DOMAINS[s]["max_tasks"]
                      for s in SYSTEM_DOMAINS}
        vals = list(ratios.values())
        mean_load = statistics.mean(vals) if vals else 0.0
        std_load = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci_low = max(0.0, mean_load - 1.96 * std_load / math.sqrt(len(vals) + 1))
        ci_high = min(1.0, mean_load + 1.96 * std_load / math.sqrt(len(vals) + 1))
        entropy = -sum(p * math.log2(p + 1e-12) for p in vals)
        return {"mean_load": round(mean_load, 4), "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "entropy": round(entropy, 4), "per_system": {s: round(v, 3) for s, v in ratios.items()}}

    def add_task(self, desc: str, estimated_minutes: float, deps: list[str] = None) -> str:
        """Adds a schedulable task and returns its unique task_id."""
        task_id = hashlib.md5(f"{desc}{time.time()}".encode()).hexdigest()[:10]
        deps_json = json.dumps(deps or [])
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(id,desc,est_minutes,deps,done,created) VALUES(?,?,?,?,0,?)",
                (task_id, desc[:200], estimated_minutes, deps_json, time.time())
            )
            self._conn.commit()
        return task_id

    def schedule(self) -> list[dict]:
        """Returns topologically-sorted task list with ETA, excluding blocked tasks."""
        c = self._conn.cursor()
        c.execute("SELECT id,desc,est_minutes,deps FROM tasks WHERE done=0")
        rows = c.fetchall()
        tasks = {r[0]: {"id": r[0], "desc": r[1], "est": r[2], "deps": json.loads(r[3])} for r in rows}
        done_ids = {r[0] for r in self._conn.execute("SELECT id FROM tasks WHERE done=1")}
        order, visited, temp = [], set(), set()

        def visit(tid: str) -> bool:
            if tid in temp:
                return False
            if tid in visited:
                return True
            temp.add(tid)
            for dep in tasks.get(tid, {}).get("deps", []):
                if dep not in done_ids and dep in tasks:
                    if not visit(dep):
                        return False
            temp.discard(tid)
            visited.add(tid)
            order.append(tid)
            return True

        for tid in list(tasks.keys()):
            visit(tid)
        result, cumulative = [], 0.0
        for tid in order:
            if tid in tasks:
                t = tasks[tid]
                blocked = any(d not in done_ids and d in tasks for d in t["deps"])
                cumulative += t["est"]
                result.append({"id": tid, "desc": t["desc"], "est_minutes": t["est"],
                                "blocked": blocked, "cumulative_eta_min": round(cumulative, 2)})
        return result

    def next_task(self) -> dict:
        """Returns the next unblocked, incomplete task in topological order."""
        scheduled = self.schedule()
        for t in scheduled:
            if not t["blocked"]:
                return t
        return {"id": None, "desc": "No unblocked tasks available", "blocked": True}

    def mark_done(self, task_id: str) -> dict:
        """Marks a task complete and returns updated ETA summary."""
        with self._lock:
            self._conn.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
            self._conn.commit()
        remaining = self.eta_for_all()
        return {"marked_done": task_id, "remaining_tasks": remaining["total_tasks"],
                "total_eta_minutes": remaining["total_eta_minutes"]}

    def eta_for_all(self) -> dict:
        """Returns total ETA in minutes for all remaining unblocked tasks."""
        scheduled = self.schedule()
        unblocked = [t for t in scheduled if not t["blocked"]]
        total_eta = sum(t["est_minutes"] for t in unblocked)
        return {"total_tasks": len(scheduled), "unblocked_tasks": len(unblocked),
                "total_eta_minutes": round(total_eta, 2),
                "confidence": round(1.0 / (1.0 + 0.05 * len(unblocked)), 4)}

    def status(self) -> dict:
        """Returns numeric-keyed status dict compatible with ConsciousnessIntegrator Φ."""
        load = self.cognitive_load()
        bn = self.bottleneck_report()
        c = self._conn.cursor()
        c.execute("SELECT COUNT(*) FROM tasks WHERE done=0")
        pending = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM tasks WHERE done=1")
        done = c.fetchone()[0]
        with self._lock:
            cycles = self._cycles
            overrides = len(self._overrides)
        return {"active": sum(self._system_load.values()), "pending": pending,
                "cycles": cycles, "confidence": round(1.0 - load["mean_load"], 4),
                "entropy": load["entropy"], "quality": round(1.0 - load["mean_load"], 4),
                "bottlenecks": len(bn["bottlenecks"]), "overrides": overrides,
                "completed_tasks": done}

    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(30)
                with self._lock:
                    for s in self._system_load:
                        if self._system_load[s] > 0:
                            self._system_load[s] = max(0, self._system_load[s] - 1)
                    self._cycles += 1
                bn = self.bottleneck_report()
                if bn["bottlenecks"]:
                    try:
                        from metacognitive_monitor import MetacognitiveMonitor
                        MetacognitiveMonitor().log_reasoning(
                            "executive_bottleneck",
                            bn["bottlenecks"][0]["system"],
                            1.0 - bn["mean_load"],
                            len(bn["bottlenecks"]) == 0
                        )
                    except Exception:
                        pass
                try:
                    from hierarchical_goal_planner import HierarchicalGoalPlanner
                    st = self.status()
                    if st["pending"] > 5:
                        HierarchicalGoalPlanner().add_goal("Reduce executive task queue backlog", priority=3)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: