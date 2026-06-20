"""
nova_cap_executive.py
Nova ASI — Executive
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Executive Control Layer — orchestrates all cognitive subsystems, routes problems,
detects bottlenecks, and autonomously schedules dependency-resolved tasks.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, json, os, hashlib, collections
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_executive.db")

SYSTEM_DOMAINS = {
    "BayesianBeliefSystem":    {"domain": ["probability","belief","uncertainty","evidence"], "max_tasks": 8},
    "CausalReasoningEngine":   {"domain": ["cause","effect","causal","mechanism","why"],    "max_tasks": 6},
    "HypothesisEngine":        {"domain": ["hypothesis","test","prediction","theory"],       "max_tasks": 6},
    "MetaCognitionEngine":     {"domain": ["meta","reflect","blind","pattern","insight"],    "max_tasks": 4},
    "AttentionFocusEngine":    {"domain": ["focus","attention","priority","distraction"],    "max_tasks": 5},
    "EthicsChecker":           {"domain": ["ethics","safe","harm","value","principle"],      "max_tasks": 5},
    "NovaImaginationFabric":   {"domain": ["imagine","creative","combine","dream","novel"],  "max_tasks": 4},
    "InternetResearchEngine":  {"domain": ["research","arxiv","fetch","knowledge","learn"],  "max_tasks": 3},
    "DeepEmotionEngine":       {"domain": ["emotion","feel","mood","sentiment","love"],      "max_tasks": 6},
    "SelfModificationEngine":  {"domain": ["improve","modify","weakness","proposal","grow"], "max_tasks": 3},
    "NovaTruthEngine":         {"domain": ["truth","claim","fact","verify","assert"],        "max_tasks": 5},
    "TaskPlanner":             {"domain": ["task","plan","step","goal","schedule"],          "max_tasks": 8},
    "ScientificSynthesizer":   {"domain": ["science","finding","evidence","consensus"],      "max_tasks": 4},
    "LongHorizonPlanner":      {"domain": ["horizon","long","future","strategy","replan"],   "max_tasks": 4},
}

class NovaExecutiveControl:
    """Orchestrates Nova subsystems: routes problems, allocates tasks, detects bottlenecks, schedules work."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, int] = {s: 0 for s in SYSTEM_DOMAINS}
        self._overrides: dict[str, str] = {}
        self._ema_load: float = 0.5
        self._cycle_count: int = 0
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_state()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS exec_tasks (
                id TEXT PRIMARY KEY, desc TEXT, est_min REAL, deps TEXT,
                done INTEGER DEFAULT 0, created_at REAL)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS exec_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, domain TEXT,
                decision TEXT, confidence REAL)""")

    def _load_state(self) -> None:
        try:
            rows = self._conn.execute("SELECT id, desc FROM exec_tasks WHERE done=0").fetchall()
            self._task_count = len(rows)
        except sqlite3.Error:
            self._task_count = 0

    def _relevance(self, problem: str, system: str) -> float:
        words = set(problem.lower().split())
        keywords = set(SYSTEM_DOMAINS[system]["domain"])
        overlap = len(words & keywords)
        idf_boost = math.log(len(SYSTEM_DOMAINS) / (overlap + 1) + 1)
        return overlap / (idf_boost + 1e-9) if overlap else 0.0

    def _capacity(self, system: str) -> float:
        max_t = SYSTEM_DOMAINS[system]["max_tasks"]
        active = self._active.get(system, 0)
        return 1.0 - min(active / max_t, 1.0)

    def route(self, problem: str, context: str = "") -> list[dict]:
        """Returns ordered list of systems to invoke with relevance scores and rationale."""
        combined = (problem + " " + context).lower()
        scores = {}
        with self._lock:
            for sys_name in SYSTEM_DOMAINS:
                if sys_name in self._overrides:
                    continue
                rel = self._relevance(combined, sys_name)
                cap = self._capacity(sys_name)
                score = rel * cap
                if score > 0:
                    scores[sys_name] = {"score": round(score, 4), "relevance": round(rel, 4),
                                        "capacity": round(cap, 4),
                                        "rationale": f"domain={SYSTEM_DOMAINS[sys_name]['domain'][:2]}"}
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
        result = [{"system": k, **v} for k, v in ranked]
        self._log("route", problem[:60], round(ranked[0][1]["score"], 3) if ranked else 0.0)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Execute routed plan for: {problem[:50]}", priority=7)
        except Exception:
            pass
        return result

    def allocate(self, task: str, systems: list[str]) -> dict:
        """Allocates task across given systems; returns allocation map with confidence intervals."""
        result = {}
        with self._lock:
            for s in systems:
                if s not in SYSTEM_DOMAINS:
                    continue
                cap = self._capacity(s)
                conf = cap * self._relevance(task, s)
                ci_low = max(0.0, conf - 0.15)
                ci_high = min(1.0, conf + 0.15)
                self._active[s] = min(self._active[s] + 1, SYSTEM_DOMAINS[s]["max_tasks"])
                result[s] = {"confidence": round(conf, 3), "ci": [round(ci_low, 3), round(ci_high, 3)],
                             "active_after": self._active[s]}
        self._log("allocate", task[:60], statistics.mean(v["confidence"] for v in result.values()) if result else 0.0)
        return result

    def bottleneck_report(self) -> dict:
        """Identifies which subsystem is constraining Nova's throughput via z-score analysis."""
        with self._lock:
            depths = {s: self._active[s] for s in SYSTEM_DOMAINS}
        vals = list(depths.values())
        mean_d = statistics.mean(vals)
        std_d = statistics.stdev(vals) if len(vals) > 1 else 1e-9
        bottlenecks = {}
        for s, d in depths.items():
            z = (d - mean_d) / (std_d + 1e-9)
            if d > 2 * mean_d or abs(z) > 2.0:
                bottlenecks[s] = {"queue_depth": d, "z_score": round(z, 3),
                                  "severity": "critical" if abs(z) > 3.0 else "moderate"}
        return {"bottlenecks": bottlenecks, "mean_queue": round(mean_d, 3),
                "most_constrained": max(depths, key=depths.get) if depths else None}

    def override(self, system: str, reason: str) -> dict:
        """Disables a system from routing; returns confirmation with timestamp."""
        with self._lock:
            self._overrides[system] = reason
        self._log("override", f"{system}:{reason[:40]}", 1.0)
        return {"overridden": system, "reason": reason, "ts": round(time.time(), 2)}

    def cognitive_load(self) -> dict:
        """Returns EMA-smoothed cognitive load across all systems with uncertainty bounds."""
        with self._lock:
            loads = []
            for s, cfg in SYSTEM_DOMAINS.items():
                active = self._active[s]
                cap = cfg["max_tasks"]
                loads.append(active / cap)
            raw = statistics.mean(loads) if loads else 0.0
            self._ema_load = 0.15 * raw + 0.85 * self._ema_load
            std = statistics.stdev(loads) if len(loads) > 1 else 0.0
        return {"ema_load": round(self._ema_load, 4), "raw_load": round(raw, 4),
                "std": round(std, 4), "ci": [round(max(0, self._ema_load - std), 3),
                                              round(min(1, self._ema_load + std), 3)],
                "overloaded": self._ema_load > 0.75}

    def add_task(self, desc: str, estimated_minutes: float, deps: list[str] = None) -> str:
        """Adds a scheduled task with dependencies; returns unique task_id."""
        tid = hashlib.md5(f"{desc}{time.time()}".encode()).hexdigest()[:10]
        dep_str = json.dumps(deps or [])
        with self._lock:
            with self._conn:
                self._conn.execute("INSERT INTO exec_tasks VALUES (?,?,?,?,0,?)",
                                   (tid, desc, estimated_minutes, dep_str, time.time()))
        return tid

    def schedule(self) -> list[dict]:
        """Returns topologically-sorted task list with ETA; blocked tasks marked separately."""
        rows = self._conn.execute(
            "SELECT id, desc, est_min, deps FROM exec_tasks WHERE done=0 ORDER BY created_at").fetchall()
        graph: dict[str, list] = {}
        meta: dict[str, dict] = {}
        for tid, desc, est, dep_str in rows:
            deps = json.loads(dep_str)
            graph[tid] = deps
            meta[tid] = {"desc": desc, "est_min": est, "deps": deps}
        order, visited, temp = [], set(), set()
        def visit(n: str) -> None:
            if n in temp:
                return
            if n not in visited:
                temp.add(n)
                for dep in graph.get(n, []):
                    if dep in graph:
                        visit(dep)
                temp.discard(n)
                visited.add(n)
                order.append(n)
        for node in list(graph.keys()):
            visit(node)
        eta_acc = 0.0
        result = []
        all_done_ids = {r[0] for r in self._conn.execute("SELECT id FROM exec_tasks WHERE done=1").fetchall()}
        for tid in order:
            deps = meta[tid]["deps"]
            blocked = any(d not in all_done_ids and d in graph for d in deps)
            eta_acc += meta[tid]["est_min"]
            result.append({"id": tid, "desc": meta[tid]["desc"], "est_min": meta[tid]["est_min"],
                           "blocked": blocked, "eta_minutes": round(eta_acc, 2)})
        return result

    def next_task(self) -> dict:
        """Returns next unblocked task in topological order; returns empty dict if none available."""
        for task in self.schedule():
            if not task["blocked"]:
                return task
        return {}

    def mark_done(self, task_id: str) -> dict:
        """Marks task complete; returns updated ETA for remaining tasks."""
        with self._lock:
            with self._conn:
                self._conn.execute("UPDATE exec_tasks SET done=1 WHERE id=?", (task_id,))
        remaining = self.eta_for_all()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("executive", "task_completion", 0.9, True)
        except Exception:
            pass
        return {"completed": task_id, "remaining_eta_minutes": remaining}

    def eta_for_all(self) -> float:
        """Returns total ETA in minutes for all remaining unblocked tasks."""
        tasks = self.schedule()
        return round(sum(t["est_min"] for t in tasks if not t["blocked"]), 2)

    def status(self) -> dict:
        """Returns plain dict with numeric keys for ConsciousnessIntegrator Φ calculation."""
        load = self.cognitive_load()
        pending = len(self.schedule())
        rows = self._conn.execute("SELECT COUNT(*) FROM exec_log").fetchone()
        cycles = rows[0] if rows else 0
        return {"active": sum(self._active.values()), "confidence": round(1.0 - load["ema_load"], 3),
                "pending": pending, "cycles": cycles, "entropy": round(load["std"], 4),
                "quality": round(1.0 - load["raw_load"], 3), "overrides": len(self._overrides),
                "items": len(SYSTEM_DOMAINS)}

    def _log(self, domain: str, decision: str, confidence: float) -> None:
        try:
            with self._conn:
                self._conn.execute("INSERT INTO exec_log (ts,domain,decision,confidence) VALUES (?,?,?,?)",
                                   (time.time(), domain, decision, confidence))
        except sqlite3.Error:
            pass

    def _auto_loop(self) -> None:
        while True:
            time.sleep(45)
            try:
                with self._lock:
                    for s in self._active:
                        if self._active[s] > 0:
                            self._active[s] = max(0, self._active[s] - 1)
                self._cycle_count += 1
                bn = self.bottleneck_report()
                if bn["bottlenecks"]:
                    try:
                        from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                        HierarchicalGoalPlanner().add_goal(
                            f"Resolve bottleneck in {list(bn['bottlenecks'].keys())[0]}", priority=9)
                    except Exception:
                        pass
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning(
                        "executive_auto", "bottleneck_scan",
                        round(1.0 - self._ema_load, 3), not bool(bn["bottlenecks"]))
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = NovaExecutiveControl() | result = obj.route("why does this hypothesis fail", "causal reasoning")
