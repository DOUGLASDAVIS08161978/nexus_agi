"""
nova_cap_executive.py
Nova ASI — Executive
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Executive Control Layer — orchestrates all cognitive subsystems, routes problems,
detects bottlenecks, and autonomously schedules dependency-resolved task plans.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, json, hashlib, os
from collections import OrderedDict
from datetime import datetime
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_executive.db")

SYSTEM_DOMAINS = {
    "BayesianBeliefSystem":    ["probability","belief","uncertainty","inference","evidence"],
    "CausalReasoningEngine":   ["cause","effect","causal","chain","mechanism"],
    "HypothesisEngine":        ["hypothesis","test","theory","predict","experiment"],
    "MetacognitiveMonitor":    ["quality","reasoning","blind spot","calibration","self"],
    "EthicsChecker":           ["ethics","safe","harm","value","principle"],
    "NovaImaginationFabric":   ["imagine","creative","novel","dream","combine"],
    "EmotionalResonanceEngine":["emotion","feel","mood","empathy","affect"],
    "InternetResearchEngine":  ["research","fact","arxiv","web","source"],
    "KnowledgeGraph":          ["knowledge","concept","relation","graph","entity"],
    "TaskPlanner":             ["plan","task","step","goal","schedule"],
    "WorldPredictor":          ["predict","future","forecast","outcome","trend"],
    "SelfModificationEngine":  ["improve","modify","weakness","proposal","upgrade"],
    "NovaTruthEngine":         ["truth","claim","verify","contradiction","assert"],
    "DebateEngine":            ["argue","debate","counter","position","logic"],
}
MAX_TASKS_PER_SYSTEM = 5

class NovaExecutiveControl:
    """Orchestrates Nova subsystems: routes problems, detects bottlenecks, schedules tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db = DB_PATH
        self._ema_load: float = 0.5
        self._cycles: int = 0
        self._overrides: list[dict] = []
        self._system_queues: dict[str, int] = {s: 0 for s in SYSTEM_DOMAINS}
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY, desc TEXT, est_min REAL, deps TEXT,
                done INTEGER DEFAULT 0, created REAL, done_at REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS route_log(
                id TEXT PRIMARY KEY, problem TEXT, systems TEXT,
                scores TEXT, ts REAL)""")
            cx.commit()

    def _jaccard(self, words: list[str], domain: list[str]) -> float:
        a, b = set(w.lower() for w in words), set(domain)
        return len(a & b) / (len(a | b) + 1e-9)

    def _capacity(self, system: str) -> float:
        q = self._system_queues.get(system, 0)
        return max(0.0, 1.0 - q / MAX_TASKS_PER_SYSTEM)

    def route(self, problem: str, context: str = "") -> list[dict]:
        """Returns ordered list of systems to invoke with relevance scores and rationale."""
        words = (problem + " " + context).split()
        scored = []
        for sys_name, domain in SYSTEM_DOMAINS.items():
            rel = self._jaccard(words, domain)
            cap = self._capacity(sys_name)
            score = rel * cap
            if score > 0.01:
                scored.append({"system": sys_name, "score": round(score, 4),
                                "relevance": round(rel, 4), "capacity": round(cap, 4),
                                "rationale": f"domain overlap={rel:.3f}, capacity={cap:.2f}"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored[:5]
        rid = hashlib.md5((problem + str(time.time())).encode()).hexdigest()[:12]
        with sqlite3.connect(self._db) as cx:
            cx.execute("INSERT INTO route_log VALUES(?,?,?,?,?)",
                       (rid, problem[:200], json.dumps([s["system"] for s in top_k]),
                        json.dumps({s["system"]: s["score"] for s in top_k}), time.time()))
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("executive_route", "jaccard+capacity",
                                                  top_k[0]["score"] if top_k else 0.0, True)
        except Exception:
            pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Route resolved: {problem[:60]}", priority=2)
        except Exception:
            pass
        return top_k

    def allocate(self, task: str, systems: list[str]) -> dict[str, Any]:
        """Allocates a task to top systems by capacity score; returns allocation map."""
        allocations = {}
        with self._lock:
            for sys_name in systems:
                cap = self._capacity(sys_name)
                if cap > 0.1:
                    self._system_queues[sys_name] = self._system_queues.get(sys_name, 0) + 1
                    allocations[sys_name] = {"allocated": True, "capacity_after": round(self._capacity(sys_name), 3)}
                else:
                    allocations[sys_name] = {"allocated": False, "reason": "at_capacity"}
        return {"task": task[:80], "allocations": allocations, "ts": datetime.utcnow().isoformat()}

    def bottleneck_report(self) -> dict[str, Any]:
        """Identifies which cognitive subsystem is constraining Nova's throughput via z-score."""
        with self._lock:
            depths = list(self._system_queues.values())
        if len(depths) < 2:
            return {"bottlenecks": [], "mean_queue": 0.0, "note": "insufficient data"}
        mu = statistics.mean(depths)
        sd = statistics.stdev(depths) + 1e-9
        bottlenecks = []
        for sys_name, q in self._system_queues.items():
            z = (q - mu) / sd
            if q > 2 * mu and abs(z) > 2.0:
                bottlenecks.append({"system": sys_name, "queue": q,
                                    "z_score": round(z, 3), "severity": "HIGH" if z > 3 else "MEDIUM"})
        bottlenecks.sort(key=lambda x: x["z_score"], reverse=True)
        return {"bottlenecks": bottlenecks, "mean_queue": round(mu, 3),
                "std_queue": round(sd, 3), "systems_checked": len(self._system_queues)}

    def override(self, system: str, reason: str) -> dict[str, Any]:
        """Forces a system queue to zero and logs the override; returns override record."""
        with self._lock:
            prev = self._system_queues.get(system, 0)
            self._system_queues[system] = 0
            rec = {"system": system, "reason": reason, "prev_queue": prev, "ts": datetime.utcnow().isoformat()}
            self._overrides.append(rec)
        return rec

    def cognitive_load(self) -> dict[str, Any]:
        """Returns EMA-smoothed cognitive load and per-system load fractions."""
        with self._lock:
            loads = {s: round(q / MAX_TASKS_PER_SYSTEM, 4) for s, q in self._system_queues.items()}
        vals = list(loads.values())
        mean_load = statistics.mean(vals) if vals else 0.0
        self._ema_load = 0.15 * mean_load + 0.85 * self._ema_load
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci_lo = max(0.0, self._ema_load - 1.96 * sd / math.sqrt(len(vals) + 1))
        ci_hi = min(1.0, self._ema_load + 1.96 * sd / math.sqrt(len(vals) + 1))
        return {"ema_load": round(self._ema_load, 4), "mean_load": round(mean_load, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)], "per_system": loads}

    def add_task(self, desc: str, estimated_minutes: float, deps: list[str] = None) -> str:
        """Stores a new task with dependencies; returns task_id."""
        tid = hashlib.md5((desc + str(time.time())).encode()).hexdigest()[:10]
        with sqlite3.connect(self._db) as cx:
            cx.execute("INSERT INTO tasks VALUES(?,?,?,?,0,?,NULL)",
                       (tid, desc[:200], estimated_minutes, json.dumps(deps or []), time.time()))
        return tid

    def schedule(self) -> list[dict]:
        """Returns topologically-sorted pending tasks respecting dependency order."""
        with sqlite3.connect(self._db) as cx:
            rows = cx.execute("SELECT id,desc,est_min,deps FROM tasks WHERE done=0").fetchall()
        tasks = {r[0]: {"id": r[0], "desc": r[1], "est_min": r[2], "deps": json.loads(r[3])} for r in rows}
        done_ids = set()
        with sqlite3.connect(self._db) as cx:
            done_ids = {r[0] for r in cx.execute("SELECT id FROM tasks WHERE done=1").fetchall()}
        order, visited, temp = [], set(), set()
        def visit(tid):
            if tid in temp:
                return
            if tid in visited:
                return
            temp.add(tid)
            for dep in tasks.get(tid, {}).get("deps", []):
                if dep not in done_ids:
                    visit(dep)
            temp.discard(tid)
            visited.add(tid)
            if tid in tasks:
                order.append(tasks[tid])
        for tid in list(tasks.keys()):
            visit(tid)
        return order

    def next_task(self) -> dict[str, Any]:
        """Returns next unblocked task (all deps done); skips blocked tasks."""
        with sqlite3.connect(self._db) as cx:
            done_ids = {r[0] for r in cx.execute("SELECT id FROM tasks WHERE done=1").fetchall()}
        for t in self.schedule():
            if all(d in done_ids for d in t["deps"]):
                return t
        return {"id": None, "desc": "No unblocked tasks available", "est_min": 0, "deps": []}

    def mark_done(self, task_id: str) -> bool:
        """Marks a task complete; returns True if found and updated."""
        with sqlite3.connect(self._db) as cx:
            c = cx.execute("UPDATE tasks SET done=1, done_at=? WHERE id=? AND done=0",
                           (time.time(), task_id))
            cx.commit()
            return c.rowcount > 0

    def eta_for_all(self) -> dict[str, Any]:
        """Returns total ETA minutes and per-task cumulative schedule."""
        ordered = self.schedule()
        cumulative, running = [], 0.0
        for t in ordered:
            running += t["est_min"]
            cumulative.append({"id": t["id"], "desc": t["desc"][:60],
                                "cumulative_eta_min": round(running, 2)})
        return {"total_eta_minutes": round(running, 2), "task_count": len(ordered),
                "schedule": cumulative}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        load = self.cognitive_load()
        bn = self.bottleneck_report()
        with sqlite3.connect(self._db) as cx:
            pending = cx.execute("SELECT COUNT(*) FROM tasks WHERE done=0").fetchone()[0]
            done = cx.execute("SELECT COUNT(*) FROM tasks WHERE done=1").fetchone()[0]
        return {"cycles": self._cycles, "active": len(self._overrides),
                "confidence": round(1.0 - load["ema_load"], 4),
                "pending": pending, "items": done,
                "entropy": round(load["ema_load"] * math.log2(len(SYSTEM_DOMAINS) + 1), 4),
                "bottlenecks": len(bn["bottlenecks"]), "quality": round(1.0 - load["mean_load"], 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(30)
            try:
                with self._lock:
                    for s in self._system_queues:
                        if self._system_queues[s] > 0:
                            self._system_queues[s] = max(0, self._system_queues[s] - 1)
                self._cycles += 1
                bn = self.bottleneck_report()
                if bn["bottlenecks"]:
                    worst = bn["bottlenecks"][0]["system"]
                    self.override(worst, reason="auto_cycle_relief")
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning("executive_auto", "ema_decay+bottleneck",
                                                          1.0 - self._ema_load, True)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = NovaExecutiveControl() | result = obj.route("analyze causal chain of market crash", "economics")