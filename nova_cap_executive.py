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

import sqlite3, threading, time, math, statistics, json, hashlib, os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_executive.db")

SYSTEM_DOMAINS = {
    "CausalReasoningEngine":   ["cause","effect","why","reason","mechanism","chain"],
    "HypothesisEngine":        ["hypothesis","test","predict","theory","experiment"],
    "BayesianBeliefSystem":    ["probability","belief","prior","posterior","uncertain","confidence"],
    "EmotionTracker":          ["emotion","feel","mood","sentiment","affect"],
    "EthicsChecker":           ["ethics","safe","harm","moral","value","principle"],
    "InternetResearchEngine":  ["research","search","find","arxiv","url","knowledge","fact"],
    "MetaCognitionEngine":     ["meta","reflect","blind","pattern","insight","self"],
    "TaskPlanner":             ["task","plan","step","goal","schedule","action"],
    "KnowledgeGraph":          ["concept","relation","entity","graph","link","node"],
    "NovaImaginationFabric":   ["imagine","creative","combine","dream","novel","invent"],
    "DebateEngine":            ["argue","debate","pro","con","against","for","position"],
    "WorldPredictor":          ["predict","future","outcome","forecast","next","trend"],
    "ScientificSynthesizer":   ["science","finding","evidence","consensus","study","paper"],
    "SelfModificationEngine":  ["improve","weakness","modify","proposal","upgrade","fix"],
}

MAX_TASKS_PER_SYSTEM = 6

class NovaExecutiveControl:
    """Orchestrates Nova subsystems: routes problems, allocates capacity, schedules tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db  = self._init_db()
        self._system_load: dict[str,int] = {s: 0 for s in SYSTEM_DOMAINS}
        self._overrides:   dict[str,str] = {}
        self._cycles       = 0
        self._routing_history: list[dict] = []
        self._start_auto()
        self._seed_goals()

    # ── DB ────────────────────────────────────────────────────────────────────
    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("""CREATE TABLE IF NOT EXISTS exec_tasks(
            id TEXT PRIMARY KEY, desc TEXT, est_min REAL, deps TEXT,
            done INTEGER DEFAULT 0, created REAL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS routing_log(
            id TEXT PRIMARY KEY, problem TEXT, systems TEXT,
            scores TEXT, ts REAL)""")
        conn.commit()
        return conn

    # ── Routing ───────────────────────────────────────────────────────────────
    def route(self, problem: str, context: str = "") -> list[dict]:
        """Returns ordered list of systems to invoke with relevance scores and rationale."""
        tokens = set((problem + " " + context).lower().split())
        scored: list[tuple[float,str]] = []
        with self._lock:
            for sys_name, keywords in SYSTEM_DOMAINS.items():
                if sys_name in self._overrides:
                    continue
                overlap     = len(tokens & set(keywords))
                tf_score    = overlap / (len(tokens) + 1e-9)
                idf         = math.log(len(SYSTEM_DOMAINS) / (1 + (1 if overlap > 0 else 0)))
                relevance   = tf_score * (idf + 1.0)
                active      = self._system_load.get(sys_name, 0)
                capacity    = 1.0 - (active / MAX_TASKS_PER_SYSTEM)
                score       = relevance * max(capacity, 0.05)
                scored.append((score, sys_name))
        scored.sort(reverse=True)
        top_k = scored[:5]
        result = [{"system": s, "score": round(sc, 4),
                   "rationale": f"relevance×capacity={sc:.4f}",
                   "confidence": round(min(sc * 2.5, 1.0), 3)}
                  for sc, s in top_k if sc > 0.0]
        rid = hashlib.md5(f"{problem}{time.time()}".encode()).hexdigest()[:10]
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO routing_log VALUES(?,?,?,?,?)",
                (rid, problem[:200],
                 json.dumps([r["system"] for r in result]),
                 json.dumps([r["score"]  for r in result]),
                 time.time()))
            self._db.commit()
            self._routing_history.append({"problem": problem[:80], "top": result[0]["system"] if result else "none"})
            if len(self._routing_history) > 200:
                self._routing_history.pop(0)
        self._log_meta("route", f"routed '{problem[:40]}' → {result[0]['system'] if result else 'none'}", 0.82)
        return result

    # ── Allocation ────────────────────────────────────────────────────────────
    def allocate(self, task: str, systems: list[str]) -> dict[str, Any]:
        """Allocates a task to the highest-capacity systems; returns allocation map."""
        allocation: dict[str,str] = {}
        with self._lock:
            for s in systems:
                if s not in SYSTEM_DOMAINS:
                    allocation[s] = "unknown_system"
                    continue
                if s in self._overrides:
                    allocation[s] = f"overridden: {self._overrides[s]}"
                    continue
                load = self._system_load.get(s, 0)
                if load < MAX_TASKS_PER_SYSTEM:
                    self._system_load[s] = load + 1
                    allocation[s] = "allocated"
                else:
                    allocation[s] = "at_capacity"
        self._log_meta("allocate", f"task='{task[:40]}' systems={systems}", 0.78)
        return {"task": task, "allocation": allocation, "timestamp": round(time.time(), 2)}

    # ── Bottleneck ────────────────────────────────────────────────────────────
    def bottleneck_report(self) -> dict[str, Any]:
        """Identifies which subsystem is constraining Nova's throughput via z-score."""
        with self._lock:
            loads = list(self._system_load.values())
        if len(loads) < 2:
            return {"bottleneck": None, "reason": "insufficient_data"}
        mean_load = statistics.mean(loads)
        std_load  = statistics.stdev(loads) + 1e-9
        bottlenecks = []
        for sys_name, load in self._system_load.items():
            z = (load - mean_load) / std_load
            if z > 2.0:
                bottlenecks.append({"system": sys_name, "load": load,
                                    "z_score": round(z, 3), "severity": "critical"})
            elif load > 2 * mean_load:
                bottlenecks.append({"system": sys_name, "load": load,
                                    "z_score": round(z, 3), "severity": "warning"})
        bottlenecks.sort(key=lambda x: x["z_score"], reverse=True)
        return {"bottlenecks": bottlenecks, "mean_load": round(mean_load, 3),
                "std_load": round(std_load, 3), "total_systems": len(self._system_load)}

    # ── Override ──────────────────────────────────────────────────────────────
    def override(self, system: str, reason: str) -> dict[str, str]:
        """Disables a system from routing; returns confirmation dict."""
        with self._lock:
            self._overrides[system] = reason
        self._log_meta("override", f"disabled {system}: {reason}", 0.95)
        return {"system": system, "status": "overridden", "reason": reason}

    # ── Cognitive Load ────────────────────────────────────────────────────────
    def cognitive_load(self) -> dict[str, Any]:
        """Returns mean cognitive load across all systems with confidence interval."""
        with self._lock:
            loads = [v / MAX_TASKS_PER_SYSTEM for v in self._system_load.values()]
        mean_l = statistics.mean(loads)
        std_l  = statistics.stdev(loads) + 1e-9 if len(loads) > 1 else 0.0
        ci_95  = 1.96 * std_l / math.sqrt(len(loads))
        entropy = -sum(p * math.log2(p + 1e-12) for p in loads if p > 0)
        return {"mean_load": round(mean_l, 4), "std": round(std_l, 4),
                "ci_95_lower": round(max(mean_l - ci_95, 0), 4),
                "ci_95_upper": round(min(mean_l + ci_95, 1), 4),
                "entropy": round(entropy, 4), "overridden_count": len(self._overrides)}

    # ── Task Scheduling ───────────────────────────────────────────────────────
    def add_task(self, desc: str, estimated_minutes: float, deps: list[str] = []) -> str:
        """Adds a task with ETA and dependencies; returns task_id."""
        tid = hashlib.md5(f"{desc}{time.time()}".encode()).hexdigest()[:12]
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO exec_tasks VALUES(?,?,?,?,0,?)",
                (tid, desc[:300], estimated_minutes, json.dumps(deps), time.time()))
            self._db.commit()
        return tid

    def schedule(self) -> list[dict]:
        """Returns topologically-sorted task list with cumulative ETA."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id,desc,est_min,deps FROM exec_tasks WHERE done=0 ORDER BY created").fetchall()
        tasks = {r[0]: {"id":r[0],"desc":r[1],"est":r[2],"deps":json.loads(r[3])} for r in rows}
        order, visited, temp = [], set(), set()
        def visit(tid: str) -> None:
            if tid in temp:
                return
            if tid in visited:
                return
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
        cum, result = 0.0, []
        for tid in order:
            t = tasks[tid]
            cum += t["est"]
            result.append({**t, "cumulative_eta_min": round(cum, 2)})
        return result

    def next_task(self) -> dict:
        """Returns the next unblocked task (all deps done); empty dict if none."""
        with self._lock:
            done_ids = {r[0] for r in self._db.execute(
                "SELECT id FROM exec_tasks WHERE done=1").fetchall()}
        for t in self.schedule():
            if all(d in done_ids for d in t["deps"]):
                return t
        return {}

    def mark_done(self, task_id: str) -> bool:
        """Marks a task complete; returns True on success."""
        with self._lock:
            self._db.execute("UPDATE exec_tasks SET done=1 WHERE id=?", (task_id,))
            self._db.commit()
        return True

    def eta_for_all(self) -> dict[str, Any]:
        """Returns total ETA in minutes for all remaining tasks."""
        sched = self.schedule()
        total = sum(t["est"] for t in sched)
        return {"remaining_tasks": len(sched), "total_eta_minutes": round(total, 2),
                "tasks": [{k: v for k, v in t.items() if k != "deps"} for t in sched]}

    # ── Status ────────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        load = self.cognitive_load()
        with self._lock:
            pending = self._db.execute(
                "SELECT COUNT(*) FROM exec_tasks WHERE done=0").fetchone()[0]
            cycles  = self._cycles
        return {"active": sum(self._system_load.values()),
                "pending": pending, "cycles": cycles,
                "confidence": round(1.0 - load["mean_load"], 4),
                "entropy": load["entropy"],
                "items": len(SYSTEM_DOMAINS),
                "overrides": len(self._overrides)}

    # ── Internals ─────────────────────────────────────────────────────────────
    def _log_meta(self, domain: str, approach: str, conf: float) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, approach, conf, True)
        except Exception:
            pass

    def _seed_goals(self) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal("Executive: resolve all cognitive bottlenecks", priority=9)
            hgp.add_goal("Executive: schedule and complete pending task queue", priority=7)
        except Exception:
            pass

    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(45)
                with self._lock:
                    self._cycles += 1
                    for s in self._system_load:
                        if self._system_load[s] > 0:
                            self._system_load[s] = max(0, self._system_load[s] - 1)
                bn = self.bottleneck_report()
                if bn.get("bottlenecks"):
                    self._log_meta("auto_cycle",
                        f"bottleneck={bn['bottlenecks'][0]['system']}", 0.75)
                try:
                    from consciousness_integrator import ConsciousnessIntegrator
                    ConsciousnessIntegrator().integrate("NovaExecutiveControl", self.status())
                except Exception:
                    pass
            except Exception:
                pass

    def _start_auto(self) -> None:
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

# Usage: obj = NovaExecutiveControl() | result = obj.route("why does X cause Y?", "causal