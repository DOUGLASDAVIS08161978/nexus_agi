"""
Nova ASI — A
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
AdaptiveGoalOrchestrator — Nova ASI capability module.
Orchestrates goals with EMA-smoothed priority, topological execution ordering,
Bayesian deadline risk, Welford online stats, and autonomous cycle.
"""

import math
import time
import json
import sqlite3
import threading
import statistics
import os
import hashlib
from collections import OrderedDict, deque
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "adaptive_goal_orchestrator.db")

class AdaptiveGoalOrchestrator:
    """
    Autonomous goal orchestration engine with EMA priority adaptation,
    topological dependency resolution, and Bayesian deadline risk scoring.
    """

    def __init__(self) -> None:
        self._goals: dict[str, dict] = {}
        self._priority_weights: dict[str, float] = {}
        self._completion_rates: dict[str, float] = {}
        self._dependency_graph: dict[str, list[str]] = {}
        self._alpha: float = 0.15
        self._obs_count: int = 0
        self._category_stats: dict[str, tuple[float, float]] = {}
        self._deadline_pressure: dict[str, float] = {}
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._welford: dict[str, dict] = {}
        self._init_db()
        self._load_state()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()
        self._self_generate_goals()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS goals(
                    goal_id TEXT PRIMARY KEY, category TEXT,
                    created_at REAL, deadline REAL, progress REAL, status TEXT);
                CREATE TABLE IF NOT EXISTS goal_dependencies(
                    goal_id TEXT, prerequisite_id TEXT,
                    PRIMARY KEY(goal_id, prerequisite_id));
                CREATE TABLE IF NOT EXISTS goal_events(
                    event_id INTEGER PRIMARY KEY, goal_id TEXT,
                    event_type TEXT, timestamp REAL, payload TEXT);
                CREATE TABLE IF NOT EXISTS category_stats(
                    category TEXT PRIMARY KEY, mean_duration REAL,
                    std_duration REAL, sample_count INTEGER, completion_rate REAL);
            """)

    def _load_state(self) -> None:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                for row in conn.execute("SELECT goal_id,category,created_at,deadline,progress,status FROM goals WHERE status='active'"):
                    gid, cat, created, deadline, prog, status = row
                    self._goals[gid] = {"category": cat, "created_at": created,
                                        "deadline": deadline, "progress": prog, "status": status}
                    self._dependency_graph.setdefault(gid, [])
                for row in conn.execute("SELECT goal_id, prerequisite_id FROM goal_dependencies"):
                    self._dependency_graph.setdefault(row[0], []).append(row[1])
                for row in conn.execute("SELECT category, mean_duration, std_duration, sample_count, completion_rate FROM category_stats"):
                    cat, mean_d, std_d, n, cr = row
                    self._category_stats[cat] = (mean_d, std_d)
                    self._completion_rates[cat] = cr
                    self._welford[cat] = {"n": n, "mean": mean_d, "M2": (std_d**2) * max(n-1, 1)}
        except sqlite3.Error:
            pass

    def _compute_deadline_pressure(self, goal_id: str, now: float) -> float:
        g = self._goals.get(goal_id, {})
        created = g.get("created_at", now)
        deadline = g.get("deadline", now + 3600)
        total_span = max(deadline - created, 1.0)
        buffer = 0.1 * total_span
        k = 5.0
        exponent = -k * (now - deadline + buffer) / total_span
        return 1.0 / (1.0 + math.exp(exponent))

    def _compute_dependency_fanout(self, goal_id: str) -> float:
        dependents = sum(1 for deps in self._dependency_graph.values() if goal_id in deps)
        max_dep = max((sum(1 for deps in self._dependency_graph.values() if gid in deps)
                       for gid in self._goals), default=1)
        return dependents / (max_dep + 1)

    def _compute_priority(self, goal_id: str, now: float) -> float:
        g = self._goals.get(goal_id, {})
        progress = g.get("progress", 0.0)
        dp = self._compute_deadline_pressure(goal_id, now)
        self._deadline_pressure[goal_id] = dp
        fanout = self._compute_dependency_fanout(goal_id)
        return 0.4 * (1.0 - progress) + 0.4 * dp + 0.2 * fanout

    def register_goal(self, goal_id: str, category: str, deadline: float, dependencies: list[str]) -> str:
        """Stores a new goal, initialises EMA weight, returns goal_id."""
        now = time.time()
        with self._lock:
            self._goals[goal_id] = {"category": category, "created_at": now,
                                    "deadline": deadline, "progress": 0.0, "status": "active"}
            self._dependency_graph[goal_id] = list(dependencies)
            p = self._compute_priority(goal_id, now)
            self._priority_weights[goal_id] = p
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("INSERT OR REPLACE INTO goals VALUES(?,?,?,?,?,?)",
                                 (goal_id, category, now, deadline, 0.0, "active"))
                    for dep in dependencies:
                        conn.execute("INSERT OR IGNORE INTO goal_dependencies VALUES(?,?)", (goal_id, dep))
                    conn.execute("INSERT INTO goal_events(goal_id,event_type,timestamp,payload) VALUES(?,?,?,?)",
                                 (goal_id, "register", now, json.dumps({"category": category})))
            except sqlite3.Error:
                pass
        return goal_id

    def update_progress(self, goal_id: str, progress: float, timestamp: float) -> dict:
        """Updates goal progress, recomputes EMA priority, returns priority snapshot dict."""
        with self._lock:
            if goal_id not in self._goals:
                return {"error": "unknown goal"}
            self._goals[goal_id]["progress"] = max(0.0, min(1.0, progress))
            p = self._compute_priority(goal_id, timestamp)
            prev = self._priority_weights.get(goal_id, p)
            self._priority_weights[goal_id] = self._alpha * p + (1 - self._alpha) * prev
            self._obs_count += 1
            snapshot = {"goal_id": goal_id, "progress": progress,
                        "ema_priority": self._priority_weights[goal_id],
                        "raw_priority": p, "timestamp": timestamp}
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("UPDATE goals SET progress=? WHERE goal_id=?", (progress, goal_id))
                    conn.execute("INSERT INTO goal_events(goal_id,event_type,timestamp,payload) VALUES(?,?,?,?)",
                                 (goal_id, "progress", timestamp, json.dumps(snapshot)))
            except sqlite3.Error:
                pass
            try:
                from nova_system import NovaSystem
                NovaSystem.memory.store(goal_id, snapshot)
            except Exception:
                pass
        return snapshot

    def get_execution_order(self) -> list[str]:
        """Returns topologically sorted goal_ids via Kahn's algorithm, ties broken by EMA priority descending."""
        with self._lock:
            active = {g for g, d in self._goals.items() if d.get("status") == "active"}
            in_degree: dict[str, int] = {g: 0 for g in active}
            for g in active:
                for dep in self._dependency_graph.get(g, []):
                    if dep in active:
                        in_degree[g] = in_degree.get(g, 0) + 1
            queue = sorted([g for g in active if in_degree[g] == 0],
                           key=lambda x: -self._priority_weights.get(x, 0.0))
            order: list[str] = []
            while queue:
                node = queue.pop(0)
                order.append(node)
                dependents = [g for g in active if node in self._dependency_graph.get(g, [])]
                for dep in dependents:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
                        queue.sort(key=lambda x: -self._priority_weights.get(x, 0.0))
        return order

    def compute_priority_scores(self) -> dict[str, float]:
        """Returns mapping goal_id -> current EMA-smoothed priority for all active goals."""
        now = time.time()
        with self._lock:
            result = {}
            for gid, gdata in self._goals.items():
                if gdata.get("status") == "active":
                    p = self._compute_priority(gid, now)
                    self._priority_weights[gid] = (self._alpha * p +
                                                   (1 - self._alpha) * self._priority_weights.get(gid, p))
                    result[gid] = self._priority_weights[gid]
        return result

    def detect_deadline_risk(self, goal_id: str, current_time: float) -> tuple[float, bool]:
        """Returns (risk_probability, is_at_risk) where is_at_risk when risk > 0.65."""
        with self._lock:
            g = self._goals.get(goal_id, {})
            cat = g.get("category", "default")
            base_fail = 1.0 - self._completion_rates.get(cat, 0.5)
            dp = self._compute_deadline_pressure(goal_id, current_time)
            risk = max(0.0, min(1.0, base_fail * 0.5 + dp * 0.5))
            is_at_risk = risk > 0.65
            try:
                from nova_system import NovaSystem
                NovaSystem.monitor.emit_alert(goal_id, risk)
            except Exception:
                pass
        return (risk, is_at_risk)

    def record_completion(self, goal_id: str, actual_duration: float, success: bool) -> None:
        """Updates completion EMA and category stats via Welford's online algorithm."""
        with self._lock:
            g = self._goals.get(goal_id, {})
            cat = g.get("category", "default")
            outcome = 1.0 if success else 0.0
            self._completion_rates[cat] = (self._alpha * outcome +
                                           (1 - self._alpha) * self._completion_rates.get(cat, 0.5))
            if cat not in self._welford:
                self._welford[cat] = {"n": 0, "mean": 0.0, "M2": 0.0}
            w = self._welford[cat]
            w["n"] += 1
            delta = actual_duration - w["mean"]
            w["mean"] += delta / w["n"]
            delta2 = actual_duration - w["mean"]
            w["M2"] += delta * delta2
            std = math.sqrt(w["M2"] / max(w["n"] - 1, 1))
            self._category_stats[cat] = (w["mean"], std)
            mean_d, std_d = w["mean"], std
            z = (actual_duration - mean_d) / (std_d + 1e-9)
            if abs(z) > 2.5:
                pass
            if goal_id in self._goals:
                self._goals[goal_id]["status"] = "done" if success else "failed"
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("UPDATE goals SET status=? WHERE goal_id=?",
                                 (self._goals.get(goal_id, {}).get("status", "done"), goal_id))
                    conn.execute("INSERT OR REPLACE INTO category_stats VALUES(?,?,?,?,?)",
                                 (cat, mean_d, std_d, w["n"], self._completion_rates[cat]))
                    conn.execute("INSERT INTO goal_events(goal_id,event_type,timestamp,payload) VALUES(?,?,?,?)",
                                 (goal_id, "completion", time.time(),
                                  json.dumps({"duration": actual_duration, "success": success, "z": z})))
            except sqlite3.Error:
                pass

    def resolve_conflicts(self, goal_ids: list[str]) -> list[str]:
        """Returns re-ranked subset with resource-competing goals penalised by 0.75."""
        with self._lock:
            now = time.time()
            cat_seen: dict[str, int] = {}
            scores: dict[str, float] = {}
            for gid in goal_ids:
                if gid not in self._goals:
                    continue
                cat = self._goals[gid].get("category", "default")
                p = self._priority_weights.get(gid, self._compute_priority(gid, now))
                penalty = 0.75 ** cat_seen.get(cat, 0)
                scores[gid] = p * penalty
                cat_seen[cat] = cat_seen.get(cat, 0) + 1
        return sorted(scores, key=lambda x: -scores[x])

    def prune_stale_goals(self, current_time: float, ttl: float) -> list[str]:
        """Removes and returns goal_ids where age > ttl and progress < 0.05."""
        pruned: list[str] = []
        with self._lock:
            for gid in list(self._goals.keys()):
                g = self._goals[gid]
                age = current_time - g.get("created_at", current_time)
                if age > ttl and g.get("progress", 0.0) < 0.05 and g.get("status") == "active":
                    self._goals[gid]["status"] = "pruned"
                    pruned.append(gid)
                    try:
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute("UPDATE goals SET status='pruned' WHERE goal_id=?", (gid,))
                            conn.execute("INSERT INTO goal_events(goal_id,event_type,timestamp,payload) VALUES(?,?,?,?)",
                                         (gid, "pruned", current_time, "{}"))
                    except sqlite3.Error:
                        pass
        return pruned

    def explain_priority(self, goal_id: str) -> dict:
        """Returns breakdown dict with w1_term, w2_term, w3_term, ema_weight, deadline_pressure, risk_probability."""
        now = time.time()
        with self._lock:
            g = self._goals.get(goal_id, {})
            progress = g.get("progress", 0.0)
            dp = self._compute_deadline_pressure(goal_id, now)
            fanout = self._compute_dependency_fanout(goal_id)
            w1 = 0.4 * (1.0 - progress)
            w2 = 0.4 * dp
            w3 = 0.2 * fanout
            cat = g.get("category", "default")
            risk = 1.0 - self._completion_rates.get(cat, 0.5)
        return {"goal_id": goal_id, "w1_term": round(w1, 4), "w2_term": round(w2, 4),
                "w3_term": round(w3, 4), "ema_weight": round(self._priority_weights.get(goal_id, 0.0), 4),
                "deadline_pressure": round(dp, 4), "risk_probability": round(risk, 4),
                "total_raw": round(w1 + w2 + w3, 4)}

    def status(self) -> dict:
        """Returns plain dict with numeric keys for ConsciousnessIntegrator Φ."""
        with self._lock:
            active = sum(1 for g in self._goals.values() if g.get("status") == "active")
            scores = list(self._priority_weights.values())
            avg_conf = statistics.mean(scores) if scores else 0.0
            entropy_val = 0.0
            if scores:
                total = sum(scores) + 1e-12
                dist = [s / total for s in scores]
                entropy_val = -sum(p * math.log2(p + 1e-12) for p in dist)
        return {"active": active, "cycles": self._cycles, "observations": self._obs_count,
                "confidence": round(avg_conf, 4), "entropy": round(entropy_val, 4),
                "items": len(self._goals), "pending": active, "quality": round(1.0 - avg_conf * 0.1, 4)}

    def _self_generate_goals(self) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            planner = HierarchicalGoalPlanner()
            planner.add_goal("AdaptiveGoalOrchestrator: prune stale goals every 6h", priority=2)
            planner.add_goal("AdaptiveGoalOrchestrator: rebalance priority weights on new evidence", priority=3)
        except Exception:
            pass

    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(300)
                now = time.time()
                with self._lock:
                    scores = self.compute_priority_scores()
                pruned = self.prune_stale_goals(now, ttl=86400)
                for gid in list(self._goals.keys())[:5]:
                    risk, at_risk = self.detect_deadline_risk(gid, now)
                self._cycles += 1
                try:
                    from metacognitive_monitor import MetacognitiveMonitor
                    mon = MetacognitiveMonitor()
                    mon.log_reasoning("goal_orchestration", "ema_priority_kahn",
                                      confidence=statistics.mean(scores.values()) if scores else 0.5,
                                      success=True)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = AdaptiveGoalOrchestrator() | result = obj.register_goal("g1", "research", time.time()+3600, [])
