"""
Nova ASI — Goal Decomposition & Prioritisation
Proposed autonomously via /evolve
"""

"""
GoalDecomposer — Bayesian working-memory-backed goal decomposition and prioritisation engine.
Decomposes goals into subgoals, scores them via urgency*importance with exponential decay,
tracks completion, self-generates sub-goals, and runs an autonomous background cycle.
"""

import sqlite3
import threading
import time
import math
import statistics
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "goal_decomposer.db")

class GoalDecomposer:
    """Autonomous goal decomposition, prioritisation, and next-action engine with Bayesian decay."""

    _CAPACITY = 200
    _DECAY_RATE = 0.0004
    _CYCLE_INTERVAL = 45

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cycles: int = 0
        self._accuracy_history: List[Tuple[float, float]] = []
        self._ema_score: float = 0.5
        self._init_db()
        self._load_from_db()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS goals (
                gid TEXT PRIMARY KEY, goal TEXT, subgoals TEXT,
                importance REAL, urgency REAL, timestamp REAL,
                access_count INTEGER, completed INTEGER)""")
            cx.commit()

    def _load_from_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            for row in cx.execute("SELECT gid,goal,subgoals,importance,urgency,timestamp,access_count,completed FROM goals"):
                gid, goal, subgoals_json, imp, urg, ts, ac, comp = row
                self._store[gid] = {
                    "goal": goal, "subgoals": json.loads(subgoals_json),
                    "importance": imp, "urgency": urg, "timestamp": ts,
                    "access_count": ac, "completed": bool(comp)}

    def _save(self, gid: str) -> None:
        item = self._store[gid]
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""INSERT OR REPLACE INTO goals VALUES (?,?,?,?,?,?,?,?)""",
                (gid, item["goal"], json.dumps(item["subgoals"]),
                 item["importance"], item["urgency"], item["timestamp"],
                 item["access_count"], int(item["completed"])))
            cx.commit()

    def _gid(self, goal: str) -> str:
        return hashlib.md5(goal.encode()).hexdigest()[:12]

    def _decayed_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _salience(self, item: Dict[str, Any]) -> float:
        decay = self._decayed_importance(item)
        return decay * item["urgency"] * math.log1p(item["access_count"] + 1)

    def decompose(self, goal: str, subgoals_list: List[str],
                  urgency: float = 0.7, importance: float = 0.8) -> Dict[str, Any]:
        """Returns the stored goal record with gid, subgoals, and initial salience score."""
        gid = self._gid(goal)
        with self._lock:
            if len(self._store) >= self._CAPACITY:
                self.forget_least_important()
            self._store[gid] = {
                "goal": goal, "subgoals": [{"text": s, "done": False} for s in subgoals_list],
                "importance": max(0.01, min(1.0, importance)),
                "urgency": max(0.01, min(1.0, urgency)),
                "timestamp": time.time(), "access_count": 0, "completed": False}
            self._save(gid)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            hgp = HGP()
            hgp.add_goal(f"GoalDecomposer: {goal}", priority=int(urgency * 10))
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("goal_decompose", "salience_decay", importance * urgency, True)
        except Exception:
            pass
        return {"gid": gid, "goal": goal, "subgoals": subgoals_list,
                "salience": self._salience(self._store[gid])}

    def prioritise(self) -> List[Dict[str, Any]]:
        """Returns all incomplete goals sorted descending by urgency*importance*decay salience."""
        with self._lock:
            ranked = []
            for gid, item in self._store.items():
                if item["completed"]:
                    continue
                score = self._salience(item)
                ranked.append({"gid": gid, "goal": item["goal"],
                                "score": round(score, 4),
                                "urgency": item["urgency"],
                                "importance": item["importance"],
                                "pending_steps": sum(1 for s in item["subgoals"] if not s["done"])})
            ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def next_action(self) -> Optional[Dict[str, Any]]:
        """Returns the highest-priority incomplete subgoal step across all active goals."""
        ranked = self.prioritise()
        with self._lock:
            for entry in ranked:
                item = self._store.get(entry["gid"])
                if not item:
                    continue
                for idx, step in enumerate(item["subgoals"]):
                    if not step["done"]:
                        item["access_count"] += 1
                        item["importance"] = min(1.0, item["importance"] * 1.05)
                        self._save(entry["gid"])
                        return {"gid": entry["gid"], "goal": item["goal"],
                                "step_index": idx, "step": step["text"],
                                "score": entry["score"],
                                "confidence": round(self._bayesian_confidence(entry["score"]), 4)}
        return None

    def _bayesian_confidence(self, score: float) -> float:
        prior = 0.5
        likelihood = min(0.99, score)
        posterior = (likelihood * prior) / max(1e-9, likelihood * prior + (1 - likelihood) * (1 - prior))
        self._ema_score = 0.15 * posterior + 0.85 * self._ema_score
        return posterior

    def complete_step(self, gid: str, step_index: int) -> Dict[str, Any]:
        """Returns updated goal record after marking a subgoal step as done."""
        with self._lock:
            if gid not in self._store:
                return {"error": "unknown gid"}
            item = self._store[gid]
            if 0 <= step_index < len(item["subgoals"]):
                item["subgoals"][step_index]["done"] = True
            all_done = all(s["done"] for s in item["subgoals"])
            if all_done:
                item["completed"] = True
            self._save(gid)
            pred, actual = self._ema_score, 1.0 if all_done else 0.5
            self._accuracy_history.append((pred, actual))
            return {"gid": gid, "completed": item["completed"],
                    "steps_remaining": sum(1 for s in item["subgoals"] if not s["done"])}

    def forget_least_important(self) -> Optional[str]:
        """Returns gid of evicted goal (lowest salience LRU eviction) or None."""
        if not self._store:
            return None
        worst = min(self._store.keys(), key=lambda g: self._salience(self._store[g]))
        del self._store[worst]
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("DELETE FROM goals WHERE gid=?", (worst,))
            cx.commit()
        return worst

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ sampling."""
        with self._lock:
            active = sum(1 for i in self._store.values() if not i["completed"])
            pending = sum(sum(1 for s in i["subgoals"] if not s["done"]) for i in self._store.values())
            mae = statistics.mean(abs(p - a) for p, a in self._accuracy_history[-50:]) if self._accuracy_history else 0.0
            scores = [self._salience(i) for i in self._store.values() if not i["completed"]]
            entropy = 0.0
            if scores:
                total = sum(scores) + 1e-12
                dist = [s / total for s in scores]
                entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
            return {"items": len(self._store), "active": active, "pending": pending,
                    "cycles": self._cycles, "confidence": round(self._ema_score, 4),
                    "accuracy": round(1.0 - mae, 4), "entropy": round(entropy, 4),
                    "quality": round(self._ema_score * (1.0 - mae), 4)}

    def auto_cycle(self) -> Dict[str, Any]:
        """Returns cycle summary: reprioritises, logs reasoning, generates autonomous sub-goals."""
        ranked = self.prioritise()
        self._cycles += 1
        if ranked:
            top = ranked[0]
            scores = [r["score"] for r in ranked]
            if len(scores) > 2:
                mean_s = statistics.mean(scores)
                std_s = statistics.stdev(scores) + 1e-9
                z = (top["score"] - mean_s) / std_s
                if z > 3.0:
                    try:
                        from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                        HGP().add_goal(f"URGENT escalation: {top['goal']}", priority=10)
                    except Exception:
                        pass
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor as MCM
                MCM().log_reasoning("goal_decomposer_cycle", "ema_bayesian",
                                    self._ema_score, len(ranked) > 0)
            except Exception:
                pass
        st = self.status()
        return {"cycle": self._cycles, "ranked_count": len(ranked), **st}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(self._CYCLE_INTERVAL)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = GoalDecomposer() | result = obj.decompose("Launch product", ["Research","Build","Test"])