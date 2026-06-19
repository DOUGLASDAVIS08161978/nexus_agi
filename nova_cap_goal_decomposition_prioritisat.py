"""
Nova ASI — Goal Decomposition & Prioritisation
Proposed autonomously via /evolve
"""

"""
GoalOrchestrator — Bayesian goal decomposition with exponential decay, LRU eviction,
and probabilistic prioritisation for Nova's goal-directed cognition layer.
"""

import sqlite3
import threading
import math
import time
import statistics
import json
import os
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class GoalOrchestrator:
    """
    Decomposes goals into subgoals, prioritises by urgency*importance with Bayesian
    confidence weighting, and tracks completion with exponential decay and LRU eviction.
    """

    _DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_orchestrator.db")
    _CAPACITY = 200
    _DECAY_RATE = 1.5e-4      # importance half-life ≈ 77 minutes
    _IMPORTANCE_BOOST = 1.15  # multiplicative boost on each access
    _PRUNE_THRESHOLD = 0.05   # causal chain paths below this are dropped

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._load_from_db()

    # ------------------------------------------------------------------ setup
    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    gid TEXT PRIMARY KEY,
                    parent_gid TEXT,
                    description TEXT NOT NULL,
                    urgency REAL DEFAULT 0.5,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    completed INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )""")

    def _load_from_db(self) -> None:
        cur = self._conn.execute(
            "SELECT gid,parent_gid,description,urgency,importance,confidence,"
            "completed,access_count,created_at,updated_at FROM goals ORDER BY created_at"
        )
        for row in cur.fetchall():
            gid, par, desc, urg, imp, conf, comp, acc, cat, uat = row
            self._cache[gid] = {
                "gid": gid, "parent_gid": par, "description": desc,
                "urgency": urg, "importance": imp, "confidence": conf,
                "completed": bool(comp), "access_count": acc,
                "created_at": cat, "updated_at": uat
            }

    def _gid(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ decay & eviction
    def _decayed_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["updated_at"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _evict_lru_if_needed(self) -> None:
        while len(self._cache) > self._CAPACITY:
            lru_key, _ = next(iter(self._cache.items()))
            self._cache.pop(lru_key)
            self._conn.execute("DELETE FROM goals WHERE gid=?", (lru_key,))
        self._conn.commit()

    def _touch(self, gid: str) -> None:
        if gid not in self._cache:
            return
        item = self._cache[gid]
        item["access_count"] += 1
        item["importance"] = min(1.0, self._decayed_importance(item) * self._IMPORTANCE_BOOST)
        item["updated_at"] = time.time()
        self._cache.move_to_end(gid)
        self._conn.execute(
            "UPDATE goals SET access_count=?,importance=?,updated_at=? WHERE gid=?",
            (item["access_count"], item["importance"], item["updated_at"], gid)
        )
        self._conn.commit()

    # ------------------------------------------------------------------ public API
    def decompose(self, goal: str, subgoals_list: List[str],
                  urgency: float = 0.5, importance: float = 0.5) -> Dict[str, Any]:
        """Returns a dict with parent gid and list of child gids after storing the decomposition."""
        with self._lock:
            now = time.time()
            pgid = self._gid(goal)
            if pgid not in self._cache:
                parent = {
                    "gid": pgid, "parent_gid": None, "description": goal,
                    "urgency": float(urgency), "importance": float(importance),
                    "confidence": 0.6, "completed": False,
                    "access_count": 0, "created_at": now, "updated_at": now
                }
                self._cache[pgid] = parent
                self._conn.execute(
                    "INSERT OR REPLACE INTO goals VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pgid, None, goal, urgency, importance, 0.6, 0, 0, now, now)
                )

            child_gids: List[str] = []
            for idx, sub in enumerate(subgoals_list):
                # urgency decays slightly for deeper subgoals (causal chain: 0.9^depth)
                child_urg = round(urgency * (0.9 ** (idx + 1)), 4)
                child_imp = round(importance * 0.95, 4)
                child_conf = 0.5
                cgid = self._gid(f"{pgid}:{sub}")
                if cgid not in self._cache:
                    child = {
                        "gid": cgid, "parent_gid": pgid, "description": sub,
                        "urgency": child_urg, "importance": child_imp,
                        "confidence": child_conf, "completed": False,
                        "access_count": 0, "created_at": now, "updated_at": now
                    }
                    self._cache[cgid] = child
                    self._conn.execute(
                        "INSERT OR REPLACE INTO goals VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (cgid, pgid, sub, child_urg, child_imp, child_conf, 0, 0, now, now)
                    )
                child_gids.append(cgid)

            self._conn.commit()
            self._evict_lru_if_needed()
            return {"parent_gid": pgid, "child_gids": child_gids, "total": len(child_gids)}

    def prioritise(self) -> List[Dict[str, Any]]:
        """Returns all incomplete goals sorted descending by urgency*importance*confidence with decay applied."""
        with self._lock:
            ranked: List[Tuple[float, Dict[str, Any]]] = []
            for item in self._cache.values():
                if item["completed"]:
                    continue
                dec_imp = self._decayed_importance(item)
                # Bayesian score: posterior ∝ urgency * decayed_importance * confidence
                score = item["urgency"] * dec_imp * item["confidence"]
                if score >= self._PRUNE_THRESHOLD:
                    ranked.append((score, {**item, "score": round(score, 6),
                                           "decayed_importance": round(dec_imp, 6)}))
            ranked.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in ranked]

    def next_action(self) -> Optional[Dict[str, Any]]:
        """Returns the highest-priority incomplete leaf subgoal (no children) or None if all complete."""
        with self._lock:
            parent_gids = {v["parent_gid"] for v in self._cache.values() if v["parent_gid"]}
            ranked = self.prioritise()
            for item in ranked:
                if item["gid"] not in parent_gids:   # leaf node
                    self._touch(item["gid"])
                    return item
            return ranked[0] if ranked else None

    def complete(self, gid: str) -> Dict[str, Any]:
        """Marks a goal complete, logs Bayesian confidence update, returns updated item."""
        with self._lock:
            if gid not in self._cache:
                return {"error": "gid not found", "gid": gid}
            item = self._cache[gid]
            item["completed"] = True
            # Bayesian update: success observation → confidence posterior rises
            prior = item["confidence"]
            likelihood_success = 0.9
            posterior = (likelihood_success * prior) / (
                likelihood_success * prior + (1 - likelihood_success) * (1 - prior)
            )
            item["confidence"] = round(posterior, 6)
            item["updated_at"] = time.time()
            self._conn.execute(
                "UPDATE goals SET completed=1,confidence=?,updated_at=? WHERE gid=?",
                (item["confidence"], item["updated_at"], gid)
            )
            self._conn.commit()
            return {**item}

    def capacity_used(self) -> Dict[str, Any]:
        """Returns dict with count, capacity, fill_ratio, and mean decayed importance."""
        with self._lock:
            n = len(self._cache)
            importances = [self._decayed_importance(v) for v in self._cache.values()]
            mean_imp = round(statistics.mean(importances), 6) if importances else 0.0
            std_imp = round(statistics.stdev(importances), 6) if len(importances) > 1 else 0.0
            return {
                "items": n,
                "capacity": self._CAPACITY,
                "fill_ratio": round(n / self._CAPACITY, 4),
                "mean_decayed_importance": mean_imp,
                "std_decayed_importance": std_imp,
                "confidence": round(mean_imp, 4)
            }

    def forget_least_important(self, n: int = 10) -> List[str]:
        """Evicts the n lowest decayed-importance items and returns their gids."""
        with self._lock:
            scored = [(self._decayed_importance(v), k) for k, v in self._cache.items()]
            scored.sort(key=lambda x: x[0])
            evicted: List[str] = []
            for _, gid in scored[:n]:
                self._cache.pop(gid, None)
                self._conn.execute("DELETE FROM goals WHERE gid=?", (gid,))
                evicted.append(gid)
            self._conn.commit()
            return evicted

    def status(self) -> Dict[str, Any]:
        """Returns health dict compatible with ConsciousnessIntegrator Φ computation."""
        cap = self.capacity_used()
        ranked = self.prioritise()
        completed = sum(1 for v in self._cache.values() if v["completed"])
        total = len(self._cache)
        scores = [r["score"] for r in ranked]
        mean_score = round(statistics.mean(scores), 6) if scores else 0.0
        entropy_val = 0.0
        if scores:
            total_s = sum(scores) + 1e-12
            dist = [s / total_s for s in scores]
            entropy_val = round(-sum(p * math.log2(p + 1e-12) for p in dist), 4)
        return {
            "items": total,
            "pending": len(ranked),
            "completed": completed,
            "confidence": cap["confidence"],
            "fill_ratio": cap["fill_ratio"],
            "mean_priority_score": mean_score,
            "entropy": entropy_val,
            "active": 1 if ranked else 0
        }

# Usage: obj = GoalOrchestrator() | result = obj.decompose("Build AGI", ["Research","Prototype","Evaluate"])