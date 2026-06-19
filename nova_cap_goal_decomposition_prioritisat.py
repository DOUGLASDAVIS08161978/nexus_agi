"""
Nova ASI — Goal Decomposition & Prioritisation
Proposed autonomously via /evolve
"""

"""
GoalOrchestrator — Bayesian goal decomposition with exponential-decay priority scoring,
LRU eviction, causal confidence propagation, and live cross-system integration.
Nova merges this module to plan, prioritise, and act on hierarchical goals in real time.
"""

import sqlite3
import threading
import math
import time
import statistics
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class GoalOrchestrator:
    """Decomposes goals into subgoals, scores by urgency*importance with Bayesian decay,
    evicts via LRU when capacity exceeded, and surfaces the highest-priority next action."""

    _DB = "nova_goal_orchestrator.db"
    _CAPACITY = 200
    _DECAY_RATE = 1.2e-5          # importance half-life ≈ 16 hours
    _ACCESS_BOOST = 0.08          # importance grows on each retrieval
    _MIN_CONFIDENCE = 0.05        # prune causal paths below this threshold

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._build_schema()
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._load_cache()

    # ------------------------------------------------------------------ schema
    def _build_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS goals (
                gid       TEXT PRIMARY KEY,
                parent    TEXT,
                text      TEXT NOT NULL,
                urgency   REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                status    TEXT DEFAULT 'pending',
                ts        REAL NOT NULL,
                accesses  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS history (
                hid       INTEGER PRIMARY KEY AUTOINCREMENT,
                gid       TEXT,
                event     TEXT,
                ts        REAL
            );
        """)
        self._conn.commit()

    def _load_cache(self) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT gid,parent,text,urgency,importance,confidence,status,ts,accesses FROM goals")
        for row in cur.fetchall():
            gid, parent, text, urgency, importance, confidence, status, ts, accesses = row
            self._cache[gid] = dict(gid=gid, parent=parent, text=text, urgency=urgency,
                                    importance=importance, confidence=confidence,
                                    status=status, ts=ts, accesses=accesses)

    def _gid(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def _decayed_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["ts"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _priority_score(self, item: Dict[str, Any]) -> float:
        d_imp = self._decayed_importance(item)
        return item["urgency"] * d_imp * item["confidence"]

    def _persist(self, item: Dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute("""INSERT OR REPLACE INTO goals
            (gid,parent,text,urgency,importance,confidence,status,ts,accesses)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (item["gid"], item["parent"], item["text"], item["urgency"],
             item["importance"], item["confidence"], item["status"],
             item["ts"], item["accesses"]))
        self._conn.commit()

    def _log(self, gid: str, event: str) -> None:
        self._conn.cursor().execute(
            "INSERT INTO history (gid,event,ts) VALUES (?,?,?)", (gid, event, time.time()))
        self._conn.commit()

    # ------------------------------------------------------------------ public
    def decompose(self, goal: str, subgoals: List[str],
                  urgency: float = 0.7, importance: float = 0.8) -> Dict[str, Any]:
        """Stores goal + subgoals as a causal chain; returns dict of created node IDs."""
        with self._lock:
            parent_id = self._gid(goal)
            now = time.time()
            root = dict(gid=parent_id, parent=None, text=goal, urgency=urgency,
                        importance=importance, confidence=1.0,
                        status="pending", ts=now, accesses=0)
            self._cache[parent_id] = root
            self._persist(root)
            self._log(parent_id, "decomposed")

            # Causal confidence propagates multiplicatively down the chain
            running_conf = 1.0
            child_ids: List[str] = []
            for i, sg in enumerate(subgoals):
                step_conf = max(1.0 - 0.07 * i, self._MIN_CONFIDENCE)
                running_conf = round(running_conf * step_conf, 6)
                if running_conf < self._MIN_CONFIDENCE:
                    break
                cid = self._gid(sg)
                child = dict(gid=cid, parent=parent_id, text=sg,
                             urgency=round(urgency * (1 - 0.05 * i), 4),
                             importance=round(importance * running_conf, 4),
                             confidence=running_conf,
                             status="pending", ts=now, accesses=0)
                self._cache[cid] = child
                self._persist(child)
                self._log(cid, "created")
                child_ids.append(cid)

            self._evict_if_needed()

            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                hgp = HGP()
                hgp.add_goal(goal, priority=int(urgency * 10))
            except Exception:
                pass

            return {"root": parent_id, "children": child_ids, "count": len(child_ids) + 1}

    def prioritise(self) -> List[Dict[str, Any]]:
        """Returns all pending goals sorted descending by urgency × decayed_importance × confidence."""
        with self._lock:
            pending = [v for v in self._cache.values() if v["status"] == "pending"]
            ranked = sorted(pending, key=self._priority_score, reverse=True)
            result = []
            for item in ranked:
                score = self._priority_score(item)
                result.append({**item, "priority_score": round(score, 6),
                                "decayed_importance": round(self._decayed_importance(item), 4)})
            return result

    def next_action(self) -> Optional[Dict[str, Any]]:
        """Returns the highest-priority incomplete leaf node (no pending children)."""
        with self._lock:
            pending = {v["gid"]: v for v in self._cache.values() if v["status"] == "pending"}
            parent_ids = {v["parent"] for v in pending.values() if v["parent"] in pending}
            leaves = [v for gid, v in pending.items() if gid not in parent_ids]
            if not leaves:
                return None
            best = max(leaves, key=self._priority_score)
            best["accesses"] += 1
            best["importance"] = min(1.0, best["importance"] + self._ACCESS_BOOST)
            self._persist(best)
            self._cache.move_to_end(best["gid"])

            try:
                from AttentionManager import AttentionManager
                am = AttentionManager()
                am.focus_on(best["text"])
            except Exception:
                pass

            return {**best, "priority_score": round(self._priority_score(best), 6)}

    def complete(self, goal_text: str) -> Dict[str, Any]:
        """Marks a goal complete, logs it, and returns Bayesian posterior confidence update."""
        with self._lock:
            gid = self._gid(goal_text)
            if gid not in self._cache:
                return {"error": "goal not found", "gid": gid}
            item = self._cache[gid]
            item["status"] = "complete"
            prior = item["confidence"]
            likelihood = 0.9
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            item["confidence"] = round(posterior, 6)
            self._persist(item)
            self._log(gid, "completed")

            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                mm = MetacognitiveMonitor()
                mm.log_reasoning("goal_completion", goal_text, item["confidence"], True)
            except Exception:
                pass

            return {"gid": gid, "prior_confidence": prior, "posterior_confidence": item["confidence"]}

    def consolidate(self) -> Dict[str, Any]:
        """Runs EMA smoothing on importance scores and prunes completed goals; returns stats."""
        with self._lock:
            ema = 0.5
            alpha = 0.15
            pruned = 0
            for item in list(self._cache.values()):
                if item["status"] == "complete":
                    del self._cache[item["gid"]]
                    pruned += 1
                    continue
                d_imp = self._decayed_importance(item)
                ema = alpha * d_imp + (1 - alpha) * ema
                item["importance"] = round(min(1.0, ema), 6)
                self._persist(item)

            scores = [self._priority_score(v) for v in self._cache.values()]
            mean_score = round(statistics.mean(scores), 6) if scores else 0.0
            std_score = round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0
            return {"remaining": len(self._cache), "pruned": pruned,
                    "mean_priority": mean_score, "std_priority": std_score}

    def capacity_used(self) -> Dict[str, Any]:
        """Returns current item count, capacity ceiling, and percentage utilisation."""
        with self._lock:
            used = len(self._cache)
            pct = round(100 * used / self._CAPACITY, 2)
            return {"used": used, "capacity": self._CAPACITY, "pct": pct}

    def forget_least_important(self, n: int = 10) -> List[str]:
        """Evicts n lowest-priority items from cache and DB; returns list of evicted IDs."""
        with self._lock:
            ranked = sorted(self._cache.values(), key=self._priority_score)
            evicted: List[str] = []
            for item in ranked[:n]:
                self._conn.cursor().execute("DELETE FROM goals WHERE gid=?", (item["gid"],))
                del self._cache[item["gid"]]
                evicted.append(item["gid"])
            self._conn.commit()
            return evicted

    def status(self) -> Dict[str, Any]:
        """Returns health dict for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            pending = sum(1 for v in self._cache.values() if v["status"] == "pending")
            complete = sum(1 for v in self._cache.values() if v["status"] == "complete")
            scores = [self._priority_score(v) for v in self._cache.values() if v["status"] == "pending"]
            mean_conf = round(statistics.mean(v["confidence"] for v in self._cache.values()), 4) \
                if self._cache else 0.0
            return {"items": len(self._cache), "pending": pending, "complete": complete,
                    "confidence": mean_conf,
                    "mean_priority": round(statistics.mean(scores), 4) if scores else 0.0,
                    "capacity_pct": round(100 * len(self._cache) / self._CAPACITY, 2)}

    # ------------------------------------------------------------------ private
    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._CAPACITY:
            lru_id, _ = next(iter(self._cache.items()))
            self._conn.cursor().execute("DELETE FROM goals WHERE gid=?", (lru_id,))
            del self._cache[lru_id]
        self._conn.commit()

# Usage: obj = GoalOrchestrator() | result = obj.decompose("Launch product", ["Research","Build","Test"])