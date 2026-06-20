"""
Nova ASI — Inter-Agent Communication
Proposed autonomously via /evolve
"""

"""
AgentMessageBus — Nova's inter-agent communication fabric with Bayesian working memory,
exponential decay, LRU eviction, probabilistic routing, and autonomous goal generation.
"""

import sqlite3
import threading
import time
import math
import statistics
import json
import hashlib
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

class AgentMessageBus:
    """Persistent, self-monitoring inter-agent message bus with Bayesian memory and autonomous operation."""

    _DECAY_RATE: float = 0.0005
    _CAPACITY: int = 200
    _DB_PATH: str = "nova_agent_bus.db"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._agent_registry: Dict[str, float] = {}   # agent_id -> last_seen
        self._cycles: int = 0
        self._sent: int = 0
        self._received: int = 0
        self._confidence: float = 0.85
        self._conn = sqlite3.connect(self._DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_from_db()
        self._start_daemon()
        self._bootstrap_goals()

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS                                                    #
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS messages (
            msg_id TEXT PRIMARY KEY, agent_id TEXT, payload TEXT,
            importance REAL, ts REAL, access_count INTEGER)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY, last_seen REAL)""")
        self._conn.commit()

    def _load_from_db(self) -> None:
        cur = self._conn.cursor()
        for row in cur.execute("SELECT msg_id,agent_id,payload,importance,ts,access_count FROM messages"):
            mid, aid, payload, imp, ts, ac = row
            self._memory[mid] = {"agent_id": aid, "payload": json.loads(payload),
                                 "importance": imp, "ts": ts, "access_count": ac}
        for row in cur.execute("SELECT agent_id, last_seen FROM agents"):
            self._agent_registry[row[0]] = row[1]

    def _persist_message(self, mid: str, item: Dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute("""INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?)""",
                    (mid, item["agent_id"], json.dumps(item["payload"]),
                     item["importance"], item["ts"], item["access_count"]))
        self._conn.commit()

    def _persist_agent(self, agent_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO agents VALUES (?,?)",
                    (agent_id, time.time()))
        self._conn.commit()

    def _decay_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["ts"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _msg_id(self, agent_id: str, payload: Any) -> str:
        raw = f"{agent_id}{json.dumps(payload, sort_keys=True)}{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _register_agent(self, agent_id: str) -> None:
        self._agent_registry[agent_id] = time.time()
        self._persist_agent(agent_id)

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _auto_loop(self) -> None:
        while True:
            time.sleep(30)
            self.auto_cycle()

    def _bootstrap_goals(self) -> None:
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal("Maintain reliable inter-agent message delivery with <1% loss", priority=8)
            hgp.add_goal("Expand known agent registry by monitoring broadcast replies", priority=5)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #
    def store(self, key: str, value: Any, importance: float = 0.5) -> str:
        """Stores a keyed value in Bayesian working memory; returns the storage key."""
        with self._lock:
            if len(self._memory) >= self._CAPACITY:
                self.forget_least_important()
            self._memory[key] = {"agent_id": "__wm__", "payload": value,
                                 "importance": max(0.0, min(1.0, importance)),
                                 "ts": time.time(), "access_count": 0}
            self._memory.move_to_end(key)
            self._persist_message(key, self._memory[key])
        return key

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieves a value from working memory, boosting importance on access; returns value or None."""
        with self._lock:
            item = self._memory.get(key)
            if item is None:
                return None
            item["importance"] = min(1.0, self._decay_importance(item) + 0.05)
            item["access_count"] += 1
            item["ts"] = time.time()
            self._memory.move_to_end(key)
            self._persist_message(key, item)
        return item["payload"]

    def send(self, agent_id: str, message: Any) -> str:
        """Queues a message for the target agent; returns the message ID."""
        with self._lock:
            self._register_agent(agent_id)
            mid = self._msg_id(agent_id, message)
            if len(self._memory) >= self._CAPACITY:
                self.forget_least_important()
            item = {"agent_id": agent_id, "payload": message,
                    "importance": 0.75, "ts": time.time(), "access_count": 0}
            self._memory[mid] = item
            self._memory.move_to_end(mid)
            self._persist_message(mid, item)
            self._sent += 1
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("agent_bus", "send", self._confidence, True)
        except Exception:
            pass
        return mid

    def receive(self, agent_id: str) -> List[Dict[str, Any]]:
        """Returns and clears all queued messages for agent_id; list may be empty."""
        results: List[Dict[str, Any]] = []
        with self._lock:
            self._register_agent(agent_id)
            to_delete = []
            for mid, item in self._memory.items():
                if item["agent_id"] == agent_id:
                    decayed_imp = self._decay_importance(item)
                    if decayed_imp > 0.01:
                        results.append({"id": mid, "payload": item["payload"],
                                        "importance": round(decayed_imp, 4),
                                        "age_s": round(time.time() - item["ts"], 1)})
                    to_delete.append(mid)
            for mid in to_delete:
                del self._memory[mid]
                self._conn.cursor().execute("DELETE FROM messages WHERE msg_id=?", (mid,))
            self._conn.commit()
            self._received += len(results)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("agent_bus", "receive", self._confidence, bool(results))
        except Exception:
            pass
        return results

    def broadcast(self, message: Any) -> Dict[str, str]:
        """Sends message to all known agents; returns dict of agent_id -> message_id."""
        with self._lock:
            agents = list(self._agent_registry.keys())
        result: Dict[str, str] = {}
        for aid in agents:
            result[aid] = self.send(aid, message)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Verify broadcast delivery to {len(agents)} agents", priority=6)
        except Exception:
            pass
        return result

    def consolidate(self) -> Dict[str, int]:
        """Prunes decayed messages and persists survivors; returns counts of kept and pruned."""
        kept, pruned = 0, 0
        with self._lock:
            to_prune = []
            for mid, item in list(self._memory.items()):
                if self._decay_importance(item) < 0.02:
                    to_prune.append(mid)
                else:
                    kept += 1
            for mid in to_prune:
                del self._memory[mid]
                self._conn.cursor().execute("DELETE FROM messages WHERE msg_id=?", (mid,))
                pruned += 1
            self._conn.commit()
        return {"kept": kept, "pruned": pruned}

    def capacity_used(self) -> float:
        """Returns fraction of message capacity currently in use (0.0–1.0)."""
        with self._lock:
            return round(len(self._memory) / self._CAPACITY, 4)

    def forget_least_important(self) -> Optional[str]:
        """Evicts the lowest-decayed-importance item (LRU-weighted); returns evicted key or None."""
        if not self._memory:
            return None
        scores = {mid: self._decay_importance(item) for mid, item in self._memory.items()}
        victim = min(scores, key=lambda k: scores[k])
        del self._memory[victim]
        self._conn.cursor().execute("DELETE FROM messages WHERE msg_id=?", (victim,))
        self._conn.commit()
        return victim

    def status(self) -> Dict[str, Any]:
        """Returns numeric health dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            importances = [self._decay_importance(i) for i in self._memory.values()]
            avg_imp = statistics.mean(importances) if importances else 0.0
            try:
                entropy = -sum(p * math.log2(p + 1e-12)
                               for p in [v / (sum(importances) + 1e-12) for v in importances])
            except (ValueError, ZeroDivisionError):
                entropy = 0.0
            return {"items": len(self._memory), "confidence": round(self._confidence, 4),
                    "active": len(self._agent_registry), "cycles": self._cycles,
                    "sent": self._sent, "received": self._received,
                    "entropy": round(entropy, 4), "capacity": round(self.capacity_used(), 4),
                    "avg_importance": round(avg_imp, 4)}

    def auto_cycle(self) -> Dict[str, Any]:
        """Runs one autonomous maintenance cycle: decay, consolidate, anomaly check; returns cycle report."""
        self._cycles += 1
        consolidated = self.consolidate()
        cap = self.capacity_used()
        anomaly = cap > 0.9
        if anomaly:
            try:
                from PatternDetector import PatternDetector
                PatternDetector().anomaly(f"AgentMessageBus capacity critical: {cap:.1%}")
            except Exception:
                pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "agent_bus", "auto_cycle", self._confidence, not anomaly)
        except Exception:
            pass
        return {"cycle": self._cycles, "consolidated": consolidated,
                "capacity": cap, "anomaly": anomaly}

# Usage: obj = AgentMessageBus() | result = obj.send("agent_42", {"task": "summarise"})
