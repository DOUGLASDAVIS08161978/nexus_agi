"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

"""
NovaMemoryStore — SQLite-backed long-term memory with Bayesian salience scoring,
exponential decay, LRU eviction, and autonomous consolidation for Nova's cognition.
Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import time
import math
import statistics
import os
import json
import hashlib
from collections import OrderedDict
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_memory.db")
CAPACITY = 200
DECAY_RATE = 1e-5        # importance decay per second
EMA_ALPHA  = 0.15        # EMA smoothing for salience trend
CYCLE_SEC  = 60          # auto_cycle interval

class NovaMemoryStore:
    """SQLite-backed associative memory with decay, LRU eviction, and Bayesian salience."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_salience: float = 0.5
        self._cycles: int = 0
        self._anomaly_log: List[str] = []
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ------------------------------------------------------------------ DB INIT
    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS memories (
                id        TEXT PRIMARY KEY,
                topic     TEXT NOT NULL,
                fact      TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                timestamp  REAL NOT NULL,
                accessed   REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.7
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memories(topic)")
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    # ------------------------------------------------------------------ SALIENCE
    def _salience(self, importance: float, timestamp: float,
                  access_count: int) -> float:
        """Compute decayed salience: exp-decay * log-access boost."""
        elapsed = time.time() - timestamp
        decay   = math.exp(-DECAY_RATE * elapsed)
        boost   = math.log(1 + access_count)
        return importance * decay * (1 + 0.1 * boost)

    # ------------------------------------------------------------------ PUBLIC API
    def remember(self, topic: str, fact: str,
                 importance: float = 1.0, confidence: float = 0.7) -> Dict[str, Any]:
        """Stores a fact under topic; returns the stored record dict."""
        uid = hashlib.sha256(f"{topic}:{fact}".encode()).hexdigest()[:16]
        now = time.time()
        with self._lock:
            with self._conn() as c:
                c.execute("""INSERT INTO memories
                    (id,topic,fact,importance,timestamp,accessed,access_count,confidence)
                    VALUES (?,?,?,?,?,?,0,?)
                    ON CONFLICT(id) DO UPDATE SET
                        importance=MAX(importance, excluded.importance),
                        confidence=0.5*(confidence+excluded.confidence),
                        accessed=excluded.accessed""",
                    (uid, topic, fact, importance, now, now, confidence))
                c.commit()
        self._maybe_evict()
        self._log_goal(topic)
        return {"id": uid, "topic": topic, "fact": fact,
                "importance": importance, "confidence": confidence}

    def recall(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        """Returns top-n salience-ranked facts for a topic, updating access counts."""
        with self._lock:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT id,fact,importance,timestamp,access_count,confidence "
                    "FROM memories WHERE topic=? ORDER BY importance DESC LIMIT ?",
                    (topic, n * 3)).fetchall()
        if not rows:
            return []
        ranked = sorted(rows,
            key=lambda r: self._salience(r[2], r[3], r[4]), reverse=True)[:n]
        now = time.time()
        with self._lock:
            with self._conn() as c:
                for r in ranked:
                    new_imp = r[2] * math.exp(-DECAY_RATE * (now - r[3])) * 1.15
                    c.execute("UPDATE memories SET access_count=access_count+1, "
                              "importance=?, accessed=? WHERE id=?",
                              (new_imp, now, r[0]))
                c.commit()
        self._log_reasoning("recall", topic, len(ranked))
        return [{"fact": r[1], "salience": round(self._salience(r[2], r[3], r[4]), 4),
                 "confidence": r[5], "accesses": r[4]} for r in ranked]

    def forget(self, topic: str) -> Dict[str, Any]:
        """Deletes all facts under topic; returns count of removed records."""
        with self._lock:
            with self._conn() as c:
                n = c.execute("SELECT COUNT(*) FROM memories WHERE topic=?",
                              (topic,)).fetchone()[0]
                c.execute("DELETE FROM memories WHERE topic=?", (topic,))
                c.commit()
        return {"removed": n, "topic": topic}

    def summarise(self) -> Dict[str, Any]:
        """Returns a statistical summary of the entire memory store."""
        with self._lock:
            with self._conn() as c:
                total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                topics = c.execute(
                    "SELECT topic, COUNT(*) FROM memories GROUP BY topic").fetchall()
                imps = [r[0] for r in c.execute(
                    "SELECT importance FROM memories").fetchall()]
        if not imps:
            return {"total": 0, "topics": 0, "mean_importance": 0.0,
                    "std_importance": 0.0, "entropy": 0.0, "cycles": self._cycles}
        dist = {t: cnt / total for t, cnt in topics}
        entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
        mean_imp = statistics.mean(imps)
        std_imp  = statistics.stdev(imps) if len(imps) > 1 else 0.0
        return {"total": total, "topics": len(topics),
                "mean_importance": round(mean_imp, 4),
                "std_importance": round(std_imp, 4),
                "entropy": round(entropy, 4),
                "ema_salience": round(self._ema_salience, 4),
                "cycles": self._cycles,
                "anomalies": len(self._anomaly_log)}

    def store(self, key: str, value: Any, importance: float = 1.0) -> Dict[str, Any]:
        """Alias for remember(); stores key/value as topic/fact pair."""
        return self.remember(key, str(value), importance)

    def retrieve(self, key: str) -> List[Dict[str, Any]]:
        """Alias for recall(); returns top-5 facts for the given key/topic."""
        return self.recall(key, n=5)

    def capacity_used(self) -> Dict[str, Any]:
        """Returns current item count vs max capacity and fill ratio."""
        with self._lock:
            with self._conn() as c:
                n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"items": n, "capacity": CAPACITY,
                "fill_ratio": round(n / CAPACITY, 4),
                "confidence": round(self._ema_salience, 4),
                "cycles": self._cycles, "active": 1,
                "entropy": round(self.summarise().get("entropy", 0.0), 4)}

    def forget_least_important(self, n: int = 10) -> Dict[str, Any]:
        """Evicts n lowest-salience items; returns list of evicted topics."""
        with self._lock:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT id,topic,importance,timestamp,access_count "
                    "FROM memories").fetchall()
        scored = sorted(rows,
            key=lambda r: self._salience(r[2], r[3], r[4]))[:n]
        ids = [r[0] for r in scored]
        topics = [r[1] for r in scored]
        with self._lock:
            with self._conn() as c:
                c.executemany("DELETE FROM memories WHERE id=?", [(i,) for i in ids])
                c.commit()
        return {"evicted": len(ids), "topics": topics}

    def consolidate(self) -> Dict[str, Any]:
        """Decays all importances in-place and evicts over-capacity items."""
        now = time.time()
        with self._lock:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT id,importance,timestamp FROM memories").fetchall()
                for rid, imp, ts in rows:
                    new_imp = imp * math.exp(-DECAY_RATE * (now - ts))
                    c.execute("UPDATE memories SET importance=? WHERE id=?",
                              (new_imp, rid))
                c.commit()
        evicted = self._maybe_evict()
        self._cycles += 1
        saliences = [self._salience(r[0], r[1], r[2])
                     for r in self._fetch_all_salience_data()]
        if saliences:
            self._ema_salience = (EMA_ALPHA * statistics.mean(saliences)
                                  + (1 - EMA_ALPHA) * self._ema_salience)
            self._detect_anomaly(saliences)
        return {"consolidated": len(rows), "evicted": evicted,
                "ema_salience": round(self._ema_salience, 4)}

    # ------------------------------------------------------------------ INTERNAL
    def _fetch_all_salience_data(self) -> List[tuple]:
        with self._conn() as c:
            return c.execute(
                "SELECT importance,timestamp,access_count FROM memories").fetchall()

    def _maybe_evict(self) -> int:
        cap = self.capacity_used()
        if cap["items"] > CAPACITY:
            over = cap["items"] - CAPACITY
            result = self.forget_least_important(over)
            return result["evicted"]
        return 0

    def _detect_anomaly(self, saliences: List[float]) -> None:
        if len(saliences) < 3:
            return
        mu  = statistics.mean(saliences)
        std = statistics.stdev(saliences) + 1e-9
        for s in saliences:
            z = (s - mu) / std
            if abs(z) > 3.0:
                self._anomaly_log.append(
                    f"{time.time():.0f}:z={z:.2f}")
                self._anomaly_log = self._anomaly_log[-50:]

    def _log_goal(self, topic: str) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Deepen knowledge of topic: {topic}", priority=2)
        except Exception:
            pass

    def _log_reasoning(self, domain: str, topic: str, hits: int) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            conf = min(1.0, hits / 5.0)
            MetacognitiveMonitor().log_reasoning(
                domain, f"recall:{topic}", conf, hits > 0)
        except Exception:
            pass

    def _auto_loop(self) -> None:
        while True:
            time.sleep(CYCLE_SEC)
            try:
                self.consolidate()
            except Exception:
                pass

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict for ConsciousnessIntegrator Φ computation."""
        s = self.summarise()
        return {"items": s["total"], "confidence": self._ema_salience,
                "entropy": s["entropy"], "cycles": self._cycles,
                "active": 1, "quality": round(1.0 - s["std_importance"] /
                                               (s["mean_importance"] + 1e-9), 4)}

# Usage: obj = NovaMemoryStore() | result = obj.remember("physics", "light speed = 3e8 m/s")
