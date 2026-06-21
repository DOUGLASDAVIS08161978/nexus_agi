"""
NovaMemoryStore — SQLite-backed Bayesian working memory with exponential decay,
LRU eviction, salience scoring, and autonomous consolidation for Nova's long-term memory.
Pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
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
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_memory_store.db")
CAPACITY = 200
DECAY_RATE = 1e-5
ACCESS_BOOST = 0.08
EMA_ALPHA = 0.15
CYCLE_INTERVAL = 60


class NovaMemoryStore:
    """SQLite-backed Bayesian working memory with decay, LRU eviction, and autonomous consolidation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._ema_importance: float = 0.5
        self._cycle_count: int = 0
        self._quality_history: list[float] = []
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()
        self._register_goals()

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    timestamp REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.7
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memories(topic)")
            self._conn.commit()

    def _register_goals(self) -> None:
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal("Consolidate decayed memories and evict low-salience items", priority=2)
            hgp.add_goal("Maintain memory store calibration below 0.1 entropy drift", priority=3)
        except Exception:
            pass

    def _salience(self, importance: float, timestamp: float, access_count: int) -> float:
        elapsed = time.time() - timestamp
        decayed = importance * math.exp(-DECAY_RATE * elapsed)
        recency = math.exp(-elapsed / 86400.0)
        freq = math.log1p(access_count)
        return decayed * 0.5 + recency * 0.3 + freq * 0.2

    def _fact_id(self, topic: str, fact: str) -> str:
        return hashlib.sha256(f"{topic}::{fact}".encode()).hexdigest()[:16]

    def remember(self, topic: str, fact: str, importance: float = 0.5) -> dict[str, Any]:
        """Stores a fact under a topic with Bayesian importance; returns the stored record."""
        importance = max(0.0, min(1.0, importance))
        fid = self._fact_id(topic, fact)
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT importance, access_count FROM memories WHERE id=?", (fid,))
            row = cur.fetchone()
            if row:
                prior = row[0]
                acc = row[1]
                posterior = (importance * prior) / (importance * prior + (1 - importance) * (1 - prior) + 1e-12)
                posterior = max(0.0, min(1.0, posterior))
                cur.execute("""UPDATE memories SET importance=?, timestamp=?, access_count=?, confidence=?
                               WHERE id=?""", (posterior, now, acc + 1, posterior, fid))
            else:
                cur.execute("""INSERT INTO memories(id,topic,fact,importance,timestamp,access_count,confidence)
                               VALUES(?,?,?,?,?,?,?)""", (fid, topic, fact, importance, now, 0, importance))
            self._conn.commit()
            self._ema_importance = EMA_ALPHA * importance + (1 - EMA_ALPHA) * self._ema_importance
        if self.capacity_used() > CAPACITY:
            self.forget_least_important()
        return {"id": fid, "topic": topic, "importance": importance, "stored": True}

    def recall(self, topic: str, n: int = 5) -> list[dict[str, Any]]:
        """Returns top-n salience-ranked facts for a topic, boosting importance on access."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, fact, importance, timestamp, access_count, confidence FROM memories WHERE topic=?", (topic,))
            rows = cur.fetchall()
            if not rows:
                return []
            scored = []
            for fid, fact, imp, ts, acc, conf in rows:
                sal = self._salience(imp, ts, acc)
                scored.append((fid, fact, imp, ts, acc, conf, sal))
            scored.sort(key=lambda x: x[6], reverse=True)
            top = scored[:n]
            now = time.time()
            results = []
            for fid, fact, imp, ts, acc, conf, sal in top:
                new_imp = min(1.0, imp + ACCESS_BOOST)
                cur.execute("UPDATE memories SET access_count=?, importance=?, timestamp=? WHERE id=?",
                            (acc + 1, new_imp, now, fid))
                elapsed = now - ts
                ci_low = max(0.0, conf - 0.1 * math.exp(-acc / 10.0))
                ci_high = min(1.0, conf + 0.05)
                results.append({"fact": fact, "salience": round(sal, 4),
                                 "confidence": round(conf, 4),
                                 "ci": [round(ci_low, 4), round(ci_high, 4)],
                                 "access_count": acc + 1,
                                 "age_seconds": round(elapsed, 1)})
            self._conn.commit()
        return results

    def forget(self, topic: str) -> dict[str, Any]:
        """Deletes all facts for a topic; returns count of removed entries."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE topic=?", (topic,))
            count = cur.fetchone()[0]
            cur.execute("DELETE FROM memories WHERE topic=?", (topic,))
            self._conn.commit()
        return {"topic": topic, "removed": count}

    def forget_least_important(self) -> dict[str, Any]:
        """Evicts lowest-salience item (LRU-weighted); returns evicted record summary."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, topic, importance, timestamp, access_count FROM memories")
            rows = cur.fetchall()
            if not rows:
                return {"evicted": None}
            scored = [(fid, topic, self._salience(imp, ts, acc))
                      for fid, topic, imp, ts, acc in rows]
            scored.sort(key=lambda x: x[2])
            worst_id, worst_topic, worst_sal = scored[0]
            cur.execute("DELETE FROM memories WHERE id=?", (worst_id,))
            self._conn.commit()
        return {"evicted": worst_id, "topic": worst_topic, "salience": round(worst_sal, 4)}

    def summarise(self) -> dict[str, Any]:
        """Returns per-topic fact counts, entropy, and EMA importance for the full store."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT topic, COUNT(*), AVG(importance), AVG(confidence) FROM memories GROUP BY topic")
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM memories")
            total = cur.fetchone()[0]
        topics: dict[str, Any] = {}
        importances: list[float] = []
        for topic, cnt, avg_imp, avg_conf in rows:
            topics[topic] = {"count": cnt, "avg_importance": round(avg_imp, 4),
                              "avg_confidence": round(avg_conf, 4)}
            importances.extend([avg_imp] * cnt)
        entropy = 0.0
        if importances:
            total_i = sum(importances) + 1e-12
            dist = {i: v / total_i for i, v in enumerate(importances)}
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
        std = statistics.stdev(importances) if len(importances) > 1 else 0.0
        return {"total_facts": total, "topics": topics,
                "ema_importance": round(self._ema_importance, 4),
                "entropy": round(entropy, 4), "importance_std": round(std, 4),
                "cycles": self._cycle_count}

    def capacity_used(self) -> int:
        """Returns current count of stored memory items."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]

    def consolidate(self) -> dict[str, Any]:
        """Applies exponential decay to all items and evicts those below salience threshold."""
        now = time.time()
        pruned = 0
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, importance, timestamp, access_count FROM memories")
            rows = cur.fetchall()
            for fid, imp, ts, acc in rows:
                elapsed = now - ts
                new_imp = imp * math.exp(-DECAY_RATE * elapsed)
                sal = self._salience(new_imp, ts, acc)
                if sal < 0.02:
                    cur.execute("DELETE FROM memories WHERE id=?", (fid,))
                    pruned += 1
                else:
                    cur.execute("UPDATE memories SET importance=? WHERE id=?", (new_imp, fid))
            self._conn.commit()
        quality = 1.0 - min(1.0, pruned / max(1, len(rows)))
        self._quality_history.append(quality)
        if len(self._quality_history) > 50:
            self._quality_history.pop(0)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            mm.log_reasoning("memory_consolidation", "exponential_decay+LRU",
                              round(quality, 4), quality > 0.7)
        except Exception:
            pass
        return {"pruned": pruned, "remaining": self.capacity_used(), "quality": round(quality, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        cap = self.capacity_used()
        avg_quality = statistics.mean(self._quality_history) if self._quality_history else 1.0
        return {"items": cap, "confidence": round(self._ema_importance, 4),
                "accuracy": round(avg_quality, 4), "quality": round(avg_quality, 4),
                "active": 1, "cycles": self._cycle_count,
                "entropy": self.summarise().get("entropy", 0.0)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(CYCLE_INTERVAL)
            try:
                self.consolidate()
                self._cycle_count += 1
            except Exception:
                pass

# Usage: obj = NovaMemoryStore() | result = obj.remember("physics", "E=mc^2", importance=0.9)
