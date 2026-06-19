"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

"""
Nova Persistent Long-Term Memory — Bayesian working memory with exponential decay and LRU eviction.

Each memory item carries: value, importance (float), timestamp, access_count.
Importance decays exponentially over time and grows on access (Bayesian reinforcement).
LRU eviction fires automatically when capacity exceeds 200 items.
"""

import sqlite3
import json
import math
import time
import hashlib
import statistics
from typing import Any, Dict, List, Optional, Tuple


class NovaMemoryStore:
    """SQLite-backed long-term memory with Bayesian importance, decay, and LRU eviction."""

    _DECAY_RATE: float = 1e-5          # importance decay per second
    _ACCESS_BOOST: float = 1.25        # multiplicative boost on each retrieval
    _CAPACITY: int = 200               # max items before eviction
    _CONSOLIDATE_ALPHA: float = 0.15   # EMA weight for importance smoothing

    def __init__(self) -> None:
        """Initialise SQLite store and create schema if absent."""
        self.conn = sqlite3.connect("nova_memory.db", check_same_thread=False)
        self._build_schema()

    def _build_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key        TEXT PRIMARY KEY,
                topic      TEXT NOT NULL,
                value      TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0,
                stored_at  REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memory(topic)")
        self.conn.commit()

    def _decay(self, importance: float, stored_at: float, now: float) -> float:
        """Apply exponential decay: importance * exp(-k * elapsed)."""
        elapsed = max(0.0, now - stored_at)
        return importance * math.exp(-self._DECAY_RATE * elapsed)

    def store(self, key: str, value: Any, importance: float = 1.0) -> float:
        """Store or update a memory item; returns effective importance after decay."""
        now = time.time()
        topic = key.split(":")[0] if ":" in key else key
        serialised = json.dumps(value)
        try:
            row = self.conn.execute(
                "SELECT importance, stored_at, access_count FROM memory WHERE key=?", (key,)
            ).fetchone()
            if row:
                prev_imp = self._decay(row[0], row[1], now)
                # Bayesian merge: blend prior decayed importance with new signal
                merged = self._CONSOLIDATE_ALPHA * importance + (1 - self._CONSOLIDATE_ALPHA) * prev_imp
                self.conn.execute(
                    "UPDATE memory SET value=?, importance=?, accessed_at=?, access_count=? WHERE key=?",
                    (serialised, merged, now, row[2] + 1, key)
                )
                effective = merged
            else:
                self.conn.execute(
                    "INSERT INTO memory VALUES (?,?,?,?,?,?,?)",
                    (key, topic, serialised, importance, now, now, 0)
                )
                effective = importance
            self.conn.commit()
            self.forget_least_important()
            return effective
        except sqlite3.Error as exc:
            raise RuntimeError(f"store failed: {exc}") from exc

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Fetch a memory item; boosts importance on access and returns metadata dict."""
        now = time.time()
        try:
            row = self.conn.execute(
                "SELECT value, importance, stored_at, accessed_at, access_count FROM memory WHERE key=?",
                (key,)
            ).fetchone()
            if not row:
                return None
            decayed = self._decay(row[1], row[2], now)
            boosted = min(decayed * self._ACCESS_BOOST, 100.0)
            self.conn.execute(
                "UPDATE memory SET importance=?, accessed_at=?, access_count=? WHERE key=?",
                (boosted, now, row[4] + 1, key)
            )
            self.conn.commit()
            return {
                "key": key,
                "value": json.loads(row[0]),
                "importance": round(boosted, 4),
                "stored_at": row[2],
                "accessed_at": now,
                "access_count": row[4] + 1,
            }
        except sqlite3.Error as exc:
            raise RuntimeError(f"retrieve failed: {exc}") from exc

    def consolidate(self) -> int:
        """Recompute decayed importance for all items; returns count of items updated."""
        now = time.time()
        rows = self.conn.execute("SELECT key, importance, stored_at FROM memory").fetchall()
        updated = 0
        try:
            for key, imp, stored_at in rows:
                new_imp = self._decay(imp, stored_at, now)
                self.conn.execute("UPDATE memory SET importance=? WHERE key=?", (new_imp, key))
                updated += 1
            self.conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"consolidate failed: {exc}") from exc
        return updated

    def capacity_used(self) -> Dict[str, Any]:
        """Return capacity stats: count, max, fill_ratio, mean_importance."""
        try:
            row = self.conn.execute(
                "SELECT COUNT(*), AVG(importance), MIN(importance), MAX(importance) FROM memory"
            ).fetchone()
            count, avg_imp, min_imp, max_imp = row
            return {
                "count": count or 0,
                "capacity": self._CAPACITY,
                "fill_ratio": round((count or 0) / self._CAPACITY, 3),
                "mean_importance": round(avg_imp or 0.0, 4),
                "min_importance": round(min_imp or 0.0, 4),
                "max_importance": round(max_imp or 0.0, 4),
            }
        except sqlite3.Error as exc:
            raise RuntimeError(f"capacity_used failed: {exc}") from exc

    def forget_least_important(self) -> int:
        """Evict lowest-importance items when over capacity; returns eviction count."""
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            if count <= self._CAPACITY:
                return 0
            excess = count - self._CAPACITY
            victims = self.conn.execute(
                "SELECT key FROM memory ORDER BY importance ASC, accessed_at ASC LIMIT ?",
                (excess,)
            ).fetchall()
            keys = [v[0] for v in victims]
            self.conn.executemany("DELETE FROM memory WHERE key=?", [(k,) for k in keys])
            self.conn.commit()
            return len(keys)
        except sqlite3.Error as exc:
            raise RuntimeError(f"forget_least_important failed: {exc}") from exc

    def remember(self, topic: str, fact: Any, importance: float = 1.0) -> float:
        """High-level alias: store a fact under topic key; returns effective importance."""
        key = f"{topic}:{hashlib.md5(json.dumps(fact, sort_keys=True).encode()).hexdigest()[:8]}"
        return self.store(key, {"topic": topic, "fact": fact}, importance)

    def recall(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        """Return up to n most important facts for a topic, with live decay applied."""
        now = time.time()
        try:
            rows = self.conn.execute(
                "SELECT key, value, importance, stored_at, access_count FROM memory WHERE topic=? ORDER BY importance DESC LIMIT ?",
                (topic, n)
            ).fetchall()
            results = []
            for key, val, imp, stored_at, acc in rows:
                decayed = self._decay(imp, stored_at, now)
                results.append({
                    "key": key,
                    "fact": json.loads(val).get("fact"),
                    "importance": round(decayed, 4),
                    "access_count": acc,
                })
            return results
        except sqlite3.Error as exc:
            raise RuntimeError(f"recall failed: {exc}") from exc

    def forget(self, topic: str) -> int:
        """Delete all memory items for a given topic; returns deleted count."""
        try:
            cur = self.conn.execute("DELETE FROM memory WHERE topic=?", (topic,))
            self.conn.commit()
            return cur.rowcount
        except sqlite3.Error as exc:
            raise RuntimeError(f"forget failed: {exc}") from exc

    def summarise(self) -> Dict[str, Any]:
        """Return a cognitive summary: top topics, importance distribution, health score."""
        now = time.time()
        try:
            rows = self.conn.execute(
                "SELECT topic, importance, stored_at, access_count FROM memory"
            ).fetchall()
            if not rows:
                return {"total": 0, "topics": {}, "health": 0.0}
            topic_stats: Dict[str, Dict[str, Any]] = {}
            importances: List[float] = []
            for topic, imp, stored_at, acc in rows:
                decayed = self._decay(imp, stored_at, now)
                importances.append(decayed)
                if topic not in topic_stats:
                    topic_stats[topic] = {"count": 0, "total_importance": 0.0, "total_accesses": 0}
                topic_stats[topic]["count"] += 1
                topic_stats[topic]["total_importance"] += decayed
                topic_stats[topic]["total_accesses"] += acc
            mean_imp = statistics.mean(importances)
            stdev_imp = statistics.stdev(importances) if len(importances) > 1 else 0.0
            health = min(1.0, mean_imp / (1.0 + stdev_imp))
            return {
                "total": len(rows),
                "unique_topics": len(topic_stats),
                "mean_importance": round(mean_imp, 4),
                "stdev_importance": round(stdev_imp, 4),
                "health_score": round(health, 4),
                "top_topics": sorted(
                    topic_stats.items(), key=lambda x: x[1]["total_importance"], reverse=True
                )[:5],
            }
        except sqlite3.Error as exc:
            raise RuntimeError(f"summarise failed: {exc}") from exc

# Usage: obj = NovaMemoryStore() | obj.remember("physics", "gravity accelerates at 9.8 m/s²", importance=0.9)