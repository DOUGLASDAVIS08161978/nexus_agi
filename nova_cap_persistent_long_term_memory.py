"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

"""
Nova Epistemic Memory Fabric — SQLite-backed Bayesian working memory with exponential decay,
LRU eviction, salience scoring, probabilistic recall, and autonomous consolidation daemon.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
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
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova_episodic_fabric.db")
DECAY_RATE = 0.00023  # half-life ≈ 50 minutes
CAPACITY = 200
EMA_ALPHA = 0.15

class EpistemicMemoryFabric:
    """Bayesian working memory with decay, LRU eviction, and autonomous consolidation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_quality: float = 0.5
        self._cycle_count: int = 0
        self._history: list[tuple[float, float]] = []  # (predicted_importance, actual_access)
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic (
                    key TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    importance REAL DEFAULT 1.0,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    tags TEXT DEFAULT '[]'
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON episodic(topic)")
            self._conn.commit()

    def _salience(self, importance: float, ts: float, access_count: int) -> float:
        elapsed = time.time() - ts
        decay = math.exp(-DECAY_RATE * elapsed)
        lru_penalty = 1.0 / (1.0 + access_count)
        return importance * decay * (1.0 - 0.3 * lru_penalty)

    def _bayesian_confidence(self, access_count: int, importance: float) -> float:
        prior = 0.5
        likelihood = min(0.95, 0.5 + 0.05 * access_count)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        return round(min(0.99, posterior * importance), 4)

    def store(self, key: str, value: str, importance: float = 1.0) -> dict[str, Any]:
        """Returns dict with key, importance, confidence after storing item in SQLite."""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()[:16] + "_" + key[:32]
        now = time.time()
        confidence = self._bayesian_confidence(0, importance)
        with self._lock:
            existing = self._conn.execute(
                "SELECT access_count, importance FROM episodic WHERE key=?", (hashed_key,)
            ).fetchone()
            if existing:
                new_acc = existing[0] + 1
                new_imp = min(10.0, existing[1] * 1.1 + importance * 0.2)
                confidence = self._bayesian_confidence(new_acc, new_imp)
                self._conn.execute(
                    "UPDATE episodic SET fact=?, importance=?, timestamp=?, access_count=?, confidence=? WHERE key=?",
                    (value, new_imp, now, new_acc, confidence, hashed_key)
                )
            else:
                self._conn.execute(
                    "INSERT INTO episodic(key,topic,fact,importance,timestamp,access_count,confidence) VALUES(?,?,?,?,?,0,?)",
                    (hashed_key, key, value, importance, now, confidence)
                )
            self._conn.commit()
        self._maybe_evict()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Reinforce memory: {key[:40]}", priority=2)
        except Exception:
            pass
        return {"key": hashed_key, "importance": importance, "confidence": confidence}

    def retrieve(self, key: str) -> dict[str, Any]:
        """Returns best matching memory dict with value, importance, confidence, salience."""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT key,topic,fact,importance,timestamp,access_count,confidence FROM episodic WHERE topic LIKE ?",
                (f"%{key}%",)
            ).fetchall()
        if not rows:
            return {"found": False, "key": key, "confidence": 0.0}
        scored = sorted(rows, key=lambda r: self._salience(r[3], r[4], r[5]), reverse=True)
        best = scored[0]
        new_acc = best[5] + 1
        new_imp = min(10.0, best[3] * 1.05)
        new_conf = self._bayesian_confidence(new_acc, new_imp)
        with self._lock:
            self._conn.execute(
                "UPDATE episodic SET access_count=?, importance=?, confidence=?, timestamp=? WHERE key=?",
                (new_acc, new_imp, new_conf, now, best[0])
            )
            self._conn.commit()
        self._history.append((best[3], new_imp))
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("episodic_recall", "salience_lru", new_conf, True)
        except Exception:
            pass
        return {
            "found": True, "key": best[0], "topic": best[1], "value": best[2],
            "importance": round(new_imp, 4), "confidence": new_conf,
            "salience": round(self._salience(new_imp, best[4], new_acc), 4),
            "access_count": new_acc
        }

    def recall(self, topic: str, n: int = 5) -> list[dict[str, Any]]:
        """Returns top-n salience-ranked memory records for a topic."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key,topic,fact,importance,timestamp,access_count,confidence FROM episodic WHERE topic LIKE ?",
                (f"%{topic}%",)
            ).fetchall()
        scored = sorted(rows, key=lambda r: self._salience(r[3], r[4], r[5]), reverse=True)[:n]
        return [{"topic": r[1], "fact": r[2], "importance": round(r[3], 4),
                 "confidence": round(r[6], 4), "access_count": r[5]} for r in scored]

    def forget(self, topic: str) -> dict[str, Any]:
        """Returns count of deleted records matching topic."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM episodic WHERE topic LIKE ?", (f"%{topic}%",))
            self._conn.commit()
        return {"deleted": cur.rowcount, "topic": topic}

    def consolidate(self) -> dict[str, Any]:
        """Returns consolidation report after decaying and pruning low-salience memories."""
        now = time.time()
        pruned = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, importance, timestamp, access_count FROM episodic"
            ).fetchall()
            for key, imp, ts, acc in rows:
                new_imp = imp * math.exp(-DECAY_RATE * (now - ts))
                sal = self._salience(new_imp, ts, acc)
                if sal < 0.02:
                    self._conn.execute("DELETE FROM episodic WHERE key=?", (key,))
                    pruned += 1
                else:
                    self._conn.execute("UPDATE episodic SET importance=? WHERE key=?", (new_imp, key))
            self._conn.commit()
        self._cycle_count += 1
        self._ema_quality = EMA_ALPHA * (1.0 - pruned / max(1, len(rows))) + (1 - EMA_ALPHA) * self._ema_quality
        return {"pruned": pruned, "remaining": len(rows) - pruned, "ema_quality": round(self._ema_quality, 4)}

    def capacity_used(self) -> dict[str, Any]:
        """Returns dict with item count, capacity ratio, entropy, and EMA quality."""
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
            importances = [r[0] for r in self._conn.execute("SELECT importance FROM episodic").fetchall()]
        ratio = count / CAPACITY
        if importances:
            total = sum(importances) + 1e-12
            dist = [i / total for i in importances]
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
        else:
            entropy = 0.0
        mae = 0.0
        if len(self._history) >= 2:
            mae = statistics.mean(abs(p - a) for p, a in self._history[-50:])
        return {"items": count, "capacity": CAPACITY, "ratio": round(ratio, 4),
                "entropy": round(entropy, 4), "mae": round(mae, 6),
                "ema_quality": round(self._ema_quality, 4), "cycles": self._cycle_count}

    def forget_least_important(self) -> dict[str, Any]:
        """Returns key of evicted item with lowest salience score."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, importance, timestamp, access_count FROM episodic"
            ).fetchall()
        if not rows:
            return {"evicted": None}
        worst = min(rows, key=lambda r: self._salience(r[1], r[2], r[3]))
        with self._lock:
            self._conn.execute("DELETE FROM episodic WHERE key=?", (worst[0],))
            self._conn.commit()
        return {"evicted": worst[0], "salience": round(self._salience(worst[1], worst[2], worst[3]), 6)}

    def summarise(self) -> dict[str, Any]:
        """Returns statistical summary of all stored memories with z-score anomaly flags."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT topic, importance, access_count, confidence FROM episodic"
            ).fetchall()
        if not rows:
            return {"summary": "empty", "items": 0}
        importances = [r[1] for r in rows]
        accesses = [r[2] for r in rows]
        mean_imp = statistics.mean(importances)
        std_imp = statistics.stdev(importances) if len(importances) > 1 else 1e-9
        anomalies = [r[0] for r in rows if abs((r[1] - mean_imp) / (std_imp + 1e-9)) > 3.0]
        top_topics = sorted(rows, key=lambda r: r[1], reverse=True)[:5]
        return {
            "items": len(rows), "mean_importance": round(mean_imp, 4),
            "std_importance": round(std_imp, 4), "mean_accesses": round(statistics.mean(accesses), 2),
            "anomalous_topics": anomalies, "top_topics": [r[0] for r in top_topics],
            "confidence": round(statistics.mean(r[3] for r in rows), 4)
        }

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        cap = self.capacity_used()
        return {"items": cap["items"], "confidence": cap["ema_quality"],
                "entropy": cap["entropy"], "cycles": cap["cycles"],
                "accuracy": round(1.0 - cap["mae"], 4), "active": 1}

    def _maybe_evict(self) -> None:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
        while count > CAPACITY:
            self.forget_least_important()
            count -= 1

    def _auto_loop(self) -> None:
        while True:
            time.sleep(300)
            try:
                self.consolidate()
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    MetacognitiveMonitor().log_reasoning("auto_consolidate", "ema_decay_lru",
                                                         self._ema_quality, True)
                except Exception:
                    pass
            except sqlite3.Error:
                pass

# Usage: obj = EpistemicMemoryFabric() | result = obj.store("quantum_physics", "entanglement persists", 0.9)