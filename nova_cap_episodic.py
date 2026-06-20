"""
nova_cap_episodic.py
Nova ASI — Episodic
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
Nova Episodic Memory Fabric — stores, retrieves, and narrates temporally structured
episodes with emotional salience, Bayesian decay, and LRU working memory.
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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

class EpisodicMemoryFabric:
    """Episodic memory with emotional salience retrieval and Bayesian working memory."""

    _DB = "nova_episodic.db"
    _CAPACITY = 200
    _DECAY_RATE = 0.0003
    _GAP_SECONDS = 300

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wm: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cycles = 0
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._start_daemon()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    context TEXT,
                    emotion TEXT,
                    valence REAL DEFAULT 0.0,
                    importance REAL DEFAULT 0.5,
                    timestamp REAL NOT NULL,
                    group_id TEXT
                )
            """)

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            self.consolidate()
            self._decay_working_memory()
            self._cycles += 1
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning(
                    "episodic", "decay+consolidate", self._calibration_confidence(), True
                )
            except Exception:
                pass

    def _calibration_confidence(self) -> float:
        with self._lock:
            if not self._wm:
                return 0.5
            vals = [v["importance"] for v in self._wm.values()]
            return float(min(1.0, statistics.mean(vals))) if vals else 0.5

    def _make_id(self, event: str, ts: float) -> str:
        raw = f"{event}{ts}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _valence_from_emotion(self, emotion: str) -> float:
        pos = {"joy": 0.9, "love": 0.85, "pride": 0.7, "calm": 0.4, "hope": 0.6}
        neg = {"fear": -0.8, "anger": -0.75, "sadness": -0.7, "disgust": -0.65, "shame": -0.6}
        e = emotion.lower().strip()
        return pos.get(e, neg.get(e, 0.0))

    def _assign_group(self, ts: float) -> str:
        cur = self._conn.execute(
            "SELECT group_id, timestamp FROM episodes ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if cur and (ts - cur[1]) < self._GAP_SECONDS:
            return cur[0]
        return self._make_id(str(ts), ts)

    def record(self, event: str, context: str, emotion: str, importance: float) -> str:
        """Stores an episode and returns its episode_id."""
        ts = time.time()
        eid = self._make_id(event, ts)
        valence = self._valence_from_emotion(emotion)
        importance = max(0.0, min(1.0, importance))
        with self._lock:
            group_id = self._assign_group(ts)
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?)",
                    (eid, event, context, emotion, valence, importance, ts, group_id)
                )
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            if abs(valence) > 0.7:
                HierarchicalGoalPlanner().add_goal(
                    f"Reflect on high-salience episode: {event[:60]}", priority=2
                )
        except Exception:
            pass
        self.store(eid, {"event": event, "emotion": emotion}, importance)
        return eid

    def recall(self, cue: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns top_k episodes sorted by retrieval score (importance×emotion×recency×relevance)."""
        now = time.time()
        cue_tokens = set(cue.lower().split())
        rows = self._conn.execute(
            "SELECT episode_id, event, context, emotion, valence, importance, timestamp FROM episodes"
        ).fetchall()
        scored = []
        for eid, event, context, emotion, valence, importance, ts in rows:
            elapsed_min = (now - ts) / 60.0
            recency = math.exp(-0.001 * elapsed_min)
            emotion_weight = 1.0 + abs(valence)
            ev_tokens = set((event + " " + (context or "")).lower().split())
            overlap = len(cue_tokens & ev_tokens) / (len(cue_tokens) + 1e-9)
            tf_idf_boost = math.log(1 + overlap * 10)
            score = importance * emotion_weight * recency * (0.5 + tf_idf_boost)
            scored.append({
                "episode_id": eid, "event": event, "context": context,
                "emotion": emotion, "valence": valence, "importance": importance,
                "timestamp": ts, "score": round(score, 4)
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def replay(self, episode_id: str) -> Dict[str, Any]:
        """Returns full episode detail and boosts its working-memory importance."""
        row = self._conn.execute(
            "SELECT episode_id, event, context, emotion, valence, importance, timestamp, group_id "
            "FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if not row:
            return {"error": "episode_not_found"}
        keys = ["episode_id", "event", "context", "emotion", "valence", "importance", "timestamp", "group_id"]
        ep = dict(zip(keys, row))
        with self._lock:
            if episode_id in self._wm:
                self._wm[episode_id]["importance"] = min(1.0, self._wm[episode_id]["importance"] * 1.2)
                self._wm[episode_id]["access_count"] += 1
                self._wm.move_to_end(episode_id)
        return ep

    def emotional_highlights(self) -> List[Dict[str, Any]]:
        """Returns episodes with abs(valence) > 0.6, sorted by emotional intensity."""
        rows = self._conn.execute(
            "SELECT episode_id, event, emotion, valence, importance, timestamp "
            "FROM episodes WHERE ABS(valence) > 0.6 ORDER BY ABS(valence) DESC LIMIT 20"
        ).fetchall()
        return [
            {"episode_id": r[0], "event": r[1], "emotion": r[2],
             "valence": r[3], "importance": r[4], "timestamp": r[5]}
            for r in rows
        ]

    def temporal_summary(self, hours: float = 6.0) -> str:
        """Returns a coherent narrative string of episodes from the last N hours."""
        cutoff = time.time() - hours * 3600
        rows = self._conn.execute(
            "SELECT event, context, emotion, valence, timestamp, group_id "
            "FROM episodes WHERE timestamp >= ? ORDER BY timestamp ASC", (cutoff,)
        ).fetchall()
        if not rows:
            return f"No episodes recorded in the last {hours} hours."
        groups: Dict[str, List] = {}
        for event, context, emotion, valence, ts, gid in rows:
            groups.setdefault(gid, []).append((event, context, emotion, valence, ts))
        parts = []
        for gid, events in groups.items():
            t0 = datetime.fromtimestamp(events[0][4]).strftime("%H:%M")
            summary_events = "; ".join(f"{e[0]} ({e[2]})" for e in events[:3])
            avg_valence = statistics.mean(e[3] for e in events)
            tone = "positively" if avg_valence > 0.2 else ("negatively" if avg_valence < -0.2 else "neutrally")
            parts.append(f"[{t0}] {tone.capitalize()} toned cluster: {summary_events}.")
        return " → ".join(parts)

    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Stores a key-value pair in Bayesian working memory with decay tracking."""
        with self._lock:
            if key in self._wm:
                self._wm.move_to_end(key)
                self._wm[key]["importance"] = min(1.0, importance + self._wm[key]["importance"] * 0.3)
                self._wm[key]["access_count"] += 1
            else:
                self._wm[key] = {
                    "value": value, "importance": importance,
                    "timestamp": time.time(), "access_count": 0
                }
            if len(self._wm) > self._CAPACITY:
                self.forget_least_important()

    def retrieve(self, key: str) -> Optional[Any]:
        """Returns value from working memory and boosts its importance on access."""
        with self._lock:
            if key not in self._wm:
                return None
            item = self._wm[key]
            item["importance"] = min(1.0, item["importance"] * 1.15)
            item["access_count"] += 1
            self._wm.move_to_end(key)
            return item["value"]

    def consolidate(self) -> int:
        """Decays all working-memory items and evicts below-threshold entries; returns evicted count."""
        return self._decay_working_memory()

    def _decay_working_memory(self) -> int:
        now = time.time()
        evicted = 0
        with self._lock:
            to_delete = []
            for key, item in self._wm.items():
                elapsed = now - item["timestamp"]
                item["importance"] *= math.exp(-self._DECAY_RATE * elapsed)
                item["timestamp"] = now
                if item["importance"] < 0.01:
                    to_delete.append(key)
            for k in to_delete:
                del self._wm[k]
                evicted += 1
        return evicted

    def capacity_used(self) -> Dict[str, Any]:
        """Returns dict with working-memory usage stats and confidence interval."""
        with self._lock:
            n = len(self._wm)
            imps = [v["importance"] for v in self._wm.values()] or [0.0]
            mean_imp = statistics.mean(imps)
            std_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
            ci_low = max(0.0, mean_imp - 1.96 * std_imp)
            ci_high = min(1.0, mean_imp + 1.96 * std_imp)
        total_eps = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return {
            "items": n, "capacity": self._CAPACITY,
            "pct_full": round(n / self._CAPACITY * 100, 1),
            "mean_importance": round(mean_imp, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "total_episodes": total_eps
        }

    def forget_least_important(self) -> Optional[str]:
        """Evicts the least important working-memory item; returns its key or None."""
        with self._lock:
            if not self._wm:
                return None
            worst = min(self._wm, key=lambda k: self._wm[k]["importance"])
            del self._wm[worst]
            return worst

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        cap = self.capacity_used()
        highlights = self.emotional_highlights()
        mean_valence = statistics.mean(h["valence"] for h in highlights) if highlights else 0.0
        entropy_val = 0.0
        if cap["items"] > 1:
            imps = [v["importance"] for v in self._wm.values()]
            total = sum(imps) + 1e-12
            probs = [i / total for i in imps]
            entropy_val = -sum(p * math.log2(p + 1e-12) for p in probs)
        return {
            "items": cap["items"],
            "confidence": cap["mean_importance"],
            "accuracy": round(1.0 - cap["pct_full"] / 100, 3),
            "active": cap["total_episodes"],
            "cycles": self._cycles,
            "entropy": round(entropy_val, 4),
            "mean_valence": round(mean_valence, 4),
            "quality": round(cap["mean_importance"] * (1 - entropy_val / (math.log2(cap["items"] + 2))), 4)
        }

# Usage: obj = EpisodicMemoryFabric() | result = obj.record("Solved hard problem", "research session", "joy", 0.9)