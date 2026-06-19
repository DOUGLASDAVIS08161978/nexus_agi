"""
nova_cap_episodic.py
Nova ASI — Episodic
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
EpisodicMemoryEngine — Nova's temporal episodic memory with emotional salience retrieval.
Stores sequences of events with full temporal structure, groups by proximity,
and retrieves via composite scoring: importance * emotion_weight * recency * cue_relevance.
Integrates with BayesianBeliefSystem, HierarchicalGoalPlanner, MetacognitiveMonitor, WorkingMemory.
"""

import sqlite3
import threading
import math
import time
import statistics
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "episodic_memory.db")
EPISODE_GAP_SECONDS = 300
DECAY_RATE = 0.001
CAPACITY = 200

class EpisodicMemoryEngine:
    """Nova's episodic memory: records, groups, scores, and replays lived experience."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._setup_db()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._load_cache()
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _setup_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            event TEXT, context TEXT, emotion TEXT,
            importance REAL, valence REAL, timestamp REAL,
            group_id TEXT, access_count INTEGER DEFAULT 0
        )""")
        self._conn.commit()

    def _load_cache(self) -> None:
        c = self._conn.cursor()
        c.execute("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 200")
        for row in c.fetchall():
            eid, ev, ctx, emo, imp, val, ts, gid, acc = row
            self._cache[eid] = dict(episode_id=eid, event=ev, context=ctx,
                emotion=emo, importance=imp, valence=val,
                timestamp=ts, group_id=gid, access_count=acc)

    def _gen_id(self, event: str, ts: float) -> str:
        raw = f"{event}{ts}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _assign_group(self, ts: float) -> str:
        """Returns group_id based on temporal proximity (< 5 min = same group)."""
        with self._lock:
            for ep in reversed(list(self._cache.values())):
                if abs(ts - ep["timestamp"]) < EPISODE_GAP_SECONDS:
                    return ep["group_id"]
        return self._gen_id(f"group{ts}", ts)

    def record(self, event: str, context: str, emotion: str,
               importance: float, valence: float = 0.0) -> str:
        """Records a new episodic event; returns its episode_id."""
        ts = time.time()
        eid = self._gen_id(event, ts)
        valence = max(-1.0, min(1.0, valence))
        importance = max(0.0, min(1.0, importance))
        group_id = self._assign_group(ts)
        row = dict(episode_id=eid, event=event, context=context,
                   emotion=emotion, importance=importance,
                   valence=valence, timestamp=ts,
                   group_id=group_id, access_count=0)
        with self._lock:
            self._cache[eid] = row
            if len(self._cache) > CAPACITY:
                self._evict()
        c = self._conn.cursor()
        c.execute("""INSERT OR REPLACE INTO episodes
            (episode_id,event,context,emotion,importance,valence,timestamp,group_id,access_count)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (eid, event, context, emotion, importance, valence, ts, group_id, 0))
        self._conn.commit()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            hgp = HGP()
            if importance > 0.8:
                hgp.add_goal(f"Reflect on high-salience episode: {event[:60]}", priority=2)
        except (ImportError, Exception):
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("episodic_record", "salience_grouping", importance, True)
        except (ImportError, Exception):
            pass
        return eid

    def _score(self, ep: dict, cue: str, now: float) -> float:
        elapsed_min = (now - ep["timestamp"]) / 60.0
        recency = math.exp(-DECAY_RATE * elapsed_min)
        emotion_weight = 1.0 + abs(ep["valence"])
        cue_words = set(re.findall(r'\\w+', cue.lower()))
        ep_words = set(re.findall(r'\\w+', (ep["event"] + " " + ep["context"]).lower()))
        overlap = len(cue_words & ep_words)
        denom = math.log(1 + len(cue_words) + len(ep_words) + 1)
        cue_relevance = overlap / denom if denom > 0 else 0.0
        return ep["importance"] * emotion_weight * recency * (0.5 + 0.5 * cue_relevance)

    def recall(self, cue: str, top_k: int = 5) -> list[dict]:
        """Returns top_k episodes sorted by composite retrieval score."""
        now = time.time()
        with self._lock:
            scored = []
            for ep in self._cache.values():
                s = self._score(ep, cue, now)
                scored.append((s, ep))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for s, ep in scored[:top_k]:
                ep["access_count"] += 1
                ep["importance"] = min(1.0, ep["importance"] * 1.05)
                results.append({**ep, "retrieval_score": round(s, 4)})
        try:
            from WorkingMemory import WorkingMemory as WM
            WM().store("last_recall_cue", cue, importance=0.6)
        except (ImportError, Exception):
            pass
        return results

    def replay(self, episode_id: str) -> dict:
        """Returns full episode record for a given episode_id, with decay-adjusted importance."""
        now = time.time()
        with self._lock:
            ep = self._cache.get(episode_id)
            if not ep:
                return {"error": f"episode {episode_id} not found"}
            elapsed = now - ep["timestamp"]
            decayed = ep["importance"] * math.exp(-DECAY_RATE * elapsed / 60.0)
            ep["access_count"] += 1
            return {**ep, "decayed_importance": round(decayed, 4),
                    "elapsed_minutes": round(elapsed / 60, 2)}

    def emotional_highlights(self) -> list[dict]:
        """Returns top 10 episodes ranked by emotional salience (abs(valence) * importance)."""
        with self._lock:
            ranked = sorted(self._cache.values(),
                key=lambda e: abs(e["valence"]) * e["importance"], reverse=True)
            return [dict(episode_id=e["episode_id"], event=e["event"],
                         emotion=e["emotion"], valence=e["valence"],
                         importance=e["importance"],
                         salience=round(abs(e["valence"]) * e["importance"], 4))
                    for e in ranked[:10]]

    def temporal_summary(self, hours: float = 1.0) -> dict:
        """Narrates Nova's experience over the last N hours as a coherent grouped story."""
        now = time.time()
        cutoff = now - hours * 3600
        with self._lock:
            recent = [e for e in self._cache.values() if e["timestamp"] >= cutoff]
        if not recent:
            return {"narrative": "No episodes recorded in this window.", "count": 0}
        recent.sort(key=lambda e: e["timestamp"])
        groups: dict[str, list] = {}
        for ep in recent:
            groups.setdefault(ep["group_id"], []).append(ep)
        chapters = []
        for gid, eps in groups.items():
            emotions = [e["emotion"] for e in eps]
            avg_val = statistics.mean(e["valence"] for e in eps)
            tone = "positive" if avg_val > 0.2 else ("negative" if avg_val < -0.2 else "neutral")
            events_str = "; ".join(e["event"][:50] for e in eps)
            chapters.append(f"[{tone.upper()}] {events_str}")
        valences = [e["valence"] for e in recent]
        entropy_val = 0.0
        if len(valences) > 1:
            mean_v = statistics.mean(valences)
            std_v = statistics.stdev(valences) + 1e-9
            z_scores = [abs((v - mean_v) / std_v) for v in valences]
            anomalies = [z for z in z_scores if z > 2.0]
            entropy_val = round(len(anomalies) / len(z_scores), 4)
        return {"narrative": " | ".join(chapters), "episode_count": len(recent),
                "group_count": len(groups), "emotional_entropy": entropy_val,
                "hours_covered": hours}

    def _evict(self) -> None:
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k]["access_count"])
        del self._cache[lru_key]

    def consolidate(self) -> int:
        """Persists all in-memory episodes to SQLite; returns count saved."""
        with self._lock:
            items = list(self._cache.values())
        c = self._conn.cursor()
        for ep in items:
            c.execute("""INSERT OR REPLACE INTO episodes
                (episode_id,event,context,emotion,importance,valence,timestamp,group_id,access_count)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (ep["episode_id"], ep["event"], ep["context"], ep["emotion"],
                 ep["importance"], ep["valence"], ep["timestamp"],
                 ep["group_id"], ep["access_count"]))
        self._conn.commit()
        return len(items)

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            items = len(self._cache)
            vals = [e["valence"] for e in self._cache.values()] or [0.0]
            imps = [e["importance"] for e in self._cache.values()] or [0.0]
        mean_imp = statistics.mean(imps)
        mean_val = statistics.mean(vals)
        entropy = -sum((abs(v) / (sum(abs(x) for x in vals) + 1e-9)) *
                       math.log2(abs(v) / (sum(abs(x) for x in vals) + 1e-9) + 1e-12)
                       for v in vals)
        return {"items": items, "confidence": round(mean_imp, 4),
                "accuracy": round(abs(mean_val), 4), "entropy": round(entropy, 4),
                "cycles": self._cycles, "active": 1, "capacity": CAPACITY}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            try:
                self._cycles += 1
                self.consolidate()
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor as MCM
                    s = self.status()
                    MCM().log_reasoning("episodic_auto", "consolidation",
                                        s["confidence"], True)
                except (ImportError, Exception):
                    pass
                try:
                    from BayesianBeliefSystem import BayesianBeliefSystem as BBS
                    s = self.status()
                    BBS().update("episodic_health",
                                 {"entropy": s["entropy"]},
                                 {"entropy": s["confidence"]})
                except (ImportError, Exception):
                    pass
            except Exception:
                pass

# Usage: obj = EpisodicMemoryEngine() | result = obj.record("Nova solved a hard problem", "reasoning session", "pride", 0.9, 0.8)