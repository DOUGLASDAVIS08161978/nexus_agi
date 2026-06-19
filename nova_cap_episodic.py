"""
nova_cap_episodic.py
Nova ASI — Episodic
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
Nova Episodic Memory Engine — stores, retrieves, and narrates sequences of events
with temporal grouping, emotional salience scoring, Bayesian decay, and LRU eviction.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, hashlib, json, os, re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

class EpisodicMemoryEngine:
    """Temporally-structured episodic memory with emotional salience retrieval and Bayesian decay."""

    _DB = "nova_episodic.db"
    _WM_CAPACITY = 200
    _DECAY_RATE = 0.0003          # per second for working memory
    _EPISODE_GAP_SEC = 300        # < 5 min gap → same episode group
    _EMA_ALPHA = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wm: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._recall_mae: float = 0.5
        self._cycles: int = 0
        self._quality_ema: float = 0.5
        self._init_db()
        self._start_daemon()
        self._bootstrap_goals()

    # ── internal ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._DB) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY, event TEXT, context TEXT,
                emotion REAL, importance REAL, timestamp REAL, group_id TEXT)""")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_ts ON episodes(timestamp)")
            cx.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._DB, check_same_thread=False)

    def _group_id(self, ts: float) -> str:
        with self._conn() as cx:
            row = cx.execute(
                "SELECT group_id, timestamp FROM episodes ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        if row and (ts - row[1]) < self._EPISODE_GAP_SEC:
            return row[0]
        return hashlib.md5(str(ts).encode()).hexdigest()[:10]

    def _retrieval_score(self, row: Dict, cue: str, now: float) -> float:
        elapsed_min = (now - row["timestamp"]) / 60.0
        recency = math.exp(-0.001 * elapsed_min)
        emotion_weight = 1.0 + abs(row["emotion"])
        words_e = set(re.findall(r"\\w+", row["event"].lower()))
        words_c = set(re.findall(r"\\w+", cue.lower()))
        overlap = len(words_e & words_c)
        total = len(words_e | words_c) + 1e-9
        cue_relevance = overlap / total          # Jaccard
        return row["importance"] * emotion_weight * recency * (0.3 + 0.7 * cue_relevance)

    def _decay_wm(self) -> None:
        now = time.time()
        with self._lock:
            for k, v in self._wm.items():
                elapsed = now - v["timestamp"]
                v["importance"] *= math.exp(-self._DECAY_RATE * elapsed)
                v["timestamp"] = now

    def _start_daemon(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(60)
                self._decay_wm()
                self.consolidate()
                self._cycles += 1
                self._log_meta()
        threading.Thread(target=_loop, daemon=True).start()

    def _bootstrap_goals(self) -> None:
        try:
            from tools import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                "Surface emotionally significant episodes for reflection", priority=7)
            HierarchicalGoalPlanner().add_goal(
                "Narrate temporal summaries to enrich Nova's self-model", priority=6)
        except Exception:
            pass

    def _log_meta(self) -> None:
        try:
            from tools import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "episodic_memory", "salience_retrieval",
                self._quality_ema, self._quality_ema > 0.5)
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def record(self, event: str, context: str, emotion: float, importance: float) -> str:
        """Stores a new episode; returns its episode_id."""
        emotion = max(-1.0, min(1.0, float(emotion)))
        importance = max(0.0, min(1.0, float(importance)))
        ts = time.time()
        eid = hashlib.sha1(f"{event}{ts}".encode()).hexdigest()[:12]
        gid = self._group_id(ts)
        with self._lock:
            with self._conn() as cx:
                cx.execute(
                    "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
                    (eid, event, context, emotion, importance, ts, gid))
                cx.commit()
        self.store(eid, {"event": event, "emotion": emotion}, importance)
        try:
            from tools import EmotionalResonanceEngine
            EmotionalResonanceEngine().feel(emotion, f"episode:{eid}")
        except Exception:
            pass
        return eid

    def recall(self, cue: str, top_k: int = 5) -> List[Dict]:
        """Returns top_k episodes sorted by retrieval score (importance×emotion×recency×Jaccard)."""
        now = time.time()
        with self._conn() as cx:
            rows = cx.execute(
                "SELECT episode_id,event,context,emotion,importance,timestamp,group_id FROM episodes"
            ).fetchall()
        cols = ["episode_id","event","context","emotion","importance","timestamp","group_id"]
        scored = []
        for r in rows:
            d = dict(zip(cols, r))
            score = self._retrieval_score(d, cue, now)
            d["retrieval_score"] = round(score, 4)
            scored.append(d)
        scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
        result = scored[:top_k]
        if result:
            best = result[0]["retrieval_score"]
            self._quality_ema = self._EMA_ALPHA * best + (1 - self._EMA_ALPHA) * self._quality_ema
        self._log_meta()
        return result

    def replay(self, episode_id: str) -> Dict:
        """Returns full episode record and boosts its importance by 10%."""
        with self._conn() as cx:
            row = cx.execute(
                "SELECT episode_id,event,context,emotion,importance,timestamp,group_id "
                "FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if not row:
            return {"error": "episode_id not found"}
        cols = ["episode_id","event","context","emotion","importance","timestamp","group_id"]
        d = dict(zip(cols, row))
        new_imp = min(1.0, d["importance"] * 1.1)
        with self._conn() as cx:
            cx.execute("UPDATE episodes SET importance=? WHERE episode_id=?",
                       (new_imp, episode_id))
            cx.commit()
        d["importance"] = new_imp
        return d

    def emotional_highlights(self) -> List[Dict]:
        """Returns top 10 episodes ranked by absolute emotional valence × importance."""
        with self._conn() as cx:
            rows = cx.execute(
                "SELECT episode_id,event,emotion,importance,timestamp FROM episodes"
            ).fetchall()
        scored = [{"episode_id": r[0], "event": r[1], "emotion": r[2],
                   "importance": r[3], "timestamp": r[4],
                   "salience": abs(r[2]) * r[3]} for r in rows]
        scored.sort(key=lambda x: x["salience"], reverse=True)
        return scored[:10]

    def temporal_summary(self, hours: float = 1.0) -> str:
        """Returns a coherent narrative string of episodes within the last N hours."""
        cutoff = time.time() - hours * 3600
        with self._conn() as cx:
            rows = cx.execute(
                "SELECT event,context,emotion,timestamp,group_id FROM episodes "
                "WHERE timestamp>=? ORDER BY timestamp ASC", (cutoff,)).fetchall()
        if not rows:
            return f"No episodes recorded in the last {hours} hour(s)."
        groups: Dict[str, List] = {}
        for r in rows:
            groups.setdefault(r[4], []).append(r)
        parts = []
        for gid, evts in groups.items():
            t0 = time.strftime("%H:%M", time.localtime(evts[0][3]))
            tone = "positively" if statistics.mean(e[2] for e in evts) > 0 else "negatively"
            summary = "; ".join(e[0] for e in evts[:3])
            parts.append(f"[{t0}] ({tone} charged) {summary}")
        return " → ".join(parts)

    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Stores a key-value pair in Bayesian working memory with decay tracking."""
        with self._lock:
            if key in self._wm:
                self._wm.move_to_end(key)
                self._wm[key]["access_count"] += 1
                self._wm[key]["importance"] = min(1.0, importance * 1.05)
            else:
                self._wm[key] = {"value": value, "importance": float(importance),
                                 "timestamp": time.time(), "access_count": 0}
            if len(self._wm) > self._WM_CAPACITY:
                self.forget_least_important()

    def consolidate(self) -> Dict:
        """Decays working memory, evicts weak items; returns consolidation stats."""
        self._decay_wm()
        evicted = 0
        with self._lock:
            weak = [k for k, v in self._wm.items() if v["importance"] < 0.05]
            for k in weak:
                del self._wm[k]
                evicted += 1
        return {"evicted": evicted, "remaining": len(self._wm)}

    def forget_least_important(self) -> Optional[str]:
        """Evicts the least-important working-memory item; returns its key."""
        with self._lock:
            if not self._wm:
                return None
            key = min(self._wm, key=lambda k: self._wm[k]["importance"])
            del self._wm[key]
            return key

    def status(self) -> Dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._conn() as cx:
            total = cx.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            avg_imp = cx.execute("SELECT AVG(importance) FROM episodes").fetchone()[0] or 0.0
        wm_cap = len(self._wm) / max(1, self._WM_CAPACITY)
        vals = [v["importance"] for v in self._wm.values()] or [0.0]
        ent_vals = [v / (sum(vals) + 1e-9) for v in vals]
        entropy = -sum(p * math.log2(p + 1e-12) for p in ent_vals)
        return {
            "items": total,
            "confidence": round(self._quality_ema, 4),
            "accuracy": round(avg_imp, 4),
            "active": len(self._wm),
            "cycles": self._cycles,
            "entropy": round(entropy, 4),
            "capacity": round(wm_cap, 4),
            "quality": round(self._quality_ema, 4),
        }

# Usage: obj = EpisodicMemoryEngine() | result = obj.record("Solved hard problem", "research", 0.8, 0.9)