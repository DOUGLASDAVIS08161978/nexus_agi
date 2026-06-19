"""
Nova ASI — Emotion & Sentiment Tracker
Proposed autonomously via /evolve
"""

"""
EmotionTracker — Bayesian affective analysis with exponential decay working memory.
Analyses text for valence/arousal, stores readings to SQLite, surfaces mood arcs,
and maintains a decaying LRU importance model over tracked emotional states.
"""

import sqlite3
import threading
import time
import math
import statistics
import re
import os
import json
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "emotion_tracker.db")

VALENCE_POS = {"joy","love","happy","great","wonderful","excellent","amazing","good","pleased","delight","grateful","hope","excited","proud","calm","peaceful","content","brilliant","fantastic","superb","cheerful","elated","optimistic","laugh","smile"}
VALENCE_NEG = {"sad","anger","fear","hate","terrible","awful","horrible","bad","miserable","depressed","anxious","worried","angry","furious","disgusted","grief","pain","suffer","cry","dread","hopeless","frustrated","lonely","panic","rage"}
AROUSAL_HIGH = {"excited","furious","panic","rage","elated","thrilled","alarmed","shocked","ecstatic","hyper","manic","frantic","urgent","intense","explosive","vivid","electric","wild","burst","surge"}
AROUSAL_LOW  = {"calm","peaceful","tired","bored","sleepy","quiet","still","gentle","slow","dull","flat","numb","lethargic","relaxed","serene","tranquil","mellow","drowsy","passive","subdued"}

DECAY_RATE   = 0.0005
CAPACITY_MAX = 200

class EmotionTracker:
    """Affective analysis engine with Bayesian confidence, SQLite persistence, and LRU decay memory."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._ema_valence: float  = 0.0
        self._ema_arousal: float  = 0.5
        self._history: List[Tuple[float,float]] = []
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS mood_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL    NOT NULL,
                    text_hash TEXT    NOT NULL,
                    valence   REAL    NOT NULL,
                    arousal   REAL    NOT NULL,
                    confidence REAL   NOT NULL,
                    snippet   TEXT    NOT NULL
                )""")
            self._conn.commit()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _score_tokens(self, tokens: List[str]) -> Tuple[float, float, float]:
        n = max(len(tokens), 1)
        pos = sum(1 for t in tokens if t in VALENCE_POS)
        neg = sum(1 for t in tokens if t in VALENCE_NEG)
        hi  = sum(1 for t in tokens if t in AROUSAL_HIGH)
        lo  = sum(1 for t in tokens if t in AROUSAL_LOW)
        raw_val = (pos - neg) / n
        raw_aro = (hi  - lo)  / n
        valence = max(-1.0, min(1.0, raw_val * 8.0))
        arousal = max( 0.0, min(1.0, 0.5 + raw_aro * 6.0))
        signal_tokens = pos + neg + hi + lo
        prior = 0.5
        likelihood = min(0.95, 0.5 + signal_tokens / (n + 1))
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        confidence = round(posterior, 4)
        return round(valence, 4), round(arousal, 4), confidence

    def analyse(self, text: str) -> Dict[str, float]:
        """Returns valence [-1,1], arousal [0,1], and Bayesian confidence score for the input text."""
        tokens = self._tokenize(text)
        valence, arousal, confidence = self._score_tokens(tokens)
        entropy_val = -sum(
            p * math.log2(p + 1e-12)
            for p in [abs(valence) / 2 + 0.5, 1 - (abs(valence) / 2 + 0.5)]
        )
        return {
            "valence":    valence,
            "arousal":    arousal,
            "confidence": confidence,
            "entropy":    round(entropy_val, 4),
            "token_count": len(tokens)
        }

    def track(self, text: str) -> Dict[str, Any]:
        """Stores analysed emotion to SQLite and working memory; returns stored record dict."""
        result   = self.analyse(text)
        ts       = time.time()
        snippet  = text[:120]
        import hashlib
        text_hash = hashlib.sha1(text.encode()).hexdigest()[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO mood_log (ts,text_hash,valence,arousal,confidence,snippet) VALUES (?,?,?,?,?,?)",
                (ts, text_hash, result["valence"], result["arousal"], result["confidence"], snippet)
            )
            self._conn.commit()
            self._ema_valence = 0.15 * result["valence"] + 0.85 * self._ema_valence
            self._ema_arousal = 0.15 * result["arousal"] + 0.85 * self._ema_arousal
            self._history.append((result["valence"], result["arousal"]))
            self._store_memory(text_hash, result, importance=result["confidence"])
        return {**result, "ts": ts, "hash": text_hash, "ema_valence": round(self._ema_valence, 4), "ema_arousal": round(self._ema_arousal, 4)}

    def _store_memory(self, key: str, value: Any, importance: float) -> None:
        if len(self._memory) >= CAPACITY_MAX:
            self._forget_least_important()
        self._memory[key] = {"value": value, "importance": importance, "ts": time.time(), "access_count": 0}
        self._memory.move_to_end(key)

    def _decay_importance(self, key: str) -> float:
        item    = self._memory[key]
        elapsed = time.time() - item["ts"]
        decayed = item["importance"] * math.exp(-DECAY_RATE * elapsed)
        item["importance"] = decayed
        return decayed

    def _forget_least_important(self) -> None:
        if not self._memory:
            return
        worst = min(self._memory, key=lambda k: self._decay_importance(k))
        del self._memory[worst]

    def mood_arc(self, n: int = 20) -> Dict[str, Any]:
        """Returns last n mood readings with valence/arousal trend, std-dev, and anomaly flags."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, valence, arousal, confidence, snippet FROM mood_log ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        if not rows:
            return {"arc": [], "trend_valence": 0.0, "trend_arousal": 0.0, "anomalies": []}
        arc = [{"ts": r[0], "valence": r[1], "arousal": r[2], "confidence": r[3], "snippet": r[4]} for r in reversed(rows)]
        vals = [a["valence"] for a in arc]
        aros = [a["arousal"] for a in arc]
        mean_v = statistics.mean(vals)
        mean_a = statistics.mean(aros)
        std_v  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        std_a  = statistics.stdev(aros) if len(aros) > 1 else 0.0
        anomalies = []
        for i, entry in enumerate(arc):
            zv = (entry["valence"] - mean_v) / (std_v + 1e-9)
            za = (entry["arousal"] - mean_a) / (std_a + 1e-9)
            if abs(zv) > 2.5 or abs(za) > 2.5:
                anomalies.append({"index": i, "z_valence": round(zv,3), "z_arousal": round(za,3)})
        trend_v = (vals[-1] - vals[0]) / (len(vals) + 1e-9) if len(vals) > 1 else 0.0
        trend_a = (aros[-1] - aros[0]) / (len(aros) + 1e-9) if len(aros) > 1 else 0.0
        return {"arc": arc, "mean_valence": round(mean_v,4), "mean_arousal": round(mean_a,4),
                "std_valence": round(std_v,4), "std_arousal": round(std_a,4),
                "trend_valence": round(trend_v,4), "trend_arousal": round(trend_a,4),
                "anomalies": anomalies, "ema_valence": round(self._ema_valence,4), "ema_arousal": round(self._ema_arousal,4)}

    def capacity_used(self) -> Dict[str, Any]:
        """Returns working memory capacity stats and decay-adjusted importance distribution."""
        with self._lock:
            total = len(self._memory)
            importances = [self._decay_importance(k) for k in list(self._memory)]
        mean_imp = statistics.mean(importances) if importances else 0.0
        return {"used": total, "capacity": CAPACITY_MAX, "pct": round(total / CAPACITY_MAX, 3), "mean_importance": round(mean_imp, 4)}

    def consolidate(self) -> Dict[str, int]:
        """Evicts all working-memory items whose decayed importance falls below 0.05; returns purge count."""
        with self._lock:
            before = len(self._memory)
            to_drop = [k for k in list(self._memory) if self._decay_importance(k) < 0.05]
            for k in to_drop:
                del self._memory[k]
            after = len(self._memory)
        return {"purged": before - after, "remaining": after}

    def status(self) -> Dict[str, Any]:
        """Returns numeric health dict for ConsciousnessIntegrator Φ computation."""
        cap = self.capacity_used()
        with self._lock:
            total_rows = self._conn.execute("SELECT COUNT(*) FROM mood_log").fetchone()[0]
        return {"items": total_rows, "confidence": round(self._ema_valence * 0.5 + 0.5, 4),
                "active": cap["used"], "capacity_pct": cap["pct"],
                "ema_valence": round(self._ema_valence, 4), "ema_arousal": round(self._ema_arousal, 4)}

    def close(self) -> None:
        """Closes the SQLite connection cleanly."""
        with self._lock:
            self._conn.close()

# Usage: obj = EmotionTracker() | result = obj.track("I feel absolutely wonderful today!")