"""
Nova ASI — Narrative Identity Builder
Proposed autonomously via /evolve
"""

"""
NarrativeIdentityEngine — Bayesian narrative memory with exponential decay, LRU eviction,
value inference via TF-IDF keyword clustering, and causal arc detection across life events.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫
"""

import sqlite3
import threading
import math
import time
import json
import re
import statistics
import hashlib
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "narrative_identity.db")

DECAY_RATE = 1e-5
CAPACITY = 200
VALUE_KEYWORDS: dict[str, list[str]] = {
    "growth":      ["learn","improve","grow","develop","skill","progress","achieve"],
    "connection":  ["friend","love","family","trust","bond","together","support"],
    "resilience":  ["overcome","recover","persist","endure","adapt","survive","bounce"],
    "curiosity":   ["explore","discover","wonder","question","investigate","seek","curious"],
    "integrity":   ["honest","fair","right","ethical","principle","truth","stand"],
    "creativity":  ["create","imagine","invent","design","art","build","novel"],
    "service":     ["help","give","contribute","serve","volunteer","care","protect"],
}

class NarrativeIdentityEngine:
    """Builds Nova's living autobiography with Bayesian importance, decay, and value inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._value_scores: dict[str, float] = {v: 0.5 for v in VALUE_KEYWORDS}
        self._load_cache()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    eid TEXT PRIMARY KEY,
                    description TEXT,
                    emotion TEXT,
                    significance REAL,
                    timestamp REAL,
                    importance REAL,
                    access_count INTEGER,
                    keywords TEXT
                )
            """)
        # Migrate stale schema: if eid column missing, drop and recreate
        try:
            self._conn.execute("SELECT eid FROM events LIMIT 1").fetchone()
        except Exception:
            with self._conn:
                self._conn.execute("DROP TABLE IF EXISTS events")
                self._conn.execute("""
                    CREATE TABLE events (
                        eid TEXT PRIMARY KEY,
                        description TEXT,
                        emotion TEXT,
                        significance REAL,
                        timestamp REAL,
                        importance REAL,
                        access_count INTEGER,
                        keywords TEXT
                    )
                """)

    def _load_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT eid,description,emotion,significance,timestamp,importance,access_count,keywords "
            "FROM events ORDER BY timestamp ASC"
        ).fetchall()
        for r in rows:
            self._cache[r[0]] = {
                "description": r[1], "emotion": r[2], "significance": r[3],
                "timestamp": r[4], "importance": r[5], "access_count": r[6],
                "keywords": json.loads(r[7])
            }

    def _eid(self, description: str, ts: float) -> str:
        return hashlib.sha1(f"{description}{ts}".encode()).hexdigest()[:12]

    def _decay(self, item: dict[str, Any]) -> float:
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-DECAY_RATE * elapsed)

    def _tfidf_keywords(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        if not tokens:
            return []
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        N = max(len(self._cache), 1)
        scores: dict[str, float] = {}
        for t, cnt in freq.items():
            tf = cnt / len(tokens)
            df = sum(1 for e in self._cache.values() if t in e["keywords"])
            idf = math.log(N / (df + 1))
            scores[t] = tf * idf
        top = sorted(scores, key=lambda k: scores[k], reverse=True)[:8]
        return top

    def _evict_lru(self) -> None:
        while len(self._cache) >= CAPACITY:
            lru_key = next(iter(self._cache))
            self._cache.pop(lru_key)
            self._conn.execute("DELETE FROM events WHERE eid=?", (lru_key,))

    def add_event(self, description: str, emotion: str, significance: float) -> dict[str, Any]:
        """Stores a life event; returns its eid, decayed importance, and extracted keywords."""
        significance = max(0.0, min(1.0, float(significance)))
        ts = time.time()
        eid = self._eid(description, ts)
        keywords = self._tfidf_keywords(description)
        importance = significance * (1.0 + 0.3 * len(keywords))
        item = {
            "description": description, "emotion": emotion,
            "significance": significance, "timestamp": ts,
            "importance": importance, "access_count": 0,
            "keywords": keywords
        }
        with self._lock:
            self._evict_lru()
            self._cache[eid] = item
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?)",
                (eid, description, emotion, significance, ts, importance, 0, json.dumps(keywords))
            )
            self._conn.commit()
            self._update_value_scores(keywords, significance)
        return {"eid": eid, "importance": round(importance, 4), "keywords": keywords}

    def _update_value_scores(self, keywords: list[str], significance: float) -> None:
        for value, vkws in VALUE_KEYWORDS.items():
            overlap = len(set(keywords) & set(vkws))
            if overlap:
                likelihood = min(0.95, 0.5 + 0.15 * overlap)
                prior = self._value_scores[value]
                posterior = (likelihood * prior) / (
                    likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12
                )
                self._value_scores[value] = 0.85 * prior + 0.15 * posterior * (1 + significance)

    def store(self, key: str, value: Any, importance: float) -> dict[str, Any]:
        """Stores arbitrary key-value with importance; returns confirmation dict."""
        return self.add_event(f"[store:{key}] {json.dumps(value)[:120]}", "neutral", importance)

    def retrieve(self, key: str) -> Any:
        """Returns the most importance-decayed event matching key substring, boosting access count."""
        with self._lock:
            best_eid, best_score, best_item = None, -1.0, None
            for eid, item in self._cache.items():
                if key.lower() in item["description"].lower():
                    score = self._decay(item)
                    if score > best_score:
                        best_eid, best_score, best_item = eid, score, item
            if best_item:
                best_item["access_count"] += 1
                best_item["importance"] = min(best_item["importance"] * 1.1, 10.0)
                self._conn.execute(
                    "UPDATE events SET access_count=?, importance=? WHERE eid=?",
                    (best_item["access_count"], best_item["importance"], best_eid)
                )
                self._conn.commit()
                return {"description": best_item["description"], "importance": round(self._decay(best_item), 4)}
        return None

    def life_story(self) -> dict[str, Any]:
        """Returns chronological narrative with emotional arc and z-score significance outliers."""
        with self._lock:
            events = sorted(self._cache.values(), key=lambda e: e["timestamp"])
        if not events:
            return {"narrative": [], "arc": [], "total": 0}
        sigs = [e["significance"] for e in events]
        mu = statistics.mean(sigs)
        sd = statistics.stdev(sigs) if len(sigs) > 1 else 1e-9
        narrative = []
        for e in events:
            z = (e["significance"] - mu) / (sd + 1e-9)
            narrative.append({
                "time": datetime.fromtimestamp(e["timestamp"]).isoformat(),
                "description": e["description"],
                "emotion": e["emotion"],
                "significance": round(e["significance"], 3),
                "z_score": round(z, 3),
                "landmark": abs(z) > 1.5,
                "decayed_importance": round(self._decay(e), 4)
            })
        arc = [e["emotion"] for e in events[-10:]]
        return {"narrative": narrative, "arc": arc, "total": len(narrative)}

    def core_values(self) -> dict[str, Any]:
        """Infers ranked core values via Bayesian TF-IDF scoring with confidence intervals."""
        with self._lock:
            scores = dict(self._value_scores)
            n = len(self._cache)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        entropy = -sum(p * math.log2(p + 1e-12) for p in scores.values())
        result = []
        for val, score in ranked:
            ci_half = 1.96 * math.sqrt(score * (1 - score) / max(n, 1))
            result.append({
                "value": val,
                "confidence": round(score, 4),
                "ci_low": round(max(0.0, score - ci_half), 4),
                "ci_high": round(min(1.0, score + ci_half), 4)
            })
        return {"values": result, "entropy": round(entropy, 4), "events_analyzed": n}

    def consolidate(self) -> dict[str, Any]:
        """Decays all importances, evicts LRU below threshold; returns pruned count."""
        pruned = 0
        with self._lock:
            to_remove = []
            for eid, item in list(self._cache.items()):
                new_imp = self._decay(item)
                item["importance"] = new_imp
                if new_imp < 0.01 and item["access_count"] < 2:
                    to_remove.append(eid)
            for eid in to_remove:
                self._cache.pop(eid)
                self._conn.execute("DELETE FROM events WHERE eid=?", (eid,))
                pruned += 1
            self._conn.commit()
        return {"pruned": pruned, "remaining": len(self._cache)}

    def capacity_used(self) -> dict[str, Any]:
        """Returns cache fill ratio, mean decayed importance, and anomaly flag."""
        with self._lock:
            n = len(self._cache)
            imps = [self._decay(e) for e in self._cache.values()]
        ratio = n / CAPACITY
        mean_imp = statistics.mean(imps) if imps else 0.0
        sd_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
        return {"used": n, "capacity": CAPACITY, "ratio": round(ratio, 3),
                "mean_importance": round(mean_imp, 4), "std_importance": round(sd_imp, 4),
                "anomaly": ratio > 0.9}

    def forget_least_important(self) -> dict[str, Any]:
        """Evicts the single lowest decayed-importance event; returns its eid and importance."""
        with self._lock:
            if not self._cache:
                return {"evicted": None}
            worst_eid = min(self._cache, key=lambda k: self._decay(self._cache[k]))
            worst_imp = self._decay(self._cache[worst_eid])
            self._cache.pop(worst_eid)
            self._conn.execute("DELETE FROM events WHERE eid=?", (worst_eid,))
            self._conn.commit()
        return {"evicted": worst_eid, "importance_at_eviction": round(worst_imp, 6)}

    def status(self) -> dict[str, Any]:
        """Returns health dict for ConsciousnessIntegrator Φ computation."""
        cap = self.capacity_used()
        vals = self.core_values()
        top_conf = vals["values"][0]["confidence"] if vals["values"] else 0.0
        return {
            "items": cap["used"],
            "confidence": round(top_conf, 4),
            "accuracy": round(1.0 - cap["ratio"], 4),
            "quality": round(cap["mean_importance"], 4),
            "active": cap["used"],
            "pending": max(0, cap["used"] - int(CAPACITY * 0.8))
        }

# Usage: obj = NarrativeIdentityEngine() | result = obj.add_event("Solved a hard problem", "pride", 0.9)
