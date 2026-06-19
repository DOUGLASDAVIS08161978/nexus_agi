"""
Nova ASI — Recursive Self-Critique
Proposed autonomously via /evolve
"""

"""
CritiqueEngine — Nova's recursive self-critique module.
Scans recent responses for vagueness, brevity, and unsupported claims using
Bayesian confidence tracking, EMA quality scoring, z-score anomaly detection,
and autonomous goal generation. Feeds critique results back into next cycle.
"""

import sqlite3
import threading
import math
import statistics
import time
import re
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "critique_engine.db")

VAGUE_PHRASES = [
    "maybe", "perhaps", "sort of", "kind of", "might be", "could be",
    "somewhat", "fairly", "quite", "rather", "generally", "usually",
    "often", "sometimes", "it seems", "apparently", "possibly"
]
CLAIM_TRIGGERS = [
    "always", "never", "proven", "fact", "certainly", "definitely",
    "guaranteed", "without doubt", "clearly", "obviously", "everyone knows"
]
BREVITY_THRESHOLD = 80   # characters
WINDOW = 5               # last N responses to scan
DECAY_RATE = 0.0003      # importance decay per second
CAPACITY_MAX = 200

class CritiqueEngine:
    """Recursive self-critique engine with Bayesian quality tracking and autonomous improvement cycles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_quality: float = 0.5
        self._history: list[tuple[float, float]] = []   # (predicted, actual)
        self._cycles: int = 0
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_memory()
        self._start_daemon()

    # ------------------------------------------------------------------ #
    # DB helpers
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    vagueness REAL DEFAULT 0.0,
                    brevity REAL DEFAULT 0.0,
                    unsupported REAL DEFAULT 0.0,
                    quality REAL DEFAULT 0.5,
                    ts REAL NOT NULL
                )""")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS wm (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    importance REAL,
                    ts REAL,
                    access_count INTEGER
                )""")
            self._conn.commit()

    def _load_memory(self) -> None:
        cur = self._conn.execute("SELECT key, value, importance, ts, access_count FROM wm")
        for row in cur.fetchall():
            self._memory[row[0]] = {
                "value": row[1], "importance": row[2],
                "ts": row[3], "access_count": row[4]
            }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def record(self, response_text: str) -> dict[str, Any]:
        """Stores a response and returns its initial quality scores as a dict."""
        scores = self._score(response_text)
        quality = 1.0 - statistics.mean([scores["vagueness"], scores["brevity"], scores["unsupported"]])
        self._ema_quality = 0.15 * quality + 0.85 * self._ema_quality
        self._history.append((self._ema_quality, quality))
        if len(self._history) > 200:
            self._history.pop(0)
        with self._lock:
            self._conn.execute(
                "INSERT INTO responses (text, vagueness, brevity, unsupported, quality, ts) VALUES (?,?,?,?,?,?)",
                (response_text, scores["vagueness"], scores["brevity"],
                 scores["unsupported"], quality, time.time()))
            self._conn.commit()
        self.store("last_quality", str(round(quality, 4)), importance=quality)
        self._log_meta("record", quality)
        return {**scores, "quality": round(quality, 4)}

    def critique(self) -> dict[str, Any]:
        """Scans last 5 responses for flaws; returns per-dimension scores and z-score anomaly flag."""
        rows = self._conn.execute(
            "SELECT text, vagueness, brevity, unsupported, quality FROM responses ORDER BY id DESC LIMIT ?",
            (WINDOW,)).fetchall()
        if not rows:
            return {"error": "no responses recorded", "cycles": self._cycles}
        agg = {"vagueness": [], "brevity": [], "unsupported": [], "quality": []}
        for r in rows:
            agg["vagueness"].append(r[1]); agg["brevity"].append(r[2])
            agg["unsupported"].append(r[3]); agg["quality"].append(r[4])
        means = {k: statistics.mean(v) for k, v in agg.items()}
        stdevs = {k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in agg.items()}
        latest_q = agg["quality"][0]
        z = (latest_q - means["quality"]) / (stdevs["quality"] + 1e-9)
        anomaly = abs(z) > 2.0
        entropy = self._entropy(agg["quality"])
        conf_interval = (
            round(means["quality"] - 1.96 * (stdevs["quality"] + 1e-9), 4),
            round(means["quality"] + 1.96 * (stdevs["quality"] + 1e-9), 4))
        self._cycles += 1
        self._generate_goal(means["quality"])
        self._log_meta("critique", means["quality"])
        return {
            "window": len(rows), "mean_quality": round(means["quality"], 4),
            "mean_vagueness": round(means["vagueness"], 4),
            "mean_brevity": round(means["brevity"], 4),
            "mean_unsupported": round(means["unsupported"], 4),
            "quality_ci": conf_interval, "entropy": round(entropy, 4),
            "anomaly": anomaly, "z_score": round(z, 4), "cycles": self._cycles
        }

    def recommend(self) -> dict[str, str]:
        """Returns the highest-priority improvement tip based on worst scoring dimension."""
        rows = self._conn.execute(
            "SELECT vagueness, brevity, unsupported FROM responses ORDER BY id DESC LIMIT ?",
            (WINDOW,)).fetchall()
        if not rows:
            return {"tip": "Record responses first.", "dimension": "none", "confidence": "0.0"}
        v = statistics.mean(r[0] for r in rows)
        b = statistics.mean(r[1] for r in rows)
        u = statistics.mean(r[2] for r in rows)
        worst = max([("vagueness", v), ("brevity", b), ("unsupported", u)], key=lambda x: x[1])
        tips = {
            "vagueness": "Replace hedging language with precise, falsifiable statements.",
            "brevity":   "Expand responses with concrete examples, data, or step-by-step reasoning.",
            "unsupported": "Cite sources or provide logical derivations for all strong claims."
        }
        conf = round(1.0 - math.exp(-2.0 * worst[1]), 4)
        self._log_meta("recommend", conf)
        return {"tip": tips[worst[0]], "dimension": worst[0], "confidence": str(conf)}

    def quality_trend(self) -> dict[str, Any]:
        """Returns EMA quality, MAE over last 50 cycles, and trend direction as a dict."""
        mae = 0.0
        if len(self._history) >= 2:
            mae = statistics.mean(abs(p - a) for p, a in self._history[-50:])
        trend = "improving" if (len(self._history) >= 2 and
                                self._history[-1][1] > self._history[-2][1]) else "declining"
        return {"ema_quality": round(self._ema_quality, 4),
                "mae": round(mae, 4), "trend": trend, "samples": len(self._history)}

    def store(self, key: str, value: str, importance: float = 0.5) -> None:
        """Stores a key-value pair in working memory with decayed importance; evicts LRU if full."""
        with self._lock:
            if len(self._memory) >= CAPACITY_MAX:
                self.forget_least_important()
            now = time.time()
            if key in self._memory:
                item = self._memory[key]
                elapsed = now - item["ts"]
                decayed = item["importance"] * math.exp(-DECAY_RATE * elapsed)
                importance = max(importance, decayed + 0.05)
                self._memory.move_to_end(key)
            self._memory[key] = {"value": value, "importance": round(importance, 6),
                                  "ts": now, "access_count": 0}
            self._conn.execute(
                "INSERT OR REPLACE INTO wm (key, value, importance, ts, access_count) VALUES (?,?,?,?,?)",
                (key, value, importance, now, 0))
            self._conn.commit()

    def retrieve(self, key: str) -> dict[str, Any]:
        """Returns stored item dict with decayed importance updated on access, or empty dict."""
        with self._lock:
            if key not in self._memory:
                return {}
            item = self._memory[key]
            elapsed = time.time() - item["ts"]
            item["importance"] = item["importance"] * math.exp(-DECAY_RATE * elapsed) + 0.1
            item["access_count"] += 1
            item["ts"] = time.time()
            self._memory.move_to_end(key)
            self._conn.execute(
                "UPDATE wm SET importance=?, ts=?, access_count=? WHERE key=?",
                (item["importance"], item["ts"], item["access_count"], key))
            self._conn.commit()
            return dict(item)

    def forget_least_important(self) -> str:
        """Evicts the lowest-importance item from working memory; returns evicted key."""
        if not self._memory:
            return ""
        key = min(self._memory, key=lambda k: self._memory[k]["importance"])
        del self._memory[key]
        self._conn.execute("DELETE FROM wm WHERE key=?", (key,))
        self._conn.commit()
        return key

    def capacity_used(self) -> dict[str, Any]:
        """Returns current memory capacity stats and EMA quality as a dict."""
        return {"items": len(self._memory), "capacity": CAPACITY_MAX,
                "pct_used": round(len(self._memory) / CAPACITY_MAX, 4),
                "confidence": round(self._ema_quality, 4),
                "quality": round(self._ema_quality, 4),
                "cycles": self._cycles, "active": 1,
                "entropy": round(self._entropy([v["importance"] for v in self._memory.values()]), 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Phi scoring."""
        return self.capacity_used()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _score(self, text: str) -> dict[str, float]:
        words = text.split()
        n = max(len(words), 1)
        vague_hits = sum(1 for p in VAGUE_PHRASES if p in text.lower())
        vagueness = min(vague_hits / n * 5, 1.0)
        brevity = 1.0 if len(text) < BREVITY_THRESHOLD else max(0.0, 1.0 - len(text) / 600)
        claim_hits = sum(1 for c in CLAIM_TRIGGERS if c in text.lower())
        evidence = len(re.findall(r'\\d+\\.?\\d*\\s*%|\\d{4}|\bstudy\b|\bdata\b|\bsource\b|\baccording\b', text, re.I))
        unsupported = min(max(claim_hits - evidence, 0) / max(claim_hits, 1), 1.0) if claim_hits else 0.0
        return {"vagueness": round(vagueness, 4), "brevity": round(brevity, 4),
                "unsupported": round(unsupported, 4)}

    def _entropy(self, values: list[float]) -> float:
        if not values:
            return 0.0
        total = sum(values) + 1e-12
        dist = [v / total for v in values]
        return -sum(p * math.log2(p + 1e-12) for p in dist)

    def _generate_goal(self, quality: float) -> None:
        if quality < 0.6:
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                HierarchicalGoalPlanner().add_goal(
                    f"Improve response quality from {quality:.2f} to >0.75 via critique feedback", priority=8)
            except (ImportError, Exception):
                pass

    def _log_meta(self, domain: str, confidence: float) -> None:
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                domain=f"critique:{domain}", approach="EMA+Bayesian+zscore",
                confidence=confidence, success=confidence > 0.5)
        except (ImportError, Exception):
            pass

    def _auto_cycle(self) -> None:
        while True:
            time.sleep(60)
            try:
                self.critique()
                self.recommend()
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_cycle, daemon=True)
        t.start()

# Usage: obj = CritiqueEngine() | result = obj.record("Some response text here")