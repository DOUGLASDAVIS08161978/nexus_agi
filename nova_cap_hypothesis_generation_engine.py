"""
Nova ASI — Hypothesis Generation Engine
Proposed autonomously via /evolve
"""

"""
HypothesisEngine: Bayesian hypothesis generation, tracking, and calibration engine.
Generates falsifiable predictions per topic, records outcomes, and autonomously
improves confidence via EMA, Bayesian updates, z-score anomaly detection, and
LRU-eviction working memory. Integrates with Nova's live cognitive systems.
"""

import sqlite3
import threading
import math
import statistics
import hashlib
import time
import random
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "hypothesis_engine.db")

TEMPLATES = [
    "If {topic} increases, then measurable output will rise by >10% within 7 days.",
    "Under {topic} conditions, the failure rate will drop below 5% with confidence 0.80.",
    "{topic} will exhibit a statistically significant trend (p<0.05) within 14 observations.",
    "Introducing {topic} as a variable will shift the mean outcome by at least 1 std deviation.",
    "{topic} correlates positively with downstream performance at r>0.6 over 30 cycles.",
    "Removing {topic} will cause entropy to increase by >15% within 5 cycles.",
    "{topic} predicts the next state with accuracy >70% using a linear model.",
]

class HypothesisEngine:
    """Generates falsifiable hypotheses, tracks outcomes, and self-improves via Bayesian EMA."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_accuracy: float = 0.5
        self._history: list[tuple[float, float]] = []
        self._wm: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._wm_capacity: int = 200
        self._decay_rate: float = 0.001
        self._cycles: int = 0
        self._init_db()
        self._start_daemon()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY, topic TEXT, prediction TEXT,
                confidence REAL, timestamp REAL, outcome INTEGER DEFAULT -1,
                accesses INTEGER DEFAULT 0)""")
            cx.commit()

    def _hid(self, topic: str, ts: float) -> str:
        raw = f"{topic}:{ts}:{random.random()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _wm_store(self, key: str, value: Any, importance: float) -> None:
        with self._lock:
            if key in self._wm:
                self._wm.move_to_end(key)
                self._wm[key]["access_count"] += 1
                self._wm[key]["importance"] = min(1.0, importance + 0.05 * self._wm[key]["access_count"])
            else:
                if len(self._wm) >= self._wm_capacity:
                    self._wm.popitem(last=False)
                self._wm[key] = {"value": value, "importance": importance,
                                 "timestamp": time.time(), "access_count": 0}

    def _wm_retrieve(self, key: str) -> Any:
        with self._lock:
            if key not in self._wm:
                return None
            entry = self._wm[key]
            elapsed = time.time() - entry["timestamp"]
            entry["importance"] *= math.exp(-self._decay_rate * elapsed)
            entry["access_count"] += 1
            entry["importance"] = min(1.0, entry["importance"] + 0.05)
            self._wm.move_to_end(key)
            return entry["value"]

    def generate(self, topic: str) -> dict[str, Any]:
        """Returns a new falsifiable hypothesis dict with id, prediction, and confidence."""
        ts = time.time()
        hid = self._hid(topic, ts)
        template = random.choice(TEMPLATES)
        prediction = template.format(topic=topic)
        prior = self._ema_accuracy
        noise = random.gauss(0, 0.05)
        confidence = max(0.1, min(0.99, prior + noise))
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?)",
                       (hid, topic, prediction, confidence, ts, -1, 0))
            cx.commit()
        self._wm_store(hid, {"topic": topic, "prediction": prediction,
                             "confidence": confidence}, importance=confidence)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Validate hypothesis {hid} on topic '{topic}'", priority=3)
        except Exception:
            pass
        return {"id": hid, "topic": topic, "prediction": prediction,
                "confidence": round(confidence, 4), "timestamp": ts}

    def record_outcome(self, hid: str, was_correct: bool) -> dict[str, Any]:
        """Returns updated EMA accuracy and Bayesian posterior after recording outcome."""
        outcome = 1 if was_correct else 0
        with sqlite3.connect(DB_PATH) as cx:
            row = cx.execute("SELECT confidence, accesses FROM hypotheses WHERE id=?", (hid,)).fetchone()
            if row is None:
                return {"error": "unknown id", "id": hid}
            prior_conf, accesses = row
            cx.execute("UPDATE hypotheses SET outcome=?, accesses=? WHERE id=?",
                       (outcome, accesses + 1, hid))
            cx.commit()
        likelihood = prior_conf if was_correct else (1.0 - prior_conf)
        prior = self._ema_accuracy
        posterior = (likelihood * prior) / max(1e-9, likelihood * prior + (1 - likelihood) * (1 - prior))
        self._ema_accuracy = 0.15 * outcome + 0.85 * self._ema_accuracy
        self._history.append((prior_conf, float(outcome)))
        self._wm_store(hid, {"outcome": outcome, "posterior": posterior}, importance=0.7)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("hypothesis", "bayesian_ema",
                                                 self._ema_accuracy, was_correct)
        except Exception:
            pass
        return {"id": hid, "was_correct": was_correct, "posterior": round(posterior, 4),
                "ema_accuracy": round(self._ema_accuracy, 4)}

    def accuracy(self) -> dict[str, Any]:
        """Returns prediction accuracy stats including EMA, MAE, CI bounds, and entropy."""
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute("SELECT confidence, outcome FROM hypotheses WHERE outcome != -1").fetchall()
        if not rows:
            return {"ema_accuracy": round(self._ema_accuracy, 4), "total": 0,
                    "mae": None, "ci_low": None, "ci_high": None, "entropy": None}
        confs = [r[0] for r in rows]
        outcomes = [r[1] for r in rows]
        mae = statistics.mean(abs(c - o) for c, o in zip(confs, outcomes))
        acc = statistics.mean(outcomes)
        n = len(outcomes)
        z = 1.96
        ci_half = z * math.sqrt(acc * (1 - acc) / max(1, n))
        dist = {1: acc + 1e-12, 0: (1 - acc) + 1e-12}
        total = sum(dist.values())
        entropy = -sum((v / total) * math.log2(v / total) for v in dist.values())
        if len(confs) > 1:
            std = statistics.stdev(confs)
            mean_c = statistics.mean(confs)
            anomalies = [c for c in confs if abs((c - mean_c) / (std + 1e-9)) > 3.0]
        else:
            anomalies = []
        return {"ema_accuracy": round(self._ema_accuracy, 4), "total": n,
                "raw_accuracy": round(acc, 4), "mae": round(mae, 4),
                "ci_low": round(max(0, acc - ci_half), 4),
                "ci_high": round(min(1, acc + ci_half), 4),
                "entropy": round(entropy, 4), "anomaly_count": len(anomalies)}

    def consolidate(self) -> dict[str, Any]:
        """Returns count of weak hypotheses pruned from DB (confidence drift correction)."""
        pruned = 0
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute("SELECT id, confidence FROM hypotheses WHERE outcome=-1").fetchall()
            for hid, conf in rows:
                if conf < 0.05:
                    cx.execute("DELETE FROM hypotheses WHERE id=?", (hid,))
                    pruned += 1
            cx.commit()
        return {"pruned": pruned, "ema_accuracy": round(self._ema_accuracy, 4)}

    def forget_least_important(self) -> dict[str, Any]:
        """Returns id of evicted hypothesis with lowest decayed importance score."""
        with sqlite3.connect(DB_PATH) as cx:
            rows = cx.execute(
                "SELECT id, confidence, timestamp, accesses FROM hypotheses ORDER BY confidence ASC LIMIT 1"
            ).fetchall()
            if not rows:
                return {"evicted": None}
            hid, conf, ts, acc = rows[0]
            elapsed = time.time() - ts
            decayed = conf * math.exp(-self._decay_rate * elapsed) * (1 / (1 + acc))
            cx.execute("DELETE FROM hypotheses WHERE id=?", (hid,))
            cx.commit()
        return {"evicted": hid, "decayed_importance": round(decayed, 6)}

    def capacity_used(self) -> dict[str, Any]:
        """Returns current DB row count and working-memory utilization ratio."""
        with sqlite3.connect(DB_PATH) as cx:
            total = cx.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
            pending = cx.execute("SELECT COUNT(*) FROM hypotheses WHERE outcome=-1").fetchone()[0]
        wm_ratio = len(self._wm) / self._wm_capacity
        return {"items": total, "pending": pending, "wm_ratio": round(wm_ratio, 4),
                "cycles": self._cycles, "confidence": round(self._ema_accuracy, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ scoring."""
        acc = self.accuracy()
        cap = self.capacity_used()
        return {"items": cap["items"], "confidence": cap["confidence"],
                "accuracy": acc.get("raw_accuracy", 0.0) or 0.0,
                "entropy": acc.get("entropy", 0.0) or 0.0,
                "pending": cap["pending"], "cycles": self._cycles,
                "active": 1, "quality": round(1.0 - (acc.get("mae") or 0.5), 4)}

    def auto_cycle(self) -> dict[str, Any]:
        """Returns cycle summary after autonomous consolidation and goal generation."""
        self._cycles += 1
        consolidated = self.consolidate()
        cap = self.capacity_used()
        if cap["items"] > 500:
            self.forget_least_important()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            if self._cycles % 10 == 0:
                HierarchicalGoalPlanner().add_goal(
                    f"Review hypothesis calibration — EMA={self._ema_accuracy:.3f}", priority=2)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("hypothesis_auto_cycle", "ema+bayesian",
                                                 self._ema_accuracy, self._ema_accuracy > 0.6)
        except Exception:
            pass
        return {"cycle": self._cycles, "pruned": consolidated["pruned"],
                "ema_accuracy": round(self._ema_accuracy, 4), "items": cap["items"]}

    def _start_daemon(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(60)
                try:
                    self.auto_cycle()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

# Usage: obj = HypothesisEngine() | result = obj.generate("market_volatility")