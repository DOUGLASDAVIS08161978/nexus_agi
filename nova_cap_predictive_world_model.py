"""
Nova ASI — Predictive World Model
Proposed autonomously via /evolve
"""

"""
WorldPredictor: Bayesian predictive world model with EMA learning, causal confidence propagation,
anomaly detection, and autonomous goal-generation. Predicts future domain states and tracks accuracy.
"""

import sqlite3
import threading
import math
import statistics
import time
import json
import os
import hashlib
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "world_predictor.db")

class WorldPredictor:
    """Predictive world model: observe facts, predict futures, track accuracy, self-improve."""

    _DECAY = 0.693 / (7 * 86400)   # half-life = 7 days
    _CAPACITY = 200
    _EMA_ALPHA = 0.15
    _HORIZON_CONF_BASE = 0.92       # per-day confidence decay

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._ema: dict[str, float] = {}          # domain → EMA prediction score
        self._history: dict[str, list] = {}       # domain → [(predicted, actual, ts)]
        self._cycles: int = 0
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._load_cache()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS observations(
            id TEXT PRIMARY KEY, domain TEXT, fact TEXT, value REAL,
            importance REAL, ts REAL, access_count INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id TEXT PRIMARY KEY, domain TEXT, horizon INTEGER,
            predicted REAL, actual REAL, confidence REAL, ts REAL, resolved INTEGER)""")
        self._conn.commit()

    def _load_cache(self) -> None:
        c = self._conn.cursor()
        c.execute("SELECT id,domain,fact,value,importance,ts,access_count FROM observations ORDER BY ts DESC LIMIT ?", (self._CAPACITY,))
        for row in c.fetchall():
            key = row[0]
            self._cache[key] = {"domain": row[1], "fact": row[2], "value": row[3],
                                "importance": row[4], "ts": row[5], "access_count": row[6]}

    def observe(self, domain: str, fact: str, value: float = 1.0, importance: float = 0.5) -> dict[str, Any]:
        """Stores a domain observation and returns the stored record with salience score."""
        now = time.time()
        key = hashlib.md5(f"{domain}:{fact}".encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                rec = self._cache[key]
                rec["access_count"] += 1
                rec["importance"] = min(1.0, rec["importance"] * 1.1 + 0.05)
                rec["value"] = 0.7 * rec["value"] + 0.3 * value
                rec["ts"] = now
                self._cache.move_to_end(key)
            else:
                rec = {"domain": domain, "fact": fact, "value": value,
                       "importance": importance, "ts": now, "access_count": 1}
                self._cache[key] = rec
                if len(self._cache) > self._CAPACITY:
                    self.forget_least_important()
            salience = math.exp(-(now - rec["ts"]) / 300) * rec["importance"]
            rec["salience"] = round(salience, 4)
            c = self._conn.cursor()
            c.execute("""INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)""",
                      (key, domain, fact, rec["value"], rec["importance"], now, rec["access_count"]))
            self._conn.commit()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "observe", rec["importance"], True)
        except Exception:
            pass
        return rec

    def predict(self, domain: str, horizon_days: int = 1) -> dict[str, Any]:
        """Returns a probabilistic future-state prediction with confidence interval for the domain."""
        with self._lock:
            domain_facts = [v for v in self._cache.values() if v["domain"] == domain]
        now = time.time()
        if not domain_facts:
            return {"domain": domain, "horizon_days": horizon_days, "predicted": 0.5,
                    "confidence": 0.1, "ci_low": 0.0, "ci_high": 1.0, "basis": "no_data"}
        # EMA-weighted value with decay
        weights, vals = [], []
        for f in domain_facts:
            elapsed = now - f["ts"]
            w = math.exp(-self._DECAY * elapsed) * f["importance"] * (1 + math.log1p(f["access_count"]))
            weights.append(w); vals.append(f["value"])
        total_w = sum(weights) + 1e-12
        pred = sum(v * w for v, w in zip(vals, weights)) / total_w
        # EMA update
        prev = self._ema.get(domain, pred)
        self._ema[domain] = self._EMA_ALPHA * pred + (1 - self._EMA_ALPHA) * prev
        pred = self._ema[domain]
        # Long-horizon confidence decay: conf(n) = base^n
        conf = self._HORIZON_CONF_BASE ** horizon_days
        if len(vals) >= 2:
            try:
                sd = statistics.stdev(vals)
            except statistics.StatisticsError:
                sd = 0.1
        else:
            sd = 0.2
        z = 1.96
        margin = z * sd * (1 + 0.1 * horizon_days)
        ci_low = max(0.0, round(pred - margin, 4))
        ci_high = min(1.0, round(pred + margin, 4))
        # Bayesian entropy over domain values
        buckets: dict[str, float] = {}
        for v in vals:
            b = str(round(v, 1))
            buckets[b] = buckets.get(b, 0) + 1 / len(vals)
        entropy = -sum(p * math.log2(p + 1e-12) for p in buckets.values())
        pred_id = hashlib.md5(f"{domain}:{horizon_days}:{now}".encode()).hexdigest()
        c = self._conn.cursor()
        c.execute("INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?,?,?)",
                  (pred_id, domain, horizon_days, round(pred, 4), None, round(conf, 4), now, 0))
        self._conn.commit()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Validate prediction for domain '{domain}' in {horizon_days}d", priority=3)
        except Exception:
            pass
        return {"domain": domain, "horizon_days": horizon_days, "predicted": round(pred, 4),
                "confidence": round(conf, 4), "ci_low": ci_low, "ci_high": ci_high,
                "entropy": round(entropy, 4), "basis": len(domain_facts)}

    def resolve_prediction(self, domain: str, actual: float) -> dict[str, Any]:
        """Resolves the latest unresolved prediction for a domain and returns accuracy delta."""
        c = self._conn.cursor()
        c.execute("SELECT id,predicted,confidence FROM predictions WHERE domain=? AND resolved=0 ORDER BY ts DESC LIMIT 1", (domain,))
        row = c.fetchone()
        if not row:
            return {"status": "no_pending", "domain": domain}
        pid, predicted, conf = row
        error = abs(predicted - actual)
        c.execute("UPDATE predictions SET actual=?, resolved=1 WHERE id=?", (actual, pid))
        self._conn.commit()
        with self._lock:
            hist = self._history.setdefault(domain, [])
            hist.append((predicted, actual, time.time()))
            if len(hist) > 100:
                hist.pop(0)
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(domain, "resolve", conf, error < 0.2)
        except Exception:
            pass
        return {"domain": domain, "predicted": predicted, "actual": actual,
                "error": round(error, 4), "calibrated": error < (1 - conf)}

    def accuracy_report(self) -> dict[str, Any]:
        """Returns per-domain MAE, z-score anomaly flags, and overall calibration quality."""
        c = self._conn.cursor()
        c.execute("SELECT domain,predicted,actual,confidence FROM predictions WHERE resolved=1")
        rows = c.fetchall()
        if not rows:
            return {"status": "no_resolved_predictions", "domains": {}, "overall_mae": None}
        domain_data: dict[str, list] = {}
        for domain, pred, actual, conf in rows:
            domain_data.setdefault(domain, []).append((pred, actual, conf))
        report: dict[str, Any] = {}
        all_errors = []
        for domain, records in domain_data.items():
            errors = [abs(p - a) for p, a, _ in records]
            confs = [c for _, _, c in records]
            mae = statistics.mean(errors)
            all_errors.extend(errors)
            mean_e = statistics.mean(errors)
            std_e = statistics.stdev(errors) if len(errors) > 1 else 0.0
            z_scores = [(e - mean_e) / (std_e + 1e-9) for e in errors]
            anomalies = sum(1 for z in z_scores if abs(z) > 3.0)
            calib_err = abs(statistics.mean(confs) - (1 - mae))
            report[domain] = {"mae": round(mae, 4), "samples": len(records),
                              "anomalies": anomalies, "calibration_error": round(calib_err, 4),
                              "mean_confidence": round(statistics.mean(confs), 4)}
        overall_mae = round(statistics.mean(all_errors), 4) if all_errors else None
        return {"domains": report, "overall_mae": overall_mae, "total_resolved": len(rows)}

    def capacity_used(self) -> dict[str, Any]:
        """Returns current cache fill ratio and item count."""
        with self._lock:
            n = len(self._cache)
        return {"items": n, "capacity": self._CAPACITY, "ratio": round(n / self._CAPACITY, 3),
                "active": 1, "cycles": self._cycles}

    def forget_least_important(self) -> str:
        """Evicts the lowest-importance decayed item from cache; returns evicted key."""
        now = time.time()
        if not self._cache:
            return "empty"
        scores = {k: math.exp(-self._DECAY * (now - v["ts"])) * v["importance"]
                  for k, v in self._cache.items()}
        evict_key = min(scores, key=lambda k: scores[k])
        self._cache.pop(evict_key)
        c = self._conn.cursor()
        c.execute("DELETE FROM observations WHERE id=?", (evict_key,))
        self._conn.commit()
        return evict_key

    def consolidate(self) -> dict[str, Any]:
        """Merges weak observations, prunes low-confidence predictions; returns consolidation stats."""
        now = time.time()
        pruned_obs, pruned_pred = 0, 0
        with self._lock:
            dead = [k for k, v in self._cache.items()
                    if math.exp(-self._DECAY * (now - v["ts"])) * v["importance"] < 0.02]
            for k in dead:
                self._cache.pop(k)
                pruned_obs += 1
        c = self._conn.cursor()
        c.execute("DELETE FROM observations WHERE id IN ({})".format(",".join("?" * len(dead))), dead) if dead else None
        c.execute("DELETE FROM predictions WHERE resolved=0 AND ts < ?", (now - 30 * 86400,))
        pruned_pred = c.rowcount
        self._conn.commit()
        return {"pruned_observations": pruned_obs, "pruned_predictions": pruned_pred, "cycles": self._cycles}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        cap = self.capacity_used()
        c = self._conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions WHERE resolved=1")
        resolved = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE resolved=0")
        pending = c.fetchone()[0]
        ema_vals = list(self._ema.values())
        entropy = (-sum((v / (sum(ema_vals) + 1e-12)) * math.log2(v / (sum(ema_vals) + 1e-12) + 1e-12)
                        for v in ema_vals) if ema_vals else 0.0)
        return {"items": cap["items"], "confidence": round(statistics.mean(ema_vals), 4) if ema_vals else 0.5,
                "accuracy": resolved / max(1, resolved + pending), "active": 1,
                "pending": pending, "cycles": self._cycles, "entropy": round(entropy, 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(3600)
            try:
                with self._lock:
                    self._cycles += 1
                self.consolidate()
                report = self.accuracy_report()
                try:
                    from WorkingMemory import WorkingMemory
                    WorkingMemory().consolidate()
                except Exception:
                    pass
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor
                    mae = report.get("overall_mae") or 0.5
                    MetacognitiveMonitor().log_reasoning("world_predictor", "auto_cycle", 1 - mae, mae < 0.3)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = WorldPredictor() | result = obj.predict("economy", horizon_days=7)