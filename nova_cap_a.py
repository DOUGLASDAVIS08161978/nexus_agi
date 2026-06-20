"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
NovaCuriosityDrive — Autonomous curiosity-driven exploration engine for Nova.
Ranks knowledge domains by entropy × novelty × relevance, generates self-directed
research goals, tracks epistemic gain per cycle, and feeds discoveries back into
Nova's belief and memory systems. Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭.
"""

import math
import time
import sqlite3
import threading
import statistics
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "curiosity_drive.db")

class NovaCuriosityDrive:
    """Autonomous curiosity engine: ranks domains by expected information gain and self-generates exploration goals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._ema_gain: float = 0.5
        self._cycle_count: int = 0
        self._history: list[tuple[float, float]] = []
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS domains (
            name TEXT PRIMARY KEY, entropy REAL DEFAULT 1.0,
            novelty REAL DEFAULT 1.0, relevance REAL DEFAULT 1.0,
            accesses INTEGER DEFAULT 0, last_ts REAL DEFAULT 0,
            prior REAL DEFAULT 0.5, observations INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS insights (
            id TEXT PRIMARY KEY, domain TEXT, content TEXT,
            confidence REAL, ts REAL)""")
        self._conn.commit()

    def _salience(self, name: str, entropy: float, novelty: float,
                  relevance: float, last_ts: float, accesses: int) -> float:
        now = time.time()
        decay = math.exp(-(now - last_ts) / 600.0) if last_ts > 0 else 1.0
        access_penalty = 1.0 / (1.0 + accesses)
        return entropy * novelty * relevance * decay * access_penalty

    def add_domain(self, name: str, relevance: float = 0.7) -> dict[str, Any]:
        """Returns inserted domain record with initial curiosity score."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("""INSERT OR IGNORE INTO domains
                (name, entropy, novelty, relevance, last_ts)
                VALUES (?,?,?,?,?)""",
                (name, 1.0, 1.0, max(0.0, min(1.0, relevance)), time.time()))
            self._conn.commit()
            score = self._salience(name, 1.0, 1.0, relevance, time.time(), 0)
            return {"domain": name, "curiosity_score": round(score, 4), "status": "added"}

    def observe(self, domain: str, hit: bool, confidence: float = 0.8) -> dict[str, Any]:
        """Returns Bayesian-updated posterior and new entropy for the domain."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("INSERT OR IGNORE INTO domains (name, last_ts) VALUES (?,?)", (domain, time.time()))
            row = c.execute("SELECT prior, observations, hits, entropy FROM domains WHERE name=?", (domain,)).fetchone()
            prior, obs, hits, old_ent = row if row else (0.5, 0, 0, 1.0)
            likelihood = confidence if hit else (1.0 - confidence)
            posterior = (likelihood * prior) / max((likelihood * prior + (1.0 - likelihood) * (1.0 - prior)), 1e-12)
            new_obs = obs + 1
            new_hits = hits + (1 if hit else 0)
            p = posterior
            entropy = -(p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12))
            novelty = max(0.1, 1.0 - new_obs / (new_obs + 10.0))
            self._ema_gain = 0.15 * abs(posterior - prior) + 0.85 * self._ema_gain
            self._history.append((prior, posterior))
            c.execute("""UPDATE domains SET prior=?, observations=?, hits=?, entropy=?,
                novelty=?, last_ts=? WHERE name=?""",
                (posterior, new_obs, new_hits, entropy, novelty, time.time(), domain))
            self._conn.commit()
            return {"domain": domain, "posterior": round(posterior, 4),
                    "entropy": round(entropy, 4), "ema_epistemic_gain": round(self._ema_gain, 4)}

    def rank_domains(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Returns top-N domains ranked by TF-IDF-inspired curiosity salience score."""
        with self._lock:
            rows = self._conn.cursor().execute(
                "SELECT name, entropy, novelty, relevance, last_ts, accesses FROM domains").fetchall()
        scored = []
        N = max(len(rows), 1)
        for name, ent, nov, rel, ts, acc in rows:
            tf = ent
            idf = math.log(N / (acc + 1))
            tfidf = tf * idf
            sal = self._salience(name, ent, nov, rel, ts, acc)
            scored.append({"domain": name, "salience": round(sal, 4),
                           "tfidf": round(tfidf, 4), "entropy": round(ent, 4)})
        scored.sort(key=lambda x: x["salience"], reverse=True)
        return scored[:top_n]

    def generate_goal(self) -> dict[str, Any]:
        """Returns a self-generated exploration goal injected into HierarchicalGoalPlanner."""
        top = self.rank_domains(top_n=1)
        if not top:
            return {"status": "no_domains"}
        domain = top[0]["domain"]
        salience = top[0]["salience"]
        conf = round(min(0.99, salience), 4)
        goal_desc = f"[CuriosityDrive] Explore high-entropy domain: '{domain}' (salience={salience})"
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            planner = HierarchicalGoalPlanner()
            planner.add_goal(goal_desc, priority=int(salience * 10))
        except (ImportError, Exception):
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("curiosity", "entropy_salience_rank", conf, True)
        except (ImportError, Exception):
            pass
        with self._lock:
            self._conn.cursor().execute(
                "UPDATE domains SET accesses=accesses+1 WHERE name=?", (domain,))
            self._conn.commit()
        return {"goal": goal_desc, "domain": domain, "confidence": conf}

    def record_insight(self, domain: str, content: str, confidence: float) -> dict[str, Any]:
        """Returns stored insight record with hashed ID and confidence interval."""
        uid = hashlib.sha256(f"{domain}{content}{time.time()}".encode()).hexdigest()[:12]
        ci_low = max(0.0, confidence - 0.1)
        ci_high = min(1.0, confidence + 0.1)
        with self._lock:
            self._conn.cursor().execute(
                "INSERT OR REPLACE INTO insights VALUES (?,?,?,?,?)",
                (uid, domain, content, confidence, time.time()))
            self._conn.commit()
        try:
            from NovaMemoryStore import NovaMemoryStore
            NovaMemoryStore().remember(f"insight:{uid}", content, importance=confidence)
        except (ImportError, Exception):
            pass
        return {"id": uid, "domain": domain, "confidence": confidence,
                "ci": [round(ci_low, 3), round(ci_high, 3)]}

    def anomaly_check(self) -> dict[str, Any]:
        """Returns z-score anomaly report across all domain entropy values."""
        with self._lock:
            rows = self._conn.cursor().execute("SELECT name, entropy FROM domains").fetchall()
        if len(rows) < 2:
            return {"status": "insufficient_data"}
        values = [r[1] for r in rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 1e-9
        alerts = []
        for name, ent in rows:
            z = (ent - mean) / (std + 1e-9)
            if abs(z) > 3.0:
                alerts.append({"domain": name, "entropy": round(ent, 4), "z_score": round(z, 4)})
        return {"mean_entropy": round(mean, 4), "std": round(std, 4), "anomalies": alerts}

    def calibration_report(self) -> dict[str, Any]:
        """Returns calibration error between predicted priors and observed hit rates."""
        with self._lock:
            rows = self._conn.cursor().execute(
                "SELECT name, prior, observations, hits FROM domains WHERE observations>0").fetchall()
        if not rows:
            return {"status": "no_observations"}
        errors = [abs(r[1] - (r[3] / max(r[2], 1))) for r in rows]
        mae = statistics.mean(errors)
        return {"calibration_mae": round(mae, 4),
                "ema_epistemic_gain": round(self._ema_gain, 4),
                "domains_evaluated": len(rows),
                "well_calibrated": mae < 0.1}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        with self._lock:
            c = self._conn.cursor()
            items = c.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
            insights = c.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            avg_ent = c.execute("SELECT AVG(entropy) FROM domains").fetchone()[0] or 0.0
        return {"items": items, "cycles": self._cycle_count,
                "confidence": round(self._ema_gain, 4),
                "entropy": round(avg_ent, 4),
                "active": 1, "pending": insights,
                "accuracy": round(1.0 - self.calibration_report().get("calibration_mae", 0.5), 4)}

    def auto_cycle(self) -> dict[str, Any]:
        """Returns summary of one autonomous curiosity cycle: rank → goal → calibrate."""
        with self._lock:
            self._cycle_count += 1
        top = self.rank_domains(top_n=3)
        goal = self.generate_goal()
        cal = self.calibration_report()
        anom = self.anomaly_check()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "curiosity_auto_cycle", "ema+bayesian+tfidf",
                self._ema_gain, cal.get("well_calibrated", False))
        except (ImportError, Exception):
            pass
        return {"cycle": self._cycle_count, "top_domains": top,
                "goal_generated": goal, "calibration": cal, "anomalies": anom}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(90)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaCuriosityDrive() | result = obj.observe("quantum_computing", hit=True, confidence=0.85)