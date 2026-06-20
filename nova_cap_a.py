"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
NovaCuriosityDrive — autonomous curiosity and epistemic hunger engine.

Ranks knowledge domains by information-gain potential (entropy × novelty × relevance),
generates self-directed research goals, tracks calibration of curiosity predictions,
and feeds discoveries back into Nova's belief and memory systems.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import math
import time
import threading
import sqlite3
import statistics
import hashlib
import os
import json
import re
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "curiosity_drive.db")

class NovaCuriosityDrive:
    """Autonomous epistemic curiosity engine that self-generates research goals and tracks information gain."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._boot()
        self._ema_gain: float = 0.5
        self._history: list[tuple[float, float]] = []
        self._cycle_count: int = 0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _boot(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS domains (
                name TEXT PRIMARY KEY,
                entropy REAL DEFAULT 1.0,
                novelty REAL DEFAULT 1.0,
                relevance REAL DEFAULT 1.0,
                accesses INTEGER DEFAULT 0,
                last_ts REAL DEFAULT 0,
                prior REAL DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                insight TEXT,
                gain REAL,
                ts REAL
            );
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted REAL,
                actual REAL,
                ts REAL
            );
        """)
        self._conn.commit()

    def _salience(self, name: str, entropy: float, novelty: float,
                  relevance: float, last_ts: float, accesses: int) -> float:
        now = time.time()
        decay = math.exp(-(now - last_ts) / 3600.0)
        access_penalty = 1.0 / (1.0 + accesses)
        return entropy * novelty * relevance * decay * access_penalty

    def observe_domain(self, name: str, evidence: float, relevance: float = 0.8) -> dict[str, float]:
        """Returns updated Bayesian posterior and curiosity score for a domain."""
        with self._lock:
            c = self._conn.cursor()
            row = c.execute("SELECT * FROM domains WHERE name=?", (name,)).fetchone()
            if row is None:
                c.execute("INSERT INTO domains(name,last_ts) VALUES(?,?)", (name, time.time()))
                self._conn.commit()
                row = c.execute("SELECT * FROM domains WHERE name=?", (name,)).fetchone()
            _, entropy, novelty, rel, accesses, last_ts, prior = row
            likelihood = max(0.05, min(0.95, evidence))
            posterior = (likelihood * prior) / max(1e-12, likelihood * prior + (1 - likelihood) * (1 - prior))
            novelty = max(0.01, novelty * 0.9 + (1.0 - posterior) * 0.1)
            dist = {name: posterior, "_other": 1.0 - posterior}
            new_entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
            score = self._salience(name, new_entropy, novelty, relevance, last_ts, accesses)
            c.execute("""UPDATE domains SET entropy=?,novelty=?,relevance=?,
                         accesses=accesses+1,last_ts=?,prior=? WHERE name=?""",
                      (new_entropy, novelty, relevance, time.time(), posterior, name))
            self._conn.commit()
            return {"domain": name, "posterior": round(posterior, 4),
                    "entropy": round(new_entropy, 4), "curiosity_score": round(score, 4)}

    def top_curiosities(self, k: int = 5) -> list[dict[str, Any]]:
        """Returns top-k domains ranked by TF-IDF-weighted curiosity salience score."""
        with self._lock:
            c = self._conn.cursor()
            rows = c.execute("SELECT name,entropy,novelty,relevance,accesses,last_ts FROM domains").fetchall()
            if not rows:
                return []
            N = len(rows)
            scored = []
            for name, entropy, novelty, relevance, accesses, last_ts in rows:
                tf = 1.0 / (1.0 + accesses)
                idf = math.log(N / (accesses + 1))
                tfidf = tf * idf
                score = self._salience(name, entropy, novelty, relevance, last_ts, accesses) * (1.0 + tfidf)
                ci_low = max(0.0, score - 0.1 * (1.0 - entropy))
                ci_high = min(1.0, score + 0.1 * entropy)
                scored.append({"domain": name, "score": round(score, 4),
                               "ci": [round(ci_low, 4), round(ci_high, 4)]})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:k]

    def record_discovery(self, domain: str, insight: str, actual_gain: float) -> dict[str, float]:
        """Returns calibration delta after comparing predicted vs actual information gain and updates EMA."""
        with self._lock:
            predicted = self._ema_gain
            self._ema_gain = 0.15 * actual_gain + 0.85 * self._ema_gain
            self._history.append((predicted, actual_gain))
            c = self._conn.cursor()
            c.execute("INSERT INTO discoveries(domain,insight,gain,ts) VALUES(?,?,?,?)",
                      (domain, insight, actual_gain, time.time()))
            c.execute("INSERT INTO calibration(predicted,actual,ts) VALUES(?,?,?)",
                      (predicted, actual_gain, time.time()))
            self._conn.commit()
            mae = statistics.mean(abs(p - a) for p, a in self._history[-50:])
            return {"predicted": round(predicted, 4), "actual": round(actual_gain, 4),
                    "ema_gain": round(self._ema_gain, 4), "mae": round(mae, 4)}

    def anomaly_check(self) -> dict[str, Any]:
        """Returns z-score anomaly report across curiosity scores, flagging outliers above threshold."""
        with self._lock:
            c = self._conn.cursor()
            rows = c.execute("SELECT name, entropy*novelty*relevance FROM domains").fetchall()
            if len(rows) < 3:
                return {"status": "insufficient_data", "anomalies": []}
            scores = [r[1] for r in rows]
            mu = statistics.mean(scores)
            sigma = statistics.stdev(scores) + 1e-9
            anomalies = []
            for name, score in rows:
                z = (score - mu) / sigma
                if abs(z) > 3.0:
                    anomalies.append({"domain": name, "z_score": round(z, 3)})
            return {"mean": round(mu, 4), "std": round(sigma, 4), "anomalies": anomalies}

    def generate_goal(self) -> dict[str, Any]:
        """Returns a newly created research goal injected into HierarchicalGoalPlanner for the top curiosity domain."""
        top = self.top_curiosities(k=1)
        if not top:
            return {"status": "no_domains"}
        domain = top[0]["domain"]
        score = top[0]["score"]
        goal_desc = f"Research high-curiosity domain: '{domain}' (score={score})"
        result = {"domain": domain, "score": score, "goal": goal_desc, "planner_status": "skipped"}
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner as HGP
            hgp = HGP()
            hgp.add_goal(goal_desc, priority=int(score * 10))
            result["planner_status"] = "injected"
        except (ImportError, Exception) as e:
            result["planner_error"] = str(e)
        try:
            from metacognitive_monitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("curiosity", "entropy_salience_tfidf", score, score > 0.3)
        except (ImportError, Exception):
            pass
        return result

    def calibration_report(self) -> dict[str, Any]:
        """Returns calibration statistics including MAE, z-score of error, and confidence interval."""
        with self._lock:
            if len(self._history) < 2:
                return {"status": "insufficient_history", "samples": len(self._history)}
            errors = [abs(p - a) for p, a in self._history]
            mae = statistics.mean(errors)
            sigma = statistics.stdev(errors) + 1e-9
            z = (mae - 0.0) / sigma
            ci_low = mae - 1.96 * sigma / math.sqrt(len(errors))
            ci_high = mae + 1.96 * sigma / math.sqrt(len(errors))
            return {"samples": len(errors), "mae": round(mae, 4), "z_score": round(z, 4),
                    "ci_95": [round(ci_low, 4), round(ci_high, 4)], "ema_gain": round(self._ema_gain, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            c = self._conn.cursor()
            items = c.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
            pending = c.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
            avg_entropy_row = c.execute("SELECT AVG(entropy) FROM domains").fetchone()[0]
            avg_entropy = avg_entropy_row if avg_entropy_row else 0.0
            return {"items": items, "confidence": round(self._ema_gain, 4),
                    "accuracy": round(1.0 - min(1.0, self._ema_gain), 4),
                    "entropy": round(avg_entropy, 4), "cycles": self._cycle_count,
                    "pending": pending, "active": 1}

    def auto_cycle(self) -> dict[str, Any]:
        """Returns cycle summary after autonomously re-scoring domains and injecting a new curiosity goal."""
        with self._lock:
            self._cycle_count += 1
        report = self.anomaly_check()
        goal = self.generate_goal()
        tops = self.top_curiosities(k=3)
        try:
            from metacognitive_monitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("curiosity_auto_cycle", "salience_ema_bayesian",
                                self._ema_gain, len(report.get("anomalies", [])) == 0)
        except (ImportError, Exception):
            pass
        return {"cycle": self._cycle_count, "anomalies": report.get("anomalies", []),
                "goal": goal, "top_domains": tops}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaCuriosityDrive() | result = obj.observe_domain("quantum_computing", evidence=0.7)