"""
nova_cap_introspection.py
Nova ASI — Introspection
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Self-Model Engine — live introspective capability tracker.
Maintains EMA proficiency, gap detection, growth rates, and calibrated
confidence per capability. Satisfies pillars: ①②③④⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import math
import statistics
import time
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_self_model.db")

EMA_ALPHA = 0.10
GAP_THRESHOLD = 0.40
GAP_MIN_ATTEMPTS = 3
HISTORY_WINDOW = 30
ANOMALY_Z = 2.5
AUTO_INTERVAL = 60.0


class NovaSelfModelEngine:
    """Live self-model: tracks capability proficiency, gaps, and growth via EMA."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _build_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS capabilities (
                    name TEXT PRIMARY KEY,
                    ema REAL NOT NULL DEFAULT 0.5,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    first_seen REAL NOT NULL,
                    last_updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ema_history (
                    name TEXT NOT NULL,
                    episode INTEGER NOT NULL,
                    ema REAL NOT NULL,
                    ts REAL NOT NULL,
                    PRIMARY KEY (name, episode)
                );
            """)
            self._conn.commit()

    def _row(self, name: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT name,ema,attempts,successes,first_seen,last_updated FROM capabilities WHERE name=?",
            (name,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(["name","ema","attempts","successes","first_seen","last_updated"], row))

    def update_capability(self, name: str, proficiency: float, evidence: float) -> dict[str, Any]:
        """Updates EMA proficiency for a capability; returns updated record dict."""
        proficiency = max(0.0, min(1.0, proficiency))
        evidence = max(0.0, min(1.0, evidence))
        now = time.time()
        with self._lock:
            row = self._row(name)
            if row is None:
                ema = 0.5 * EMA_ALPHA + 0.5 * (1 - EMA_ALPHA)
                ema = EMA_ALPHA * proficiency + (1 - EMA_ALPHA) * 0.5
                self._conn.execute(
                    "INSERT INTO capabilities VALUES (?,?,1,?,?,?)",
                    (name, ema, int(evidence >= 0.5), now, now)
                )
                episode = 1
            else:
                ema = EMA_ALPHA * proficiency + (1 - EMA_ALPHA) * row["ema"]
                successes = row["successes"] + int(evidence >= 0.5)
                attempts = row["attempts"] + 1
                episode = attempts
                self._conn.execute(
                    "UPDATE capabilities SET ema=?,attempts=?,successes=?,last_updated=? WHERE name=?",
                    (ema, attempts, successes, now, name)
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO ema_history VALUES (?,?,?,?)",
                (name, episode, ema, now)
            )
            self._conn.commit()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("self_model", f"update:{name}", ema, evidence >= 0.5)
        except Exception:
            pass
        return self._row(name) or {}

    def known_gaps(self) -> list[dict[str, Any]]:
        """Returns list of capabilities with proficiency < 0.4 and attempts > 3."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT name,ema,attempts,successes FROM capabilities WHERE ema<? AND attempts>?",
                (GAP_THRESHOLD, GAP_MIN_ATTEMPTS)
            )
            rows = cur.fetchall()
        gaps = []
        for r in rows:
            success_rate = r[3] / max(r[2], 1)
            calibration_error = abs(r[1] - success_rate)
            gaps.append({
                "capability": r[0],
                "ema_proficiency": round(r[1], 4),
                "attempts": r[2],
                "success_rate": round(success_rate, 4),
                "calibration_error": round(calibration_error, 4)
            })
        gaps.sort(key=lambda x: x["ema_proficiency"])
        return gaps

    def strongest_domains(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Returns top-N capabilities ranked by EMA proficiency with confidence intervals."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT name,ema,attempts,successes FROM capabilities WHERE attempts>1 ORDER BY ema DESC LIMIT ?",
                (top_n,)
            )
            rows = cur.fetchall()
        result = []
        for r in rows:
            p = r[3] / max(r[2], 1)
            n = max(r[2], 1)
            ci = 1.96 * math.sqrt(p * (1 - p) / n)
            result.append({
                "capability": r[0],
                "ema_proficiency": round(r[1], 4),
                "success_rate": round(p, 4),
                "ci_95": round(ci, 4),
                "attempts": r[2]
            })
        return result

    def confidence_in(self, capability: str) -> float:
        """Returns calibrated confidence float [0,1] matching actual success rate via Bayesian blend."""
        with self._lock:
            row = self._row(capability)
        if row is None:
            return 0.5
        n = max(row["attempts"], 1)
        success_rate = row["successes"] / n
        ema = row["ema"]
        weight = min(n / (n + 10.0), 0.9)
        calibrated = weight * success_rate + (1 - weight) * ema
        entropy = -sum(
            p * math.log2(p + 1e-12)
            for p in [calibrated, 1 - calibrated]
        )
        uncertainty_penalty = entropy / 2.0
        return round(max(0.0, min(1.0, calibrated - 0.05 * uncertainty_penalty)), 4)

    def growth_rate(self, capability: str) -> float:
        """Returns per-episode growth rate of EMA over last 30 episodes; positive = improving."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT ema FROM ema_history WHERE name=? ORDER BY episode DESC LIMIT ?",
                (capability, HISTORY_WINDOW + 1)
            )
            rows = [r[0] for r in cur.fetchall()]
        if len(rows) < 2:
            return 0.0
        current = rows[0]
        oldest = rows[-1]
        span = max(len(rows) - 1, 1)
        return round((current - oldest) / span, 6)

    def anomaly_scan(self) -> list[dict[str, Any]]:
        """Returns capabilities showing anomalous EMA drift via z-score > 2.5."""
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT name FROM ema_history")
            names = [r[0] for r in cur.fetchall()]
        alerts = []
        for name in names:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT ema FROM ema_history WHERE name=? ORDER BY episode ASC",
                    (name,)
                )
                vals = [r[0] for r in cur.fetchall()]
            if len(vals) < 5:
                continue
            try:
                mu = statistics.mean(vals)
                sigma = statistics.stdev(vals) + 1e-9
                z = (vals[-1] - mu) / sigma
                if abs(z) > ANOMALY_Z:
                    alerts.append({"capability": name, "z_score": round(z, 3), "latest_ema": round(vals[-1], 4)})
            except statistics.StatisticsError:
                continue
        return alerts

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ scoring."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*), AVG(ema), SUM(attempts), SUM(successes) FROM capabilities")
            row = cur.fetchone()
        total = row[0] or 0
        avg_ema = row[1] or 0.5
        attempts = row[2] or 0
        successes = row[3] or 0
        accuracy = successes / max(attempts, 1)
        gaps = len(self.known_gaps())
        entropy = -sum(
            p * math.log2(p + 1e-12)
            for p in [avg_ema, 1 - avg_ema]
        )
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Improve {gaps} capability gaps identified by SelfModelEngine", priority=2
            )
        except Exception:
            pass
        return {
            "items": total,
            "confidence": round(avg_ema, 4),
            "accuracy": round(accuracy, 4),
            "active": total - gaps,
            "pending": gaps,
            "entropy": round(entropy, 4),
            "cycles": attempts,
            "quality": round(avg_ema * accuracy, 4)
        }

    def _auto_loop(self) -> None:
        while True:
            time.sleep(AUTO_INTERVAL)
            try:
                self.anomaly_scan()
                self.status()
            except Exception:
                pass

    def auto_cycle(self) -> dict[str, Any]:
        """Triggers one introspection cycle; returns status dict."""
        alerts = self.anomaly_scan()
        s = self.status()
        s["anomalies"] = alerts
        return s

# Usage: obj = NovaSelfModelEngine() | result = obj.update_capability("reasoning", 0.8, 0.9)