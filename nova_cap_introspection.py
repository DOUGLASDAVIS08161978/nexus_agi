"""
nova_cap_introspection.py
Nova ASI — Introspection
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Self-Model Engine — live introspective capability tracker.
Maintains EMA proficiency, gap detection, growth-rate analysis, and
calibration so Nova can answer 'What am I good at?' and 'Where do I need
to improve?' with mathematically grounded confidence.
"""

import sqlite3
import threading
import math
import statistics
import time
import os
import json
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_selfmodel.db")

class SelfModelEngine:
    """Nova's live self-model: tracks capabilities, gaps, growth, and calibration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._bootstrap_db()
        self._cycle_count: int = 0
        self._start_daemon()

    # ------------------------------------------------------------------ #
    #  DB bootstrap                                                        #
    # ------------------------------------------------------------------ #
    def _bootstrap_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS capabilities (
                    name        TEXT PRIMARY KEY,
                    ema         REAL DEFAULT 0.5,
                    attempts    INTEGER DEFAULT 0,
                    successes   INTEGER DEFAULT 0,
                    first_seen  REAL,
                    history_json TEXT DEFAULT '[]'
                )
            """)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _fetch(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT name,ema,attempts,successes,first_seen,history_json "
            "FROM capabilities WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {
            "name": row[0], "ema": row[1], "attempts": row[2],
            "successes": row[3], "first_seen": row[4],
            "history": json.loads(row[5])
        }

    def _upsert(self, cap: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute("""
                INSERT INTO capabilities(name,ema,attempts,successes,first_seen,history_json)
                VALUES(:name,:ema,:attempts,:successes,:first_seen,:history_json)
                ON CONFLICT(name) DO UPDATE SET
                    ema=excluded.ema, attempts=excluded.attempts,
                    successes=excluded.successes, history_json=excluded.history_json
            """, {
                "name": cap["name"], "ema": cap["ema"],
                "attempts": cap["attempts"], "successes": cap["successes"],
                "first_seen": cap["first_seen"],
                "history_json": json.dumps(cap["history"][-60:])
            })

    def _all_caps(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name,ema,attempts,successes,first_seen,history_json FROM capabilities"
        ).fetchall()
        return [{"name": r[0], "ema": r[1], "attempts": r[2],
                 "successes": r[3], "first_seen": r[4],
                 "history": json.loads(r[5])} for r in rows]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def update_capability(self, name: str, proficiency: float, evidence: float) -> dict[str, Any]:
        """Updates EMA proficiency for a capability and returns updated record."""
        proficiency = max(0.0, min(1.0, proficiency))
        evidence    = max(0.0, min(1.0, evidence))
        with self._lock:
            cap = self._fetch(name) or {
                "name": name, "ema": 0.5, "attempts": 0,
                "successes": 0, "first_seen": time.time(), "history": []
            }
            cap["ema"]      = 0.1 * evidence + 0.9 * cap["ema"]
            cap["attempts"] += 1
            if proficiency >= 0.5:
                cap["successes"] += 1
            cap["history"].append(round(cap["ema"], 4))
            self._upsert(cap)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "self_model", "EMA_update", cap["ema"], proficiency >= 0.5)
        except Exception:
            pass
        return {"capability": name, "ema": round(cap["ema"], 4),
                "attempts": cap["attempts"], "successes": cap["successes"]}

    def known_gaps(self) -> list[dict[str, Any]]:
        """Returns capabilities with EMA < 0.4 and more than 3 attempts (true gaps)."""
        with self._lock:
            caps = self._all_caps()
        gaps = [
            {"name": c["name"], "ema": round(c["ema"], 4), "attempts": c["attempts"]}
            for c in caps if c["ema"] < 0.4 and c["attempts"] > 3
        ]
        gaps.sort(key=lambda x: x["ema"])
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            for g in gaps[:2]:
                HierarchicalGoalPlanner().add_goal(
                    f"Improve capability: {g['name']} (EMA={g['ema']})", priority=7)
        except Exception:
            pass
        return gaps

    def strongest_domains(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Returns top-N capabilities ranked by EMA proficiency."""
        with self._lock:
            caps = self._all_caps()
        ranked = sorted(caps, key=lambda c: c["ema"], reverse=True)
        return [{"name": c["name"], "ema": round(c["ema"], 4),
                 "success_rate": round(c["successes"] / max(c["attempts"], 1), 4)}
                for c in ranked[:top_n]]

    def confidence_in(self, capability: str) -> dict[str, Any]:
        """Returns calibrated confidence dict with EMA, success_rate, and CI bounds."""
        with self._lock:
            cap = self._fetch(capability)
        if cap is None:
            return {"capability": capability, "confidence": 0.5,
                    "ci_low": 0.0, "ci_high": 1.0, "calibrated": False}
        n  = max(cap["attempts"], 1)
        sr = cap["successes"] / n
        z  = 1.96
        margin = z * math.sqrt((sr * (1 - sr)) / n + 1e-9)
        calib_err = abs(cap["ema"] - sr)
        calibrated_conf = cap["ema"] * math.exp(-calib_err)
        return {
            "capability": capability,
            "confidence": round(calibrated_conf, 4),
            "ema": round(cap["ema"], 4),
            "success_rate": round(sr, 4),
            "ci_low": round(max(0.0, sr - margin), 4),
            "ci_high": round(min(1.0, sr + margin), 4),
            "calibration_error": round(calib_err, 4),
            "calibrated": calib_err < 0.1
        }

    def growth_rate(self, capability: str) -> dict[str, Any]:
        """Returns per-episode growth rate over last 30 episodes via EMA delta."""
        with self._lock:
            cap = self._fetch(capability)
        if cap is None or len(cap["history"]) < 2:
            return {"capability": capability, "growth_rate": 0.0, "episodes": 0}
        hist  = cap["history"]
        span  = min(30, len(hist))
        delta = hist[-1] - hist[-span]
        rate  = delta / span
        trend = "improving" if rate > 0.005 else "declining" if rate < -0.005 else "stable"
        return {"capability": capability, "growth_rate": round(rate, 6),
                "trend": trend, "episodes_analysed": span,
                "ema_now": round(hist[-1], 4), "ema_then": round(hist[-span], 4)}

    def anomaly_check(self) -> list[dict[str, Any]]:
        """Returns capabilities where recent EMA z-score > 2.0 (sudden change)."""
        with self._lock:
            caps = self._all_caps()
        alerts: list[dict[str, Any]] = []
        for c in caps:
            hist = c["history"]
            if len(hist) < 5:
                continue
            try:
                mean = statistics.mean(hist)
                std  = statistics.stdev(hist) + 1e-9
                z    = (hist[-1] - mean) / std
                if abs(z) > 2.0:
                    alerts.append({"name": c["name"], "z_score": round(z, 3),
                                   "ema": round(c["ema"], 4)})
            except statistics.StatisticsError:
                pass
        return alerts

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            caps = self._all_caps()
        emas = [c["ema"] for c in caps] or [0.5]
        gaps = [c for c in caps if c["ema"] < 0.4 and c["attempts"] > 3]
        try:
            entropy = -sum(e * math.log2(e + 1e-12) + (1-e) * math.log2(1-e+1e-12)
                           for e in emas) / max(len(emas), 1)
        except ValueError:
            entropy = 0.0
        return {
            "items": len(caps),
            "confidence": round(statistics.mean(emas), 4),
            "accuracy": round(
                sum(c["successes"] for c in caps) /
                max(sum(c["attempts"] for c in caps), 1), 4),
            "active": len(caps) - len(gaps),
            "pending": len(gaps),
            "cycles": self._cycle_count,
            "entropy": round(entropy, 4),
            "quality": round(1.0 - len(gaps) / max(len(caps), 1), 4)
        }

    # ------------------------------------------------------------------ #
    #  Autonomous daemon                                                   #
    # ------------------------------------------------------------------ #
    def auto_cycle(self) -> dict[str, Any]:
        """Runs one autonomous introspection cycle; returns summary of findings."""
        gaps   = self.known_gaps()
        strong = self.strongest_domains(3)
        anomalies = self.anomaly_check()
        self._cycle_count += 1
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                "self_model_auto", "gap+strength+anomaly",
                self.status()["confidence"], len(gaps) == 0)
        except Exception:
            pass
        return {"cycle": self._cycle_count, "gaps_found": len(gaps),
                "top_domains": [s["name"] for s in strong],
                "anomalies": len(anomalies)}

    def _start_daemon(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(300)
                try:
                    self.auto_cycle()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

# Usage: obj = SelfModelEngine() | result = obj.update_capability("reasoning", 0.8, 0.9)
