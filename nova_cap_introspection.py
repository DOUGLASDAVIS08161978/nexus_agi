"""
nova_cap_introspection.py
Nova ASI — Introspection
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Self-Model Engine — live introspective capability tracker.
Maintains EMA proficiency, growth rates, calibration, and gap detection
so Nova can answer 'What am I good at?' and 'Where do I need to improve?'
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import math
import time
import statistics
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_self_model.db")

class NovaSelfModelEngine:
    """Live self-model: tracks capabilities, gaps, growth, and calibration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._boot()
        self._cycles: int = 0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _boot(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS capabilities (
                name TEXT PRIMARY KEY,
                ema REAL DEFAULT 0.5,
                attempts INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                first_seen REAL,
                last_updated REAL
            );
            CREATE TABLE IF NOT EXISTS ema_history (
                name TEXT,
                episode INTEGER,
                ema REAL,
                ts REAL
            );
        """)
        self._conn.commit()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get(self, name: str) -> dict | None:
        c = self._conn.cursor()
        c.execute("SELECT name,ema,attempts,successes,first_seen,last_updated FROM capabilities WHERE name=?", (name,))
        row = c.fetchone()
        if row is None:
            return None
        return {"name": row[0], "ema": row[1], "attempts": row[2],
                "successes": row[3], "first_seen": row[4], "last_updated": row[5]}

    def _all(self) -> list[dict]:
        c = self._conn.cursor()
        c.execute("SELECT name,ema,attempts,successes,first_seen,last_updated FROM capabilities")
        return [{"name": r[0], "ema": r[1], "attempts": r[2],
                 "successes": r[3], "first_seen": r[4], "last_updated": r[5]} for r in c.fetchall()]

    def _upsert(self, name: str, ema: float, attempts: int, successes: int,
                first_seen: float) -> None:
        now = time.time()
        self._conn.execute(
            """INSERT INTO capabilities(name,ema,attempts,successes,first_seen,last_updated)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
               ema=excluded.ema, attempts=excluded.attempts,
               successes=excluded.successes, last_updated=excluded.last_updated""",
            (name, ema, attempts, successes, first_seen, now))
        episode = attempts
        self._conn.execute(
            "INSERT INTO ema_history(name,episode,ema,ts) VALUES(?,?,?,?)",
            (name, episode, ema, now))
        self._conn.commit()

    def _ema_30_ago(self, name: str, current_episode: int) -> float:
        target = max(0, current_episode - 30)
        c = self._conn.cursor()
        c.execute(
            "SELECT ema FROM ema_history WHERE name=? AND episode<=? ORDER BY episode DESC LIMIT 1",
            (name, target))
        row = c.fetchone()
        return row[0] if row else 0.5

    def _calibration_error(self, cap: dict) -> float:
        if cap["attempts"] == 0:
            return 0.0
        actual_rate = cap["successes"] / cap["attempts"]
        return abs(cap["ema"] - actual_rate)

    # ── public API ────────────────────────────────────────────────────────────

    def update_capability(self, name: str, proficiency: float, evidence: float) -> dict[str, Any]:
        """Updates EMA proficiency for a capability; returns updated record."""
        proficiency = max(0.0, min(1.0, proficiency))
        evidence = max(0.0, min(1.0, evidence))
        with self._lock:
            rec = self._get(name)
            if rec is None:
                ema = 0.5
                attempts = 0
                successes = 0
                first_seen = time.time()
            else:
                ema = rec["ema"]
                attempts = rec["attempts"]
                successes = rec["successes"]
                first_seen = rec["first_seen"]
            new_ema = 0.1 * evidence + 0.9 * ema
            attempts += 1
            if proficiency >= 0.5:
                successes += 1
            self._upsert(name, new_ema, attempts, successes, first_seen)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("self_model", "EMA_update", new_ema, proficiency >= 0.5)
        except Exception:
            pass
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            if new_ema < 0.4 and attempts > 3:
                HierarchicalGoalPlanner().add_goal(f"Improve capability: {name}", priority=7)
        except Exception:
            pass
        return {"name": name, "ema": round(new_ema, 4), "attempts": attempts, "successes": successes}

    def known_gaps(self) -> list[dict[str, Any]]:
        """Returns capabilities with EMA < 0.4 AND attempts > 3, sorted by severity."""
        with self._lock:
            all_caps = self._all()
        gaps = [c for c in all_caps if c["ema"] < 0.4 and c["attempts"] > 3]
        gaps.sort(key=lambda c: c["ema"])
        return [{"name": g["name"], "ema": round(g["ema"], 4),
                 "attempts": g["attempts"],
                 "calibration_error": round(self._calibration_error(g), 4)} for g in gaps]

    def strongest_domains(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Returns top-N capabilities by EMA proficiency with confidence intervals."""
        with self._lock:
            all_caps = self._all()
        ranked = sorted(all_caps, key=lambda c: c["ema"], reverse=True)[:top_n]
        result = []
        for c in ranked:
            n = c["attempts"]
            p = c["ema"]
            margin = 1.96 * math.sqrt(p * (1 - p) / (n + 1e-9))
            result.append({
                "name": c["name"],
                "ema": round(p, 4),
                "ci_low": round(max(0.0, p - margin), 4),
                "ci_high": round(min(1.0, p + margin), 4),
                "attempts": n
            })
        return result

    def confidence_in(self, capability: str) -> float:
        """Returns calibrated confidence float matching actual success rate."""
        with self._lock:
            rec = self._get(capability)
        if rec is None:
            return 0.5
        actual = rec["successes"] / (rec["attempts"] + 1e-9)
        alpha = min(1.0, rec["attempts"] / 20.0)
        calibrated = alpha * actual + (1 - alpha) * rec["ema"]
        entropy = -(calibrated * math.log2(calibrated + 1e-12) +
                    (1 - calibrated) * math.log2(1 - calibrated + 1e-12))
        uncertainty_penalty = entropy / 8.0
        return round(max(0.0, min(1.0, calibrated - uncertainty_penalty)), 4)

    def growth_rate(self, capability: str) -> float:
        """Returns EMA growth per episode over last 30 episodes (positive = improving)."""
        with self._lock:
            rec = self._get(capability)
        if rec is None or rec["attempts"] < 2:
            return 0.0
        old_ema = self._ema_30_ago(capability, rec["attempts"])
        rate = (rec["ema"] - old_ema) / 30.0
        return round(rate, 6)

    def calibration_report(self) -> dict[str, Any]:
        """Returns mean calibration error and z-score anomalies across all capabilities."""
        with self._lock:
            all_caps = self._all()
        if not all_caps:
            return {"mean_cal_error": 0.0, "anomalies": []}
        errors = [self._calibration_error(c) for c in all_caps]
        mean_err = statistics.mean(errors)
        std_err = statistics.stdev(errors) if len(errors) > 1 else 1e-9
        anomalies = []
        for c, e in zip(all_caps, errors):
            z = (e - mean_err) / (std_err + 1e-9)
            if abs(z) > 2.0:
                anomalies.append({"name": c["name"], "cal_error": round(e, 4), "z": round(z, 3)})
        return {"mean_cal_error": round(mean_err, 4), "anomalies": anomalies,
                "total_capabilities": len(all_caps)}

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one introspection sweep; returns gaps found and goals generated."""
        gaps = self.known_gaps()
        strongest = self.strongest_domains(3)
        self._cycles += 1
        try:
            from working_memory import WorkingMemory
            WorkingMemory().store("self_model_gaps", gaps, importance=0.8)
        except Exception:
            pass
        try:
            from nova_omnisynthengine import NovaOmnisynthEngine
            NovaOmnisynthEngine().observe("self_model_cycle", {"gaps": len(gaps), "cycle": self._cycles})
        except Exception:
            pass
        return {"cycle": self._cycles, "gaps_found": len(gaps), "top_domains": [s["name"] for s in strongest]}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            all_caps = self._all()
        emas = [c["ema"] for c in all_caps] or [0.5]
        entropy = -sum((p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12))
                       for p in emas) / (len(emas) + 1e-9)
        return {
            "items": len(all_caps),
            "confidence": round(statistics.mean(emas), 4),
            "accuracy": round(sum(c["successes"] for c in all_caps) /
                              max(1, sum(c["attempts"] for c in all_caps)), 4),
            "entropy": round(entropy, 4),
            "cycles": self._cycles,
            "active": len([c for c in all_caps if c["attempts"] > 0]),
            "pending": len(self.known_gaps())
        }

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaSelfModelEngine() | result = obj.update_capability("causal_reasoning", 0.8, 0.9)