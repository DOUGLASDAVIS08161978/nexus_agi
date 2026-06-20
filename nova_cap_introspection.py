"""
nova_cap_introspection.py
Nova ASI — Introspection
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Self-Model Engine — live introspective capability tracker.
Maintains EMA proficiency, gap detection, growth rates, and calibration
so Nova can answer 'What am I good at?' and 'Where do I need to improve?'
Satisfies pillars: ①②③④⑥⑦⑧⑨⑪⑫⑬⑭
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

EMA_FAST = 0.10
EMA_SLOW = 0.90
GAP_THRESHOLD = 0.40
GAP_MIN_ATTEMPTS = 3
ANOMALY_Z = 2.5
AUTO_INTERVAL = 90          # seconds between autonomous cycles
HISTORY_WINDOW = 30         # episodes for growth_rate denominator


class SelfModelEngine:
    """Nova's live self-model: tracks capability proficiency, gaps, and growth autonomously."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = self._init_db()
        self._cycles: int = 0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ------------------------------------------------------------------ DB --
    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                name        TEXT PRIMARY KEY,
                ema         REAL DEFAULT 0.5,
                attempts    INTEGER DEFAULT 0,
                successes   INTEGER DEFAULT 0,
                first_seen  REAL,
                last_seen   REAL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cap_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                ema_snap    REAL,
                ts          REAL
            )""")
        conn.commit()
        return conn

    def _row(self, name: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT name,ema,attempts,successes,first_seen,last_seen FROM capabilities WHERE name=?",
            (name,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(["name","ema","attempts","successes","first_seen","last_seen"], row))

    # -------------------------------------------------------- public API ----
    def update_capability(self, name: str, proficiency: float, evidence: float) -> dict[str, Any]:
        """Returns updated EMA proficiency dict after absorbing new evidence."""
        now = time.time()
        clipped = max(0.0, min(1.0, evidence))
        with self._lock:
            row = self._row(name)
            if row is None:
                ema = EMA_FAST * clipped + EMA_SLOW * 0.5
                self._conn.execute(
                    "INSERT INTO capabilities VALUES (?,?,1,?,?,?)",
                    (name, ema, int(clipped >= 0.5), now, now))
            else:
                ema = EMA_FAST * clipped + EMA_SLOW * row["ema"]
                new_att = row["attempts"] + 1
                new_suc = row["successes"] + int(clipped >= 0.5)
                self._conn.execute(
                    "UPDATE capabilities SET ema=?,attempts=?,successes=?,last_seen=? WHERE name=?",
                    (ema, new_att, new_suc, now, name))
            self._conn.execute(
                "INSERT INTO cap_history (name,ema_snap,ts) VALUES (?,?,?)", (name, ema, now))
            self._conn.commit()
        # Feedback: log to MetacognitiveMonitor
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(name, "EMA_update", ema, clipped >= 0.5)
        except Exception:
            pass
        return {"name": name, "ema": round(ema, 4), "evidence": clipped}

    def known_gaps(self) -> list[dict[str, Any]]:
        """Returns list of capabilities with proficiency < 0.4 and attempts > 3."""
        cur = self._conn.execute(
            "SELECT name,ema,attempts,successes FROM capabilities WHERE ema<? AND attempts>?",
            (GAP_THRESHOLD, GAP_MIN_ATTEMPTS))
        gaps = []
        for row in cur.fetchall():
            name, ema, att, suc = row
            calibration_err = abs(ema - (suc / max(att, 1)))
            gaps.append({"name": name, "ema": round(ema, 4),
                         "attempts": att, "calibration_err": round(calibration_err, 4)})
        gaps.sort(key=lambda x: x["ema"])
        # Self-generate improvement goals
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            for g in gaps[:3]:
                hgp.add_goal(f"Improve capability: {g['name']} (ema={g['ema']})", priority=8)
        except Exception:
            pass
        return gaps

    def strongest_domains(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Returns top-N capabilities ranked by EMA proficiency descending."""
        cur = self._conn.execute(
            "SELECT name,ema,attempts,successes FROM capabilities ORDER BY ema DESC LIMIT ?",
            (top_n,))
        results = []
        for row in cur.fetchall():
            name, ema, att, suc = row
            success_rate = suc / max(att, 1)
            calibration = 1.0 - abs(ema - success_rate)
            results.append({"name": name, "ema": round(ema, 4),
                            "success_rate": round(success_rate, 4),
                            "calibration": round(calibration, 4)})
        return results

    def confidence_in(self, capability: str) -> float:
        """Returns calibrated confidence float [0,1] matching actual success rate via shrinkage."""
        row = self._row(capability)
        if row is None:
            return 0.5
        att = row["attempts"]
        suc = row["successes"]
        empirical = suc / max(att, 1)
        # Shrinkage toward EMA — weight by sqrt(attempts) for calibration
        weight = math.sqrt(att) / (math.sqrt(att) + 5.0)
        calibrated = weight * empirical + (1.0 - weight) * row["ema"]
        # Uncertainty: 95% Wilson CI half-width
        if att > 0:
            p = calibrated
            ci_half = 1.96 * math.sqrt(max(p * (1 - p) / att, 1e-9))
        else:
            ci_half = 0.5
        return round(max(0.0, min(1.0, calibrated)), 4)

    def growth_rate(self, capability: str) -> dict[str, Any]:
        """Returns per-episode growth rate and trend direction over last 30 episodes."""
        cur = self._conn.execute(
            "SELECT ema_snap FROM cap_history WHERE name=? ORDER BY ts DESC LIMIT ?",
            (capability, HISTORY_WINDOW + 1))
        snaps = [r[0] for r in cur.fetchall()]
        if len(snaps) < 2:
            return {"capability": capability, "growth_rate": 0.0, "trend": "insufficient_data"}
        recent = snaps[0]
        oldest = snaps[-1]
        n = len(snaps) - 1
        rate = (recent - oldest) / n
        # Z-score anomaly on snapshot series
        if len(snaps) >= 3:
            mu = statistics.mean(snaps)
            sigma = statistics.stdev(snaps) + 1e-9
            z = (recent - mu) / sigma
            anomaly = abs(z) > ANOMALY_Z
        else:
            z, anomaly = 0.0, False
        trend = "improving" if rate > 0.005 else ("declining" if rate < -0.005 else "stable")
        return {"capability": capability, "growth_rate": round(rate, 6),
                "trend": trend, "z_score": round(z, 3), "anomaly": anomaly}

    def calibration_report(self) -> dict[str, Any]:
        """Returns system-wide calibration error and entropy across all capabilities."""
        cur = self._conn.execute("SELECT ema,attempts,successes FROM capabilities")
        rows = cur.fetchall()
        if not rows:
            return {"calibration_mae": 0.0, "entropy": 0.0, "n": 0}
        errors, emas = [], []
        for ema, att, suc in rows:
            sr = suc / max(att, 1)
            errors.append(abs(ema - sr))
            emas.append(ema)
        mae = statistics.mean(errors)
        # Entropy over EMA distribution (binned to 10 buckets)
        bins = [0.0] * 10
        for e in emas:
            bins[min(int(e * 10), 9)] += 1
        total = sum(bins) + 1e-12
        dist = [b / total for b in bins]
        entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
        return {"calibration_mae": round(mae, 4), "entropy": round(entropy, 4), "n": len(rows)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        cur = self._conn.execute("SELECT COUNT(*),AVG(ema),SUM(attempts),SUM(successes) FROM capabilities")
        row = cur.fetchone()
        n, avg_ema, total_att, total_suc = row if row[0] else (0, 0.5, 0, 0)
        gaps = self._conn.execute(
            "SELECT COUNT(*) FROM capabilities WHERE ema<? AND attempts>?",
            (GAP_THRESHOLD, GAP_MIN_ATTEMPTS)).fetchone()[0]
        cal = self.calibration_report()
        return {
            "items": n or 0,
            "confidence": round(avg_ema or 0.5, 4),
            "accuracy": round((total_suc or 0) / max(total_att or 1, 1), 4),
            "active": gaps,
            "cycles": self._cycles,
            "entropy": cal["entropy"],
            "calibration_mae": cal["calibration_mae"],
        }

    # ------------------------------------------------- autonomous loop ----
    def _auto_loop(self) -> None:
        """Background daemon: runs calibration and gap checks every AUTO_INTERVAL seconds."""
        while True:
            time.sleep(AUTO_INTERVAL)
            try:
                with self._lock:
                    self._cycles += 1
                gaps = self.known_gaps()
                strongest = self.strongest_domains(3)
                try:
                    from bayesian_belief_system import BayesianBeliefSystem
                    bbs = BayesianBeliefSystem()
                    for g in gaps[:2]:
                        bbs.add_causal_edge("low_proficiency", g["name"], weight=0.7)
                except Exception:
                    pass
                try:
                    from metacognitive_monitor import MetacognitiveMonitor
                    mm = MetacognitiveMonitor()
                    s = self.status()
                    mm.log_reasoning("SelfModelEngine", "auto_cycle",
                                     s["confidence"], s["accuracy"] > 0.5)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = SelfModelEngine() | result = obj.update_capability("reasoning", 0.8, 0.9)