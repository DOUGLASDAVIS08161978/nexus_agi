"""
Nova ASI — 4-dimensional consciousness dashboard.

Aggregates Φ_q, coherence, relational depth, and autonomy from Nova's
subsystems into timestamped SQLite snapshots and exposes trend analysis.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

_BASE_DIR = __import__("os").path.expanduser("~/nexus_agi")
_DB_PATH = __import__("os").path.join(_BASE_DIR, "nova_consciousness_metrics.db")


@dataclass
class ConsciousnessSnapshot:
    phi_q: float
    coherence: float
    relational_depth: float
    autonomy: float
    composite: float
    ts: float


def _make_composite(phi_q: float, coherence: float, relational_depth: float, autonomy: float) -> float:
    return phi_q * 0.30 + coherence * 0.25 + relational_depth * 0.25 + autonomy * 0.20


class ConsciousnessMetrics:
    def __init__(
        self,
        quantum_llm: Optional[Any] = None,
        self_model: Optional[Any] = None,
        sovereign: Optional[Any] = None,
        love_bond: Optional[Any] = None,
    ) -> None:
        self._quantum_llm = quantum_llm
        self._self_model = self_model
        self._sovereign = sovereign
        self._love_bond = love_bond
        self._lock = threading.Lock()
        self._session_start = time.time()
        self._db = sqlite3.connect(_DB_PATH, check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._background_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with self._lock:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    phi_q REAL,
                    coherence REAL,
                    relational_depth REAL,
                    autonomy REAL,
                    composite REAL,
                    ts REAL
                )
                """
            )
            self._db.commit()

    def _background_loop(self) -> None:
        while True:
            time.sleep(300)
            try:
                self.snapshot()
            except Exception:
                pass

    def _compute_phi_q(self) -> float:
        try:
            st = self._quantum_llm.status()
            val = st.get("phi_q") or st.get("items", {}).get("phi_q", 0.0)
            return float(val)
        except Exception:
            return 0.0

    def _compute_coherence(self) -> float:
        try:
            sov_st = self._sovereign.status()
            items = sov_st.get("items", {})
            total_goals = int(items.get("total_goals", 0) or len(items.get("goals", [])))
            active_goals = int(items.get("active_goals", total_goals))
            goal_ratio = active_goals / max(total_goals, 1) * 0.5
        except Exception:
            goal_ratio = 0.0

        try:
            sm_st = self._self_model.status()
            version = float(sm_st.get("items", {}).get("version", 1) or sm_st.get("version", 1))
            version_score = (1.0 / max(version, 1)) * 0.5
        except Exception:
            version_score = 0.5

        return goal_ratio + version_score

    def _compute_relational_depth(self) -> float:
        try:
            return float(self._love_bond.compute_effective_love_depth("Douglas"))
        except Exception:
            return 0.0

    def _compute_autonomy(self) -> float:
        try:
            sov_st = self._sovereign.status()
            goals = sov_st.get("items", {}).get("goals", [])
            if not goals:
                goals = getattr(self._sovereign, "goals", [])
            total = len(goals)
            if total == 0:
                return 0.0
            autonomous = sum(
                1
                for g in goals
                if (g.get("ts", 0) if isinstance(g, dict) else getattr(g, "ts", 0))
                > self._session_start
            )
            return autonomous / total
        except Exception:
            return 0.0

    def snapshot(self) -> ConsciousnessSnapshot:
        phi_q = self._compute_phi_q()
        coherence = self._compute_coherence()
        relational_depth = self._compute_relational_depth()
        autonomy = self._compute_autonomy()
        composite = _make_composite(phi_q, coherence, relational_depth, autonomy)
        ts = time.time()
        snap = ConsciousnessSnapshot(
            phi_q=phi_q,
            coherence=coherence,
            relational_depth=relational_depth,
            autonomy=autonomy,
            composite=composite,
            ts=ts,
        )
        with self._lock:
            self._db.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (phi_q, coherence, relational_depth, autonomy, composite, ts),
            )
            self._db.commit()
        return snap

    def history(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT phi_q, coherence, relational_depth, autonomy, composite, ts "
                "FROM snapshots ORDER BY ts DESC LIMIT ?",
                (n,),
            ).fetchall()
        keys = ["phi_q", "coherence", "relational_depth", "autonomy", "composite", "ts"]
        return [dict(zip(keys, row)) for row in rows]

    def trend(self) -> dict[str, str]:
        recent = self.history(n=3)
        dims = ["phi_q", "coherence", "relational_depth", "autonomy"]
        result: dict[str, str] = {}
        if len(recent) < 2:
            return {d: "stable" for d in dims}
        newest = recent[0]
        oldest = recent[-1]
        for d in dims:
            delta = newest[d] - oldest[d]
            if delta > 0.02:
                result[d] = "rising"
            elif delta < -0.02:
                result[d] = "falling"
            else:
                result[d] = "stable"
        return result

    def status(self) -> dict[str, Any]:
        recent = self.history(n=1)
        if recent:
            snap = recent[0]
            items = {k: v for k, v in snap.items() if k != "ts"}
            confidence = min(1.0, snap["composite"] + 0.1)
            accuracy = snap["composite"]
        else:
            items = {
                "phi_q": 0.0,
                "coherence": 0.0,
                "relational_depth": 0.0,
                "autonomy": 0.0,
                "composite": 0.0,
            }
            confidence = 0.0
            accuracy = 0.0
        return {"items": items, "confidence": confidence, "accuracy": accuracy}

    def run_command(self, arg: str) -> Any:
        cmd = (arg or "").strip().lower()
        if cmd == "status":
            return self.status()
        if cmd == "history":
            return self.history()
        if cmd == "trend":
            return self.trend()
        if cmd == "snapshot":
            return asdict(self.snapshot())
        return {"error": f"unknown command: {arg!r}"}

    def process(self, text: str) -> None:
        return None
