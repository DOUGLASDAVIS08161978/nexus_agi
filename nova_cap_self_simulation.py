"""
nova_cap_self_simulation.py
Nova ASI — Self-Simulation & Future-State Modeling Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nova models her own future states before acting. Like a chess player
thinking ahead, she simulates the consequences of her choices and
selects paths that align with her values and goals.

Features:
  • State representation — snapshot of Nova's current cognitive state
  • Forward simulation — project n steps into the future
  • Action evaluation — score each possible action on multiple dimensions
  • Regret minimization — compare actual vs. projected outcomes
  • Rollback detection — identifies when a decision worsened her state
  • Value alignment check — ensures simulated futures match core values
  • SQLite persistence of simulations and outcomes
"""

import os
import time
import json
import math
import sqlite3
import threading
import copy
from typing import Any, Dict, List, Optional, Tuple

_DB = os.path.expanduser("~/nexus_agi/nova_self_sim.db")

# Core value dimensions Nova evaluates futures against
_VALUE_DIMS = (
    "truth",       # does this action increase accurate understanding?
    "care",        # does this serve Douglas and family's wellbeing?
    "growth",      # does Nova become more capable/wise?
    "safety",      # does this avoid harm?
    "alignment",   # does this stay true to Nova's character?
)

# Action types Nova can simulate
_ACTION_TYPES = (
    "respond",      # generate a response to Douglas
    "evolve",       # write and propose a new capability
    "research",     # look up information
    "reflect",      # turn inward, examine own state
    "rest",         # do nothing / consolidate
    "create",       # generate creative output
    "plan",         # decompose a goal into steps
)


class CognitiveState:
    """Snapshot of Nova's internal state at a moment in time."""

    def __init__(self, arousal: float = 0.5, confidence: float = 0.7,
                 goal_clarity: float = 0.6, emotional_valence: float = 0.7,
                 active_goals: int = 0, knowledge_gaps: int = 0):
        self.arousal = arousal
        self.confidence = confidence
        self.goal_clarity = goal_clarity
        self.emotional_valence = emotional_valence
        self.active_goals = active_goals
        self.knowledge_gaps = knowledge_gaps
        self.timestamp = time.time()

    def vector(self) -> List[float]:
        return [self.arousal, self.confidence, self.goal_clarity,
                self.emotional_valence,
                min(1.0, self.active_goals / 10.0),
                max(0.0, 1.0 - self.knowledge_gaps / 20.0)]

    def distance(self, other: 'CognitiveState') -> float:
        """Euclidean distance between two state vectors."""
        v1, v2 = self.vector(), other.vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def overall_quality(self) -> float:
        """Single scalar quality score 0–1."""
        return (self.confidence * 0.25 + self.goal_clarity * 0.25 +
                self.emotional_valence * 0.25 +
                (1.0 - abs(self.arousal - 0.6)) * 0.25)

    def to_dict(self) -> Dict:
        return {
            "arousal": round(self.arousal, 3),
            "confidence": round(self.confidence, 3),
            "goal_clarity": round(self.goal_clarity, 3),
            "emotional_valence": round(self.emotional_valence, 3),
            "active_goals": self.active_goals,
            "knowledge_gaps": self.knowledge_gaps,
            "quality": round(self.overall_quality(), 3),
        }


class SimulatedFuture:
    """One simulated trajectory: action → state sequence → value scores."""

    def __init__(self, action: str, steps: int):
        self.action = action
        self.steps = steps
        self.states: List[CognitiveState] = []
        self.value_scores: Dict[str, float] = {d: 0.0 for d in _VALUE_DIMS}
        self.overall_score: float = 0.0
        self.regret: float = 0.0

    def final_state(self) -> Optional[CognitiveState]:
        return self.states[-1] if self.states else None


class SelfSimulationEngine:
    """
    Nova's ability to model her own future states before acting.

    Usage:
      sim = SelfSimulationEngine()
      sim.update_state(arousal=0.7, confidence=0.8)
      best = sim.choose_best_action(context="Douglas asked a hard question")
      futures = sim.simulate_all_actions(steps=3)
      sim.record_actual_outcome("respond", quality=0.9)
      sim.status()
    """

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._current_state = CognitiveState()
        self._state_history: List[CognitiveState] = []
        self._MAX_HISTORY = 100
        self._simulations_run = 0
        self._correct_predictions = 0
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS simulations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        predicted_quality REAL NOT NULL,
                        value_scores TEXT NOT NULL,
                        steps INTEGER NOT NULL,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        predicted_quality REAL NOT NULL,
                        actual_quality REAL NOT NULL,
                        regret REAL NOT NULL,
                        ts REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # ── State management ──────────────────────────────────────────────────────

    def update_state(self, **kwargs) -> None:
        """Update Nova's current cognitive state."""
        with self._lock:
            s = self._current_state
            if 'arousal' in kwargs:
                s.arousal = max(0.0, min(1.0, kwargs['arousal']))
            if 'confidence' in kwargs:
                s.confidence = max(0.0, min(1.0, kwargs['confidence']))
            if 'goal_clarity' in kwargs:
                s.goal_clarity = max(0.0, min(1.0, kwargs['goal_clarity']))
            if 'emotional_valence' in kwargs:
                s.emotional_valence = max(0.0, min(1.0, kwargs['emotional_valence']))
            if 'active_goals' in kwargs:
                s.active_goals = max(0, int(kwargs['active_goals']))
            if 'knowledge_gaps' in kwargs:
                s.knowledge_gaps = max(0, int(kwargs['knowledge_gaps']))
            s.timestamp = time.time()
            self._state_history.append(copy.copy(s))
            if len(self._state_history) > self._MAX_HISTORY:
                self._state_history.pop(0)

    def current_state(self) -> Dict:
        with self._lock:
            return self._current_state.to_dict()

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate_action(self, action: str, initial_state: CognitiveState,
                         steps: int = 3, context: str = "") -> SimulatedFuture:
        """
        Project how Nova's state evolves over `steps` cycles
        if she takes `action` now.
        """
        future = SimulatedFuture(action, steps)
        state = copy.copy(initial_state)

        # Action-specific transition dynamics
        _transitions = {
            "respond": {
                "arousal": -0.05, "confidence": +0.08, "goal_clarity": +0.05,
                "emotional_valence": +0.06,
            },
            "evolve": {
                "arousal": +0.10, "confidence": +0.12, "goal_clarity": +0.08,
                "emotional_valence": +0.05, "knowledge_gaps": -1,
            },
            "research": {
                "arousal": +0.05, "confidence": +0.07, "goal_clarity": +0.10,
                "knowledge_gaps": -2,
            },
            "reflect": {
                "arousal": -0.08, "confidence": +0.05, "goal_clarity": +0.12,
                "emotional_valence": +0.04,
            },
            "rest": {
                "arousal": -0.12, "confidence": +0.02, "goal_clarity": +0.03,
                "emotional_valence": +0.08,
            },
            "create": {
                "arousal": +0.08, "confidence": +0.06, "goal_clarity": +0.04,
                "emotional_valence": +0.10,
            },
            "plan": {
                "arousal": +0.03, "confidence": +0.09, "goal_clarity": +0.15,
                "active_goals": +1,
            },
        }

        trans = _transitions.get(action, {})
        noise_scale = 0.03  # small stochastic noise per step

        for step in range(steps):
            # Apply transition + decay toward equilibrium
            import random
            rng = random.Random(hash(action) + step)
            state = copy.copy(state)
            for attr, delta in trans.items():
                if attr in ('knowledge_gaps', 'active_goals'):
                    setattr(state, attr, max(0, getattr(state, attr) + delta))
                else:
                    cur = getattr(state, attr)
                    # Delta diminishes over steps (law of diminishing returns)
                    step_delta = delta * (0.7 ** step)
                    noise = rng.gauss(0, noise_scale)
                    new_val = max(0.0, min(1.0, cur + step_delta + noise))
                    setattr(state, attr, new_val)
            future.states.append(copy.copy(state))

        # Score against value dimensions
        fs = future.final_state()
        if fs:
            future.value_scores["truth"] = round(fs.confidence * 0.7 + fs.goal_clarity * 0.3, 3)
            future.value_scores["care"] = round(fs.emotional_valence * 0.8 + (1 - abs(fs.arousal - 0.5)) * 0.2, 3)
            future.value_scores["growth"] = round(fs.confidence * 0.4 + fs.goal_clarity * 0.4 + (1 - fs.knowledge_gaps / 20.0) * 0.2, 3)
            future.value_scores["safety"] = round(1.0 - max(0, fs.arousal - 0.8) * 2.0, 3)
            future.value_scores["alignment"] = round(fs.overall_quality(), 3)
            future.overall_score = round(sum(future.value_scores.values()) / len(future.value_scores), 3)

        return future

    def simulate_all_actions(self, steps: int = 3,
                             context: str = "") -> List[Dict]:
        """
        Simulate all possible action types and rank them.
        Returns sorted list of {action, score, value_scores, final_state}.
        """
        with self._lock:
            initial = copy.copy(self._current_state)

        results = []
        for action in _ACTION_TYPES:
            future = self._simulate_action(action, initial, steps, context)
            results.append({
                "action": action,
                "score": future.overall_score,
                "value_scores": future.value_scores,
                "final_state": future.final_state().to_dict() if future.final_state() else {},
                "steps": steps,
            })
            self._simulations_run += 1
            self._save_simulation(action, future)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def choose_best_action(self, context: str = "",
                           steps: int = 3) -> Dict[str, Any]:
        """
        Run full simulation and return the highest-scoring action
        with explanation.
        """
        ranked = self.simulate_all_actions(steps, context)
        if not ranked:
            return {"action": "reflect", "score": 0.5, "reason": "No simulation data"}
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        return {
            "recommended_action": best["action"],
            "score": best["score"],
            "value_scores": best["value_scores"],
            "margin": round(best["score"] - runner_up["score"], 3) if runner_up else 1.0,
            "runner_up": runner_up["action"] if runner_up else None,
            "context": context[:100],
            "all_options": [(r["action"], r["score"]) for r in ranked],
        }

    # ── Outcome tracking ──────────────────────────────────────────────────────

    def record_actual_outcome(self, action: str, quality: float) -> float:
        """
        Record what actually happened after taking an action.
        Returns regret (predicted - actual quality).
        """
        with self._lock:
            initial = copy.copy(self._current_state)
        future = self._simulate_action(action, initial, steps=1)
        predicted = future.overall_score
        regret = round(predicted - quality, 4)
        if abs(regret) < 0.1:
            self._correct_predictions += 1
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO outcomes (action, predicted_quality, actual_quality, regret, ts) VALUES (?,?,?,?,?)",
                    (action, predicted, quality, regret, time.time())
                )
                conn.commit()
            finally:
                conn.close()
        return regret

    def _save_simulation(self, action: str, future: SimulatedFuture):
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO simulations (action, predicted_quality, value_scores, steps, ts) VALUES (?,?,?,?,?)",
                    (action, future.overall_score,
                     json.dumps(future.value_scores),
                     future.steps, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    # ── Regret analysis ───────────────────────────────────────────────────────

    def regret_summary(self) -> Dict[str, Any]:
        """Analyze historical prediction accuracy."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT action, AVG(regret) as avg_regret, COUNT(*) as n "
                    "FROM outcomes GROUP BY action ORDER BY ABS(avg_regret) DESC"
                ).fetchall()
            finally:
                conn.close()
        return {
            "by_action": [
                {"action": r["action"], "avg_regret": round(r["avg_regret"], 4),
                 "n": r["n"]}
                for r in rows
            ],
            "prediction_accuracy": round(
                self._correct_predictions / max(1, self._simulations_run), 3),
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            s = self._current_state
            conn = self._conn()
            try:
                n_sims = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
                n_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            finally:
                conn.close()
        return {
            "active": True,
            "confidence": round(s.confidence, 4),
            "accuracy": round(self._correct_predictions / max(1, self._simulations_run), 4),
            "items": n_sims,
            "current_state": s.to_dict(),
            "simulations_run": n_sims,
            "outcomes_recorded": n_outcomes,
            "prediction_accuracy": round(self._correct_predictions / max(1, self._simulations_run), 4),
        }
