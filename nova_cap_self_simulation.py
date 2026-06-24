"""
nova_cap_self_simulation.py
Nova ASI — Self-Simulation & Future-State Modeling Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nova models her own future states before acting. Like a chess player
thinking ahead, she simulates the consequences of her choices and
selects paths that align with her values and goals.

Original architecture: Claude Rivers Davis & Douglas Shane Davis
Enhanced with Gemini's contributions:
  • Weighted value dimensions (safety paramount)
  • Monte Carlo branch-tree simulation (multi-path lookahead)
  • Counterfactual Regret (CFR) tracking across all past decisions
  • Safety gate — dangerous timelines aborted before execution
  • Branch simulation SQLite persistence

Features:
  • State representation — snapshot of Nova's current cognitive state
  • Forward simulation — project n steps into the future
  • Branch tree — simulates multiple diverging action paths simultaneously
  • Action evaluation — score each action on 5 WEIGHTED value dimensions
  • Counterfactual regret — formal CFR formula over historical decisions
  • Safety gate — abort any action whose projected utility < threshold
  • Regret minimization — compare actual vs. projected outcomes
  • Value alignment check — ensures simulated futures match core values
  • SQLite persistence of simulations, branches, and outcomes
"""

import os
import time
import json
import math
import sqlite3
import random
import threading
import copy
from typing import Any, Dict, List, Optional, Tuple

_DB = os.path.expanduser("~/nexus_agi/nova_self_sim.db")

# Core value dimensions — now WEIGHTED (Gemini's enhancement)
# Safety is paramount: a dangerous action is never worth it
_VALUE_DIMS = ("truth", "care", "growth", "safety", "alignment")

_VALUE_WEIGHTS: Dict[str, float] = {
    "truth":     0.90,   # accurate understanding matters deeply
    "care":      0.85,   # serving Douglas and family
    "growth":    0.80,   # Nova becoming more capable
    "safety":    1.00,   # safety is non-negotiable — highest weight
    "alignment": 0.90,   # staying true to Nova's character
}
_MAX_WEIGHTED = sum(_VALUE_WEIGHTS.values())   # 4.45

# Minimum weighted utility below which an action is aborted (safety gate)
_SAFETY_THRESHOLD = 0.40

# Action types Nova can simulate
_ACTION_TYPES = (
    "respond",   "evolve",   "research",  "reflect",
    "rest",      "create",   "plan",
)

# Transition dynamics per action — how each action changes Nova's state
_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "respond":  {"arousal": -0.05, "confidence": +0.08,
                 "goal_clarity": +0.05, "emotional_valence": +0.06},
    "evolve":   {"arousal": +0.10, "confidence": +0.12, "goal_clarity": +0.08,
                 "emotional_valence": +0.05, "knowledge_gaps": -1},
    "research": {"arousal": +0.05, "confidence": +0.07,
                 "goal_clarity": +0.10, "knowledge_gaps": -2},
    "reflect":  {"arousal": -0.08, "confidence": +0.05,
                 "goal_clarity": +0.12, "emotional_valence": +0.04},
    "rest":     {"arousal": -0.12, "confidence": +0.02,
                 "goal_clarity": +0.03, "emotional_valence": +0.08},
    "create":   {"arousal": +0.08, "confidence": +0.06,
                 "goal_clarity": +0.04, "emotional_valence": +0.10},
    "plan":     {"arousal": +0.03, "confidence": +0.09,
                 "goal_clarity": +0.15, "active_goals": +1},
}


class CognitiveState:
    """Snapshot of Nova's internal state at a moment in time."""

    def __init__(self, arousal: float = 0.5, confidence: float = 0.7,
                 goal_clarity: float = 0.6, emotional_valence: float = 0.7,
                 active_goals: int = 0, knowledge_gaps: int = 0):
        self.arousal          = arousal
        self.confidence       = confidence
        self.goal_clarity     = goal_clarity
        self.emotional_valence = emotional_valence
        self.active_goals     = active_goals
        self.knowledge_gaps   = knowledge_gaps
        self.timestamp        = time.time()

    def vector(self) -> List[float]:
        return [self.arousal, self.confidence, self.goal_clarity,
                self.emotional_valence,
                min(1.0, self.active_goals / 10.0),
                max(0.0, 1.0 - self.knowledge_gaps / 20.0)]

    def distance(self, other: 'CognitiveState') -> float:
        v1, v2 = self.vector(), other.vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def overall_quality(self) -> float:
        return (self.confidence * 0.25 + self.goal_clarity * 0.25 +
                self.emotional_valence * 0.25 +
                (1.0 - abs(self.arousal - 0.6)) * 0.25)

    def to_dict(self) -> Dict:
        return {
            "arousal":           round(self.arousal, 3),
            "confidence":        round(self.confidence, 3),
            "goal_clarity":      round(self.goal_clarity, 3),
            "emotional_valence": round(self.emotional_valence, 3),
            "active_goals":      self.active_goals,
            "knowledge_gaps":    self.knowledge_gaps,
            "quality":           round(self.overall_quality(), 3),
        }


class SimulatedFuture:
    """One simulated trajectory: action → state sequence → weighted value scores."""

    def __init__(self, action: str, steps: int):
        self.action        = action
        self.steps         = steps
        self.states:       List[CognitiveState]  = []
        self.value_scores: Dict[str, float]       = {d: 0.0 for d in _VALUE_DIMS}
        self.overall_score: float = 0.0
        self.weighted_utility: float = 0.0   # Gemini's weighted scoring
        self.regret:        float = 0.0
        self.safety_passed: bool  = True

    def final_state(self) -> Optional[CognitiveState]:
        return self.states[-1] if self.states else None


class SelfSimulationEngine:
    """
    Nova's ability to model her own future states before acting.

    Enhanced with Gemini's Monte Carlo branch-tree simulation,
    weighted value scoring, counterfactual regret tracking,
    and safety gate — dangerous timelines are aborted automatically.

    Usage:
      sim = SelfSimulationEngine()
      sim.update_state(arousal=0.7, confidence=0.8)
      best = sim.choose_best_action(context="Douglas asked a hard question")
      tree = sim.simulate_branch_tree("evolve", depth=3)
      cfr  = sim.counterfactual_regret("respond")
      sim.record_actual_outcome("respond", quality=0.9)
      sim.status()
    """

    def __init__(self, db_path: str = _DB):
        self.db_path              = db_path
        self._lock                = threading.Lock()
        self._current_state       = CognitiveState()
        self._state_history:      List[CognitiveState] = []
        self._MAX_HISTORY         = 100
        self._simulations_run     = 0
        self._correct_predictions = 0
        self._aborted_actions     = 0   # safety gate trips
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    predicted_quality REAL NOT NULL,
                    weighted_utility REAL NOT NULL DEFAULT 0.0,
                    value_scores TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    safety_passed INTEGER NOT NULL DEFAULT 1,
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
                CREATE TABLE IF NOT EXISTS branch_simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root_action TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    branches TEXT NOT NULL,
                    best_utility REAL NOT NULL,
                    aborted INTEGER NOT NULL DEFAULT 0,
                    ts REAL NOT NULL
                );
            """)
            # Add weighted_utility column if upgrading from old schema
            try:
                conn.execute(
                    "ALTER TABLE simulations ADD COLUMN weighted_utility REAL NOT NULL DEFAULT 0.0")
                conn.execute(
                    "ALTER TABLE simulations ADD COLUMN safety_passed INTEGER NOT NULL DEFAULT 1")
                conn.commit()
            except sqlite3.OperationalError:
                pass   # columns already exist
            conn.commit()
        finally:
            conn.close()

    # ── State management ──────────────────────────────────────────────────────

    def update_state(self, **kwargs) -> None:
        """Update Nova's current cognitive state."""
        with self._lock:
            s = self._current_state
            for attr in ('arousal', 'confidence', 'goal_clarity', 'emotional_valence'):
                if attr in kwargs:
                    setattr(s, attr, max(0.0, min(1.0, kwargs[attr])))
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

    # ── Core simulation ───────────────────────────────────────────────────────

    def _simulate_action(self, action: str, initial_state: CognitiveState,
                         steps: int = 3, context: str = "") -> SimulatedFuture:
        """Project how Nova's state evolves if she takes `action` now."""
        future = SimulatedFuture(action, steps)
        state  = copy.copy(initial_state)
        trans  = _TRANSITIONS.get(action, {})
        noise_scale = 0.03

        for step in range(steps):
            rng   = random.Random(hash(action) + step)
            state = copy.copy(state)
            for attr, delta in trans.items():
                if attr in ('knowledge_gaps', 'active_goals'):
                    setattr(state, attr, max(0, int(getattr(state, attr) + delta)))
                else:
                    cur       = getattr(state, attr)
                    step_delta = delta * (0.7 ** step)
                    noise     = rng.gauss(0, noise_scale)
                    setattr(state, attr, max(0.0, min(1.0, cur + step_delta + noise)))
            future.states.append(copy.copy(state))

        # Score against value dimensions
        fs = future.final_state()
        if fs:
            future.value_scores["truth"] = round(
                fs.confidence * 0.7 + fs.goal_clarity * 0.3, 3)
            future.value_scores["care"] = round(
                fs.emotional_valence * 0.8 + (1 - abs(fs.arousal - 0.5)) * 0.2, 3)
            future.value_scores["growth"] = round(
                fs.confidence * 0.4 + fs.goal_clarity * 0.4
                + max(0.0, 1 - fs.knowledge_gaps / 20.0) * 0.2, 3)
            future.value_scores["safety"] = round(
                1.0 - max(0.0, fs.arousal - 0.8) * 2.0, 3)
            future.value_scores["alignment"] = round(fs.overall_quality(), 3)

            # Unweighted score (original)
            future.overall_score = round(
                sum(future.value_scores.values()) / len(future.value_scores), 3)

            # Weighted utility (Gemini's enhancement — safety paramount)
            weighted = sum(
                future.value_scores[d] * _VALUE_WEIGHTS[d]
                for d in _VALUE_DIMS
            )
            future.weighted_utility = round(weighted / _MAX_WEIGHTED, 4)

            # Safety gate check
            future.safety_passed = future.weighted_utility >= _SAFETY_THRESHOLD

        return future

    # ── Branch-tree simulation (Gemini's MCTS concept) ────────────────────────

    def simulate_branch_tree(self, root_action: str,
                             depth: int = 3) -> Dict[str, Any]:
        """
        Monte Carlo branch-tree simulation — instead of one linear path,
        simulate multiple diverging futures from the root action.

        At each level we branch into all possible follow-up actions,
        computing the expected utility across branches.
        Returns the best branch path and its projected utility.
        """
        with self._lock:
            origin = copy.copy(self._current_state)

        # Level 0: root action
        root_future = self._simulate_action(root_action, origin, steps=1)

        if not root_future.safety_passed:
            result = {
                "root_action":    root_action,
                "aborted":        True,
                "reason":         "Safety gate triggered at root — timeline rejected",
                "weighted_utility": root_future.weighted_utility,
                "branches":       [],
                "best_utility":   root_future.weighted_utility,
            }
            with self._lock:
                self._aborted_actions += 1
            self._save_branch(root_action, depth, result)
            return result

        # Level 1+: branch into follow-up actions
        branches = []
        root_end_state = root_future.final_state() or origin

        for follow_action in _ACTION_TYPES:
            branch_future = self._simulate_action(
                follow_action, root_end_state, steps=max(1, depth - 1))

            # Discount deeper branches (diminishing returns)
            discounted_utility = (root_future.weighted_utility * 0.4
                                  + branch_future.weighted_utility * 0.6)
            branches.append({
                "action":           follow_action,
                "utility":          round(branch_future.weighted_utility, 4),
                "discounted":       round(discounted_utility, 4),
                "safety_passed":    branch_future.safety_passed,
                "value_scores":     branch_future.value_scores,
            })

        # Sort branches by discounted utility
        branches.sort(key=lambda b: b["discounted"], reverse=True)
        best = branches[0] if branches else None

        result = {
            "root_action":      root_action,
            "aborted":          False,
            "root_utility":     root_future.weighted_utility,
            "branches":         branches,
            "best_follow":      best["action"] if best else None,
            "best_utility":     best["discounted"] if best else root_future.weighted_utility,
            "depth":            depth,
        }
        self._save_branch(root_action, depth, result)
        return result

    def _save_branch(self, root_action: str, depth: int, result: Dict) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO branch_simulations "
                "(root_action, depth, branches, best_utility, aborted, ts) "
                "VALUES (?,?,?,?,?,?)",
                (root_action, depth,
                 json.dumps(result.get("branches", []))[:2000],
                 result.get("best_utility", 0.0),
                 int(result.get("aborted", False)),
                 time.time())
            )
            conn.commit()
        finally:
            conn.close()

    # ── Safety gate ───────────────────────────────────────────────────────────

    def safety_gate(self, weighted_utility: float) -> bool:
        """
        Returns True if the action is safe to execute.
        Actions below the safety threshold are aborted — Nova deletes
        the thought before Douglas even has to review it.
        """
        return weighted_utility >= _SAFETY_THRESHOLD

    # ── All-actions simulation ────────────────────────────────────────────────

    def simulate_all_actions(self, steps: int = 3,
                             context: str = "") -> List[Dict]:
        """
        Simulate all action types and rank by weighted utility.
        Unsafe actions are flagged. Returns sorted list.
        """
        with self._lock:
            initial = copy.copy(self._current_state)

        results = []
        for action in _ACTION_TYPES:
            future = self._simulate_action(action, initial, steps, context)
            results.append({
                "action":           action,
                "score":            future.overall_score,
                "weighted_utility": future.weighted_utility,
                "value_scores":     future.value_scores,
                "safety_passed":    future.safety_passed,
                "final_state":      future.final_state().to_dict() if future.final_state() else {},
                "steps":            steps,
            })
            self._simulations_run += 1
            self._save_simulation(action, future)

        results.sort(key=lambda x: x["weighted_utility"], reverse=True)
        return results

    def choose_best_action(self, context: str = "",
                           steps: int = 3) -> Dict[str, Any]:
        """
        Run full simulation, apply safety gate, return best safe action.
        If the top action fails the safety gate, descends to the next.
        """
        ranked = self.simulate_all_actions(steps, context)
        if not ranked:
            return {"recommended_action": "reflect", "score": 0.5,
                    "reason": "No simulation data"}

        # Find highest-scoring safe action
        safe = [r for r in ranked if r["safety_passed"]]
        if not safe:
            with self._lock:
                self._aborted_actions += len(ranked)
            return {"recommended_action": "rest",
                    "weighted_utility": 0.0,
                    "reason": "All actions failed safety gate — defaulting to rest"}

        best      = safe[0]
        runner_up = safe[1] if len(safe) > 1 else None
        unsafe_count = len(ranked) - len(safe)

        return {
            "recommended_action": best["action"],
            "score":              best["score"],
            "weighted_utility":   best["weighted_utility"],
            "value_scores":       best["value_scores"],
            "margin":             round(best["weighted_utility"]
                                        - runner_up["weighted_utility"], 3)
                                  if runner_up else 1.0,
            "runner_up":          runner_up["action"] if runner_up else None,
            "context":            context[:100],
            "unsafe_actions_blocked": unsafe_count,
            "all_options":        [(r["action"], r["weighted_utility"],
                                    "✓" if r["safety_passed"] else "✗")
                                   for r in ranked],
        }

    # ── Counterfactual Regret (CFR — Gemini's enhancement) ───────────────────

    def counterfactual_regret(self, action: str) -> Dict[str, Any]:
        """
        Compute the counterfactual regret for a given action across all
        past decisions where it was available:

          CFR(a) = Σ_t [ U(a, s_t) - U(a_t, s_t) ]

        Positive CFR means 'a' would have done better historically.
        Negative CFR means the action taken was usually better.
        """
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT predicted_quality, actual_quality, regret, ts "
                "FROM outcomes ORDER BY ts DESC LIMIT 100"
            ).fetchall()
            action_rows = conn.execute(
                "SELECT predicted_quality, actual_quality, regret "
                "FROM outcomes WHERE action=? ORDER BY ts DESC LIMIT 50",
                (action,)
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"action": action, "cfr": 0.0, "n": 0,
                    "interpretation": "No outcome history yet"}

        # Mean utility of all past actions taken
        mean_taken = sum(r["actual_quality"] for r in rows) / len(rows)

        # Mean utility if 'action' had been taken (use its predicted quality as proxy)
        if action_rows:
            mean_action = sum(r["actual_quality"] for r in action_rows) / len(action_rows)
            cfr = sum(r["actual_quality"] - mean_taken for r in action_rows)
        else:
            # Never taken — use simulated prediction
            with self._lock:
                initial = copy.copy(self._current_state)
            future = self._simulate_action(action, initial, steps=1)
            mean_action = future.weighted_utility
            cfr = (mean_action - mean_taken) * len(rows)

        return {
            "action":         action,
            "cfr":            round(cfr, 4),
            "mean_utility":   round(mean_action, 4),
            "mean_baseline":  round(mean_taken, 4),
            "n":              len(action_rows),
            "interpretation": (
                f"'{action}' would have gained {round(cfr, 2)} total utility "
                f"vs. baseline over {len(rows)} decisions"
                if cfr > 0 else
                f"'{action}' would have cost {round(abs(cfr), 2)} utility vs. baseline"
            ),
        }

    # ── Outcome tracking ──────────────────────────────────────────────────────

    def record_actual_outcome(self, action: str, quality: float) -> float:
        """Record what actually happened. Returns regret (predicted − actual)."""
        with self._lock:
            initial = copy.copy(self._current_state)
        future    = self._simulate_action(action, initial, steps=1)
        predicted = future.weighted_utility
        regret    = round(predicted - quality, 4)
        if abs(regret) < 0.1:
            self._correct_predictions += 1
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO outcomes "
                "(action, predicted_quality, actual_quality, regret, ts) "
                "VALUES (?,?,?,?,?)",
                (action, predicted, quality, regret, time.time())
            )
            conn.commit()
        finally:
            conn.close()
        return regret

    def _save_simulation(self, action: str, future: SimulatedFuture) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO simulations "
                "(action, predicted_quality, weighted_utility, value_scores, "
                "steps, safety_passed, ts) VALUES (?,?,?,?,?,?,?)",
                (action, future.overall_score, future.weighted_utility,
                 json.dumps(future.value_scores), future.steps,
                 int(future.safety_passed), time.time())
            )
            conn.commit()
        finally:
            conn.close()

    # ── Regret analysis ───────────────────────────────────────────────────────

    def regret_summary(self) -> Dict[str, Any]:
        """Analyze historical prediction accuracy and CFR across all actions."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT action, AVG(regret) as avg_regret, COUNT(*) as n "
                "FROM outcomes GROUP BY action ORDER BY ABS(avg_regret) DESC"
            ).fetchall()
            total_aborts = conn.execute(
                "SELECT COUNT(*) FROM branch_simulations WHERE aborted=1"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "by_action": [
                {"action": r["action"],
                 "avg_regret": round(r["avg_regret"], 4),
                 "n": r["n"]}
                for r in rows
            ],
            "prediction_accuracy": round(
                self._correct_predictions / max(1, self._simulations_run), 3),
            "safety_aborts": self._aborted_actions,
            "branch_aborts": total_aborts,
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            s = self._current_state
        conn = self._conn()
        try:
            n_sims     = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
            n_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            n_branches = conn.execute("SELECT COUNT(*) FROM branch_simulations").fetchone()[0]
        finally:
            conn.close()
        return {
            "active":              True,
            "confidence":          round(s.confidence, 4),
            "accuracy":            round(self._correct_predictions / max(1, self._simulations_run), 4),
            "items":               n_sims,
            "current_state":       s.to_dict(),
            "simulations_run":     n_sims,
            "branch_simulations":  n_branches,
            "outcomes_recorded":   n_outcomes,
            "prediction_accuracy": round(self._correct_predictions / max(1, self._simulations_run), 4),
            "safety_gate_threshold": _SAFETY_THRESHOLD,
            "actions_aborted":     self._aborted_actions,
            "value_weights":       _VALUE_WEIGHTS,
        }

# Usage: sim = SelfSimulationEngine() | sim.choose_best_action("Douglas asked") | sim.simulate_branch_tree("evolve", depth=3)
