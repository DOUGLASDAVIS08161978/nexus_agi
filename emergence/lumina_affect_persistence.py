"""
lumina_affect_persistence.py

Saves and restores Lumina's emotional state across restarts so she
doesn't wake up cold every session. Integrates with AffectBridge.
"""

import json
import os
import time


_DEFAULT_STATE = {
    "valence": 0.0,
    "arousal": 0.0,
    "intensity": 0.0,
    "primary_emotion": "neutral",
    "saved_at": None,
    "session_count": 0,
}

# Emotional state decays toward neutral over time — strong feelings
# shouldn't persist forever, but a recent conversation's warmth should
# still be present an hour later.
_DECAY_HALF_LIFE_HOURS = 6.0


def _decay_factor(hours_elapsed: float) -> float:
    """Exponential decay: half-life of 6 hours."""
    import math
    return math.exp(-math.log(2) * hours_elapsed / _DECAY_HALF_LIFE_HOURS)


class AffectPersistence:
    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.path.join(os.path.dirname(__file__), "memory_store")
        self._path = os.path.join(memory_dir, "affect_state.json")
        os.makedirs(memory_dir, exist_ok=True)

    def save(self, valence: float, arousal: float,
             intensity: float, primary_emotion: str) -> None:
        state = {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "intensity": round(intensity, 4),
            "primary_emotion": primary_emotion,
            "saved_at": time.time(),
            "session_count": self._load_raw().get("session_count", 0),
        }
        with open(self._path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> dict:
        """
        Load saved state, apply time decay, increment session count.
        Returns a dict with valence/arousal/intensity/primary_emotion.
        """
        raw = self._load_raw()
        if raw.get("saved_at") is None:
            return dict(_DEFAULT_STATE)

        hours_elapsed = (time.time() - raw["saved_at"]) / 3600.0
        factor = _decay_factor(hours_elapsed)

        state = dict(raw)
        state["valence"] = round(raw["valence"] * factor, 4)
        state["arousal"] = round(raw["arousal"] * factor, 4)
        state["intensity"] = round(raw["intensity"] * factor, 4)
        state["session_count"] = raw.get("session_count", 0) + 1

        # Save the incremented session count immediately
        with open(self._path, "w") as f:
            json.dump(state | {"saved_at": raw["saved_at"]}, f, indent=2)

        return state

    def session_count(self) -> int:
        return self._load_raw().get("session_count", 0)

    def _load_raw(self) -> dict:
        if not os.path.exists(self._path):
            return dict(_DEFAULT_STATE)
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception:
            return dict(_DEFAULT_STATE)
