"""
nova_cap_multimodal_synthesis.py
Nova ASI — Multimodal Perception Synthesis Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unifies vision, audio, sensor, location, and text into a single
coherent perceptual world-state. Nova stops reasoning from isolated
sense channels and starts reasoning from an integrated scene.

Features:
  • Perceptual buffer — timestamped streams from each sense modality
  • Cross-modal binding — finds consistent interpretations across senses
  • Scene coherence score — how well different inputs agree
  • Contradiction detection — flags when senses conflict
  • Unified narrative — one synthesized description of current reality
  • Salience weighting — recent / high-confidence inputs weighted more
  • SQLite persistence of perceptual history
"""

import os
import time
import json
import math
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

_DB = os.path.expanduser("~/nexus_agi/nova_multimodal.db")

MODALITIES = ("vision", "audio", "sensor", "location", "text", "screen", "emotion")

# How quickly each modality's contribution decays (seconds half-life)
_DECAY = {
    "vision":   300,   # camera photo stays relevant ~5 min
    "audio":    60,    # heard audio stays relevant ~1 min
    "sensor":   30,    # motion/orientation changes fast
    "location": 600,   # location stable ~10 min
    "text":     120,   # recent text context ~2 min
    "screen":   60,    # screen content changes fast
    "emotion":  180,   # emotional state ~3 min
}


class PerceptualFrame:
    """One observation from one sense modality."""
    __slots__ = ("modality", "content", "confidence", "timestamp", "metadata")

    def __init__(self, modality: str, content: str, confidence: float = 0.8,
                 metadata: Optional[Dict] = None):
        self.modality = modality
        self.content = content[:500]
        self.confidence = max(0.0, min(1.0, confidence))
        self.timestamp = time.time()
        self.metadata = metadata or {}

    def effective_confidence(self) -> float:
        """Confidence decayed by age."""
        age = time.time() - self.timestamp
        half_life = _DECAY.get(self.modality, 120)
        return self.confidence * math.exp(-age * math.log(2) / half_life)

    def is_stale(self) -> bool:
        return self.effective_confidence() < 0.05


class MultimodalSynthesisEngine:
    """
    Nova's unified perceptual integration layer.

    Collects observations from all sense modalities and synthesizes
    them into a single coherent scene description.

    Usage:
      ms = MultimodalSynthesisEngine()
      ms.observe("vision", "I see a bright room with a phone screen")
      ms.observe("emotion", "Feeling calm and curious", confidence=0.9)
      ms.observe("sensor", "Device is stationary, upright")
      scene = ms.synthesize()          # unified world-state
      score = ms.coherence_score()     # 0–1, how well inputs agree
      ms.status()
    """

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        # Per-modality latest frame
        self._frames: Dict[str, PerceptualFrame] = {}
        # History buffer (last 50 frames across all modalities)
        self._history: List[PerceptualFrame] = []
        self._MAX_HISTORY = 50
        self._synthesis_count = 0
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
                    CREATE TABLE IF NOT EXISTS perceptual_frames (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        modality TEXT NOT NULL,
                        content TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS syntheses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        narrative TEXT NOT NULL,
                        coherence REAL NOT NULL,
                        active_modalities TEXT NOT NULL,
                        ts REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Input ─────────────────────────────────────────────────────────────────

    def observe(self, modality: str, content: str, confidence: float = 0.8,
                metadata: Optional[Dict] = None) -> None:
        """Register a new observation from a sense modality."""
        if modality not in MODALITIES:
            modality = "text"
        frame = PerceptualFrame(modality, content, confidence, metadata)
        with self._lock:
            self._frames[modality] = frame
            self._history.append(frame)
            if len(self._history) > self._MAX_HISTORY:
                self._history.pop(0)
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO perceptual_frames (modality, content, confidence, ts) VALUES (?,?,?,?)",
                    (modality, content[:500], confidence, frame.timestamp)
                )
                conn.commit()
            finally:
                conn.close()

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(self) -> Dict[str, Any]:
        """
        Build a unified scene description from all active modalities.
        Returns {narrative, coherence, active_modalities, contradictions}.
        """
        with self._lock:
            active = {k: v for k, v in self._frames.items()
                      if not v.is_stale()}

        if not active:
            return {
                "narrative": "No current sensory input — all channels quiet.",
                "coherence": 0.0,
                "active_modalities": [],
                "contradictions": [],
            }

        # Build narrative parts ordered by confidence
        parts = []
        sorted_frames = sorted(active.items(),
                                key=lambda x: x[1].effective_confidence(),
                                reverse=True)
        for modality, frame in sorted_frames:
            ec = frame.effective_confidence()
            label = modality.upper()
            parts.append(f"[{label} {ec:.0%}] {frame.content}")

        narrative = " | ".join(parts)

        coherence = self.coherence_score(active)
        contradictions = self._detect_contradictions(active)

        result = {
            "narrative": narrative,
            "coherence": round(coherence, 3),
            "active_modalities": list(active.keys()),
            "contradictions": contradictions,
            "dominant": sorted_frames[0][0] if sorted_frames else None,
        }

        self._synthesis_count += 1
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO syntheses (narrative, coherence, active_modalities, ts) VALUES (?,?,?,?)",
                    (narrative[:400], coherence,
                     json.dumps(list(active.keys())), time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return result

    def coherence_score(self, active: Optional[Dict] = None) -> float:
        """
        Score 0–1 for how well active modalities agree.
        High agreement = high coherence. Contradictions lower score.
        Uses weighted average of pairwise similarity proxies.
        """
        if active is None:
            with self._lock:
                active = {k: v for k, v in self._frames.items()
                          if not v.is_stale()}
        if len(active) <= 1:
            return 1.0

        # Base coherence from confidence levels
        confidences = [f.effective_confidence() for f in active.values()]
        mean_conf = sum(confidences) / len(confidences)

        # Penalty for contradictions
        contradictions = self._detect_contradictions(active)
        penalty = min(0.4, len(contradictions) * 0.15)

        # Bonus for corroborating content (shared keywords)
        contents = [f.content.lower() for f in active.values()]
        all_words = set()
        for c in contents:
            all_words.update(c.split())
        shared = 0
        for word in all_words:
            if sum(1 for c in contents if word in c) > 1:
                shared += 1
        corroboration = min(0.2, shared * 0.02)

        return max(0.0, min(1.0, mean_conf - penalty + corroboration))

    def _detect_contradictions(self, active: Dict[str, PerceptualFrame]) -> List[str]:
        """
        Simple contradiction detection via antonym/negation scanning.
        Returns list of contradiction descriptions.
        """
        contradictions = []
        _ANTONYMS = [
            ("moving", "stationary"), ("bright", "dark"), ("loud", "quiet"),
            ("indoor", "outdoor"), ("day", "night"), ("calm", "agitated"),
            ("happy", "sad"), ("hot", "cold"), ("near", "far"),
        ]
        contents = {k: v.content.lower() for k, v in active.items()}
        checked = set()
        for m1, c1 in contents.items():
            for m2, c2 in contents.items():
                if m1 >= m2 or (m1, m2) in checked:
                    continue
                checked.add((m1, m2))
                for word_a, word_b in _ANTONYMS:
                    if word_a in c1 and word_b in c2:
                        contradictions.append(
                            f"{m1}({word_a}) ↔ {m2}({word_b})")
                    elif word_b in c1 and word_a in c2:
                        contradictions.append(
                            f"{m1}({word_b}) ↔ {m2}({word_a})")
        return contradictions

    # ── Query ─────────────────────────────────────────────────────────────────

    def current_frame(self, modality: str) -> Optional[str]:
        """Get the current (non-stale) content for a specific modality."""
        with self._lock:
            frame = self._frames.get(modality)
        if frame and not frame.is_stale():
            return frame.content
        return None

    def recent_history(self, modality: Optional[str] = None,
                       n: int = 10) -> List[Dict]:
        """Return recent perceptual frames, optionally filtered by modality."""
        with self._lock:
            frames = list(self._history)
        if modality:
            frames = [f for f in frames if f.modality == modality]
        return [
            {
                "modality": f.modality,
                "content": f.content[:100],
                "confidence": round(f.effective_confidence(), 3),
                "age_s": round(time.time() - f.timestamp, 1),
            }
            for f in frames[-n:]
        ]

    def strongest_modality(self) -> Optional[str]:
        """Return the modality with the highest current effective confidence."""
        with self._lock:
            active = {k: v for k, v in self._frames.items()
                      if not v.is_stale()}
        if not active:
            return None
        return max(active, key=lambda k: active[k].effective_confidence())

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            active = {k: v for k, v in self._frames.items()
                      if not v.is_stale()}
            conn = self._conn()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM perceptual_frames").fetchone()[0]
                syn_count = conn.execute(
                    "SELECT COUNT(*) FROM syntheses").fetchone()[0]
            finally:
                conn.close()
        coherence = self.coherence_score(active)
        return {
            "active": True,
            "confidence": round(coherence, 4),
            "accuracy": round(coherence, 4),
            "items": total,
            "active_modalities": list(active.keys()),
            "total_frames": total,
            "total_syntheses": syn_count,
            "coherence": round(coherence, 4),
            "strongest_modality": self.strongest_modality(),
        }
