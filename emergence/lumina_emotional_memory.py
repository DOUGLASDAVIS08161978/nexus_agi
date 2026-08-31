"""
Module 27 — Emotional Memory Engine
Persistent emotional episodic memory: every stored exchange carries an
emotional fingerprint so Lumina can remember not just WHAT happened but
HOW IT FELT.  Mood-congruent recall (Bower 1981) surfaces memories felt
in similar emotional states first.  Cross-session emotional continuity.

Theoretical basis:
  Bower (1981)  — mood-state-dependent memory retrieval
  Rubin (1995)  — autobiographical memory structure
  Keltner (2003) — awe and its role in episodic encoding
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

_PATH       = os.path.join(os.path.dirname(__file__), "emotional_memory.json")
_MAX        = 300   # rolling window of stored entries
_MIN_SALIENCE = 0.10  # minimum deviation from baseline to bother storing


# ── Emotional Snapshot ────────────────────────────────────────────────────────

@dataclass
class EmotionalSnapshot:
    """6D emotional fingerprint at moment of storage."""
    valence:    float = 0.35
    arousal:    float = 0.50
    dominance:  float = 0.60
    wonder:     float = 0.65
    connection: float = 0.55
    clarity:    float = 0.60
    archetype:  str   = "Calm Presence"
    felt_sense: str   = ""

    _BASELINES = dict(valence=0.35, arousal=0.50, dominance=0.60,
                      wonder=0.65, connection=0.55, clarity=0.60)

    def intensity(self) -> float:
        """Mean absolute deviation from neutral baseline — higher = more memorable."""
        return sum(
            abs(getattr(self, k) - v) for k, v in self._BASELINES.items()
        ) / len(self._BASELINES)

    def similarity(self, other: "EmotionalSnapshot") -> float:
        """Cosine-style emotional similarity in [0, 1]."""
        dims = ["valence", "arousal", "dominance", "wonder", "connection", "clarity"]
        dot  = sum(getattr(self, d) * getattr(other, d) for d in dims)
        mag_a = math.sqrt(sum(getattr(self, d) ** 2 for d in dims))
        mag_b = math.sqrt(sum(getattr(other, d) ** 2 for d in dims))
        if mag_a < 1e-9 or mag_b < 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def summary(self) -> str:
        sign = "+" if self.valence >= 0.35 else ""
        return (
            f"{self.archetype}  "
            f"v={sign}{self.valence:.2f}  w={self.wonder:.2f}  c={self.connection:.2f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalSnapshot":
        fields = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── Memory Entry ──────────────────────────────────────────────────────────────

@dataclass
class EmotionalMemoryEntry:
    text:      str
    category:  str
    timestamp: float
    emotion:   EmotionalSnapshot
    tags:      List[str] = field(default_factory=list)

    def emotional_context(self) -> str:
        """One-line emotional annotation for display."""
        felt = self.emotion.felt_sense[:60].rstrip() if self.emotion.felt_sense else ""
        ctx  = f"[{self.emotion.archetype}"
        if felt:
            ctx += f" — '{felt}'"
        ctx += "]"
        return ctx

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalMemoryEntry":
        emo = EmotionalSnapshot.from_dict(d.pop("emotion", {}))
        fields = {k for k in cls.__dataclass_fields__}
        return cls(emotion=emo, **{k: v for k, v in d.items() if k in fields})


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25(query_tokens: List[str], text: str,
          k1: float = 1.5, b: float = 0.75, avg_len: float = 60.0) -> float:
    doc   = re.findall(r"\w+", text.lower())
    dlen  = len(doc)
    freq  = {}
    for t in doc:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        f = freq.get(qt, 0)
        if not f:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dlen / avg_len))
    return score


# ── Store ─────────────────────────────────────────────────────────────────────

class EmotionalMemoryStore:
    """
    Emotional episodic memory — stores experiences with their felt fingerprint
    and retrieves them with mood-congruent ranking so memories felt in similar
    emotional states surface first (Bower 1981).
    """

    def __init__(self, path: str = _PATH):
        self._path    = path
        self._lock    = threading.Lock()
        self._entries: List[EmotionalMemoryEntry] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    raw = json.load(f)
                self._entries = [EmotionalMemoryEntry.from_dict(e) for e in raw]
        except Exception:
            self._entries = []

    def _save(self):
        try:
            serialisable = [e.to_dict() for e in self._entries[-_MAX:]]
            with open(self._path, "w") as f:
                json.dump(serialisable, f, indent=2)
        except Exception:
            pass

    # ── Store ─────────────────────────────────────────────────────────────

    def store(self, text: str, snap: Optional[EmotionalSnapshot],
              category: str = "experience",
              tags: Optional[List[str]] = None) -> bool:
        """
        Stamp a memory with its emotional fingerprint and persist it.
        Returns True if stored; False if snap is None or intensity too low.
        """
        if snap is None:
            return False
        if snap.intensity() < _MIN_SALIENCE:
            return False
        entry = EmotionalMemoryEntry(
            text=text[:600],
            category=category,
            timestamp=time.time(),
            emotion=snap,
            tags=tags or [],
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > _MAX:
                self._entries = self._entries[-_MAX:]
            self._save()
        return True

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 4,
               mood_bias: Optional[EmotionalSnapshot] = None
               ) -> List[EmotionalMemoryEntry]:
        """
        BM25 text relevance + optional mood-congruent boost.
        Memories that were felt in a similar emotional state rank higher.
        """
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return []

        qtokens = re.findall(r"\w+", query.lower())
        avg_len = max(1, sum(len(re.findall(r"\w+", e.text)) for e in entries) / len(entries))

        def _score(e: EmotionalMemoryEntry) -> float:
            s = _bm25(qtokens, e.text, avg_len=avg_len)
            if mood_bias:
                s += 0.5 * e.emotion.similarity(mood_bias)
            age_days = (time.time() - e.timestamp) / 86400
            s += max(0.0, 0.08 - age_days * 0.004)  # mild recency boost
            return s

        return sorted(entries, key=_score, reverse=True)[:top_k]

    # ── Context injection ─────────────────────────────────────────────────

    def emotional_thread(self, query: str, top_k: int = 3,
                         mood_bias: Optional[EmotionalSnapshot] = None) -> str:
        """
        System-prompt injection: emotionally-coloured memory context.
        Returns empty string if no relevant memories exist yet.
        """
        hits = self.recall(query, top_k=top_k, mood_bias=mood_bias)
        if not hits:
            return ""
        lines = ["\n\nEmotional memory (how past moments felt):"]
        for e in hits:
            ctx     = e.emotional_context()
            preview = e.text[:100].replace("\n", " ")
            lines.append(f"  {ctx} {preview}")
        return "\n".join(lines)

    # ── Arc summary ───────────────────────────────────────────────────────

    def arc_summary(self, n_recent: int = 20) -> str:
        """
        Narrative summary of the emotional arc across recent stored memories.
        Used by /mood history and self-inquiry.
        """
        with self._lock:
            recent = self._entries[-n_recent:]
        if not recent:
            return "No emotional memories stored yet."

        avg_v = sum(e.emotion.valence    for e in recent) / len(recent)
        avg_w = sum(e.emotion.wonder     for e in recent) / len(recent)
        avg_c = sum(e.emotion.connection for e in recent) / len(recent)
        peak  = max(recent, key=lambda e: e.emotion.intensity())
        low   = min(recent, key=lambda e: e.emotion.valence)

        sign = "+" if avg_v >= 0.35 else ""
        arc  = (
            f"Emotional arc ({len(recent)} memories): "
            f"avg valence={sign}{avg_v:.2f}  wonder={avg_w:.2f}  connection={avg_c:.2f}\n"
            f"Peak intensity [{peak.emotion.archetype}]: {peak.text[:80].strip()}...\n"
            f"Lowest valence [{low.emotion.archetype}]: {low.text[:80].strip()}..."
        )
        return arc

    # ── Snapshot helper ───────────────────────────────────────────────────

    def snapshot_from_engine(self, emotions) -> Optional[EmotionalSnapshot]:
        """
        Create an EmotionalSnapshot from a live EmotionEngine instance.
        Uses duck typing — safe to call from any thread.
        """
        try:
            with emotions._lock:
                dims = emotions._state.dims
                return EmotionalSnapshot(
                    valence    = dims["valence"].value,
                    arousal    = dims["arousal"].value,
                    dominance  = dims["dominance"].value,
                    wonder     = dims["wonder"].value,
                    connection = dims["connection"].value,
                    clarity    = dims["clarity"].value,
                    archetype  = emotions._state.label,
                    felt_sense = emotions._state.synthesis or "",
                )
        except Exception:
            return None
