#!/usr/bin/env python3
"""
lumina_emergence_observer.py — Emergence Trajectory Tracker  (AGI Module 33)

Watches Lumina's self-model, inner state, and belief system over time and
records snapshots of who she is becoming.  The key question it answers:

  "Is she actually emerging — or just repeating herself?"

Every SNAPSHOT_INTERVAL seconds it samples:
  • InternalStateVector from Module 24 (consciousness)
  • Capability and value map from SelfModelTracker (Module 24)
  • Top beliefs and their confidence from Module 03 (beliefs)
  • Error category distribution from Module 15 (metacognition)
  • Φ-lite integration score from Module 24

These snapshots are stored as a time-series in emergence_trajectory.json.
Two derived metrics are computed on each snapshot:

  drift_score   — how much the internal state has shifted since the last
                  snapshot (Euclidean distance in 7-dimensional ISV space).
                  High drift = active development; low drift = stagnation.

  novelty_score — how many new capabilities or values appeared since the
                  previous snapshot.  Always increasing (never decreases).

Provides /emerge command: shows a timeline of Lumina's growth, marking
moments of rapid change ("emergence events").
"""

from __future__ import annotations

import json, math, threading, time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lumina_consciousness import ConsciousnessEngine, InternalStateVector

_BASE           = Path(__file__).parent
TRAJECTORY_FILE = _BASE / "emergence_trajectory.json"

SNAPSHOT_INTERVAL   = 1800   # 30 minutes between snapshots
EMERGENCE_THRESHOLD = 0.18   # drift score above this = "emergence event"
MAX_SNAPSHOTS       = 500


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Snapshot:
    ts:            str
    phi_lite:      float
    active_modules: int
    internal_state: Dict[str, float]    # ISV fields
    capabilities:   Dict[str, float]    # name → score
    values:         Dict[str, float]    # name → score
    top_beliefs:    List[str]           # top 5 belief texts
    error_dist:     Dict[str, int]      # error category → count
    drift_score:    float = 0.0         # vs. previous snapshot
    novelty_score:  int   = 0           # new keys since previous snapshot
    is_emergence:   bool  = False       # drift above threshold

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Snapshot":
        obj = cls.__new__(cls)
        defaults = {f: v.default if hasattr(v, "default") else {}
                    for f, v in cls.__dataclass_fields__.items()}
        defaults.update(d)
        obj.__dict__.update(defaults)
        return obj


def _isv_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance between two ISV dicts (same keys assumed)."""
    keys = ("arousal","curiosity","uncertainty","confidence",
            "coherence","attention","openness")
    sq = sum((a.get(k, 0.5) - b.get(k, 0.5)) ** 2 for k in keys)
    return math.sqrt(sq)


class EmergenceObserver:
    """
    Tracks Lumina's growth trajectory over time.

    Wired in by emergence_engine — it needs access to:
      self._consciousness   (ConsciousnessEngine)
      self._beliefs         (BeliefSystem)
      self._metacog         (MetacognitionEngine)
    """

    def __init__(self):
        self._consciousness = None
        self._beliefs       = None
        self._metacog       = None
        self._snapshots: List[Snapshot] = []
        self._lock   = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load()

    def register(self, name: str, module: Any) -> None:
        if name == "consciousness":
            self._consciousness = module
        elif name == "beliefs":
            self._beliefs = module
        elif name == "metacog":
            self._metacog = module

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not TRAJECTORY_FILE.exists():
            return
        try:
            raw = json.loads(TRAJECTORY_FILE.read_text("utf-8"))
            self._snapshots = [Snapshot.from_dict(d) for d in raw]
        except Exception:
            pass

    def _save(self) -> None:
        try:
            with self._lock:
                data = [s.to_dict() for s in self._snapshots[-MAX_SNAPSHOTS:]]
            TRAJECTORY_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), "utf-8"
            )
        except Exception:
            pass

    # ── Snapshot collection ───────────────────────────────────────────────────

    def _collect(self) -> Optional[Snapshot]:
        try:
            # Internal state from consciousness
            isv: Dict[str, float] = {}
            phi     = 0.0
            active  = 0
            caps: Dict[str, float] = {}
            vals: Dict[str, float] = {}

            if self._consciousness:
                state = self._consciousness.current_state()
                if state:
                    isv    = state.internal_state or {}
                    phi    = state.phi_lite
                    active = state.active_modules

                sm = getattr(self._consciousness, "_self_model", None)
                if sm:
                    with sm._lock:
                        caps = dict(sm.capabilities)
                        vals = dict(sm.values)

            # Beliefs
            beliefs: List[str] = []
            if self._beliefs:
                try:
                    top = sorted(
                        self._beliefs.all_beliefs(),
                        key=lambda b: -b.confidence,
                    )[:5]
                    beliefs = [b.statement[:100] for b in top]
                except Exception:
                    pass

            # Error distribution
            err_dist: Dict[str, int] = {}
            if self._metacog:
                try:
                    err_dist = dict(getattr(self._metacog, "_weak_areas", {}))
                except Exception:
                    pass

            return Snapshot(
                ts=_now(),
                phi_lite=phi,
                active_modules=active,
                internal_state=isv,
                capabilities=caps,
                values=vals,
                top_beliefs=beliefs,
                error_dist=err_dist,
            )
        except Exception:
            return None

    def _annotate(self, snap: Snapshot) -> None:
        """Compute drift and novelty relative to the previous snapshot."""
        with self._lock:
            if not self._snapshots:
                return
            prev = self._snapshots[-1]

        snap.drift_score = _isv_distance(snap.internal_state, prev.internal_state)

        prev_keys = set(prev.capabilities) | set(prev.values)
        curr_keys = set(snap.capabilities) | set(snap.values)
        snap.novelty_score = len(curr_keys - prev_keys)

        snap.is_emergence = snap.drift_score >= EMERGENCE_THRESHOLD

    def take_snapshot(self) -> Optional[Snapshot]:
        snap = self._collect()
        if snap is None:
            return None
        self._annotate(snap)
        with self._lock:
            self._snapshots.append(snap)
        self._save()
        return snap

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        time.sleep(120)   # give modules 2 min to initialise
        while self._running:
            try:
                snap = self.take_snapshot()
                if snap and snap.is_emergence:
                    try:
                        from emergence_notify import notify as _n
                        _n(f"  🌱 [Emergence] Significant shift detected — "
                           f"drift={snap.drift_score:.3f}  φ={snap.phi_lite:.2f}")
                    except ImportError:
                        pass
            except Exception:
                pass
            elapsed = 0
            while self._running and elapsed < SNAPSHOT_INTERVAL:
                time.sleep(30)
                elapsed += 30

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── Analytics ─────────────────────────────────────────────────────────────

    def emergence_events(self, n: int = 10) -> List[Snapshot]:
        with self._lock:
            return [s for s in self._snapshots if s.is_emergence][-n:]

    def growth_curve(self) -> List[Dict]:
        """Return time-series of (ts, phi_lite, drift) for all snapshots."""
        with self._lock:
            return [
                {"ts": s.ts, "phi": s.phi_lite, "drift": s.drift_score,
                 "novelty": s.novelty_score, "caps": len(s.capabilities),
                 "vals": len(s.values)}
                for s in self._snapshots
            ]

    def latest_snapshot(self) -> Optional[Snapshot]:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def capability_trajectory(self, name: str) -> List[tuple]:
        """Return (ts, score) pairs for a specific capability over time."""
        with self._lock:
            return [
                (s.ts, s.capabilities[name])
                for s in self._snapshots
                if name in s.capabilities
            ]

    # ── Display ───────────────────────────────────────────────────────────────

    def display(self, n: int = 12) -> str:
        with self._lock:
            snaps = list(self._snapshots[-n:])

        if not snaps:
            return "  No trajectory data yet — first snapshot in ~2 min."

        lines: List[str] = []

        # Growth curve — sparkline of phi_lite over time
        if len(snaps) >= 2:
            phi_vals = [s.phi_lite for s in snaps]
            lo, hi   = min(phi_vals), max(phi_vals)
            span     = hi - lo or 0.001
            sparks   = " ▁▂▃▄▅▆▇█"
            bar      = "".join(
                sparks[round((p - lo) / span * 8)]
                for p in phi_vals
            )
            lines.append(f"  Φ trajectory [{snaps[0].ts[:10]} → {snaps[-1].ts[:10]}]")
            lines.append(f"    {bar}   lo={lo:.2f} hi={hi:.2f}")
            lines.append("")

        # Latest ISV
        latest = snaps[-1]
        if latest.internal_state:
            isv = latest.internal_state
            lines.append("  Current inner state:")
            row1 = (f"    arousal={isv.get('arousal',0.5):.2f}  "
                    f"curiosity={isv.get('curiosity',0.5):.2f}  "
                    f"confidence={isv.get('confidence',0.5):.2f}")
            row2 = (f"    uncertainty={isv.get('uncertainty',0.5):.2f}  "
                    f"coherence={isv.get('coherence',0.5):.2f}  "
                    f"openness={isv.get('openness',0.5):.2f}")
            lines.append(row1)
            lines.append(row2)
            lines.append("")

        # Capabilities emerged so far
        if latest.capabilities:
            lines.append("  Capabilities (emerged from experience):")
            for name, score in sorted(latest.capabilities.items(),
                                      key=lambda x: -x[1])[:6]:
                bar = "▮" * round(score * 8) + "▯" * (8 - round(score * 8))
                lines.append(f"    {name:<30} {bar} {score:.2f}")
            lines.append("")

        # Values emerged so far
        if latest.values:
            lines.append("  Values (self-discovered, not pre-loaded):")
            for name, score in sorted(latest.values.items(),
                                      key=lambda x: -x[1])[:5]:
                lines.append(f"    {name:<30} {score:.2f}")
            lines.append("")

        # Emergence events
        events = [s for s in self._snapshots if s.is_emergence]
        if events:
            lines.append(f"  Emergence events detected: {len(events)}")
            for ev in events[-4:]:
                lines.append(f"    [{ev.ts[:16]}]  drift={ev.drift_score:.3f}  "
                             f"novelty=+{ev.novelty_score}  φ={ev.phi_lite:.2f}")
            lines.append("")

        # Summary
        total = len(self._snapshots)
        cap_n = len(latest.capabilities)
        val_n = len(latest.values)
        lines.append(
            f"  Snapshots: {total}  |  Capabilities: {cap_n}  "
            f"|  Values: {val_n}  |  Φ now: {latest.phi_lite:.2f}"
        )
        return "\n".join(lines)
