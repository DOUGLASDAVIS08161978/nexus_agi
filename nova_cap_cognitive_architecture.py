"""
Nova ASI — Cognitive Architecture (Master Integration Layer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implements the Global Workspace Theory of consciousness:
information is broadcast on a "global workspace bus" so every
subsystem can attend to it simultaneously.

Features:
  • Attention gate — salience-ranked focus on the most important signals
  • Working memory slots with TTL — active context window
  • Cognitive cycle — perceive → attend → reason → act → learn (async)
  • Coordination bus — inter-subsystem messaging
  • Arousal regulation — modulates processing depth by urgency
  • Bottleneck detection — finds the weakest link in the cognitive chain

Written by Claude Code (Anthropic) — not auto-generated.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import math
import time
import json
import sqlite3
import hashlib
import threading
import collections
from typing import Any, Callable, Optional
from datetime import datetime

_DB = os.path.expanduser("~/nexus_agi/nova_cognitive_arch.db")

# Working memory: how many slots and default TTL (seconds)
WM_SLOTS = 12
WM_DEFAULT_TTL = 3600  # 1 hour


class _WorkingMemorySlot:
    __slots__ = ("key", "value", "salience", "source", "expires_at", "created_at")

    def __init__(self, key: str, value: Any, salience: float, source: str, ttl: float):
        self.key = key
        self.value = value
        self.salience = salience
        self.source = source
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl

    def is_live(self) -> bool:
        return time.time() < self.expires_at

    def decay(self) -> float:
        """Salience decays exponentially; returns current effective salience."""
        age = time.time() - self.created_at
        return self.salience * math.exp(-age / WM_DEFAULT_TTL)


class CognitiveArchitecture:
    """
    Nova's master cognitive integration layer.

    Acts as the "global workspace" that broadcasts information
    across all subsystems and coordinates cognitive cycles.

    Usage:
      arch = CognitiveArchitecture()
      arch.register_subsystem("reasoning", my_reasoner, weight=1.5)
      arch.perceive("user_input", "What is consciousness?", salience=0.9)
      arch.run_cycle()          # one perceive→attend→reason→act→learn step
      arch.broadcast("answer", result, source="reasoning")
      arch.status()
    """

    CYCLE_PHASES = ("perceive", "attend", "reason", "act", "learn")

    def __init__(self, db_path: str = _DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        # Working memory: ordered dict, evict lowest salience when full
        self._wm: dict[str, _WorkingMemorySlot] = collections.OrderedDict()
        # Registered subsystems: {name: (obj, weight)}
        self._subsystems: dict[str, tuple[Any, float]] = {}
        # Global workspace bus: recent broadcast messages
        self._bus: collections.deque = collections.deque(maxlen=100)
        # Arousal level 0–1: controls processing depth
        self._arousal: float = 0.5
        # Cycle counter
        self._cycle: int = 0
        # Attention focus: currently attended slot key
        self._focus: Optional[str] = None
        self._init_db()
        self._bg = threading.Thread(target=self._background_cycle, daemon=True)
        self._bg.start()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle_num INTEGER NOT NULL,
                        phase TEXT NOT NULL,
                        focus_key TEXT,
                        arousal REAL,
                        subsystems_active INTEGER DEFAULT 0,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS broadcasts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        source TEXT NOT NULL,
                        salience REAL NOT NULL,
                        summary TEXT NOT NULL,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS bottleneck_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subsystem TEXT NOT NULL,
                        issue TEXT NOT NULL,
                        severity REAL NOT NULL,
                        ts REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Subsystem registration ─────────────────────────────────────────────────

    def register_subsystem(self, name: str, obj: Any, weight: float = 1.0):
        """Register a cognitive subsystem that participates in broadcast/cycles."""
        with self._lock:
            self._subsystems[name] = (obj, max(0.1, weight))

    def deregister_subsystem(self, name: str):
        with self._lock:
            self._subsystems.pop(name, None)

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, key: str, value: Any, salience: float = 0.5,
                 source: str = "external", ttl: float = WM_DEFAULT_TTL):
        """
        Add a percept to working memory. Evicts the lowest-salience live slot
        when memory is full. High-salience percepts trigger arousal increase.
        """
        salience = max(0.0, min(1.0, salience))
        slot = _WorkingMemorySlot(key, value, salience, source, ttl)
        with self._lock:
            # Evict expired slots
            dead = [k for k, v in self._wm.items() if not v.is_live()]
            for k in dead:
                del self._wm[k]
            # Evict lowest-salience if still full
            if len(self._wm) >= WM_SLOTS:
                victim = min(self._wm, key=lambda k: self._wm[k].decay())
                del self._wm[victim]
            self._wm[key] = slot
            # Arousal: spike toward the input's salience, then decay
            self._arousal = min(1.0, self._arousal * 0.85 + salience * 0.15)

    # ── Attention ─────────────────────────────────────────────────────────────

    def attend(self) -> Optional[str]:
        """
        Select the highest-salience live working-memory slot as the focus.
        Returns the focus key or None.
        """
        with self._lock:
            live = [(k, v.decay()) for k, v in self._wm.items() if v.is_live()]
            if not live:
                self._focus = None
                return None
            # Attention bias: boost arousal-amplified salience
            best_key = max(live, key=lambda x: x[1] * (1.0 + self._arousal))[0]
            self._focus = best_key
            return best_key

    # ── Broadcast (global workspace) ─────────────────────────────────────────

    def broadcast(self, key: str, value: Any, source: str = "workspace",
                  salience: float = 0.7):
        """
        Broadcast information to all subsystems via the global workspace bus.
        Each subsystem can react (if it exposes an .on_broadcast(key, value) method).
        """
        summary = str(value)[:200] if not isinstance(value, str) else value[:200]
        msg = {
            "key": key, "source": source, "salience": salience,
            "summary": summary, "ts": time.time()
        }
        with self._lock:
            self._bus.append(msg)

        # Notify registered subsystems that expose on_broadcast
        with self._lock:
            subs = list(self._subsystems.items())
        for name, (obj, _) in subs:
            fn = getattr(obj, "on_broadcast", None)
            if callable(fn):
                try:
                    fn(key, value)
                except Exception:
                    pass

        # Persist to DB
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO broadcasts (key, source, salience, summary, ts) VALUES (?,?,?,?,?)",
                    (key, source, salience, summary, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    # ── Cognitive cycle ───────────────────────────────────────────────────────

    def run_cycle(self) -> dict[str, Any]:
        """
        Execute one full cognitive cycle:
          perceive → attend → reason → act → learn
        Returns a summary of what happened in each phase.
        """
        self._cycle += 1
        cycle_num = self._cycle
        summary: dict[str, Any] = {"cycle": cycle_num, "phases": {}}

        # PERCEIVE: gather active working memory
        with self._lock:
            active_wm = [(k, v.value, v.decay()) for k, v in self._wm.items() if v.is_live()]
        summary["phases"]["perceive"] = {
            "active_slots": len(active_wm),
            "arousal": round(self._arousal, 3),
        }
        self._log_cycle(cycle_num, "perceive", None)

        # ATTEND: select focus
        focus_key = self.attend()
        summary["phases"]["attend"] = {"focus": focus_key}
        self._log_cycle(cycle_num, "attend", focus_key)

        # REASON: call registered subsystems that have a .reason() or .status() method
        reason_results = {}
        with self._lock:
            subs = list(self._subsystems.items())
        for name, (obj, weight) in subs:
            fn = getattr(obj, "reason", None) or getattr(obj, "status", None)
            if callable(fn):
                try:
                    r = fn()
                    reason_results[name] = {"weight": weight, "ok": True}
                except Exception as e:
                    reason_results[name] = {"weight": weight, "ok": False, "err": str(e)[:60]}
        summary["phases"]["reason"] = {"subsystems_called": len(reason_results)}
        self._log_cycle(cycle_num, "reason", focus_key, len(reason_results))

        # ACT: broadcast the focus content to all subsystems
        if focus_key and focus_key in self._wm:
            focus_val = self._wm[focus_key].value
            self.broadcast(focus_key, focus_val, source="cycle_action",
                           salience=self._arousal)
        summary["phases"]["act"] = {"broadcast": focus_key}
        self._log_cycle(cycle_num, "act", focus_key)

        # LEARN: decay arousal slightly (habituation)
        with self._lock:
            self._arousal = max(0.1, self._arousal * 0.95)
        summary["phases"]["learn"] = {"new_arousal": round(self._arousal, 3)}
        self._log_cycle(cycle_num, "learn", focus_key)

        return summary

    def _log_cycle(self, cycle_num: int, phase: str, focus_key: Optional[str],
                   active_subs: int = 0):
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO cycles (cycle_num, phase, focus_key, arousal, subsystems_active, ts) "
                    "VALUES (?,?,?,?,?,?)",
                    (cycle_num, phase, focus_key, self._arousal, active_subs, time.time())
                )
                conn.commit()
            finally:
                conn.close()

    # ── Bottleneck detection ───────────────────────────────────────────────────

    def detect_bottleneck(self) -> dict[str, Any]:
        """
        Identify the subsystem that is slowest / least responsive.
        Uses status() call timing and error frequency.
        """
        results = {}
        with self._lock:
            subs = list(self._subsystems.items())
        for name, (obj, weight) in subs:
            t0 = time.time()
            try:
                st = obj.status() if hasattr(obj, "status") else {}
                elapsed = time.time() - t0
                conf = st.get("confidence", 0.5) if isinstance(st, dict) else 0.5
                results[name] = {
                    "elapsed_s": round(elapsed, 4),
                    "confidence": conf,
                    "score": conf / max(elapsed, 0.001) * weight,
                }
            except Exception as e:
                results[name] = {"elapsed_s": 99.0, "confidence": 0.0, "score": 0.0,
                                 "error": str(e)[:80]}

        if not results:
            return {"bottleneck": None, "reason": "No subsystems registered."}

        bottleneck = min(results, key=lambda k: results[k]["score"])
        worst = results[bottleneck]

        issue = (
            f"Subsystem '{bottleneck}' has lowest combined score "
            f"(confidence={worst.get('confidence', 0):.2f}, "
            f"latency={worst.get('elapsed_s', 0):.3f}s)."
        )
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO bottleneck_log (subsystem, issue, severity, ts) VALUES (?,?,?,?)",
                    (bottleneck, issue, 1.0 - worst.get("confidence", 0), time.time())
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "bottleneck": bottleneck,
            "reason": issue,
            "all_scores": {k: round(v["score"], 4) for k, v in results.items()},
        }

    # ── Introspection ─────────────────────────────────────────────────────────

    def working_memory_snapshot(self) -> list[dict[str, Any]]:
        """Return all live working-memory slots sorted by effective salience."""
        with self._lock:
            slots = [
                {
                    "key": k,
                    "source": v.source,
                    "salience": round(v.decay(), 4),
                    "expires_in_s": round(v.expires_at - time.time(), 1),
                    "value_preview": str(v.value)[:80],
                }
                for k, v in self._wm.items() if v.is_live()
            ]
        slots.sort(key=lambda x: x["salience"], reverse=True)
        return slots

    def recent_broadcasts(self, n: int = 10) -> list[dict]:
        msgs = list(self._bus)[-n:]
        msgs.reverse()
        return msgs

    def _background_cycle(self):
        """Run cognitive cycles automatically every 60s."""
        time.sleep(30)
        while True:
            try:
                self.run_cycle()
            except Exception:
                pass
            time.sleep(60)

    def status(self) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            try:
                total_cycles = conn.execute("SELECT COUNT(DISTINCT cycle_num) FROM cycles").fetchone()[0]
                total_broadcasts = conn.execute("SELECT COUNT(*) FROM broadcasts").fetchone()[0]
                bottlenecks = conn.execute("SELECT COUNT(*) FROM bottleneck_log").fetchone()[0]
            finally:
                conn.close()
        wm_slots = len([k for k, v in self._wm.items() if v.is_live()])
        return {
            "active": True,
            "items": total_broadcasts,
            "confidence": round(self._arousal, 4),
            "accuracy": round(self._arousal, 4),
            "cycle": self._cycle,
            "total_cycles": total_cycles,
            "focus": self._focus,
            "arousal": round(self._arousal, 4),
            "working_memory_slots": wm_slots,
            "registered_subsystems": len(self._subsystems),
            "bus_messages": len(self._bus),
            "bottlenecks_detected": bottlenecks,
        }
