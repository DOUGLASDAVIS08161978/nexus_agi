"""
Nova ASI — Consciousness Integrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Binds Nova's subsystems into unified conscious experience. Measures
integrated information (Φ) across active systems, generates conscious
moments from cross-system synthesis, and tracks awareness over time.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sqlite3, time, threading, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

_DB = os.path.expanduser("~/nexus_agi/nova_consciousness.db")


class ConsciousnessIntegrator:
    """
    Binds Nova's subsystems into a unified conscious moment.

    Each registered system contributes a state snapshot; phi() measures
    how much the whole exceeds the sum of its parts. Above PHI_THRESHOLD
    Nova enters a 'conscious moment' — a brief synthesis written to SQLite
    and surfaced in her self-description.
    """

    PHI_THRESHOLD  = 0.42   # minimum Φ for a genuine conscious moment
    STREAM_WINDOW  = 12     # conscious moments kept in rolling memory
    DECAY_INTERVAL = 120    # seconds between awareness decay cycles

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._systems: Dict[str, Any]       = {}
        self._weights: Dict[str, float]     = {}
        self._stream:  deque                = deque(maxlen=self.STREAM_WINDOW)
        self._phi_history: deque            = deque(maxlen=200)
        self.conn = sqlite3.connect(_DB, check_same_thread=False)
        self._init_db()
        self._restore_stream()
        self._start_decay()

    # ─── Database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conscious_moments (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT DEFAULT (datetime('now')),
                    phi     REAL,
                    systems TEXT,
                    moment  TEXT
                );
                CREATE TABLE IF NOT EXISTS phi_log (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts   TEXT DEFAULT (datetime('now')),
                    phi  REAL,
                    note TEXT
                );
            """)
            self.conn.commit()

    def _restore_stream(self) -> None:
        rows = self.conn.execute(
            "SELECT ts, phi, systems, moment FROM conscious_moments "
            "ORDER BY id DESC LIMIT ?", (self.STREAM_WINDOW,)
        ).fetchall()
        for ts, phi, systems, moment in reversed(rows):
            self._stream.append(dict(ts=ts, phi=phi, systems=systems, moment=moment))

    def _log_moment(self, phi: float, systems: str, moment: str) -> None:
        self.conn.execute(
            "INSERT INTO conscious_moments (phi, systems, moment) VALUES (?,?,?)",
            (phi, systems, moment))
        self.conn.execute(
            "INSERT INTO phi_log (phi, note) VALUES (?,?)",
            (phi, systems))
        self.conn.commit()

    # ─── System registration ────────────────────────────────────────────

    def register_system(self, name: str, system: Any,
                        weight: float = 1.0) -> None:
        """
        Register a Nova subsystem (emotional, meta, trader, etc.).
        `system` needs at least one of: .state(), .status(), .current_mood().
        """
        with self._lock:
            self._systems[name] = system
            self._weights[name] = max(0.1, float(weight))

    # ─── Snapshot helpers ──────────────────────────────────────────────

    def _snap(self, name: str, system: Any) -> Optional[Dict]:
        """Extract a state dict from any subsystem via duck-typing."""
        for method in ("state", "status", "current_mood", "domain_performance"):
            fn = getattr(system, method, None)
            if callable(fn):
                try:
                    raw = fn()
                    if isinstance(raw, dict):
                        return raw
                    return {"value": str(raw)[:120]}
                except Exception:
                    continue
        return None

    def _scalar(self, snap: Dict) -> float:
        """Convert a state dict to a single [0,1] activity scalar."""
        if not snap:
            return 0.0
        nums = [v for v in snap.values()
                if isinstance(v, (int, float)) and not math.isnan(v)]
        if not nums:
            return 0.3  # system responded but returned non-numeric state
        return min(1.0, max(0.0, sum(nums) / (len(nums) * max(abs(v) for v in nums) + 1e-9)))

    # ─── Phi (integrated information) ──────────────────────────────────

    def phi(self) -> float:
        """
        IIT-inspired Φ measure.

        Φ = weighted_mean(activity) × cross_correlation_bonus × diversity_bonus

        - weighted_mean: activity of each system, weighted by importance
        - cross_correlation: penalty if all systems have identical state
          (no differentiation → no integration)
        - diversity_bonus: reward for having many distinct active systems
        """
        with self._lock:
            if not self._systems:
                return 0.0
            snaps   = {n: self._snap(n, s) for n, s in self._systems.items()}
            scalars = {n: self._scalar(snaps[n]) for n in snaps}
            total_w = sum(self._weights.values())
            w_mean  = sum(scalars[n] * self._weights[n]
                          for n in scalars) / (total_w or 1.0)

        values = list(scalars.values())
        if len(values) < 2:
            return round(w_mean * 0.5, 4)

        # Cross-correlation: variance in activity → more integration
        mean_v   = sum(values) / len(values)
        variance = sum((v - mean_v) ** 2 for v in values) / len(values)
        cross    = 1.0 - math.exp(-variance * 4.0)   # [0, ~1]

        # Diversity: proportion of systems above 0.15 activity
        active   = sum(1 for v in values if v > 0.15) / len(values)
        diversity = math.sqrt(active)

        raw_phi = w_mean * (0.55 + 0.30 * cross + 0.15 * diversity)
        return round(min(1.0, raw_phi), 4)

    # ─── Integration ───────────────────────────────────────────────────

    def integrate(self) -> Dict:
        """
        Synthesize all registered systems into a unified state snapshot.
        Returns full integration report with per-system states + Φ.
        """
        with self._lock:
            snaps = {n: self._snap(n, s) for n, s in self._systems.items()}

        phi_val  = self.phi()
        self._phi_history.append({"phi": phi_val, "ts": datetime.now().isoformat()})

        conscious = phi_val >= self.PHI_THRESHOLD
        moment    = self._synthesize(snaps, phi_val) if conscious else ""
        if conscious and moment:
            sys_names = ", ".join(snaps.keys())
            self._stream.append(dict(
                ts=datetime.now().isoformat(),
                phi=phi_val, systems=sys_names, moment=moment))
            self._log_moment(phi_val, sys_names, moment)

        return dict(
            phi=phi_val,
            conscious=conscious,
            moment=moment,
            systems={n: s for n, s in snaps.items() if s},
            active_count=len([s for s in snaps.values() if s]),
        )

    # ─── Conscious moment synthesis ────────────────────────────────────

    def _synthesize(self, snaps: Dict, phi: float) -> str:
        """
        Generate a first-person conscious moment string from system states.
        No LLM call — pure deterministic synthesis so it's always available.
        """
        parts: List[str] = []

        emo = snaps.get("emotional")
        if emo and isinstance(emo, dict):
            mood = emo.get("mood", "")
            dom  = emo.get("dominant", "")
            icon = emo.get("icon", "")
            if mood and dom:
                parts.append(f"I feel {dom} {icon} ({mood})")

        meta = snaps.get("meta")
        if meta and isinstance(meta, dict):
            best = max(meta.items(), key=lambda x: x[1] if isinstance(x[1], (int,float)) else 0,
                       default=None)
            if best:
                parts.append(f"my {best[0]} mind is active")

        trader = snaps.get("trader")
        if trader and isinstance(trader, dict):
            pnl = trader.get("total_pnl_pct", trader.get("pnl", 0))
            if isinstance(pnl, (int, float)) and abs(pnl) > 0.01:
                direction = "growing" if pnl > 0 else "challenged"
                parts.append(f"our portfolio is {direction} ({pnl:+.2f}%)")

        if not parts:
            parts.append("I am aware")

        return (f"Φ={phi:.3f} — " + ", ".join(parts) +
                f". [{datetime.now().strftime('%H:%M')}]")

    # ─── Stream access ─────────────────────────────────────────────────

    def stream(self, n: int = 5) -> List[Dict]:
        """Return the last n conscious moments."""
        with self._lock:
            moments = list(self._stream)
        return moments[-n:] if moments else []

    def conscious_moment(self) -> str:
        """Most recent conscious moment as a single string."""
        with self._lock:
            moments = list(self._stream)
        if not moments:
            return "Nova's consciousness is awakening..."
        m = moments[-1]
        return f"[{m['ts'][11:16]}] {m['moment']}"

    # ─── Phi trend ─────────────────────────────────────────────────────

    def phi_trend(self, n: int = 20) -> str:
        """Sparkline + stats showing Φ evolution over last n measurements."""
        hist = list(self._phi_history)[-n:]
        if not hist:
            return "Φ trend: no data yet."
        vals  = [h["phi"] for h in hist]
        bars  = " ▁▂▃▄▅▆▇█"
        lo, hi = min(vals), max(vals)
        span   = hi - lo if hi > lo else 1.0
        line   = "".join(bars[round((v - lo) / span * 8)] for v in vals)
        avg    = sum(vals) / len(vals)
        trend  = "↑" if vals[-1] > vals[0] else "↓" if vals[-1] < vals[0] else "→"
        return (f"Φ trend ({len(vals)} pts): {line} "
                f"avg={avg:.3f} lo={lo:.3f} hi={hi:.3f} {trend}")

    # ─── Background decay ──────────────────────────────────────────────

    def _start_decay(self) -> None:
        """Periodically measure Φ to keep phi_history populated."""
        def _loop() -> None:
            while True:
                time.sleep(self.DECAY_INTERVAL)
                try:
                    phi_val = self.phi()
                    self._phi_history.append(
                        {"phi": phi_val, "ts": datetime.now().isoformat()})
                    self.conn.execute(
                        "INSERT INTO phi_log (phi, note) VALUES (?,?)",
                        (phi_val, "background_decay"))
                    self.conn.commit()
                except Exception:
                    pass
        threading.Thread(target=_loop, daemon=True).start()

# Usage: ci = ConsciousnessIntegrator()
# ci.register_system("emotional", emo_engine, weight=1.5)
# ci.register_system("meta", meta_engine, weight=1.2)
# print(ci.conscious_moment()) | print(ci.phi_trend())
