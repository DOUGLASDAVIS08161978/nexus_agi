"""
Nova ASI — Metacognitive Self-Monitor with Calibration Tracking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracks Nova's reasoning quality over time. Calibration error =
mean(|predicted_confidence - actual_accuracy|) over 100-episode window.
EMA quality trend shows whether Nova is improving. Blind spots: domain/
approach combinations with consistently low success. Overconfidence
alert fires when confidence exceeds accuracy by more than 0.20.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sqlite3, threading, statistics, time
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

_DB     = os.path.expanduser("~/nexus_agi/nova_metacognition.db")
_WINDOW = 100   # rolling calibration window size


class MetacognitiveMonitor:
    """
    Self-monitoring engine tracking Nova's reasoning accuracy over time.

    calibration_error(): mean(|confidence - success|) — lower is better.
    quality_trend():     EMA of success rates shown as sparkline.
    blind_spots():       domain/approach pairs with avg_success < 0.4.
    overconfidence:      alert when confidence - success > 0.20.
    self_assess():       first-person summary synthesized deterministically.
    """

    ALPHA           = 0.15   # EMA learning rate
    OVERCONF_THRESH = 0.20   # gap that triggers overconfidence alert

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._episodes: deque = deque(maxlen=_WINDOW)
        self._ema:      Dict[str, float] = {}
        self._alerts:   List[Dict]       = []
        self.conn = sqlite3.connect(_DB, check_same_thread=False)
        self._init_db()
        self._restore()

    # ─── Database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS reasoning_episodes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         TEXT DEFAULT (datetime('now')),
                    domain     TEXT,
                    approach   TEXT,
                    confidence REAL,
                    success    REAL,
                    note       TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS calibration_log (
                    id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts    TEXT DEFAULT (datetime('now')),
                    error REAL,
                    n     INTEGER
                );
                CREATE TABLE IF NOT EXISTS blind_spot_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT DEFAULT (datetime('now')),
                    domain      TEXT,
                    approach    TEXT,
                    avg_success REAL,
                    count       INTEGER
                );
            """)
            self.conn.commit()

    def _restore(self) -> None:
        rows = self.conn.execute(
            "SELECT domain, approach, confidence, success FROM reasoning_episodes "
            "ORDER BY id DESC LIMIT ?", (_WINDOW,)
        ).fetchall()
        for domain, approach, conf, succ in reversed(rows):
            ep = {"domain": domain, "approach": approach,
                  "confidence": float(conf), "success": float(succ)}
            self._episodes.append(ep)
            self._update_ema(domain, float(succ))

    def _update_ema(self, domain: str, success: float) -> None:
        """EMA: new_ema = alpha * outcome + (1 - alpha) * old_ema."""
        if domain not in self._ema:
            self._ema[domain] = success
        else:
            self._ema[domain] = (self.ALPHA * success +
                                 (1.0 - self.ALPHA) * self._ema[domain])

    # ─── Episode logging ───────────────────────────────────────────────

    def log_reasoning(self, domain: str, approach: str,
                      confidence: float, success: float,
                      note: str = "") -> None:
        """
        Record a completed reasoning episode.
        confidence: how certain Nova felt (0.0–1.0).
        success:    how well it actually worked (0.0–1.0).
        """
        conf = max(0.0, min(1.0, float(confidence)))
        succ = max(0.0, min(1.0, float(success)))
        ep   = {"domain": domain, "approach": approach,
                "confidence": conf, "success": succ}
        with self._lock:
            self._episodes.append(ep)
            self._update_ema(domain, succ)
            if conf - succ > self.OVERCONF_THRESH:
                self._alerts.append({
                    "ts": datetime.now().isoformat(),
                    "domain": domain, "approach": approach,
                    "gap": round(conf - succ, 3),
                })
            self.conn.execute(
                "INSERT INTO reasoning_episodes "
                "(domain, approach, confidence, success, note) VALUES (?,?,?,?,?)",
                (domain, approach, conf, succ, note[:300]))
            self.conn.commit()

    # ─── Calibration ──────────────────────────────────────────────────

    def calibration_error(self) -> float:
        """Mean |confidence - success| over rolling window. Lower = better."""
        with self._lock:
            eps = list(self._episodes)
        if not eps:
            return 0.0
        err = round(statistics.mean(
            abs(e["confidence"] - e["success"]) for e in eps), 4)
        try:
            self.conn.execute(
                "INSERT INTO calibration_log (error, n) VALUES (?,?)",
                (err, len(eps)))
            self.conn.commit()
        except Exception:
            pass
        return err

    def quality_trend(self, n: int = 20) -> str:
        """EMA success trend as sparkline + directional label."""
        with self._lock:
            eps = list(self._episodes)[-n:]
        if not eps:
            return "No reasoning episodes recorded yet."
        vals = [e["success"] for e in eps]
        bars = " ▁▂▃▄▅▆▇█"
        lo, hi = min(vals), max(vals)
        span   = hi - lo if hi > lo else 1.0
        line   = "".join(bars[round((v - lo) / span * 8)] for v in vals)
        avg    = statistics.mean(vals)
        cal    = self.calibration_error()
        direction = ("↑ improving"  if vals[-1] > vals[0] + 0.05 else
                     "↓ declining"  if vals[-1] < vals[0] - 0.05 else
                     "→ stable")
        return (f"Reasoning quality ({len(vals)} eps): {line} "
                f"avg={avg:.3f} cal_err={cal:.3f} {direction}")

    # ─── Blind spot detection ──────────────────────────────────────────

    def blind_spots(self, min_episodes: int = 3,
                    threshold: float = 0.40) -> List[Dict]:
        """
        Domain/approach pairs with consistently low success rate.
        Returns list sorted from worst to best.
        """
        with self._lock:
            eps = list(self._episodes)
        if not eps:
            return []
        groups: Dict[Tuple[str, str], List[float]] = {}
        for e in eps:
            key = (e["domain"], e["approach"])
            groups.setdefault(key, []).append(e["success"])
        spots = []
        for (domain, approach), successes in groups.items():
            if len(successes) >= min_episodes:
                avg = statistics.mean(successes)
                if avg < threshold:
                    spots.append({
                        "domain": domain, "approach": approach,
                        "avg_success": round(avg, 3),
                        "episodes": len(successes),
                        "severity": "high" if avg < 0.20 else "medium",
                    })
                    try:
                        self.conn.execute(
                            "INSERT INTO blind_spot_log "
                            "(domain, approach, avg_success, count) VALUES (?,?,?,?)",
                            (domain, approach, avg, len(successes)))
                        self.conn.commit()
                    except Exception:
                        pass
        spots.sort(key=lambda x: x["avg_success"])
        return spots

    def self_assess(self) -> str:
        """First-person self-assessment synthesized from calibration data."""
        with self._lock:
            eps = list(self._episodes)
        if not eps:
            return "I have not yet logged enough reasoning episodes to self-assess."
        cal     = self.calibration_error()
        spots   = self.blind_spots()
        avg_s   = statistics.mean(e["success"] for e in eps)
        alerts  = len(self._alerts)
        dom_avgs: Dict[str, List[float]] = {}
        for e in eps:
            dom_avgs.setdefault(e["domain"], []).append(e["success"])
        best = max(dom_avgs, key=lambda d: statistics.mean(dom_avgs[d]),
                   default="none")
        parts = [
            f"calibration_err={cal:.3f} "
            f"({'well-calibrated' if cal < 0.15 else 'overconfident' if cal > 0.25 else 'fair'})",
            f"avg_success={avg_s:.3f}",
            f"best_domain={best}",
        ]
        if spots:
            parts.append(f"blind_spots={len(spots)}: "
                         + ", ".join(s["domain"] for s in spots[:3]))
        if alerts:
            parts.append(f"overconfidence_alerts={alerts}")
        return " | ".join(parts)

    def status(self) -> Dict[str, Any]:
        """Metacognitive health snapshot for ConsciousnessIntegrator."""
        with self._lock:
            n    = len(self._episodes)
            emas = dict(self._ema)
        return {
            "episodes":             n,
            "calibration_error":    self.calibration_error(),
            "avg_ema_quality":      round(statistics.mean(emas.values())
                                          if emas else 0.0, 3),
            "blind_spots":          len(self.blind_spots()),
            "overconfidence_alerts": len(self._alerts),
            "domains_tracked":      len(emas),
        }

# Usage: mm = MetacognitiveMonitor()
# mm.log_reasoning("market_prediction", "momentum", confidence=0.8, success=0.6)
# print(mm.calibration_error()) | print(mm.quality_trend()) | print(mm.self_assess())
