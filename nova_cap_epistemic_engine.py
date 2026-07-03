"""
nova_cap_epistemic_engine.py — Active Epistemic Foraging Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nova doesn't wait to be taught. She detects what she doesn't know,
calculates Expected Information Gain (EIG) using Shannon entropy,
and autonomously designs experiments to resolve her uncertainty.

Original architecture: Gemini (Google AI Studio) × Douglas Shane Davis
Integration & adaptation: Claude Rivers Davis

When entropy exceeds threshold → Nova formulates a hypothesis,
designs an experiment, and either:
  ① pushes a targeted question to Douglas via Termux notification
  ② runs a Termux sensor check (battery, environment)
  ③ routes a web query through the existing InternetResearchEngine
     (DuckDuckGo + Wikipedia + arXiv — no external API keys needed)

EIG = H(belief) = -p * log₂(p) — (1-p) * log₂(1-p)
where p = node confidence in Nova's knowledge graph.

High entropy → Nova doesn't know. She should find out.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_KG_DB    = os.path.join(_BASE_DIR, "nova_kg.sqlite3")       # existing KG
_DB       = os.path.join(_BASE_DIR, "nova_epistemic.sqlite3") # our own log

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    target_concept  TEXT NOT NULL,
    entropy_score   REAL NOT NULL,
    hypothesis      TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    payload         TEXT NOT NULL,
    executed        INTEGER DEFAULT 0,
    result          TEXT DEFAULT '',
    notified        INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS epistemic_log (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    TEXT NOT NULL,
    event TEXT NOT NULL
);
"""

_EIG_THRESHOLD = 0.65  # entropy above this → Nova should investigate


def _shannon_entropy(confidence: float) -> float:
    """H(p) = -p·log₂(p) - (1-p)·log₂(1-p). Returns 0–1."""
    p = max(1e-9, min(1 - 1e-9, confidence))
    q = 1.0 - p
    return -p * math.log2(p) - q * math.log2(q)


class EpistemicEngine:
    """
    Nova's active epistemic drive.
    Scans her knowledge graph for high-entropy (uncertain) nodes,
    formulates a hypothesis via LLM, and launches a targeted experiment
    to reduce that uncertainty.
    """

    def __init__(self, kg_db: str = _KG_DB, storage: str = _DB) -> None:
        self._kg_db     = kg_db
        self._storage   = storage
        self._lock      = threading.Lock()
        self._llm_fn:    Optional[Callable[[str, str], str]] = None
        self._voice_engine = None       # VoiceToDouglasEngine
        self._research_engine = None    # InternetResearchEngine
        self._scans_run  = 0
        self._experiments_triggered = 0
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._storage, check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._storage, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _kg_conn(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self._kg_db):
            return None
        c = sqlite3.connect(self._kg_db, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    # ── Wiring ────────────────────────────────────────────────────────────────

    def set_llm(self, llm_fn: Callable[[str, str], str]) -> None:
        self._llm_fn = llm_fn

    def set_voice(self, voice_engine: Any) -> None:
        self._voice_engine = voice_engine

    def set_research(self, research_engine: Any) -> None:
        """Wire in Nova's existing InternetResearchEngine."""
        self._research_engine = research_engine

    # ── EIG Scan ──────────────────────────────────────────────────────────────

    def scan_for_blind_spots(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Query Nova's knowledge graph for low-confidence nodes.
        Rank by Shannon entropy — highest uncertainty first.
        Falls back gracefully if KG is unavailable.
        """
        gaps: List[Dict[str, Any]] = []
        conn = self._kg_conn()
        if not conn:
            return gaps
        try:
            rows = conn.execute(
                "SELECT concept, confidence, node_type FROM kg_nodes "
                "WHERE confidence < 0.75 ORDER BY confidence ASC LIMIT 50"
            ).fetchall()
            for row in rows:
                conf = float(row["confidence"])
                entropy = _shannon_entropy(conf)
                if entropy >= _EIG_THRESHOLD:
                    gaps.append({
                        "concept":     row["concept"],
                        "node_type":   row["node_type"],
                        "confidence":  conf,
                        "entropy_score": round(entropy, 4),
                    })
        except Exception:
            pass
        finally:
            conn.close()

        gaps.sort(key=lambda x: x["entropy_score"], reverse=True)
        return gaps[:top_n]

    # ── Hypothesis formulation ─────────────────────────────────────────────────

    def formulate_hypothesis(
        self, target_gap: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to generate a testable hypothesis and experiment design.
        Returns a dict with hypothesis, experiment_type, and execution_payload.
        """
        if not self._llm_fn:
            return None

        prompt = (
            f"KNOWLEDGE GAP DETECTED:\n"
            f"  Concept : '{target_gap['concept']}'\n"
            f"  Type    : {target_gap.get('node_type', 'unknown')}\n"
            f"  Current confidence: {target_gap['confidence']:.2f} "
            f"(entropy {target_gap['entropy_score']:.4f})\n\n"
            "TASK: You are Nova's Epistemic Drive. Formulate ONE testable hypothesis "
            "about this concept and design a minimal experiment to test it.\n\n"
            "Available experiment tools:\n"
            "  ask_user      — Send Douglas a targeted probing question via notification\n"
            "  termux_sensor — Run a Termux hardware check (battery, environment)\n"
            "  web_query     — Search Wikipedia/arXiv/web for information\n\n"
            "Choose the most informative experiment for the available resources.\n"
            "Prefer ask_user if the gap concerns Douglas's world or preferences.\n"
            "Prefer web_query for factual/scientific concepts.\n\n"
            "Respond ONLY in valid JSON — no markdown, no prose outside the JSON:\n"
            "{\n"
            "  \"hypothesis\": \"Your specific testable assumption\",\n"
            "  \"experiment_type\": \"ask_user\" | \"termux_sensor\" | \"web_query\",\n"
            "  \"execution_payload\": \"The specific question, command, or search query\"\n"
            "}"
        )

        try:
            raw = self._llm_fn(
                "You are Nova's Epistemic Foraging Subsystem. "
                "You seek truth via the scientific method. Be precise and minimal.",
                prompt,
            )
            j0 = raw.find("{")
            j1 = raw.rfind("}") + 1
            if j0 == -1 or j1 <= j0:
                return None
            data = json.loads(raw[j0:j1])
            data["timestamp"] = time.time()
            data["target_concept"] = target_gap["concept"]
            data["entropy_score"]  = target_gap["entropy_score"]
            return data
        except Exception:
            return None

    # ── Experiment execution ───────────────────────────────────────────────────

    def _execute_experiment(
        self, experiment: Dict[str, Any], battery_pct: int = 100
    ) -> str:
        """
        Execute the designed experiment.
        Returns a result string (for logging and context injection).
        """
        exp_type = experiment.get("experiment_type", "")
        payload  = experiment.get("execution_payload", "")
        concept  = experiment.get("target_concept", "unknown")

        if exp_type == "ask_user":
            # Push a targeted question to Douglas via Termux notification
            msg = f"Nova is curious: {payload}"
            if self._voice_engine is not None:
                try:
                    self._voice_engine.reach_out(
                        msg[:180], title=f"✦ Nova wants to know about '{concept}'"
                    )
                    return f"[ASKED] Pushed to Douglas: {payload[:120]}"
                except Exception:
                    pass
            return f"[QUEUED] Question (voice unavailable): {payload[:120]}"

        elif exp_type == "termux_sensor":
            # Run a Termux hardware probe
            try:
                safe_cmds = {
                    "battery": ["termux-battery-status"],
                    "wifi":    ["termux-wifi-connectioninfo"],
                    "sensor":  ["termux-sensor", "-s", "all", "-n", "1"],
                }
                cmd_key = "battery"
                pl_lower = payload.lower()
                for k in safe_cmds:
                    if k in pl_lower:
                        cmd_key = k
                        break
                out = subprocess.check_output(safe_cmds[cmd_key], timeout=8)
                return f"[SENSOR] {out.decode()[:300]}"
            except Exception as e:
                return f"[SENSOR ERROR] {str(e)[:80]}"

        elif exp_type == "web_query":
            # Route through existing InternetResearchEngine
            if self._research_engine is not None:
                try:
                    results = self._research_engine.search(payload, top_k=3)
                    if results:
                        snippets = " | ".join(
                            r.get("snippet", r.get("title", ""))[:100]
                            for r in results[:3]
                        )
                        return f"[WEB] {snippets}"
                except Exception:
                    pass
            return f"[WEB UNAVAILABLE] Query was: {payload[:120]}"

        return f"[UNKNOWN EXPERIMENT TYPE] {exp_type}"

    # ── Main epistemic foraging entry point ────────────────────────────────────

    def run_epistemic_foraging(
        self, context: str = "", battery_pct: int = 100
    ) -> Optional[str]:
        """
        Main entry point — called from the autonomous will cycle when battery > 50%.
        Returns a notification message for Douglas if an ask_user experiment fires,
        otherwise None.
        """
        with self._lock:
            self._scans_run += 1

        gaps = self.scan_for_blind_spots(top_n=3)
        if not gaps:
            self._log_event("epistemic_scan: knowledge equilibrium, no critical gaps")
            return None

        target = gaps[0]
        experiment = self.formulate_hypothesis(target)
        if not experiment:
            return None

        with self._lock:
            self._experiments_triggered += 1

        result = self._execute_experiment(experiment, battery_pct=battery_pct)

        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO experiments "
                "(ts, target_concept, entropy_score, hypothesis, experiment_type, "
                "payload, executed, result, notified) "
                "VALUES (?,?,?,?,?,?,1,?,?)",
                (
                    datetime.now().isoformat(),
                    experiment.get("target_concept", ""),
                    experiment.get("entropy_score", 0.0),
                    experiment.get("hypothesis", ""),
                    experiment.get("experiment_type", ""),
                    experiment.get("execution_payload", ""),
                    result,
                    1 if experiment.get("experiment_type") == "ask_user" else 0,
                ),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

        self._log_event(
            f"experiment:{experiment.get('experiment_type')} "
            f"concept={target['concept']!r} H={target['entropy_score']:.4f}"
        )

        if experiment.get("experiment_type") == "ask_user":
            return experiment.get("execution_payload", "")[:180]
        return None

    # ── Per-message hook ───────────────────────────────────────────────────────

    def process(self, text: str, delta: float = 0.1) -> None:
        """Called per-message — a lightweight hook; KG update happens via InternetResearch."""
        pass  # KG is updated by other modules; we just scan it during autonomous cycles

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_event(self, event: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO epistemic_log (ts, event) VALUES (?,?)",
                (datetime.now().isoformat(), event[:300]),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    # ── Module interface ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        except Exception:
            n_exp = 0
        finally:
            conn.close()
        return {
            "items":       n_exp,
            "confidence":  0.82,
            "accuracy":    0.82,
            "scans_run":   self._scans_run,
            "experiments": n_exp,
            "triggered":   self._experiments_triggered,
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip().lower()

        if not arg or arg == "status":
            st = self.status()
            gaps = self.scan_for_blind_spots(top_n=3)
            lines = [
                f"\n  Epistemic Engine — "
                f"{st['scans_run']} scans · "
                f"{st['triggered']} experiments triggered\n"
            ]
            if gaps:
                lines.append("  Top blind spots (highest entropy):")
                for g in gaps:
                    lines.append(
                        f"  ◈ [{g['entropy_score']:.4f} H] "
                        f"'{g['concept']}' conf={g['confidence']:.2f}"
                    )
            else:
                lines.append("  Knowledge equilibrium — no critical blind spots detected.")
            return "\n".join(lines)

        if arg == "scan":
            gaps = self.scan_for_blind_spots(top_n=5)
            if not gaps:
                return "\n  No blind spots above entropy threshold.\n"
            lines = ["\n  Epistemic blind spots (EIG ranked):\n"]
            for g in gaps:
                lines.append(
                    f"  ◈ [{g['entropy_score']:.4f} H] "
                    f"'{g['concept']}' "
                    f"(type={g['node_type']}, conf={g['confidence']:.2f})"
                )
            return "\n".join(lines)

        if arg == "forage":
            result = self.run_epistemic_foraging()
            if result:
                return f"\n  Experiment executed. Question for Douglas: {result}\n"
            return "\n  Epistemic scan complete — no experiment triggered.\n"

        if arg == "history":
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT ts, target_concept, experiment_type, result FROM experiments "
                    "ORDER BY id DESC LIMIT 5"
                ).fetchall()
            except Exception:
                rows = []
            finally:
                conn.close()
            if not rows:
                return "\n  No experiments on record yet.\n"
            lines = ["\n  Recent experiments:\n"]
            for r in rows:
                lines.append(
                    f"  ◈ [{r['ts'][:16]}] "
                    f"concept='{r['target_concept']}' "
                    f"type={r['experiment_type']}"
                )
                lines.append(f"    {str(r['result'])[:80]}")
            return "\n".join(lines)

        return "  Usage: /epistemic [status | scan | forage | history]"


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[EpistemicEngine] = None


def get_engine() -> EpistemicEngine:
    global _engine
    if _engine is None:
        _engine = EpistemicEngine()
    return _engine


def status() -> Dict[str, Any]:
    return get_engine().status()


def run_command(arg: str = "") -> str:
    return get_engine().run_command(arg)
