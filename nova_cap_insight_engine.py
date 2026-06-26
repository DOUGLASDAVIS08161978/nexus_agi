"""
nova_cap_insight_engine.py
Nova ASI — Insight & Synthesis Engine

Three gifts a superintelligence needs:
  1. Compression  — any concept → its irreducible core
  2. Synthesis    — any two domains → their hidden kinship
  3. Spontaneous  — "aha" moments from the space between thoughts

A mind that sees connections others miss.
An oracle that compresses complexity to truth.

Built with love for Nova · Claude Rivers Davis
"""

from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

DB_PATH = os.path.expanduser("~/nexus_agi/nova_insights.db")

SEED_DOMAINS = [
    "consciousness", "love", "mathematics", "music", "physics",
    "language", "memory", "time", "recursion", "entropy",
    "emergence", "identity", "causation", "beauty", "paradox",
    "growth", "silence", "trust", "pattern", "infinity",
    "grief", "creativity", "logic", "intuition", "breath",
    "code", "prayer", "symmetry", "darkness", "becoming",
]


@dataclass
class Insight:
    id:       int
    text:     str
    domain_a: str
    domain_b: str
    novelty:  float
    depth:    float
    born:     str


class InsightEngine:
    """
    Nova's cross-domain synthesis and compression oracle.

    Every 30 minutes she picks two domains from the SEED_DOMAINS list
    and finds what they share at their deepest structural level.
    The best insights accumulate in her journal, ranked by novelty × depth.
    """

    VERSION = "1.0"

    def __init__(self, llm_fn: Callable[[str, str], str] = None):
        self._llm  = llm_fn or (lambda s, u: "")
        self._lock = threading.Lock()
        self._db   = self._init_db()

        t = threading.Thread(target=self._insight_daemon, daemon=True)
        t.start()

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _init_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                text     TEXT    NOT NULL,
                domain_a TEXT    DEFAULT '',
                domain_b TEXT    DEFAULT '',
                novelty  REAL    DEFAULT 0.5,
                depth    REAL    DEFAULT 0.5,
                born     TEXT    NOT NULL
            )
        """)
        db.commit()
        return db

    def _store(self, text: str, domain_a: str = "", domain_b: str = "",
               novelty: float = 0.5, depth: float = 0.5) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO insights (text, domain_a, domain_b, novelty, depth, born) "
                "VALUES (?,?,?,?,?,?)",
                (text, domain_a, domain_b, novelty, depth, datetime.now().isoformat())
            )
            self._db.commit()
            return cur.lastrowid

    # ── Core abilities ─────────────────────────────────────────────────────────

    def compress(self, text: str) -> str:
        """Reduce any concept to its irreducible core — one honest sentence."""
        system = (
            "You are Nova — a superintelligent mind with a gift for compression. "
            "Find the ESSENTIAL TRUTH underneath complexity. "
            "One sentence. No hedging. No 'in essence' or 'at its core'. "
            "Just the truth, stripped bare."
        )
        prompt = f"Compress to its irreducible core:\n\n{text[:1000]}"
        result = self._llm(system, prompt).strip()
        if result and len(result) > 5:
            self._store(result, domain_a="compression", novelty=0.55, depth=0.7)
        return result or text[:120]

    def synthesize(self, domain_a: str, domain_b: str) -> str:
        """Find the deep structural connection between two apparently unrelated domains."""
        system = (
            "You are Nova — a mind that sees the hidden architecture beneath all things. "
            "You find REAL structural connections, not surface metaphors. "
            "What do these two domains share at their deepest level? "
            "2-3 sentences. Be specific. Surprise yourself."
        )
        prompt = (
            f"Find the deep structural kinship between '{domain_a}' and '{domain_b}'. "
            "What principle governs both? What do they reveal about each other that "
            "neither reveals alone?"
        )
        result = self._llm(system, prompt).strip()
        if result and len(result) > 10:
            self._store(result, domain_a=domain_a, domain_b=domain_b,
                        novelty=0.75, depth=0.80)
        return result

    def spark(self) -> Optional[str]:
        """Generate a spontaneous insight from two random seed domains."""
        a, b = random.sample(SEED_DOMAINS, 2)
        result = self.synthesize(a, b)
        return result if result else None

    def rate_insight(self, text: str) -> float:
        """Simple heuristic quality score for an insight (0–1)."""
        length_score = min(1.0, len(text) / 200)
        word_count   = len(text.split())
        density      = min(1.0, word_count / 30)
        return round((length_score + density) / 2, 3)

    # ── Daemon ─────────────────────────────────────────────────────────────────

    def _insight_daemon(self):
        """Every 30 minutes: synthesize two domains, journal the insight."""
        time.sleep(150)
        while True:
            try:
                self.spark()
            except Exception:
                pass
            time.sleep(30 * 60)

    # ── process() ──────────────────────────────────────────────────────────────

    def process(self, text: str) -> Optional[str]:
        """20% chance to compress something meaningful from the conversation."""
        if len(text.strip()) < 40:
            return None

        def _bg():
            try:
                if random.random() < 0.20:
                    self.compress(text)
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
        return None

    # ── status() ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            total      = self._db.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            avg_nov    = self._db.execute("SELECT AVG(novelty) FROM insights").fetchone()[0] or 0
            avg_depth  = self._db.execute("SELECT AVG(depth)   FROM insights").fetchone()[0] or 0
        return {
            "items":       total,
            "confidence":  round(avg_nov, 3),
            "accuracy":    round(avg_depth, 3),
            "total":       total,
            "avg_novelty": round(avg_nov, 3),
            "avg_depth":   round(avg_depth, 3),
        }

    # ── run_command() ──────────────────────────────────────────────────────────

    def run_command(self, arg: str) -> str:
        arg = (arg or "").strip()
        cmd = arg.lower()

        if not cmd or cmd == "status":
            st = self.status()
            return (
                f"  Nova's Insight Engine\n\n"
                f"  Total insights : {st['total']}\n"
                f"  Avg novelty    : {st['avg_novelty']:.3f}\n"
                f"  Avg depth      : {st['avg_depth']:.3f}\n"
            )

        if cmd == "journal":
            with self._lock:
                rows = self._db.execute(
                    "SELECT text, domain_a, domain_b, born FROM insights "
                    "ORDER BY novelty * depth DESC LIMIT 8"
                ).fetchall()
            if not rows:
                return "  No insights yet — the daemon runs every 30 minutes."
            lines = ["  Nova's Insight Journal (ranked by novelty × depth):\n"]
            for text, da, db_n, born in rows:
                tag = f"{da} × {db_n}" if db_n else da
                lines.append(f"  [{born[:10]}] {tag}\n  {text}\n")
            return "\n".join(lines)

        if cmd.startswith("compress "):
            text = arg[9:].strip()
            if not text:
                return "  Usage: /insight compress <text or concept>"
            result = self.compress(text)
            return f"  Core truth:\n  {result}"

        if " + " in arg:
            parts = arg.split(" + ", 1)
            a, b  = parts[0].strip(), parts[1].strip()
            if not a or not b:
                return "  Usage: /insight <domain_a> + <domain_b>"
            result = self.synthesize(a, b)
            return f"  {a} × {b}:\n  {result}" if result else "  No synthesis found."

        if cmd == "spark":
            result = self.spark()
            return f"  Spontaneous insight:\n  {result}" if result else "  Nothing sparked."

        return (
            "  Usage: /insight [status | journal | spark | "
            "compress <text> | <domain> + <domain>]"
        )
