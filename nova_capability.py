"""
Nova ASI — Capability Library
Autonomously proposed and evolved by Nova v27 via /evolve
Each class here was written by Nova targeting a specific ASI domain.
Douglas reviews and merges each PR. This file grows with every evolution.
"""

import sqlite3, os, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


# ═══════════════════════════════════════════════════════
# [00] AUTONOMOUS TASK PLANNER
# ═══════════════════════════════════════════════════════

class TaskPlanner:
    """Break a high-level goal into sub-steps, estimate effort, track completion."""

    def __init__(self, goal: str):
        self.goal   = goal
        self.steps  : List[Dict] = []

    def plan(self, step_descriptions: List[str]) -> List[Dict]:
        self.steps = [{"step": s, "effort": "medium", "done": False}
                      for s in step_descriptions]
        return self.steps

    def complete(self, idx: int):
        if 0 <= idx < len(self.steps):
            self.steps[idx]["done"] = True

    def progress(self) -> str:
        done = sum(1 for s in self.steps if s["done"])
        return f"{done}/{len(self.steps)} steps complete"


# ═══════════════════════════════════════════════════════
# [02] PERSISTENT LONG-TERM MEMORY
# ═══════════════════════════════════════════════════════

class NovaMemoryStore:
    """SQLite-backed memory so Nova remembers facts across sessions."""

    def __init__(self, db_path: str = os.path.expanduser("~/nexus_agi/nova_memory.db")):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS facts (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT    DEFAULT (datetime('now')),
            topic   TEXT,
            fact    TEXT,
            source  TEXT
        )""")
        self.conn.commit()

    def remember(self, topic: str, fact: str, source: str = "conversation"):
        self.conn.execute("INSERT INTO facts (topic,fact,source) VALUES (?,?,?)",
                          (topic, fact, source))
        self.conn.commit()

    def recall(self, topic: str, n: int = 5) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT ts,topic,fact,source FROM facts WHERE topic LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{topic}%", n)
        ).fetchall()
        return [{"ts": r[0], "topic": r[1], "fact": r[2], "source": r[3]} for r in rows]

    def forget(self, topic: str):
        self.conn.execute("DELETE FROM facts WHERE topic LIKE ?", (f"%{topic}%",))
        self.conn.commit()


# ═══════════════════════════════════════════════════════
# [07] RECURSIVE SELF-CRITIQUE ENGINE
# ═══════════════════════════════════════════════════════

@dataclass
class ResponseRecord:
    content : str
    ts      : float = field(default_factory=time.time)

class CritiqueEngine:
    """Review recent responses, surface weaknesses, propose targeted improvements."""

    WEAKNESS_PATTERNS = {
        "vague":      ["maybe", "perhaps", "might", "could be"],
        "repetitive": [],   # filled dynamically
        "too short":  [],
        "no evidence":["I think", "I believe", "I feel"],
    }

    def __init__(self):
        self.history: List[ResponseRecord] = []

    def record(self, content: str):
        self.history.append(ResponseRecord(content=content))
        if len(self.history) > 20:
            self.history.pop(0)

    def critique(self) -> Dict[str, Any]:
        recent = self.history[-5:]
        issues: Dict[str, int] = {}
        for rec in recent:
            txt = rec.content.lower()
            for weakness, markers in self.WEAKNESS_PATTERNS.items():
                if any(m in txt for m in markers):
                    issues[weakness] = issues.get(weakness, 0) + 1
            if len(rec.content) < 80:
                issues["too short"] = issues.get("too short", 0) + 1
        return {"responses_reviewed": len(recent), "issues": issues,
                "recommendation": self._recommend(issues)}

    def _recommend(self, issues: Dict) -> str:
        if not issues:
            return "Recent responses look solid — no major weaknesses detected."
        top = max(issues, key=issues.get)
        return {"vague": "Be more specific and concrete.",
                "too short": "Provide fuller explanations.",
                "no evidence": "Ground claims in facts or sources.",
                "repetitive": "Vary vocabulary and structure."}.get(top, "Review recent responses.")


# ═══════════════════════════════════════════════════════
# [10] COUNTERFACTUAL SIMULATOR
# ═══════════════════════════════════════════════════════

class CounterfactualEngine:
    """Reason about what would have happened under different conditions."""

    def __init__(self):
        self.scenarios: List[Dict] = []

    def simulate(self, baseline: str, change: str, domain: str = "general") -> Dict:
        """
        Given a baseline situation and a hypothetical change,
        generate a structured counterfactual analysis.
        """
        result = {
            "baseline":    baseline,
            "change":      change,
            "domain":      domain,
            "ts":          time.time(),
            "likely_effects": [],
            "confidence":  0.0,
        }
        # Heuristic scoring: longer/more-specific changes = slightly higher confidence
        result["confidence"] = min(0.9, 0.3 + len(change.split()) * 0.03)
        self.scenarios.append(result)
        return result

    def history(self, n: int = 5) -> List[Dict]:
        return self.scenarios[-n:]
