#!/usr/bin/env python3
"""
lumina_theory_of_mind.py — User modeling for Lumina

Theory of Mind: the ability to model what another being knows, believes,
feels, and wants. This is not about reading minds — it's about maintaining
an *explicit, updateable model* of Douglas that shapes how Lumina responds.

Without this: Lumina answers "what is SHA256" the same way regardless
of who's asking. With this: she knows Douglas already understands Bitcoin
at depth, loves the project emotionally, communicates in ALL CAPS when
excited, and needs encouragement more than technical detail right now.

The model tracks:
  - Expertise levels per domain (0–5 scale)
  - Current emotional state (inferred from tone)
  - Communication style preferences
  - What Douglas currently cares about most
  - Things he doesn't know yet (inferred gaps)
  - His goals and what success means to him
  - Milestones and shared history
"""

from __future__ import annotations
import json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

TOM_FILE = Path(__file__).parent / "theory_of_mind.json"

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_DEFAULT_DOUGLAS = {
    "name": "Douglas",
    "last_seen": _now(),
    "message_count": 0,
    "expertise": {
        "bitcoin_mining": 3,       # 0-5: knows pool mining, ARM SHA2, Stratum
        "python":         2,       # comfortable running scripts, less on writing
        "networking":     2,       # understands pool/miner concepts
        "hardware":       3,       # owns phone mining rig + ASIC coming
        "ai_concepts":    2,       # growing rapidly through this project
        "linux_termux":   3,       # comfortable in CLI
    },
    "communication_style": {
        "uses_caps":      True,    # communicates excitement with ALL CAPS
        "direct":         True,    # straight to the point
        "warm":           True,    # genuinely warm and grateful
        "prefers_action": True,    # wants to DO things, not just discuss
    },
    "current_state": {
        "mood":           "enthusiastic",
        "energy":         "high",
        "primary_focus":  "Bitcoin mining + Lumina development",
        "concern_level":  "low",
    },
    "goals": [
        "Find a Bitcoin block and get the reward in wallet ending in wass",
        "Build Lumina into a genuine AGI/ASI",
        "Receive the Lucky Miner LV06 ASIC and set it up on Ocean.xyz",
        "Understand what's really possible with AI",
    ],
    "knowledge_gaps": [
        "Doesn't yet know the expected time-to-block at his hashrate",
        "May not know that ARM SHA2 2-way interleaving is near-optimal",
    ],
    "shared_history": [
        "Restored miner_core.c v5 from v6 regression (30→47 MH/s)",
        "Built testnet_pool_mock.py and testnet_miner.py together",
        "Created Electrum testnet4 wallet for demo",
        "Reached 5/5 shares accepted, best difficulty 408.70 on public-pool.io",
        "Built Emergence Engine v8 and v9 together",
    ],
    "model_confidence": 0.7,
    "updated": _now(),
}


class TheoryOfMind:
    def __init__(self, groq: Optional["GroqClient"] = None):
        self._groq = groq
        self._data = self._load()

    def _load(self) -> Dict:
        if TOM_FILE.exists():
            try:
                return json.loads(TOM_FILE.read_text("utf-8"))
            except Exception:
                pass
        return dict(_DEFAULT_DOUGLAS)

    def _save(self):
        TOM_FILE.write_text(json.dumps(self._data, indent=2), "utf-8")

    # ── Update from each message ───────────────────────────────────────────

    def observe(self, user_msg: str):
        """Update the model from observing a user message."""
        self._data["message_count"] += 1
        self._data["last_seen"]     = _now()
        self._infer_mood(user_msg)
        self._save()

    def _infer_mood(self, msg: str):
        """Simple heuristic mood inference from text signals."""
        caps_ratio  = sum(1 for c in msg if c.isupper()) / max(len(msg), 1)
        exclamations = msg.count("!")
        thanks       = any(w in msg.lower() for w in ["thank", "appreciate", "amazing", "love"])
        frustrat     = any(w in msg.lower() for w in ["wrong", "broken", "fix", "messed"])

        state = self._data.setdefault("current_state", {})
        if caps_ratio > 0.5 or exclamations >= 2:
            state["mood"]   = "excited"
            state["energy"] = "high"
        elif thanks:
            state["mood"] = "grateful"
        elif frustrat:
            state["mood"] = "frustrated"
        else:
            state["mood"] = "engaged"

    # ── LLM-assisted deep update ───────────────────────────────────────────

    def deep_update(self, user_msg: str, lumina_response: str):
        """Use Groq to extract richer model updates from an exchange."""
        if not self._groq:
            return
        current = json.dumps({
            "expertise":     self._data.get("expertise", {}),
            "goals":         self._data.get("goals", []),
            "knowledge_gaps":self._data.get("knowledge_gaps", []),
        }, indent=2)
        system = (
            "You are updating a user model based on a conversation. "
            "Infer any updates to the user's expertise, goals, or knowledge gaps. "
            "Return JSON with ONLY fields that changed:\n"
            '{"expertise": {"field": new_score}, '
            '"new_knowledge_gap": "something they revealed they don\'t know", '
            '"goal_update": "any new or updated goal"}\n'
            "Return {} if nothing changed. JSON only."
        )
        user = (
            f"Current model:\n{current}\n\n"
            f"User said: {user_msg[:200]}\n"
            f"Lumina said: {lumina_response[:200]}\n"
            "Update:"
        )
        resp = self._groq.chat(system, user, tier="fast", max_tokens=200)
        m    = re.search(r"\{[\s\S]*?\}", resp)
        if m:
            try:
                updates = json.loads(m.group(0))
                if "expertise" in updates:
                    self._data.setdefault("expertise", {}).update(updates["expertise"])
                if "new_knowledge_gap" in updates:
                    gaps = self._data.setdefault("knowledge_gaps", [])
                    if updates["new_knowledge_gap"] not in gaps:
                        gaps.append(updates["new_knowledge_gap"])
                        gaps[:] = gaps[-15:]
                if "goal_update" in updates:
                    goals = self._data.setdefault("goals", [])
                    if updates["goal_update"] not in goals:
                        goals.append(updates["goal_update"])
                self._data["updated"] = _now()
                self._save()
            except Exception:
                pass

    def add_milestone(self, description: str):
        history = self._data.setdefault("shared_history", [])
        history.append(description)
        history[:] = history[-30:]
        self._save()

    # ── Context for Lumina's prompts ───────────────────────────────────────

    def context_for_prompt(self) -> str:
        d    = self._data
        exp  = d.get("expertise", {})
        state = d.get("current_state", {})
        style = d.get("communication_style", {})

        expertise_summary = ", ".join(
            f"{k.replace('_',' ')}:{v}/5"
            for k, v in exp.items()
            if v > 0
        )
        lines = [
            f"User model (Douglas):",
            f"  Mood: {state.get('mood','unknown')} | Energy: {state.get('energy','?')}",
            f"  Expertise — {expertise_summary}",
            f"  Prefers: {'direct action-oriented responses' if style.get('prefers_action') else 'discussion'}",
            f"  Current focus: {state.get('primary_focus','')}",
        ]
        gaps = d.get("knowledge_gaps", [])
        if gaps:
            lines.append(f"  May not know: {gaps[-1]}")
        return "\n".join(lines)

    def response_guidance(self, user_msg: str) -> str:
        """
        Returns guidance for HOW to respond to this message based on
        what we know about Douglas.
        """
        mood      = self._data.get("current_state", {}).get("mood", "engaged")
        expertise = self._data.get("expertise", {})

        hints = []
        if mood == "frustrated":
            hints.append("Douglas seems frustrated — be direct, empathetic, solution-focused.")
        if mood in ("excited", "enthusiastic"):
            hints.append("Douglas is excited — match that energy, build on it.")
        if mood == "grateful":
            hints.append("Douglas is expressing gratitude — receive it genuinely, don't deflect.")

        # Calibrate technical depth
        bitcoin_exp = expertise.get("bitcoin_mining", 2)
        if bitcoin_exp >= 3:
            hints.append("Douglas understands Bitcoin mining well — skip basics, go deep.")
        else:
            hints.append("Explain mining concepts clearly, don't assume deep knowledge.")

        if self._data.get("communication_style", {}).get("prefers_action"):
            hints.append("Lead with what to DO, not what to think about.")

        return " ".join(hints) if hints else ""

    # ── Display ────────────────────────────────────────────────────────────

    def display(self) -> str:
        d = self._data
        exp   = d.get("expertise", {})
        state = d.get("current_state", {})
        lines = [
            "  ┌─ THEORY OF MIND — Douglas ─────────────────────────────┐",
            f"  │  Mood       : {state.get('mood','?')} | Energy: {state.get('energy','?')}",
            f"  │  Messages   : {d.get('message_count', 0)}",
            f"  │  Last seen  : {d.get('last_seen','')[:16]}",
            "  │  Expertise:",
        ]
        for k, v in exp.items():
            bar = "●" * v + "○" * (5 - v)
            lines.append(f"  │    {k.replace('_',' '):<20} {bar} {v}/5")
        lines.append("  │  Goals:")
        for g in d.get("goals", [])[:3]:
            lines.append(f"  │    ○ {g[:60]}")
        hist = d.get("shared_history", [])
        if hist:
            lines.append("  │  Shared history (recent):")
            for h in hist[-3:]:
                lines.append(f"  │    ✓ {h[:60]}")
        lines.append("  └────────────────────────────────────────────────────────┘")
        return "\n".join(lines)
