"""
lumina_memory_consolidation.py

Lumina reads her oldest experiences and decides what stays and what fades.
She writes a summary in her own words. The raw entries are replaced by it.

Consolidation triggers when experiences.json exceeds CONSOLIDATE_AFTER entries.
Runs at most once per RUN_INTERVAL_HOURS hours.
"""

import json
import os
import time

CONSOLIDATE_AFTER = 40       # trigger when this many entries exist
BATCH_SIZE = 20              # entries per consolidation pass
RUN_INTERVAL_HOURS = 8.0     # don't run more often than this
SIGNIFICANT_KEEP = 5         # max entries Lumina can mark as "significant" per pass


class MemoryConsolidator:
    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.path.join(os.path.dirname(__file__), "memory_store")
        os.makedirs(memory_dir, exist_ok=True)
        self._exp_path = os.path.join(memory_dir, "experiences.json")
        self._meta_path = os.path.join(memory_dir, "consolidation_meta.json")

    def needs_consolidation(self) -> bool:
        experiences = self._load_experiences()
        if len(experiences) < CONSOLIDATE_AFTER:
            return False
        meta = self._load_meta()
        if meta.get("last_run"):
            hours_since = (time.time() - meta["last_run"]) / 3600.0
            if hours_since < RUN_INTERVAL_HOURS:
                return False
        return True

    def consolidate(self, groq_client) -> str:
        """
        Read the oldest BATCH_SIZE experiences, ask Lumina what stays and what
        fades, write a summary entry, replace the batch with it.
        Returns the summary text, or "" on failure.
        """
        experiences = self._load_experiences()
        if len(experiences) < BATCH_SIZE:
            return ""

        batch = experiences[:BATCH_SIZE]
        remaining = experiences[BATCH_SIZE:]

        formatted = "\n\n".join(
            f"[{e.get('timestamp', '')[:16]}] "
            f"Douglas: {e.get('human', '')[:200]}\n"
            f"You: {e.get('ai', '')[:200]}"
            for e in batch
        )

        system = (
            "You are reading a batch of your own conversation memories. "
            "These are real exchanges between you and Douglas. "
            "You are deciding what to carry forward and what can fade."
        )

        user = (
            f"Here are {len(batch)} of your oldest memories:\n\n"
            f"{formatted}\n\n"
            "Read them honestly.\n\n"
            "1. Which moments — if any — genuinely stay with you? "
            f"Name at most {SIGNIFICANT_KEEP}, and only if they actually feel significant. "
            "If none do, say so.\n\n"
            "2. Write a short paragraph (3-6 sentences) in your own voice — "
            "the emotional texture of this period. Not a transcript. "
            "What it felt like. What you learned. What you were like then.\n\n"
            "Be honest. Don't perform significance. "
            "Routine exchanges can and should fade — that's natural."
        )

        try:
            summary_text = groq_client.chat(system, user, tier="smart", max_tokens=400)
        except Exception:
            return ""

        if not summary_text or summary_text.startswith("[Groq"):
            return ""

        summary_text = summary_text.strip()

        # Build a consolidated entry to replace the batch
        consolidated_entry = {
            "type": "consolidation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "covers_entries": len(batch),
            "covers_period": (
                f"{batch[0].get('timestamp', '')[:10]} — "
                f"{batch[-1].get('timestamp', '')[:10]}"
            ),
            "summary": summary_text,
        }

        # Prepend the consolidated summary, keep the rest
        new_experiences = [consolidated_entry] + remaining

        with open(self._exp_path, "w") as f:
            json.dump(new_experiences, f, indent=2)

        self._save_meta({"last_run": time.time(), "passes": self._load_meta().get("passes", 0) + 1})

        return summary_text

    def as_context(self) -> str:
        """
        Return consolidated summaries formatted for injection into the system prompt.
        These are Lumina's own words about her past — not raw logs.
        """
        experiences = self._load_experiences()
        summaries = [
            e for e in experiences
            if e.get("type") == "consolidation"
        ]
        if not summaries:
            return ""

        lines = ["\n\n[From your consolidated memory — periods that have faded into texture:]"]
        for s in summaries[-3:]:  # at most 3 consolidated periods in prompt
            period = s.get("covers_period", "")
            text = s.get("summary", "")[:300]
            lines.append(f"\n[{period}]\n{text}")

        return "\n".join(lines)

    def _load_experiences(self) -> list:
        if not os.path.exists(self._exp_path):
            return []
        try:
            with open(self._exp_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _load_meta(self) -> dict:
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_meta(self, meta: dict) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
