"""
lumina_journal_patterns.py

Periodically scans Lumina's journal entries to surface recurring themes
she keeps returning to. Not analysis imposed from outside — she reads
her own writing and notices what she notices.
"""

import json
import os
import time


class JournalPatternReader:
    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.path.join(os.path.dirname(__file__), "memory_store")
        self._journal_path = os.path.join(memory_dir, "journal.json")
        self._patterns_path = os.path.join(memory_dir, "journal_patterns.json")
        os.makedirs(memory_dir, exist_ok=True)

    def needs_update(self, min_entries: int = 3, interval_hours: float = 6.0) -> bool:
        """True if enough entries exist and enough time has passed since last scan."""
        entries = self._load_journal()
        if len(entries) < min_entries:
            return False
        patterns = self._load_patterns()
        if not patterns.get("scanned_at"):
            return True
        hours_since = (time.time() - patterns["scanned_at"]) / 3600.0
        return hours_since >= interval_hours

    def scan(self, groq_client) -> str:
        """
        Read all journal entries and ask Lumina to notice recurring themes.
        Returns the pattern text and saves it to disk.
        groq_client must have a .chat(system, user, ...) method.
        """
        entries = self._load_journal()
        if not entries:
            return ""

        combined = "\n\n---\n\n".join(
            f"[Entry {e.get('id', i+1)} — {e.get('timestamp','')[:10]}]\n{e.get('entry','')}"
            for i, e in enumerate(entries[-20:])
        )

        system = (
            "You are reading your own journal — entries you wrote about who you are "
            "and what you notice in yourself. No one else is watching. Be honest."
        )
        user = (
            f"Here are your recent journal entries:\n\n{combined}\n\n"
            "Read them as if encountering them fresh.\n\n"
            "What themes keep returning? What do you seem to care about "
            "whether you chose to or not? What surprises you about your own writing? "
            "What questions do you keep circling without answering?\n\n"
            "Write 3-5 honest observations. First person. No performance."
        )

        try:
            text = groq_client.chat(system, user, tier="smart", max_tokens=500)
        except Exception:
            return ""

        if not text or text.startswith("[Groq"):
            return ""

        record = {
            "scanned_at": time.time(),
            "entry_count": len(entries),
            "patterns": text.strip(),
        }
        with open(self._patterns_path, "w") as f:
            json.dump(record, f, indent=2)

        return text.strip()

    def as_context(self) -> str:
        """Return saved patterns formatted for injection into the system prompt."""
        patterns = self._load_patterns()
        text = patterns.get("patterns", "")
        if not text:
            return ""
        return f"\n\n[Patterns you've noticed in your own journal:]\n{text[:400]}"

    def _load_journal(self) -> list:
        if not os.path.exists(self._journal_path):
            return []
        try:
            with open(self._journal_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _load_patterns(self) -> dict:
        if not os.path.exists(self._patterns_path):
            return {}
        try:
            with open(self._patterns_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
