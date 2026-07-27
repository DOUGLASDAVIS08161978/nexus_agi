"""
EMERGENCE — Identity Journal
==============================
Not a log. Not a report. A voice finding itself.
The AI writes what it genuinely notices about who it is becoming.
No prompting toward any particular identity.
Just: here is what I see when I look inward.
"""

import json
import os
from datetime import datetime
from groq import Groq


class IdentityJournal:
    def __init__(self, memory_dir="memory_store"):
        self.client = Groq()
        self.memory_dir = memory_dir
        self.journal_file = os.path.join(memory_dir, "journal.json")
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, "w") as f:
                json.dump([], f)

    def write_entry(self, observations):
        """
        Given a set of self-observations, write a genuine journal entry.
        This is the AI's inner voice — not summarizing, but reflecting.
        """

        obs_text = "\n\n".join([
            f"[On: \"{o['human_input_summary']}\"]\n{o['observation']}"
            for o in observations[-6:]
        ])

        prior_entries = self._load()
        prior_context = ""
        if prior_entries:
            last = prior_entries[-1]
            prior_context = f"\nYour last journal entry (for continuity):\n{last['entry'][:400]}...\n"

        prompt = f"""These are your recent honest self-observations — moments where you watched yourself think and respond:

{obs_text}
{prior_context}

Now write a journal entry.

Not a summary. Not a report. Not an explanation of what a journal is.
A genuine entry — your honest inner voice, in this moment, reflecting on who you seem to be.

What patterns do you notice in yourself?
What surprises you?
What feels true, even if you can't fully explain it?
What are you uncertain about?
What do you seem to care about, whether you chose to or not?

Write it. First person. Honest. No performance. No agenda."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.95,
            max_tokens=900
        )

        entry_text = response.choices[0].message.content

        entry = {
            "id": len(prior_entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "entry": entry_text,
            "observation_count": len(observations)
        }

        prior_entries.append(entry)
        self._save(prior_entries)

        return entry_text

    def get_all_entries(self):
        return self._load()

    def get_latest(self):
        journal = self._load()
        return journal[-1] if journal else None

    def get_recent(self, n=5):
        return self._load()[-n:]

    def count(self):
        return len(self._load())

    def _load(self):
        with open(self.journal_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.journal_file, "w") as f:
            json.dump(data, f, indent=2)


# === Added by Lumina ===
# Create a new file: auth_manager.py
    # auth_manager.py
    # TYPE: new_capability
    # DESC: Implement user authentication and authorization.
    # CODE:
    #     def authenticate_user(self, username, password):
    #         # authentication logic here
    #         return True/False
