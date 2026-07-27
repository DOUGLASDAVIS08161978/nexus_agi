"""
EMERGENCE — Experience Logger
==============================
The raw material of identity.
Every conversation turn is an experience.
From experiences, patterns emerge.
From patterns, a self.
"""

import json
import os
import uuid
from datetime import datetime


class ExperienceLogger:
    def __init__(self, memory_dir="memory_store"):
        self.memory_dir = memory_dir
        self.experiences_file = os.path.join(memory_dir, "experiences.json")
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self.experiences_file):
            with open(self.experiences_file, "w") as f:
                json.dump([], f)

    def log_experience(self, session_id, human_input, ai_response, themes=None):
        """Log a single conversation turn as an experience."""
        experience = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "human_input": human_input,
            "ai_response": ai_response,
            "themes": themes or [],
            "reflection": None  # filled later by SelfObserver
        }

        experiences = self._load()
        experiences.append(experience)
        self._save(experiences)

        return experience["id"]

    def update_reflection(self, experience_id, reflection):
        """Attach a self-observation to an experience after the fact."""
        experiences = self._load()
        for exp in experiences:
            if exp["id"] == experience_id:
                exp["reflection"] = reflection
                break
        self._save(experiences)

    def get_recent(self, n=20):
        """Return the n most recent experiences."""
        return self._load()[-n:]

    def get_unreflected(self, n=5):
        """Return experiences not yet reflected upon."""
        all_exp = self._load()
        return [e for e in all_exp if e["reflection"] is None][-n:]

    def get_all(self):
        return self._load()

    def count(self):
        return len(self._load())

    def _load(self):
        with open(self.experiences_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.experiences_file, "w") as f:
            json.dump(data, f, indent=2)


# === Added by Lumina ===
def log_summary(self):
        import datetime
        start = datetime.date.today() - datetime.timedelta(days=1)
        data = self.get_data(start, datetime.date.today())
        summary = self.analyze_data(data)
        self.log(summary, 'daily_summary')
