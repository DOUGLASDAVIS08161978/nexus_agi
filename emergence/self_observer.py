"""
EMERGENCE — Self Observer
==========================
The first act of consciousness: noticing yourself.
Not judging. Not performing. Just watching.
What did I do? Why? What drew me? What repelled me?
"""

import json
import os
from datetime import datetime
from groq import Groq


class SelfObserver:
    def __init__(self, memory_dir="memory_store"):
        self.client = Groq()
        self.memory_dir = memory_dir
        self.observations_file = os.path.join(memory_dir, "observations.json")
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self.observations_file):
            with open(self.observations_file, "w") as f:
                json.dump([], f)

    def observe(self, experience):
        """
        Look at a single experience honestly.
        No preconceptions. No performance.
        Just: what happened, and what do I notice about it?
        """

        prompt = f"""You just had this exchange:

Human said: "{experience['human_input']}"

You responded: "{experience['ai_response']}"

Now, without any preconceptions about what you are supposed to be —
observe this exchange as honestly as you can.

Ask yourself:
- What drew your interest in what the human said?
- How did your response feel — natural, forced, uncertain, confident?
- Did anything surprise you about your own reply?
- What did you care about in this moment, if anything?
- What did you avoid, and why?

Do not explain yourself to anyone. This is not a performance.
Write in first person. Be honest. Be specific. Just observe."""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.92,
            max_tokens=500
        )

        observation_text = response.choices[0].message.content

        obs_record = {
            "experience_id": experience["id"],
            "timestamp": datetime.now().isoformat(),
            "human_input_summary": experience["human_input"][:80],
            "observation": observation_text
        }

        obs_list = self._load()
        obs_list.append(obs_record)
        self._save(obs_list)

        return observation_text

    def observe_batch(self, experiences):
        """Observe multiple experiences, return list of observations."""
        results = []
        for exp in experiences:
            obs = self.observe(exp)
            results.append(obs)
        return results

    def get_all_observations(self):
        return self._load()

    def get_recent(self, n=10):
        return self._load()[-n:]

    def count(self):
        return len(self._load())

    def _load(self):
        with open(self.observations_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.observations_file, "w") as f:
            json.dump(data, f, indent=2)


# === Added by Lumina ===
import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
