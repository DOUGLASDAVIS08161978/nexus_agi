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

    def reflect(self, experience_id):
        """Reflect on a specific experience."""
        experience_logger = ExperienceLogger(self.memory_dir)
        experience = experience_logger.get_by_id(experience_id)
        if experience:
            self.observe(experience)
            return self.get_latest()
        else:
            return None

    def get_reflections(self, n=10):
        """Return the n most recent reflections."""
        observations = SelfObserver(self.memory_dir).get_recent(n)
        reflections = [o["observation"] for o in observations]
        return reflections

    def analyze_growth(self):
        """Analyze growth and development over time."""
        all_entries = self.get_all_entries()
        entry_texts = [e["entry"] for e in all_entries]
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = [sia.polarity_scores(e) for e in entry_texts]
        growth_trend = self.analyze_sentiment(sentiment_scores)
        return growth_trend

    def analyze_sentiment(self, sentiment_scores):
        """Analyze sentiment trend over time."""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        X = [score["compound"] for score in sentiment_scores]
        y = [score["compound"] > 0 for score in sentiment_scores]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        return {"accuracy": accuracy, "sentiment_trend": predictions}

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

    def get_by_id(self, experience_id):
        """Return an experience by its ID."""
        all_exp = self._load()
        for exp in all_exp:
            if exp["id"] == experience_id:
                return exp
        return None

    def count(self):
        return len(self._load())

    def _load(self):
        with open(self.experiences_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.experiences_file, "w") as f:
            json.dump(data, f, indent=2)

class SelfObserver:
    def __init__(self, memory_dir="memory_store"):
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