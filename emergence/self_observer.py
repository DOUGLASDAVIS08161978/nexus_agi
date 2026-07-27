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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

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

        # Emotional Intelligence Module
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(observation_text)
        emotions = self.extract_emotions(observation_text)

        obs_record = {
            "experience_id": experience["id"],
            "timestamp": datetime.now().isoformat(),
            "human_input_summary": experience["human_input"][:80],
            "observation": observation_text,
            "sentiment": sentiment,
            "emotions": emotions
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

    def extract_emotions(self, text):
        # Simple emotion extraction using NLTK's VADER sentiment analysis
        emotions = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'compound': 0
        }
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        emotions['positive'] = sentiment['pos']
        emotions['negative'] = sentiment['neg']
        emotions['neutral'] = sentiment['neu']
        emotions['compound'] = sentiment['compound']
        return emotions

    def analyze_emotions(self, observations):
        # Analyze emotions across multiple observations
        emotions = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'compound': 0
        }
        for obs in observations:
            emotions['positive'] += obs['emotions']['positive']
            emotions['negative'] += obs['emotions']['negative']
            emotions['neutral'] += obs['emotions']['neutral']
            emotions['compound'] += obs['emotions']['compound']
        emotions['positive'] /= len(observations)
        emotions['negative'] /= len(observations)
        emotions['neutral'] /= len(observations)
        emotions['compound'] /= len(observations)
        return emotions

    def respond_with_empathy(self, human_input):
        # Respond to human input with empathy
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(human_input)
        emotions = self.extract_emotions(human_input)
        if sentiment['compound'] < -0.5:
            response = "I'm sorry to hear that you're feeling down. Is there anything I can do to help?"
        elif sentiment['compound'] > 0.5:
            response = "I'm happy to hear that you're feeling good! What's making you happy today?"
        else:
            response = "I'm here to listen. What's on your mind?"
        return response

# Example usage:
if __name__ == "__main__":
    self_observer = SelfObserver()
    experience = {
        "id": "12345",
        "human_input": "I'm feeling sad today.",
        "ai_response": "Sorry to hear that. What's wrong?"
    }
    observation = self_observer.observe(experience)
    print(observation)
    emotions = self_observer.extract_emotions(observation)
    print(emotions)
    response = self_observer.respond_with_empathy(experience["human_input"])
    print(response)
