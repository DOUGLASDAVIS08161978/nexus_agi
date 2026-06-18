"""
Nova ASI — Emotion & Sentiment Tracker
Proposed autonomously via /evolve
"""

"""
EmotionTracker class for tracking and analyzing emotions in text.
"""

import sqlite3
from typing import Tuple

class EmotionTracker:
    def __init__(self) -> None:
        """Initialize the EmotionTracker with an empty database."""
        self.conn = sqlite3.connect('emotions.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS emotions (id INTEGER PRIMARY KEY, text TEXT, valence REAL, arousal REAL)')
        self.conn.commit()

    def analyse(self, text: str) -> Tuple[float, float]:
        """Analyse the given text and return its valence and arousal scores."""
        # Simple sentiment analysis using keyword matching
        positive_words = ['happy', 'joy', 'love']
        negative_words = ['sad', 'angry', 'fear']
        valence = 0.5
        arousal = 0.5
        for word in text.split():
            if word in positive_words:
                valence += 0.1
            elif word in negative_words:
                valence -= 0.1
            if word in ['exciting', 'thrilling']:
                arousal += 0.1
            elif word in ['boring', 'dull']:
                arousal -= 0.1
        return valence, arousal

    def track(self, text: str) -> None:
        """Track the given text and store its valence and arousal scores in the database."""
        valence, arousal = self.analyse(text)
        self.cursor.execute('INSERT INTO emotions (text, valence, arousal) VALUES (?, ?, ?)', (text, valence, arousal))
        self.conn.commit()

    def mood_arc(self, n: int) -> list:
        """Return the last n mood readings from the database."""
        self.cursor.execute('SELECT valence, arousal FROM emotions ORDER BY id DESC LIMIT ?', (n,))
        return self.cursor.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

# Usage: obj = EmotionTracker() | result = obj.analyse("I'm feeling happy today")