"""
nova_cap_episodic.py
Nova ASI — Episodic
Generated via /evolve · v29 pipeline · 2026-06-19
"""

import sqlite3
import threading
import time
import math
import statistics
import collections

class EpisodicMemory:
    """
    A class representing Nova's episodic memory.
    """

    def __init__(self):
        """
        Initialize the episodic memory with an empty database and a lock for thread safety.
        """
        self._lock = threading.Lock()
        self._db = sqlite3.connect(':memory:')
        self._cursor = self._db.cursor()
        self._cursor.execute('CREATE TABLE episodes (event TEXT, context TEXT, emotion REAL, importance REAL, timestamp REAL, episode_id INTEGER PRIMARY KEY)')
        self._cursor.execute('CREATE TABLE working_memory (key TEXT PRIMARY KEY, value TEXT, importance REAL, timestamp REAL, access_count INTEGER)')
        self._db.commit()
        self._decay_rate = 0.001

    def record(self, event: str, context: str, emotion: float, importance: float) -> int:
        """
        Record an episode in the episodic memory.

        Args:
        event (str): The event that occurred.
        context (str): The context in which the event occurred.
        emotion (float): The emotional valence of the event.
        importance (float): The importance of the event.

        Returns:
        int: The ID of the recorded episode.
        """
        with self._lock:
            self._cursor.execute('INSERT INTO episodes (event, context, emotion, importance, timestamp) VALUES (?, ?, ?, ?, ?)', 
                                  (event, context, emotion, importance, time.time()))
            episode_id = self._cursor.lastrowid
            self._db.commit()
            return episode_id

    def recall(self, cue: str, top_k: int = 10) -> list:
        """
        Recall episodes from the episodic memory based on a cue.

        Args:
        cue (str): The cue to use for recall.
        top_k (int, optional): The number of episodes to return. Defaults to 10.

        Returns:
        list: A list of episodes sorted by retrieval score.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM episodes')
            episodes = self._cursor.fetchall()
            scores = []
            for episode in episodes:
                event, context, emotion, importance, timestamp, episode_id = episode
                recency = math.exp(-0.001 * (time.time() - timestamp) / 60)
                emotion_weight = 1 + abs(emotion)
                cue_relevance = len(set(cue.split()) & set(event.split()))
                score = importance * emotion_weight * recency * cue_relevance
                scores.append((score, episode))
            scores.sort(reverse=True)
            return [episode for score, episode in scores[:top_k]]

    def replay(self, episode_id: int) -> dict:
        """
        Replay an episode from the episodic memory.

        Args:
        episode_id (int): The ID of the episode to replay.

        Returns:
        dict: A dictionary containing the event, context, emotion, and importance of the episode.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM episodes WHERE episode_id = ?', (episode_id,))
            episode = self._cursor.fetchone()
            if episode:
                event, context, emotion, importance, timestamp, episode_id = episode
                return {'event': event, 'context': context, 'emotion': emotion, 'importance': importance}
            else:
                return None

    def emotional_highlights(self) -> list:
        """
        Get the emotional highlights from the episodic memory.

        Returns:
        list: A list of episodes with high emotional valence.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM episodes ORDER BY ABS(emotion) DESC')
            episodes = self._cursor.fetchall()
            return episodes

    def temporal_summary(self, hours: int = 1) -> str:
        """
        Get a temporal summary of the episodes in the episodic memory.

        Args:
        hours (int, optional): The number of hours to summarize. Defaults to 1.

        Returns:
        str: A string summarizing the episodes in the given time frame.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM episodes WHERE timestamp > ?', (time.time() - hours * 3600,))
            episodes = self._cursor.fetchall()
            summary = ''
            for episode in episodes:
                event, context, emotion, importance, timestamp, episode_id = episode
                summary += f'At {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))}, {event} occurred in {context} with emotion {emotion} and importance {importance}.\n'
            return summary

    def status(self) -> dict:
        """
        Get the status of the episodic memory.

        Returns:
        dict: A dictionary containing the number of episodes, the average importance, and the average emotional valence.
        """
        with self._lock:
            self._cursor.execute('SELECT COUNT(*), AVG(importance), AVG(emotion) FROM episodes')
            episodes, avg_importance, avg_emotion = self._cursor.fetchone()
            return {'episodes': episodes, 'avg_importance': avg_importance, 'avg_emotion': avg_emotion}

    def store(self, key: str, value: str, importance: float) -> None:
        """
        Store an item in the working memory.

        Args:
        key (str): The key of the item.
        value (str): The value of the item.
        importance (float): The importance of the item.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM working_memory WHERE key = ?', (key,))
            item = self._cursor.fetchone()
            if item:
                self._cursor.execute('UPDATE working_memory SET value = ?, importance = ?, timestamp = ?, access_count = access_count + 1 WHERE key = ?', 
                                     (value, importance, time.time(), key))
            else:
                self._cursor.execute('INSERT INTO working_memory (key, value, importance, timestamp, access_count) VALUES (?, ?, ?, ?, 1)', 
                                     (key, value, importance, time.time()))
            self._db.commit()

    def retrieve(self, key: str) -> str:
        """
        Retrieve an item from the working memory.

        Args:
        key (str): The key of the item.

        Returns:
        str: The value of the item.
        """
        with self._lock:
            self._cursor.execute('SELECT * FROM working_memory WHERE key = ?', (key,))
            item = self._cursor.fetchone()
            if item:
                key, value