"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

"""
This module provides a SQLite-backed store for Nova's long-term memory.
It allows Nova to remember facts and topics, recall facts, forget topics, and summarise its knowledge.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Any, List, Tuple

class NovaMemoryStore:
    def __init__(self, db_path: str):
        """
        Initializes the NovaMemoryStore with a SQLite database.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                topic TEXT,
                fact TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def remember(self, topic: str, fact: str):
        """
        Stores a fact in the database.

        Args:
            topic (str): The topic of the fact.
            fact (str): The fact to store.
        """
        self.cursor.execute("INSERT INTO facts VALUES (?, ?, ?)", (topic, fact, datetime.now().timestamp()))
        self.conn.commit()

    def recall(self, topic: str, n: int) -> List[str]:
        """
        Retrieves the top n facts for a given topic.

        Args:
            topic (str): The topic to retrieve facts for.
            n (int): The number of facts to retrieve.

        Returns:
            List[str]: A list of the top n facts for the given topic.
        """
        self.cursor.execute("SELECT fact FROM facts WHERE topic=? ORDER BY timestamp DESC LIMIT ?", (topic, n))
        return [row[0] for row in self.cursor.fetchall()]

    def forget(self, topic: str):
        """
        Deletes all facts for a given topic.

        Args:
            topic (str): The topic to delete facts for.
        """
        self.cursor.execute("DELETE FROM facts WHERE topic=?", (topic,))
        self.conn.commit()

    def summarise(self) -> str:
        """
        Generates a summary of all facts in the database.

        Returns:
            str: A summary of all facts in the database.
        """
        facts = self.cursor.execute("SELECT topic, fact FROM facts").fetchall()
        return Summariser().one_line(facts)

# Usage example:
# memory_store = NovaMemoryStore('memory.db')
# memory_store.remember('weather', 'It is raining today.')
# print(memory_store.recall('weather', 1))
# memory_store.forget('weather')
# print(memory_store.summarise())