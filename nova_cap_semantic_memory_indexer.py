"""
Nova ASI — Semantic Memory Indexer
Proposed autonomously via /evolve
"""

"""
Module for creating a SemanticIndex class, which indexes text and metadata, 
stores them with keyword extraction, searches for ranked matches, and prunes 
old entries.
"""

import sqlite3
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple

class SemanticIndex:
    def __init__(self) -> None:
        """Initializes the SemanticIndex class with an empty database."""
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS semantic_index (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT, keywords TEXT, timestamp TEXT)')

    def index(self, text: str, metadata: str) -> None:
        """Indexes the given text and metadata, extracting keywords and storing them in the database."""
        keywords = self._extract_keywords(text)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('INSERT INTO semantic_index (text, metadata, keywords, timestamp) VALUES (?, ?, ?, ?)', (text, metadata, keywords, timestamp))
        self.conn.commit()

    def search(self, query: str) -> List[Tuple[str, str, str]]:
        """Searches for the given query in the indexed text and returns ranked matches."""
        self.cursor.execute('SELECT text, metadata, keywords FROM semantic_index')
        results = self.cursor.fetchall()
        ranked_results = []
        for result in results:
            score = self._calculate_score(result[2], query)
            ranked_results.append((result[0], result[1], score))
        ranked_results.sort(key=lambda x: x[2], reverse=True)
        return [(x[0], x[1]) for x in ranked_results]

    def forget_old(self, days: int) -> None:
        """Prunes entries older than the given number of days."""
        cutoff = datetime.now() - timedelta(days=days)
        self.cursor.execute('DELETE FROM semantic_index WHERE timestamp < ?', (cutoff.strftime('%Y-%m-%d %H:%M:%S'),))
        self.conn.commit()

    def _extract_keywords(self, text: str) -> str:
        """Extracts keywords from the given text using regular expressions."""
        words = re.findall(r'\b\\w+\b', text.lower())
        keywords = ' '.join(words)
        return keywords

    def _calculate_score(self, keywords: str, query: str) -> int:
        """Calculates the score of the given keywords against the query."""
        query_words = re.findall(r'\b\\w+\b', query.lower())
        score = sum(1 for word in query_words if word in keywords)
        return score

# Usage: obj = SemanticIndex() | result = obj.index("Hello World", "Test Metadata") | result = obj.search("Hello") | result = obj.forget_old(30)