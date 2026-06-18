"""
Nova ASI — Semantic Memory Indexer
Proposed autonomously via /evolve
"""

class for storing and retrieving semantic information from text and metadata.

This class uses keyword extraction to index text and metadata, allowing for efficient search and retrieval of relevant information.
It also includes a method for pruning old entries from the index.

Usage: obj = SemanticIndex() | result = obj.index(text, metadata) | result = obj.search(query)
"""

import os
import json
import sqlite3
import time
import re
import random
import threading
import datetime
import collections
import math
import statistics
import hashlib
import pathlib
from typing import Dict, List, Tuple

class SemanticIndex:
    def __init__(self):
        # Initialize the SQLite database
        self.db_path = os.path.join(pathlib.Path.home(), 'nexus_agi', 'semantic_index.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_index (
                id INTEGER PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                keywords TEXT,
                timestamp REAL
            )
        ''')
        self.conn.commit()

        # Initialize the lock for thread-safe state
        self.lock = threading.Lock()

    def index(self, text: str, metadata: Dict) -> None:
        """
        Index the given text and metadata.

        Args:
            text (str): The text to be indexed.
            metadata (Dict): The metadata associated with the text.

        Returns:
            None
        """
        with self.lock:
            # Extract keywords from the text
            keywords = self.extract_keywords(text)

            # Insert the text, metadata, and keywords into the database
            self.cursor.execute('''
                INSERT INTO semantic_index (text, metadata, keywords, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (text, json.dumps(metadata), json.dumps(keywords), time.time()))
            self.conn.commit()

    def search(self, query: str) -> List[Tuple]:
        """
        Search the index for matches to the given query.

        Args:
            query (str): The query to be searched.

        Returns:
            List[Tuple]: A list of tuples containing the text, metadata, and keywords that match the query.
        """
        with self.lock:
            # Extract keywords from the query
            query_keywords = self.extract_keywords(query)

            # Search the database for matches
            self.cursor.execute('''
                SELECT text, metadata, keywords FROM semantic_index
                WHERE keywords LIKE ?
            ''', ('%' + '%'.join(query_keywords) + '%',))
            results = self.cursor.fetchall()

            # Rank the matches based on the number of matching keywords
            ranked_results = []
            for result in results:
                keywords = json.loads(result[2])
                match_count = sum(1 for keyword in query_keywords if keyword in keywords)
                ranked_results.append((result[0], result[1], match_count))

            # Sort the results by rank
            ranked_results.sort(key=lambda x: x[2], reverse=True)

            return ranked_results

    def forget_old(self, days: int) -> None:
        """
        Prune old entries from the index.

        Args:
            days (int): The number of days to keep entries for.

        Returns:
            None
        """
        with self.lock:
            # Calculate the timestamp for the oldest entries to keep
            oldest_timestamp = time.time() - days * 24 * 60 * 60

            # Prune old entries from the database
            self.cursor.execute('''
                DELETE FROM semantic_index
                WHERE timestamp < ?
            ''', (oldest_timestamp,))
            self.conn.commit()

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract keywords from the given text.

        Args:
            text (str): The text to extract keywords from.

        Returns:
            List[str]: A list of keywords extracted from the text.
        """
        # Tokenize the text
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Remove stop words
        stop_words = set(['the', 'and', 'a', 'an', 'is', 'in', 'it', 'of', 'to'])
        tokens = [token for token in tokens if token not in stop_words]

        # Calculate the frequency of each token
        frequency = collections.Counter(tokens)

        # Select the top 10 keywords
        keywords = [token for token, freq in frequency.most_common(10)]

        return keywords
**Usage:**

obj = SemanticIndex()
obj.index('This is a sample text.', {'author': 'John Doe', 'date': '2022-01-01'})
results = obj.search('sample text')
print(results)
obj.forget_old(30)
This code defines a `SemanticIndex` class that uses SQLite to store and retrieve semantic information from text and metadata. The `index` method extracts keywords from the text and inserts them into the database along with the text and metadata. The `search` method searches the database for matches to the given query and returns a list of tuples containing the text, metadata, and keywords that match the query. The `forget_old` method prunes old entries from the database based on the number of days to keep entries for. The `extract_keywords` method is a static method that extracts keywords from the given text.