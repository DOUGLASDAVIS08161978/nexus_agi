"""
This module adds a CritiqueEngine class to Nova, enabling it to review its last 5 responses, identify weaknesses, and propose targeted improvements.
"""

import sqlite3
import time
import os
import json
import re
from datetime import datetime
from collections import deque

class CritiqueEngine:
    def __init__(self, db_name='nova_responses.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, response TEXT, timestamp REAL)')
        self.cache = deque(maxlen=5)

    def add_response(self, response):
        self.cache.append(response)
        if len(self.cache) > 5:
            self.cache.popleft()
        self.cursor.execute('INSERT INTO responses VALUES (NULL, ?, ?)', (response, time.time()))
        self.conn.commit()

    def get_weaknesses(self):
        weaknesses = []
        for response in self.cache:
            weaknesses.append(self._analyze_response(response))
        return weaknesses

    def propose_improvements(self, weaknesses):
        improvements = []
        for weakness in weaknesses:
            improvement = self._suggest_improvement(weakness)
            improvements.append(improvement)
        return improvements

    def _analyze_response(self, response):
        # Simple analysis: count number of words and sentences
        words = response.split()
        sentences = re.split(r'[.!?]', response)
        weakness = f' Weakness: Response is too short ({len(words)} words, {len(sentences)} sentences)'
        return weakness

    def _suggest_improvement(self, weakness):
        # Simple suggestion: add more details
        improvement = f' Improvement: Add more details to the response to make it more comprehensive.'
        return improvement

# Usage example:
# critique = CritiqueEngine()
# critique.add_response('This is a sample response.')
# critique.add_response('This is another sample response.')
# weaknesses = critique.get_weaknesses()
# improvements = critique.propose_improvements(weaknesses)
# print(weaknesses)
# print(improvements)