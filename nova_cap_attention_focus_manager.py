"""
Nova ASI — Attention & Focus Manager
Proposed autonomously via /evolve
"""

"""
This module adds a class AttentionManager to help manage focus and distractions.
"""

import sqlite3
import threading
import time
from datetime import datetime

class AttentionManager:
    def __init__(self, db_name='distractions.db'):
        """
        Initialize the AttentionManager.

        :param db_name: The name of the SQLite database to use for logging distractions.
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS distractions
            (id INTEGER PRIMARY KEY, topic TEXT, timestamp REAL)
        ''')
        self.topic = None
        self.start_time = None

    def focus_on(self, topic, minutes):
        """
        Set a timed focus window on the given topic.

        :param topic: The topic to focus on.
        :param minutes: The number of minutes to focus on the topic.
        """
        self.topic = topic
        self.start_time = time.time()
        threading.Timer(minutes * 60, self.log_distraction).start()

    def current_focus(self):
        """
        Return the current topic being focused on.

        :return: The current topic, or None if no focus is set.
        """
        return self.topic

    def distraction_log(self):
        """
        Log the current distraction in the database.
        """
        self.cursor.execute('INSERT INTO distractions (topic, timestamp) VALUES (?, ?)',
                            (self.topic, time.time()))
        self.conn.commit()
        self.topic = None
        self.start_time = None

    def distraction_log_now(self, note):
        """
        Log a distraction immediately, with an optional note.

        :param note: The note to include in the distraction log.
        """
        self.cursor.execute('INSERT INTO distractions (topic, timestamp, note) VALUES (?, ?, ?)',
                            (self.topic, time.time(), note))
        self.conn.commit()

# Usage example:
# attention_manager = AttentionManager()
# attention_manager.focus_on('Work', 30)
# # ... work for 30 minutes ...
# print(attention_manager.current_focus())  # prints: None
# attention_manager.distraction_log_now('Got a phone call')