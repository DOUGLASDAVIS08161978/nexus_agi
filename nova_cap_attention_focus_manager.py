"""
Nova ASI — Attention & Focus Manager
Proposed autonomously via /evolve
"""

"""
Module for managing attention and focus in a superintelligent AI.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

class AttentionManager:
    def __init__(self) -> None:
        """
        Initialize the attention manager with an empty focus window and distraction log.
        """
        self.conn = sqlite3.connect('attention_manager.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS focus (topic TEXT, start_time TEXT, end_time TEXT)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS distractions (note TEXT, time TEXT)')
        self.conn.commit()
        self.current_focus_topic: Optional[str] = None
        self.current_focus_start_time: Optional[datetime] = None

    def focus_on(self, topic: str, minutes: int) -> None:
        """
        Set a timed focus window on a specific topic.
        """
        try:
            if self.current_focus_topic:
                self.cursor.execute('UPDATE focus SET end_time = ? WHERE topic = ?', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.current_focus_topic))
            self.cursor.execute('INSERT INTO focus (topic, start_time, end_time) VALUES (?, ?, ?)', (topic, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (datetime.now() + timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')))
            self.conn.commit()
            self.current_focus_topic = topic
            self.current_focus_start_time = datetime.now()
        except sqlite3.Error as e:
            print(f"Error setting focus: {e}")

    def current_focus(self) -> Optional[str]:
        """
        Return the currently active topic of focus.
        """
        try:
            if self.current_focus_topic and self.current_focus_start_time and datetime.now() < self.current_focus_start_time + timedelta(minutes=30):
                return self.current_focus_topic
            else:
                self.cursor.execute('SELECT topic FROM focus WHERE end_time > ?', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
                result = self.cursor.fetchone()
                if result:
                    self.current_focus_topic = result[0]
                    return self.current_focus_topic
                else:
                    self.current_focus_topic = None
                    return None
        except sqlite3.Error as e:
            print(f"Error getting current focus: {e}")

    def distraction_log(self, note: str) -> None:
        """
        Record an interruption or distraction.
        """
        try:
            self.cursor.execute('INSERT INTO distractions (note, time) VALUES (?, ?)', (note, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging distraction: {e}")

# Usage: obj = AttentionManager() | result = obj.focus_on("topic", 30)