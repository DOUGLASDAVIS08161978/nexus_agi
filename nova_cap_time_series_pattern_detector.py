"""
Nova ASI — Time-Series Pattern Detector
Proposed autonomously via /evolve
"""

"""
Module for time-series pattern detection.
"""

import sqlite3
import json
import os
import collections
import threading

class PatternDetector:
    def __init__(self, db_name: str):
        """
        Initialize the PatternDetector with a database name.

        Args:
            db_name (str): The name of the SQLite database file.
        """
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                series_name TEXT PRIMARY KEY,
                values TEXT
            )
        """)

    def record(self, series_name: str, value: float):
        """
        Record a value for a given series.

        Args:
            series_name (str): The name of the series.
            value (float): The value to record.
        """
        self.cursor.execute("SELECT values FROM patterns WHERE series_name = ?", (series_name,))
        existing_values = self.cursor.fetchone()
        if existing_values:
            existing_values = json.loads(existing_values[0])
            existing_values.append(value)
            self.cursor.execute("UPDATE patterns SET values = ? WHERE series_name = ?", (json.dumps(existing_values), series_name))
        else:
            self.cursor.execute("INSERT INTO patterns VALUES (?, ?)", (series_name, json.dumps([value])))
        self.conn.commit()

    def detect_trend(self, series_name: str) -> str:
        """
        Detect the trend of a given series.

        Args:
            series_name (str): The name of the series.

        Returns:
            str: 'rising', 'falling', or 'stable' depending on the trend.
        """
        self.cursor.execute("SELECT values FROM patterns WHERE series_name = ?", (series_name,))
        values = json.loads(self.cursor.fetchone()[0])
        if len(values) < 3:
            return 'stable'
        prev_diff = values[1] - values[0]
        curr_diff = values[-1] - values[-2]
        if prev_diff * curr_diff > 0:
            return 'falling' if prev_diff < 0 else 'rising'
        else:
            return 'stable'

    def anomaly(self, series_name: str) -> list:
        """
        Detect anomalies in a given series.

        Args:
            series_name (str): The name of the series.

        Returns:
            list: A list of outlier values.
        """
        self.cursor.execute("SELECT values FROM patterns WHERE series_name = ?", (series_name,))
        values = json.loads(self.cursor.fetchone()[0])
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return [x for x in values if abs(x - mean) > 2 * std_dev]

# Usage example:
# pattern_detector = PatternDetector('patterns.db')
# pattern_detector.record('temperature', 25.0)
# print(pattern_detector.detect_trend('temperature'))  # Output: 'stable'
# print(pattern_detector.anomaly('temperature'))  # Output: []
