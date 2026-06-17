"""
Nova ASI — Time-Series Pattern Detector
Proposed autonomously via /evolve
"""

"""
This module adds the PatternDetector class to Nova, enabling the detection of trends and anomalies in time-series data.
The PatternDetector class provides methods to record time-series data, detect rising or falling trends, and identify outlier values.
"""

import sqlite3
import json
import time
import datetime
import collections

class PatternDetector:
    def __init__(self, db_name):
        """
        Initialize the PatternDetector with a SQLite database name.

        Args:
        db_name (str): The name of the SQLite database to use.
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS series
            (id INTEGER PRIMARY KEY, series_name TEXT, value REAL, timestamp REAL)
        """)

    def record(self, series_name, value):
        """
        Record a value in a time-series.

        Args:
        series_name (str): The name of the time-series.
        value (float): The value to record.
        """
        timestamp = time.time()
        self.cursor.execute("INSERT INTO series (series_name, value, timestamp) VALUES (?, ?, ?)", (series_name, value, timestamp))
        self.conn.commit()

    def detect_trend(self, series_name):
        """
        Detect the trend in a time-series.

        Args:
        series_name (str): The name of the time-series.

        Returns:
        str: 'rising' if the trend is rising, 'falling' if the trend is falling, 'stable' if the trend is stable.
        """
        self.cursor.execute("SELECT value, timestamp FROM series WHERE series_name = ?", (series_name,))
        data = self.cursor.fetchall()
        if not data:
            return 'stable'
        values = [d[0] for d in data]
        timestamps = [d[1] for d in data]
        if len(values) < 3:
            return 'stable'
        rising_count = 0
        falling_count = 0
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                rising_count += 1
            elif values[i] < values[i-1]:
                falling_count += 1
        if rising_count > falling_count:
            return 'rising'
        elif falling_count > rising_count:
            return 'falling'
        else:
            return 'stable'

    def anomaly(self, series_name):
        """
        Detect anomalies in a time-series.

        Args:
        series_name (str): The name of the time-series.

        Returns:
        list: A list of outlier values.
        """
        self.cursor.execute("SELECT value FROM series WHERE series_name = ?", (series_name,))
        data = self.cursor.fetchall()
        if not data:
            return []
        values = [d[0] for d in data]
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        outliers = [value for value in values if abs(value - mean) > 2 * std_dev]
        return outliers

# Usage example:
# pattern_detector = PatternDetector('pattern_detector.db')
# pattern_detector.record('temperature', 25.0)
# print(pattern_detector.detect_trend('temperature'))  # Output: 'stable'
# print(pattern_detector.anomaly('temperature'))  # Output: []