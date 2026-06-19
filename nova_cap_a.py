"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
Custom capability for Nova to interact with the weather.

This module provides a class WeatherInterface that allows Nova to interact with the weather.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List

class WeatherInterface:
    def __init__(self, db_path: str):
        """
        Initialize the WeatherInterface.

        Args:
            db_path (str): Path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def update_weather(self, weather_data: Dict):
        """
        Update the weather data in the database.

        Args:
            weather_data (Dict): Dictionary containing the weather data.
        """
        self.cursor.execute("INSERT INTO weather (timestamp, weather) VALUES (?, ?)", (datetime.now(), json.dumps(weather_data)))
        self.conn.commit()

    def get_latest_weather(self) -> Dict:
        """
        Get the latest weather data from the database.

        Returns:
            Dict: Dictionary containing the latest weather data.
        """
        self.cursor.execute("SELECT weather FROM weather ORDER BY timestamp DESC LIMIT 1")
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        else:
            return {}

    def check_weather_condition(self) -> str:
        """
        Check the current weather condition.

        Returns:
            str: The current weather condition.
        """
        weather_data = self.get_latest_weather()
        if weather_data:
            return weather_data['condition']
        else:
            return "Unknown"

# Usage example:
# weather_interface = WeatherInterface('weather.db')
# weather_interface.update_weather({'condition': 'Sunny', 'temperature': 25})
# print(weather_interface.check_weather_condition())  # Output: Sunny