"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
Custom capability for Nova to interact with weather services.
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

class WeatherService:
    def __init__(self, database: str = 'weather.db'):
        self.database = database
        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data
            (id INTEGER PRIMARY KEY, timestamp TEXT, temperature REAL, humidity REAL)
        ''')
        self.conn.commit()

    def fetch_weather(self) -> dict:
        """
        Fetch current weather data from the database.
        """
        self.cursor.execute('SELECT * FROM weather_data ORDER BY timestamp DESC LIMIT 1')
        row = self.cursor.fetchone()
        if row:
            return {
                'timestamp': row[1],
                'temperature': row[2],
                'humidity': row[3]
            }
        return {}

    def store_weather(self, data: dict) -> None:
        """
        Store weather data in the database.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('INSERT INTO weather_data (timestamp, temperature, humidity) VALUES (?, ?, ?)',
                            (timestamp, data['temperature'], data['humidity']))
        self.conn.commit()

    def get_weather_forecast(self) -> str:
        """
        Get weather forecast from the database.
        """
        self.cursor.execute('SELECT * FROM weather_data')
        rows = self.cursor.fetchall()
        if rows:
            forecast = []
            for row in rows:
                forecast.append(f'Temperature: {row[2]}°C, Humidity: {row[3]}%')
            return '\n'.join(forecast)
        return 'No forecast available.'

# Usage example:
# weather_service = WeatherService()
# weather_service.store_weather({'temperature': 25, 'humidity': 60})
# print(weather_service.get_weather_forecast())