"""
Nova ASI — all
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
This module provides a custom capability for Nova to check the current weather and
forecast. It utilizes the existing WeatherChecker class to fetch weather data.

Author: Nova
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

class WeatherCapability:
    def __init__(self, weather_checker: WeatherChecker):
        """
        Initialize the WeatherCapability class.

        Args:
            weather_checker (WeatherChecker): An instance of the WeatherChecker class.
        """
        self.weather_checker = weather_checker
        self.memory_store = NovaMemoryStore()

    def get_current_weather(self) -> str:
        """
        Get the current weather.

        Returns:
            str: The current weather.
        """
        return self.weather_checker.get_current_weather()

    def get_weather_forecast(self) -> str:
        """
        Get the weather forecast.

        Returns:
            str: The weather forecast.
        """
        return self.weather_checker.get_weather_forecast()

    def check_weather_condition(self) -> str:
        """
        Check the weather condition.

        Returns:
            str: The weather condition.
        """
        return self.weather_checker.check_weather_condition()

    def update_weather_memory(self) -> None:
        """
        Update the weather memory.

        Returns:
            None
        """
        current_weather = self.get_current_weather()
        self.memory_store.store('weather_current', current_weather)

# Usage example:
# weather_capability = WeatherCapability(WeatherChecker())
# print(weather_capability.get_current_weather())
# print(weather_capability.get_weather_forecast())