"""
Nova ASI — weather checking tool
Built to order by Nova via /build
"""

"""
Weather Checking Tool
=====================

This module adds a WeatherChecker class to Nova's capabilities. It uses the OpenWeatherMap API to fetch current weather data.
"""

import os
import json
import requests
from datetime import datetime

class WeatherChecker:
    def __init__(self, api_key: str, location: str):
        """
        Initialize the WeatherChecker.

        Args:
        api_key (str): OpenWeatherMap API key.
        location (str): Location for which to fetch weather data.
        """
        self.api_key = api_key
        self.location = location

    def get_current_weather(self) -> dict:
        """
        Fetch current weather data from OpenWeatherMap.

        Returns:
        dict: Current weather data.
        """
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={self.api_key}"
        response = requests.get(url)
        return response.json()

    def get_weather_forecast(self, days: int = 5) -> list:
        """
        Fetch weather forecast data from OpenWeatherMap.

        Args:
        days (int, optional): Number of days to fetch forecast for. Defaults to 5.

        Returns:
        list: Weather forecast data.
        """
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={self.location}&appid={self.api_key}"
        response = requests.get(url)
        forecast_data = response.json()["list"]
        return [{"date": datetime.fromtimestamp(f["dt"]).strftime("%Y-%m-%d"), **f} for f in forecast_data[:days*8]]

    def check_weather_condition(self, condition: str) -> bool:
        """
        Check if the current weather condition matches the given condition.

        Args:
        condition (str): Weather condition to check for.

        Returns:
        bool: True if the current weather condition matches, False otherwise.
        """
        current_weather = self.get_current_weather()
        return condition.lower() in current_weather["weather"][0]["description"].lower()

# Usage example:
# weather_checker = WeatherChecker("YOUR_API_KEY", "London")
# print(weather_checker.get_current_weather())
# print(weather_checker.get_weather_forecast(days=3))
