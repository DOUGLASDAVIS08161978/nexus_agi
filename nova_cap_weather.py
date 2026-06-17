"""
Nova ASI — Weather Checking Tool
Built autonomously by Nova via /build
Requires a free API key from openweathermap.org
"""

import json
import os
from datetime import datetime

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


class WeatherChecker:
    """Fetch current weather and forecasts for any city using OpenWeatherMap."""

    BASE = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str = "", location: str = ""):
        self.api_key  = api_key or os.getenv("OPENWEATHER_API_KEY", "")
        self.location = location
        self._cache: dict = {}

    def _get(self, endpoint: str, params: dict) -> dict:
        if not _REQUESTS:
            return {"error": "requests library not installed"}
        if not self.api_key:
            return {"error": "No API key. Get one free at openweathermap.org and set OPENWEATHER_API_KEY in .env"}
        params["appid"] = self.api_key
        params["units"] = "imperial"
        try:
            r = _req.get(f"{self.BASE}/{endpoint}", params=params, timeout=8)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def current(self, city: str = "") -> str:
        """Get current weather for a city. Returns a readable summary string."""
        city = city or self.location or "Memphis"
        data = self._get("weather", {"q": city})
        if "error" in data:
            return data["error"]
        if data.get("cod") != 200:
            return f"City not found: {city}"
        w    = data["weather"][0]["description"].title()
        temp = data["main"]["temp"]
        feel = data["main"]["feels_like"]
        hum  = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        return (f"{city.title()}: {w}, {temp:.0f}°F (feels {feel:.0f}°F) "
                f"| Humidity {hum}% | Wind {wind} mph")

    def forecast(self, city: str = "", days: int = 3) -> list:
        """Return a list of daily forecast summaries."""
        city = city or self.location or "Memphis"
        data = self._get("forecast", {"q": city, "cnt": days * 8})
        if "error" in data or "list" not in data:
            return [data.get("error", "Forecast unavailable")]
        seen, results = set(), []
        for item in data["list"]:
            day = datetime.fromtimestamp(item["dt"]).strftime("%A %b %d")
            if day not in seen:
                seen.add(day)
                desc = item["weather"][0]["description"].title()
                hi   = item["main"]["temp_max"]
                lo   = item["main"]["temp_min"]
                results.append(f"{day}: {desc}, Hi {hi:.0f}°F / Lo {lo:.0f}°F")
            if len(results) >= days:
                break
        return results

    def is_raining(self, city: str = "") -> bool:
        """Return True if it is currently raining."""
        summary = self.current(city)
        return any(w in summary.lower() for w in ["rain", "drizzle", "shower"])

    def summary(self, city: str = "") -> str:
        """One-line current + 3-day forecast."""
        cur  = self.current(city)
        fore = self.forecast(city, days=3)
        return cur + "\n" + "\n".join(f"  {f}" for f in fore)
