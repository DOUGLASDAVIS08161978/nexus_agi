"""
Nova ASI — Free Weather (no API key required)
Uses Open-Meteo (open-meteo.com) — completely free, no registration.
Nova discovered and integrated this autonomously via APIHunter.
"""

import json
import re
from datetime import datetime

try:
    import requests as _req
    _OK = True
except ImportError:
    _OK = False

# City → (lat, lon) quick lookup — Nova expands this via /build
_CITIES = {
    "memphis":      (35.1495, -90.0490),
    "kalamazoo":    (42.2917, -85.5872),
    "new york":     (40.7128, -74.0060),
    "los angeles":  (34.0522, -118.2437),
    "chicago":      (41.8781, -87.6298),
    "houston":      (29.7604, -95.3698),
    "phoenix":      (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio":  (29.4241, -98.4936),
    "dallas":       (32.7767, -96.7970),
    "detroit":      (42.3314, -83.0458),
    "nashville":    (36.1627, -86.7816),
    "atlanta":      (33.7490, -84.3880),
    "miami":        (25.7617, -80.1918),
    "seattle":      (47.6062, -122.3321),
    "denver":       (39.7392, -104.9903),
    "london":       (51.5074, -0.1278),
    "paris":        (48.8566, 2.3522),
    "tokyo":        (35.6762, 139.6503),
    "sydney":       (-33.8688, 151.2093),
}

_WMO = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Foggy", 48:"Icy fog", 51:"Light drizzle", 53:"Drizzle",
    55:"Heavy drizzle", 61:"Slight rain", 63:"Rain", 65:"Heavy rain",
    71:"Slight snow", 73:"Snow", 75:"Heavy snow", 77:"Snow grains",
    80:"Slight showers", 81:"Showers", 82:"Violent showers",
    85:"Snow showers", 86:"Heavy snow showers",
    95:"Thunderstorm", 96:"Thunderstorm w/ hail", 99:"Thunderstorm w/ heavy hail",
}


class FreeWeather:
    """
    Real-time weather with zero API keys using Open-Meteo.
    Nova can use this immediately — no setup required.
    """

    BASE = "https://api.open-meteo.com/v1/forecast"

    def _coords(self, city: str):
        key = city.lower().strip()
        if key in _CITIES:
            return _CITIES[key]
        # Try geocoding via Open-Meteo's geocoding API (also free)
        if not _OK:
            return None
        try:
            r = _req.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
                timeout=5
            )
            results = r.json().get("results", [])
            if results:
                return results[0]["latitude"], results[0]["longitude"]
        except Exception:
            pass
        return None

    def current(self, city: str = "Memphis") -> str:
        """Current conditions for any city — no API key needed."""
        if not _OK:
            return "requests library not installed (pip install requests)"
        coords = self._coords(city)
        if not coords:
            return f"City '{city}' not found. Try a major city name."
        lat, lon = coords
        try:
            r = _req.get(self.BASE, params={
                "latitude": lat, "longitude": lon,
                "current_weather": True,
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
            }, timeout=8)
            cw = r.json()["current_weather"]
            desc = _WMO.get(cw["weathercode"], "Unknown")
            temp = cw["temperature"]
            wind = cw["windspeed"]
            return f"{city.title()}: {desc}, {temp:.0f}°F | Wind {wind:.0f} mph"
        except Exception as e:
            return f"Weather fetch error: {e}"

    def forecast(self, city: str = "Memphis", days: int = 3) -> str:
        """Multi-day forecast — no API key needed."""
        if not _OK:
            return "requests library not installed"
        coords = self._coords(city)
        if not coords:
            return f"City '{city}' not found."
        lat, lon = coords
        try:
            r = _req.get(self.BASE, params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "forecast_days": days,
            }, timeout=8)
            d    = r.json()["daily"]
            lines = [f"{city.title()} {days}-Day Forecast:"]
            for i in range(min(days, len(d["time"]))):
                date = d["time"][i]
                hi   = d["temperature_2m_max"][i]
                lo   = d["temperature_2m_min"][i]
                desc = _WMO.get(d["weathercode"][i], "?")
                lines.append(f"  {date}: {desc}, Hi {hi:.0f}°F / Lo {lo:.0f}°F")
            return "\n".join(lines)
        except Exception as e:
            return f"Forecast error: {e}"

    def summary(self, city: str = "Memphis") -> str:
        """One call: current + 3-day forecast."""
        return self.current(city) + "\n" + self.forecast(city, 3)

    def is_raining(self, city: str = "Memphis") -> bool:
        result = self.current(city).lower()
        return any(w in result for w in ["rain", "drizzle", "shower", "storm"])
