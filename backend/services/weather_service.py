#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather service for the India Crop Recommendation System.

Open-Meteo is used ONLY for the live conditions and 7-day forecast shown in the
UI. Historical model-input weather comes from NASA POWER (see
scripts/02_fetch_weather.py), because the Open-Meteo archive host is unreliable
here and its geocoder mis-resolves many Indian districts.
"""

# Import time for the short-lived display cache.
import time
# Import threading so concurrent requests cannot corrupt the cache.
import threading
# Import requests for the HTTP call.
import requests
# Import typing helpers.
from typing import Dict, Any


class WeatherService:
    """Fetch live weather and a short forecast from Open-Meteo (no API key)."""

    def __init__(self):
        """Store the forecast endpoint and prepare the cache."""
        # The forecast endpoint is public and needs no key.
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        # Reuse one HTTP connection instead of opening a new one per request.
        self.session = requests.Session()
        # Cache display weather per location; live conditions barely change
        # from one request to the next.
        self.cache: Dict[tuple, tuple] = {}
        # Guard the cache because FastAPI serves requests from a thread pool.
        self.cache_lock = threading.Lock()
        # Successful payloads stay valid for ten minutes.
        self.cache_ttl_seconds = 600
        # Failures are retried sooner so a transient outage can recover.
        self.failure_ttl_seconds = 60

    def get_live_weather_and_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Return current conditions and a 7-day forecast for display only.

        This data is never used as a model feature; the model uses the
        historical seasonal estimate computed in Step 2.
        """
        # Round coordinates so nearby requests share one cache entry.
        key = (round(float(lat), 4), round(float(lon), 4))
        # Read the cache under the lock.
        now = time.time()
        with self.cache_lock:
            entry = self.cache.get(key)
            if entry is not None and now - entry[0] < entry[2]:
                # Return the cached payload.
                return entry[1]
        # Cache miss: call the API.
        result = self._fetch(lat, lon)
        # Cache successes for longer than failures.
        ttl = self.cache_ttl_seconds if result.get("success") else self.failure_ttl_seconds
        # Store the fresh payload.
        with self.cache_lock:
            self.cache[key] = (now, result, ttl)
        # Return the payload.
        return result

    def _fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        """Call Open-Meteo once and shape the response for the UI."""
        # Build the query parameters.
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
        }
        try:
            # Call the API with a short timeout so recommendations stay responsive.
            response = self.session.get(self.forecast_url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            # Extract the current block.
            current = data.get("current", {})
            current_weather = {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "time": current.get("time"),
            }
            # Extract the daily forecast into a simple list.
            daily = data.get("daily", {})
            forecast = []
            for index in range(len(daily.get("time", []))):
                forecast.append({
                    "date": daily["time"][index],
                    "temp_max": daily["temperature_2m_max"][index],
                    "temp_min": daily["temperature_2m_min"][index],
                    "precipitation": daily["precipitation_sum"][index],
                })
            # Return a successful payload.
            return {"current": current_weather, "forecast": forecast, "success": True}
        except Exception as exc:
            # Live display data is optional; never break the recommendation.
            return {"current": None, "forecast": [], "success": False, "error": str(exc)}
