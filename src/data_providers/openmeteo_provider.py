"""
VayuSense - Open-Meteo Weather Data Provider
==============================================
Fetches real hourly meteorological observations from Open-Meteo API.
Covers: Temperature, Humidity, Wind Speed (m/s), Wind Direction, Surface Pressure, Precipitation, PBL Height (m).
Timestamps: Asia/Kolkata.
"""

import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class OpenMeteoProvider:
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        pass

    def fetch_weather_data(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches hourly weather data from Open-Meteo for specified lat/lon coordinates.
        Explicitly requests wind_speed_unit='ms' so wind speed is returned in m/s.
        Uses fast forecast endpoint for near-real-time live queries up to today.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        is_recent_query = (end_date >= today_str)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m,boundary_layer_height",
            "wind_speed_unit": "ms",
            "timezone": "Asia/Kolkata"
        }

        if is_recent_query:
            # Use fast forecast endpoint for recent live stream data
            params["past_days"] = 3
            url = self.FORECAST_URL
        else:
            params["start_date"] = start_date
            params["end_date"] = end_date
            url = self.ARCHIVE_URL

        try:
            r = requests.get(url, params=params, verify=False, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Fallback to forecast URL if archive endpoint fails
            params.pop("start_date", None)
            params.pop("end_date", None)
            params["past_days"] = 3
            r = requests.get(self.FORECAST_URL, params=params, verify=False, timeout=10)
            r.raise_for_status()
            data = r.json()

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            raise ValueError(f"Open-Meteo returned empty payload for coordinates ({lat}, {lon}).")

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "wind_speed": hourly.get("wind_speed_10m"),
            "wind_deg": hourly.get("wind_direction_10m"),
            "pressure": hourly.get("surface_pressure"),
            "rainfall": hourly.get("precipitation"),
            "pbl_height": hourly.get("boundary_layer_height")
        })
        
        # Linearly interpolate PBL height & weather parameters smoothly
        df["pbl_height"] = df["pbl_height"].interpolate(method="linear").bfill().ffill()
        df["rainfall"] = df["rainfall"].fillna(0.0)
        df = df.interpolate(method="linear").bfill().ffill()
        
        return df
