"""
VayuSense - OpenAQ Air Quality Data Provider (v3 API)
======================================================
Fetches real hourly air quality observations (PM2.5, PM10, NO2, SO2, CO, O3).
Uses OpenAQ v3 API with X-API-Key header when configured.
Provides seamless fallback to Open-Meteo Air Quality API (CAMS/ECMWF) if unconfigured or rate limited.
Timestamps: Asia/Kolkata.
"""

import os
import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class OpenAQProvider:
    OPENAQ_V3_BASE = "https://api.openaq.org/v3"
    OPENMETEO_AQ_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAQ_API_KEY")

    def fetch_air_quality_data(self, lat: float, lon: float, start_date: str, end_date: str, station_id: str = "Delhi_Station") -> pd.DataFrame:
        """
        Fetches hourly air quality data for lat/lon coordinates.
        Tries OpenAQ v3 API if API key is present; falls back to Open-Meteo AQ API.
        """
        if self.api_key:
            try:
                return self._fetch_openaq_v3(lat, lon, start_date, end_date)
            except Exception as e:
                print(f"OpenAQ v3 API notice ({e}). Falling back to real Open-Meteo CAMS Air Quality stream...")
                return self._fetch_openmeteo_aq(lat, lon, start_date, end_date)
        else:
            return self._fetch_openmeteo_aq(lat, lon, start_date, end_date)

    def _fetch_openaq_v3(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches from OpenAQ v3 locations/sensors API using X-API-Key header."""
        headers = {
            "Accept": "application/json",
            "X-API-Key": self.api_key
        }
        
        # Search candidate OpenAQ v3 locations within 25km radius
        params_loc = {
            "coordinates": f"{lat},{lon}",
            "radius": 25000,
            "limit": 5
        }
        
        r_loc = requests.get(f"{self.OPENAQ_V3_BASE}/locations", headers=headers, params=params_loc, verify=False, timeout=8)
        r_loc.raise_for_status()
        results = r_loc.json().get("results", [])
        
        if not results:
            raise ValueError(f"No OpenAQ v3 location found within 25km of ({lat}, {lon}).")
            
        records = []
        matched_loc_id = None
        matched_loc_name = None

        # Iterate through top candidate locations to find measurements
        for loc in results:
            location_id = loc["id"]
            location_name = loc.get("name", f"Location_{location_id}")
            sensors = loc.get("sensors", [])
            
            sensor_param_map = {}
            for s in sensors:
                p_name = s.get("parameter", {}).get("name", "").lower()
                if p_name in ["pm2.5", "pm25"]: param_key = "pm25"
                elif p_name in ["pm10"]: param_key = "pm10"
                elif p_name in ["no2"]: param_key = "no2"
                elif p_name in ["so2"]: param_key = "so2"
                elif p_name in ["co"]: param_key = "co"
                elif p_name in ["o3"]: param_key = "o3"
                else: continue
                sensor_param_map[s["id"]] = param_key
                
            if not sensor_param_map:
                continue
                
            for s_id, param_key in sensor_param_map.items():
                params_meas = {
                    "datetime_from": f"{start_date}T00:00:00Z",
                    "datetime_to": f"{end_date}T23:59:59Z",
                    "limit": 1000
                }
                r_meas = requests.get(f"{self.OPENAQ_V3_BASE}/sensors/{s_id}/hours", headers=headers, params=params_meas, verify=False, timeout=6)
                if r_meas.status_code == 200:
                    meas_results = r_meas.json().get("results", [])
                    for m in meas_results:
                        dt = m.get("period", {}).get("datetimeFrom", {}).get("utc")
                        val = m.get("value")
                        if dt and val is not None:
                            records.append({
                                "timestamp": pd.to_datetime(dt).tz_convert("Asia/Kolkata").tz_localize(None),
                                "parameter": param_key,
                                "value": float(val)
                            })
                            matched_loc_id = location_id
                            matched_loc_name = location_name

            if records:
                break
                
        if not records:
            raise ValueError(f"No measurements retrieved from candidate OpenAQ v3 locations near ({lat}, {lon}).")
            
        df_raw = pd.DataFrame(records)
        df_pivot = df_raw.pivot_table(index="timestamp", columns="parameter", values="value", aggfunc="mean").reset_index()
        
        # Ensure all required pollutant columns present
        for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
            if col not in df_pivot.columns:
                df_pivot[col] = np.nan
                
        df_pivot["aq_source"] = f"OpenAQ v3 API ({matched_loc_name} - ID {matched_loc_id})"
        return df_pivot

    def _fetch_openmeteo_aq(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback real air quality fetcher using Open-Meteo CAMS API."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        is_recent = (end_date >= today_str)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
            "timezone": "Asia/Kolkata"
        }

        if is_recent:
            params["past_days"] = 3
        else:
            params["start_date"] = start_date
            params["end_date"] = end_date
        
        r = requests.get(self.OPENMETEO_AQ_BASE, params=params, verify=False, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            raise ValueError(f"Open-Meteo AQ returned empty payload for coordinates ({lat}, {lon}).")

        # Carbon Monoxide handling with safe None filtering (CO from CAMS in µg/m³ -> mg/m³)
        co_raw = hourly.get("carbon_monoxide", [])
        if co_raw is not None and len(co_raw) > 0:
            co_arr = np.array([float(x) if x is not None else np.nan for x in co_raw])
            co_mg = co_arr / 1000.0
        else:
            co_mg = np.nan

        pm25_raw = [float(x) if x is not None else np.nan for x in hourly.get("pm2_5", [])]
        pm10_raw = [float(x) if x is not None else np.nan for x in hourly.get("pm10", [])]
        no2_raw = [float(x) if x is not None else np.nan for x in hourly.get("nitrogen_dioxide", [])]
        so2_raw = [float(x) if x is not None else np.nan for x in hourly.get("sulphur_dioxide", [])]
        o3_raw = [float(x) if x is not None else np.nan for x in hourly.get("ozone", [])]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "pm25": pm25_raw,
            "pm10": pm10_raw,
            "no2": no2_raw,
            "so2": so2_raw,
            "co": co_mg,
            "o3": o3_raw
        })
        
        df = df.interpolate(method="linear").bfill().ffill()
        df["aq_source"] = "Open-Meteo CAMS Air Quality Stream (ECMWF)"
        return df
