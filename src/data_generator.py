"""
VayuSense - Synthetic Data Generator (Delhi NCR Focus)
=====================================================
NOTE: This file generates DEMO / SYNTHETIC DATA for hackathon demonstration purposes.
It models realistic meteorological and pollutant patterns for Delhi NCR (Anand Vihar,
RK Puram, Punjabi Bagh, Mandir Marg) with physics-informed interactions.
It does NOT contain live CPCB/IMD physical sensor measurements.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

STATIONS = ["Anand_Vihar", "RK_Puram", "Punjabi_Bagh", "Mandir_Marg"]

def generate_delhi_ncr_data(
    start_date: str = "2025-10-01",
    days: int = 120,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly weather and air pollution DEMO / SYNTHETIC DATA for Delhi NCR stations.
    
    Physics-informed proxy interactions embedded:
    1. Diurnal Boundary Layer Cycle: Low mixing layer (PBL) during cold winter nights.
    2. Temperature Inversion: Surface temperature gradient proxy creating trapping layers.
    3. Stagnation: Calm wind speeds (< 1.5 m/s) preventing horizontal dispersion.
    4. Hygroscopic Particle Growth: Relative humidity (> 75%) enhancing aerosol concentration proxies.
    """
    np.random.seed(random_seed)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    total_hours = days * 24
    timestamps = [start_dt + timedelta(hours=i) for i in range(total_hours)]
    
    all_data = []
    
    for station in STATIONS:
        # Station-specific emission baseline offset
        station_bias = {
            "Anand_Vihar": 1.25,
            "RK_Puram": 1.0,
            "Punjabi_Bagh": 1.1,
            "Mandir_Marg": 0.9
        }[station]
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            month = ts.month
            
            # Season flag: Delhi winter smog season (Nov - Feb)
            is_winter = 1 if month in [11, 12, 1, 2] else 0
            
            # --- 1. Weather Parameter Generation ---
            # Temperature (°C)
            base_temp = 18.0 if is_winter else 30.0
            temp_diurnal = 6.0 * np.sin(np.pi * (hour - 8) / 12)
            temperature = base_temp + temp_diurnal + np.random.normal(0, 1.2)
            temperature = max(4.0, min(48.0, temperature))
            
            # Relative Humidity (%)
            base_rh = 75.0 if is_winter else 45.0
            rh_diurnal = -20.0 * np.sin(np.pi * (hour - 8) / 12)
            humidity = base_rh + rh_diurnal + np.random.normal(0, 3.5)
            humidity = max(20.0, min(98.0, humidity))
            
            # Wind Speed (m/s)
            wind_base = 1.2 if is_winter else 2.5
            wind_diurnal = 0.8 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -0.3
            wind_speed = wind_base + wind_diurnal + np.random.uniform(-0.3, 0.5)
            wind_speed = max(0.3, min(12.0, wind_speed))
            
            # Wind Direction (degrees 0-360)
            wind_deg = (300 + np.random.normal(0, 25)) % 360  # Predominantly NW during winter
            
            # Pressure (hPa)
            pressure = 1015.0 + (3.0 if is_winter else -5.0) + np.random.normal(0, 0.8)
            
            # Rainfall (mm)
            rainfall = max(0.0, np.random.choice([0.0, 0.0, 0.0, 2.0, 6.0], p=[0.96, 0.025, 0.01, 0.003, 0.002]))
            
            # Planetary Boundary Layer (PBL) Height (m)
            pbl_min = 180.0 if is_winter else 400.0
            pbl_max = 850.0 if is_winter else 1600.0
            pbl_diurnal = 0.5 * (1 + np.sin(np.pi * (hour - 7) / 12)) if 7 <= hour <= 19 else 0.0
            pbl_height = pbl_min + (pbl_max - pbl_min) * pbl_diurnal + np.random.normal(0, 25)
            pbl_height = max(120.0, min(2200.0, pbl_height))
            
            # Thermal Inversion Gradient (dT/dz °C / 100m)
            inversion_base = 1.8 if (is_winter and (hour < 8 or hour > 20)) else -0.8
            temp_gradient = inversion_base + np.random.normal(0, 0.3)
            
            # --- 2. Pollutant Generation with Physics Coupling Proxies ---
            rush_hour_boost = 1.4 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0.9
            base_emission = 75.0 * station_bias * rush_hour_boost
            
            # Physics proxy: Trapping ratio (PBL height & Wind speed inverse relationship)
            trapping_ratio = (800.0 / pbl_height) * (2.0 / wind_speed)
            if temp_gradient > 0:
                trapping_ratio *= 1.35  # Inversion enhancement factor
            if humidity > 75:
                trapping_ratio *= (1.0 + (humidity - 75) / 100.0)  # Hygroscopic growth factor
            if rainfall > 0:
                trapping_ratio *= 0.35  # Wet deposition scavenging
                
            pm25 = base_emission * trapping_ratio * (1.2 if is_winter else 0.5) + np.random.normal(0, 8)
            pm25 = max(10.0, min(750.0, pm25))
            
            # PM10
            pm10 = pm25 * np.random.uniform(1.4, 1.75) + np.random.normal(0, 10)
            pm10 = max(15.0, min(1000.0, pm10))
            
            # NO2
            no2 = (42.0 * rush_hour_boost * (1.2 / wind_speed)) + np.random.normal(0, 4)
            no2 = max(5.0, min(250.0, no2))
            
            # SO2
            so2 = (16.0 * station_bias * (1.0 / wind_speed)) + np.random.normal(0, 2)
            so2 = max(2.0, min(120.0, so2))
            
            # CO (mg/m3)
            co = (pm25 / 100.0) * 1.05 + np.random.normal(0, 0.1)
            co = max(0.2, min(10.0, co))
            
            # O3 (Ozone)
            o3_sunlight = np.sin(np.pi * (hour - 7) / 11) if 7 <= hour <= 18 else 0.0
            o3 = 18.0 + (60.0 * o3_sunlight * (temperature / 30.0)) + np.random.normal(0, 4)
            o3 = max(5.0, min(220.0, o3))
            
            all_data.append({
                "data_source": "DEMO_SYNTHETIC",
                "timestamp": ts.strftime("%Y-%m-%d %H:00"),
                "station_id": station,
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 2),
                "wind_speed": round(wind_speed, 2),
                "wind_deg": round(wind_deg, 1),
                "pressure": round(pressure, 2),
                "rainfall": round(rainfall, 2),
                "pbl_height": round(pbl_height, 1),
                "temp_gradient": round(temp_gradient, 2),
                "pm25": round(pm25, 2),
                "pm10": round(pm10, 2),
                "no2": round(no2, 2),
                "so2": round(so2, 2),
                "co": round(co, 2),
                "o3": round(o3, 2)
            })
            
    df = pd.DataFrame(all_data)
    return df

def generate_and_save_dataset(
    output_path: str = "data/raw/delhi_ncr_aqi_weather_demo.csv",
    days: int = 120
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_delhi_ncr_data(days=days)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} DEMO / SYNTHETIC hourly records for Delhi NCR stations -> Saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_and_save_dataset()
