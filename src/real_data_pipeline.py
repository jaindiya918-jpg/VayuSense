"""
VayuSense - Real Data Pipeline Execution Engine
===============================================
Orchestrates real air quality (OpenAQ v3 / Open-Meteo AQ) and meteorological (Open-Meteo) data ingestion.
Aligns hourly observations, validates schema, applies coupling engine, and saves to data/raw/real_delhi_ncr_data.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_providers.openaq_provider import OpenAQProvider
from src.data_providers.openmeteo_provider import OpenMeteoProvider
from src.coupling_engine import compute_coupling_features

load_dotenv()

REAL_DATA_PATH = "data/raw/real_delhi_ncr_data.csv"

DELHI_STATIONS = {
    "Anand_Vihar": {"lat": 28.6469, "lon": 77.3162, "name": "Anand Vihar"},
    "RK_Puram": {"lat": 28.5644, "lon": 77.1724, "name": "RK Puram"},
    "Punjabi_Bagh": {"lat": 28.6683, "lon": 77.1247, "name": "Punjabi Bagh"},
    "Mandir_Marg": {"lat": 28.6364, "lon": 77.2011, "name": "Mandir Marg"}
}

def run_real_data_pipeline(
    start_date: str = "2024-01-01",
    end_date: str = "2024-04-01"
) -> pd.DataFrame:
    """
    Executes real data fetch, alignment, validation, coupling feature injection, and CSV export.
    """
    print("="*80)
    print(f"VAYUSENSE - REAL DATA PIPELINE INGESTION ({start_date} to {end_date})")
    print("="*80)
    
    openaq_prov = OpenAQProvider()
    openmeteo_prov = OpenMeteoProvider()
    
    station_dfs = []
    
    for station_id, meta in DELHI_STATIONS.items():
        lat, lon = meta["lat"], meta["lon"]
        print(f"\n[+] Fetching Real Observations for Station: '{station_id}' ({meta['name']}, Lat: {lat}, Lon: {lon})...")
        
        # 1. Fetch Air Quality
        df_aq = openaq_prov.fetch_air_quality_data(lat, lon, start_date, end_date, station_id=station_id)
        
        # 2. Fetch Meteorology
        df_met = openmeteo_prov.fetch_weather_data(lat, lon, start_date, end_date)
        
        # 3. Merge & Hourly Alignment
        df_merged = pd.merge(df_aq, df_met, on="timestamp", how="inner")
        df_merged["station_id"] = station_id
        
        # 4. Temperature Gradient estimation (proxy for vertical inversion)
        # Climatological diurnal gradient variation
        hour = df_merged["timestamp"].dt.hour
        # Inversion typically strongest nighttime (00-06h) with positive gradient
        df_merged["temp_gradient"] = np.where((hour <= 6) | (hour >= 21), 0.8, -0.6)
        
        df_merged["is_demo"] = 0
        station_dfs.append(df_merged)
        
    # Combine all station dataframes
    df_real_all = pd.concat(station_dfs, axis=0).sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    # 5. Calculate Physics-Informed Proxy Coupling Features
    df_real_coupled = compute_coupling_features(df_real_all)
    
    # 6. Save to data/raw/real_delhi_ncr_data.csv
    os.makedirs("data/raw", exist_ok=True)
    df_real_coupled.to_csv(REAL_DATA_PATH, index=False)
    print(f"\n[SUCCESS] Saved Real Data to '{REAL_DATA_PATH}' ({len(df_real_coupled)} records).")
    
    # 7. Print Comprehensive Summary Report
    print_real_data_summary(df_real_coupled)
    
    return df_real_coupled

def print_real_data_summary(df: pd.DataFrame):
    print("\n" + "="*80)
    print("REAL DATASET SUMMARY REPORT")
    print("="*80)
    
    stations = df["station_id"].unique().tolist()
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    avail_pollutants = [p for p in pollutants if p in df.columns]
    
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    num_rows = len(df)
    
    print(f"1. Available Stations ({len(stations)}) : {stations}")
    print(f"2. Available Pollutants ({len(avail_pollutants)})  : {avail_pollutants}")
    print(f"3. Timestamp Range (Asia/Kolkata): {ts_min}  -->  {ts_max}")
    print(f"4. Total Dataset Rows            : {num_rows}")
    
    print("\n5. Missing Value Percentages per Column:")
    print("-" * 50)
    null_pct = (df.isnull().sum() / len(df)) * 100.0
    for col, pct in null_pct.items():
        print(f"   • {col:<26}: {pct:6.2f}%")
        
    print("\n6. Final Dataframe Schema Columns:")
    print("-" * 50)
    print(list(df.columns))
    print("="*80 + "\n")

def fetch_live_data_pipeline(days_back: int = 3) -> pd.DataFrame:
    """
    Fetches current/recent air-quality (OpenAQ v3 / Open-Meteo fallback) and weather observations (Open-Meteo API).
    Dynamically computes start_date and end_date for recent window to compute required 24h lag & rolling features.
    Returns processed feature matrix containing live observations.
    """
    openaq_prov = OpenAQProvider()
    openmeteo_prov = OpenMeteoProvider()
    
    now_dt = datetime.now()
    start_dt = now_dt - timedelta(days=days_back)
    
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = now_dt.strftime("%Y-%m-%d")
    
    station_dfs = []
    for station_id, meta in DELHI_STATIONS.items():
        lat, lon = meta["lat"], meta["lon"]
        try:
            df_aq = openaq_prov.fetch_air_quality_data(lat, lon, start_date, end_date, station_id=station_id)
            df_met = openmeteo_prov.fetch_weather_data(lat, lon, start_date, end_date)
            
            df_merged = pd.merge(df_aq, df_met, on="timestamp", how="inner")
            if df_merged.empty:
                continue
            df_merged["station_id"] = station_id
            
            hour = df_merged["timestamp"].dt.hour
            df_merged["temp_gradient"] = np.where((hour <= 6) | (hour >= 21), 0.8, -0.6)
            df_merged["is_demo"] = 0
            station_dfs.append(df_merged)
        except Exception as e:
            print(f"Notice: Failed to fetch live data for station {station_id}: {e}")
            continue
            
    if not station_dfs:
        raise ValueError("Current live data is temporarily unavailable.")
        
    df_live_all = pd.concat(station_dfs, axis=0).sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    from src.feature_engineering import create_feature_pipeline
    df_live_feat = create_feature_pipeline(df_live_all, forecast_horizon=6, is_live=True)
    
    return df_live_feat

if __name__ == "__main__":
    run_real_data_pipeline()

