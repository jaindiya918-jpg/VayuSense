"""
VayuSense - Preprocessing & Dataset Validation Pipeline
========================================================
NOTE: Operates on DEMO / SYNTHETIC DATA for hackathon demonstration.
Validates numerical schemas, handles missing values, and enforces
chronological time ordering before feature extraction.
"""

import os
import pandas as pd
import numpy as np
from src.data_generator import generate_and_save_dataset
from src.feature_engineering import create_feature_pipeline

RAW_DATA_PATH = "data/raw/delhi_ncr_aqi_weather_demo.csv"
PROCESSED_DATA_PATH = "data/processed/coupled_features.csv"

def load_and_preprocess_data(
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    forecast_horizon: int = 6,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Loads raw Delhi NCR synthetic data, validates schema, runs feature engineering,
    and saves processed dataset.
    """
    if force_regenerate or not os.path.exists(raw_path):
        print(f"Generating synthetic Delhi NCR dataset...")
        df_raw = generate_and_save_dataset(raw_path, days=120)
    else:
        print(f"Loading raw dataset from {raw_path}...")
        df_raw = pd.read_csv(raw_path)
        
    num_cols = ["temperature", "humidity", "wind_speed", "wind_deg", "pressure", 
                "rainfall", "pbl_height", "temp_gradient", "pm25", "pm10", "no2", "so2", "co", "o3"]
    
    for col in num_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw[col] = df_raw[col].ffill().bfill()
            
    print(f"Constructing feature matrix with forecast horizon = +{forecast_horizon}h...")
    df_features = create_feature_pipeline(df_raw, forecast_horizon=forecast_horizon)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_features.to_csv(processed_path, index=False)
    print(f"Processed dataset ({len(df_features)} records) saved to {processed_path}")
    
    return df_features

if __name__ == "__main__":
    load_and_preprocess_data()
