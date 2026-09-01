"""
VayuSense - Feature Engineering & Final Feature Selection
=========================================================
NOTE: Strictly enforces chronological time ordering per station.
Phase 3.6 removes seasonal shortcut variables (is_winter, month)
to prevent tree models from relying on static seasonal heuristics.
"""

import numpy as np
import pandas as pd
from src.coupling_engine import compute_coupling_features

# Legacy features (including seasonal shortcuts)
BASELINE_FEATURES = [
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure", "rainfall", "pbl_height", "temp_gradient",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_24h",
    "pm25_roll_mean_6h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
    "hour_sin", "hour_cos", "month", "is_winter"
]

# Explicit Physics-Informed Coupling Terms
COUPLING_FEATURES = [
    "ventilation_coeff",
    "pm25_x_humidity",
    "pm25_x_temp",
    "pm25_x_wind_speed",
    "pm25_div_pbl",
    "stagnation_indicator",
    "inversion_indicator",
    "lagged_pm25_coupling"
]

COUPLED_FEATURES_FULL = BASELINE_FEATURES + COUPLING_FEATURES

# ------------------ PHASE 3.6 FINAL FEATURE SETS (NO SEASONAL SHORTCUTS) ------------------
FINAL_BASELINE_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_deg",
    "pressure",
    "pbl_height",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_24h",
    "pm25_roll_mean_6h",
    "pm25_roll_mean_24h",
    "pm25_roll_std_24h",
    "hour_sin",
    "hour_cos"
]

FINAL_COUPLED_FEATURES = FINAL_BASELINE_FEATURES + COUPLING_FEATURES

def create_feature_pipeline(
    df: pd.DataFrame,
    forecast_horizon: int = 6,
    is_live: bool = False
) -> pd.DataFrame:
    """
    Builds feature matrix while strictly preventing data leakage:
    1. Sorts chronologically by (station_id, timestamp).
    2. Computes historical lags using only past values.
    3. Computes historical rolling statistics.
    4. Injects physics-informed proxy coupling terms.
    5. Creates target variable (PM2.5 concentration t + forecast_horizon hours ahead).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # STRICT TIME ORDERING - NO RANDOM SHUFFLING
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    # Cyclical temporal encodings
    hours = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["month"] = df["timestamp"].dt.month
    df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    
    station_dfs = []
    for station, group in df.groupby("station_id", sort=False):
        group = group.copy()
        
        # Historical Lags (Strictly past observations)
        group["pm25_lag_1h"] = group["pm25"].shift(1)
        group["pm25_lag_3h"] = group["pm25"].shift(3)
        group["pm25_lag_6h"] = group["pm25"].shift(6)
        group["pm25_lag_24h"] = group["pm25"].shift(24)
        
        # Historical Rolling Statistics (Strictly past windows)
        group["pm25_roll_mean_6h"] = group["pm25"].rolling(window=6, min_periods=1).mean()
        group["pm25_roll_mean_24h"] = group["pm25"].rolling(window=24, min_periods=1).mean()
        group["pm25_roll_std_24h"] = group["pm25"].rolling(window=24, min_periods=1).std().fillna(0)
        
        # Target Variable (Future lead step: t + forecast_horizon)
        group["pm25_target"] = group["pm25"].shift(-forecast_horizon)
        
        station_dfs.append(group)
        
    df_processed = pd.concat(station_dfs, axis=0).reset_index(drop=True)
    
    # Add physics-informed proxy coupling terms
    df_processed = compute_coupling_features(df_processed)
    
    # Drop rows with insufficient lag history (and NaN targets if not live)
    if is_live:
        df_clean = df_processed.dropna(subset=["pm25_lag_24h"]).reset_index(drop=True)
    else:
        df_clean = df_processed.dropna(subset=["pm25_lag_24h", "pm25_target"]).reset_index(drop=True)
    
    return df_clean

def get_baseline_and_coupled_matrices(df_features: pd.DataFrame, use_final_features: bool = True):
    """
    Extracts feature matrices X_baseline, X_coupled and target vector y.
    If use_final_features=True (default), excludes seasonal shortcut variables (is_winter, month).
    """
    if use_final_features:
        X_baseline = df_features[FINAL_BASELINE_FEATURES]
        X_coupled = df_features[FINAL_COUPLED_FEATURES]
    else:
        X_baseline = df_features[BASELINE_FEATURES]
        X_coupled = df_features[COUPLED_FEATURES_FULL]
        
    y = df_features["pm25_target"]
    
    return X_baseline, X_coupled, y
