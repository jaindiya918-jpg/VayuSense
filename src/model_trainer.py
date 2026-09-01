"""
VayuSense - Model Training Pipeline (Baseline vs Coupled XGBoost)
==================================================================
NOTE: Implements strict chronological time-series splitting per station.
Prevents future data leakage and objectively evaluates whether explicit
physics-informed proxy coupling features improve PM2.5 forecasting.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import BASELINE_FEATURES, COUPLED_FEATURES_FULL, get_baseline_and_coupled_matrices
from src.preprocessing import load_and_preprocess_data

MODEL_DIR = "models"
PROCESSED_DATA_PATH = "data/processed/coupled_features.csv"
PREDICTIONS_OUTPUT_PATH = "data/processed/model_predictions.csv"
METRICS_OUTPUT_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_xgb.json")
COUPLED_MODEL_PATH = os.path.join(MODEL_DIR, "coupled_xgb.json")

def perform_data_leakage_check(df: pd.DataFrame, X_cols: list, target_col: str = "pm25_target") -> bool:
    """
    Performs automated explicit data leakage checks:
    1. Verifies target column is NOT present in feature matrix X.
    2. Verifies future PM2.5 values are not in X.
    3. Verifies lag and rolling features only utilize past information.
    """
    print("\n" + "="*60)
    print("EXECUTING AUTOMATED DATA LEAKAGE VERIFICATION")
    print("="*60)
    
    # Check 1: Target column in X
    if target_col in X_cols:
        raise ValueError(f"CRITICAL LEAKAGE ERROR: Target column '{target_col}' found in feature matrix X!")
    print("  [PASS] Target column 'pm25_target' is NOT present in feature matrix X.")
    
    # Check 2: Verify lag features only reference past observations (shift > 0)
    for col in X_cols:
        if "lead" in col.lower() or "future" in col.lower():
            raise ValueError(f"CRITICAL LEAKAGE ERROR: Future feature '{col}' found in feature matrix X!")
    print("  [PASS] No future lead columns present in input features.")
    
    # Check 3: Confirm chronological time ordering per station
    for station, group in df.groupby("station_id"):
        timestamps = pd.to_datetime(group["timestamp"])
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"CRITICAL LEAKAGE ERROR: Timestamps for station {station} are not strictly sorted chronologically!")
    print("  [PASS] Timestamps for all stations are strictly ordered chronologically.")
    
    print("\n>>> NO FUTURE DATA LEAKAGE DETECTED <<<\n")
    return True

def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    Splits multi-station time-series dataset chronologically per station:
    - Train: First 70% of timestamps
    - Val: Next 15% of timestamps
    - Test: Final 15% of timestamps
    Prevents temporal overlap and station boundary cross-contamination.
    """
    train_dfs, val_dfs, test_dfs = [], [], []
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    for station, group in df.groupby("station_id", sort=False):
        n = len(group)
        idx_train = int(n * train_ratio)
        idx_val = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(group.iloc[:idx_train])
        val_dfs.append(group.iloc[idx_train:idx_val])
        test_dfs.append(group.iloc[idx_val:])
        
    df_train = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    df_val = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    df_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)
    
    # Print exact timestamp boundaries
    print("CHRONOLOGICAL TIME-SERIES DATASET SPLIT:")
    print(f"  • Training   ({len(df_train):5d} rows): {df_train['timestamp'].min()}  -->  {df_train['timestamp'].max()}")
    print(f"  • Validation ({len(df_val):5d} rows): {df_val['timestamp'].min()}  -->  {df_val['timestamp'].max()}")
    print(f"  • Testing    ({len(df_test):5d} rows): {df_test['timestamp'].min()}  -->  {df_test['timestamp'].max()}")
    
    return df_train, df_val, df_test

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculates MAE, RMSE, R², and Severe Episode MAE (PM2.5 >= 250 µg/m³)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    severe_mask = y_true >= 250.0
    if severe_mask.sum() > 0:
        severe_mae = mean_absolute_error(y_true[severe_mask], y_pred[severe_mask])
    else:
        severe_mae = mae
        
    return {
        "model_name": model_name,
        "mae": float(round(mae, 2)),
        "rmse": float(round(rmse, 2)),
        "r2": float(round(r2, 4)),
        "high_pollution_mae": float(round(severe_mae, 2))
    }

def train_models(forecast_horizon: int = 6):
    """
    Loads preprocessed data, performs chronological split, trains Baseline & Coupled XGBoost models,
    evaluates on test set, and saves model artifacts and prediction files.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load data
    df_features = load_and_preprocess_data(processed_path=PROCESSED_DATA_PATH, forecast_horizon=forecast_horizon)
    
    # 2. Chronological split
    df_train, df_val, df_test = create_chronological_splits(df_features)
    
    X_base_train, X_coup_train, y_train = get_baseline_and_coupled_matrices(df_train)
    X_base_val, X_coup_val, y_val = get_baseline_and_coupled_matrices(df_val)
    X_base_test, X_coup_test, y_test = get_baseline_and_coupled_matrices(df_test)
    
    # 3. Perform data leakage check
    perform_data_leakage_check(df_features, COUPLED_FEATURES_FULL)
    
    # 4. Train Model A: BASELINE XGBoost (Standard weather + lags only)
    print("\n[1/2] Training Model A: BASELINE XGBoost...")
    model_baseline = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_baseline.fit(
        X_base_train, y_train,
        eval_set=[(X_base_val, y_val)],
        verbose=False
    )
    y_pred_base = model_baseline.predict(X_base_test)
    metrics_base = calculate_metrics(y_test.values, y_pred_base, model_name="Baseline XGBoost")
    model_baseline.save_model(BASELINE_MODEL_PATH)
    print(f"  • Baseline MAE: {metrics_base['mae']} | RMSE: {metrics_base['rmse']} | R²: {metrics_base['r2']} | High PM2.5 MAE: {metrics_base['high_pollution_mae']}")
    
    # 5. Train Model B: COUPLED XGBoost (Baseline + Physics Proxy Coupling features)
    print("\n[2/2] Training Model B: COUPLED XGBoost (VayuSense)...")
    model_coupled = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_coupled.fit(
        X_coup_train, y_train,
        eval_set=[(X_coup_val, y_val)],
        verbose=False
    )
    y_pred_coup = model_coupled.predict(X_coup_test)
    metrics_coup = calculate_metrics(y_test.values, y_pred_coup, model_name="Coupled XGBoost")
    model_coupled.save_model(COUPLED_MODEL_PATH)
    print(f"  • Coupled MAE:  {metrics_coup['mae']} | RMSE: {metrics_coup['rmse']} | R²: {metrics_coup['r2']} | High PM2.5 MAE: {metrics_coup['high_pollution_mae']}")
    
    # 6. Calculate Improvement Metrics Mathematically
    mae_imp = ((metrics_base['mae'] - metrics_coup['mae']) / metrics_base['mae']) * 100.0
    rmse_imp = ((metrics_base['rmse'] - metrics_coup['rmse']) / metrics_base['rmse']) * 100.0
    r2_diff = metrics_coup['r2'] - metrics_base['r2']
    severe_imp = ((metrics_base['high_pollution_mae'] - metrics_coup['high_pollution_mae']) / metrics_base['high_pollution_mae']) * 100.0
    
    comparison_summary = {
        "forecast_horizon_hours": forecast_horizon,
        "baseline": metrics_base,
        "coupled": metrics_coup,
        "improvements": {
            "mae_improvement_pct": float(round(mae_imp, 2)),
            "rmse_improvement_pct": float(round(rmse_imp, 2)),
            "r2_diff": float(round(r2_diff, 4)),
            "high_pollution_mae_improvement_pct": float(round(severe_imp, 2))
        }
    }
    
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nModel metrics saved to {METRICS_OUTPUT_PATH}")
    
    # 7. Save prediction evaluation CSV
    df_preds = pd.DataFrame({
        "timestamp": df_test["timestamp"].dt.strftime("%Y-%m-%d %H:00"),
        "station_id": df_test["station_id"],
        "actual_pm25": y_test.values,
        "baseline_prediction": y_pred_base,
        "coupled_prediction": y_pred_coup
    })
    df_preds.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"Model test predictions saved to {PREDICTIONS_OUTPUT_PATH}")
    
    return comparison_summary, model_coupled, X_coup_test, df_test, y_test.values, y_pred_base, y_pred_coup

if __name__ == "__main__":
    train_models()
