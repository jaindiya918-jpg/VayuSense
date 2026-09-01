"""
VayuSense - Model Evaluation & Visualization Suite
===================================================
Generates comparative evaluation metrics, plots, and feature importances.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from src.feature_engineering import COUPLED_FEATURES_FULL

FEATURE_NAME_MAP = {
    "temperature": "Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "wind_speed": "Wind Speed (m/s)",
    "wind_deg": "Wind Direction (°)",
    "pressure": "Pressure (hPa)",
    "rainfall": "Rainfall (mm)",
    "pbl_height": "Boundary Layer Height (m)",
    "temp_gradient": "Inversion Temperature Gradient (°C/100m)",
    "pm25_lag_1h": "PM2.5 (1h ago)",
    "pm25_lag_3h": "PM2.5 (3h ago)",
    "pm25_lag_6h": "PM2.5 (6h ago)",
    "pm25_lag_24h": "PM2.5 (24h ago)",
    "pm25_roll_mean_6h": "6h PM2.5 Moving Avg",
    "pm25_roll_mean_24h": "24h PM2.5 Moving Avg",
    "hour_sin": "Time of Day (Sin)",
    "hour_cos": "Time of Day (Cos)",
    "month": "Month",
    "is_winter": "Winter Season Flag",
    "ventilation_coeff": "Ventilation Capacity (m²/s)",
    "pm25_x_humidity": "Humidity-PM2.5 Interaction",
    "pm25_x_temp": "Temperature-PM2.5 Interaction",
    "pm25_x_wind_speed": "Wind Transport Interaction",
    "pm25_div_pbl": "Boundary Density Ratio",
    "stagnation_indicator": "Air Stagnation Event Flag",
    "inversion_indicator": "Thermal Inversion Layer Flag",
    "lagged_pm25_coupling": "Lagged Pollutant Trapping Index"
}

MODEL_DIR = "models"
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
PREDICTIONS_PATH = "data/processed/model_predictions.csv"
COUPLED_MODEL_PATH = os.path.join(MODEL_DIR, "coupled_xgb.json")

def generate_evaluation_plots(df_preds: pd.DataFrame):
    """
    Generates and saves 4 comparative evaluation plots to models/ directory.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamps = pd.to_datetime(df_preds["timestamp"])
    actual = df_preds["actual_pm25"].values
    base_pred = df_preds["baseline_prediction"].values
    coup_pred = df_preds["coupled_prediction"].values
    
    # ------------------ PLOT 1: Actual vs Baseline ------------------
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, actual, label="Actual PM2.5", color="#1f77b4", alpha=0.8, linewidth=1.5)
    plt.plot(timestamps, base_pred, label="Baseline XGBoost", color="#ff7f0e", linestyle="--", linewidth=1.5)
    plt.axhline(y=250, color="red", linestyle=":", label="Severe Threshold (250 µg/m³)")
    plt.title("Plot 1: Actual PM2.5 vs Baseline XGBoost Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 Concentration (µg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_path = os.path.join(MODEL_DIR, "eval_plot1_actual_vs_baseline.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    
    # ------------------ PLOT 2: Actual vs Coupled ------------------
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, actual, label="Actual PM2.5", color="#1f77b4", alpha=0.8, linewidth=1.5)
    plt.plot(timestamps, coup_pred, label="Coupled XGBoost (VayuSense)", color="#2ca02c", linewidth=1.8)
    plt.axhline(y=250, color="red", linestyle=":", label="Severe Threshold (250 µg/m³)")
    plt.title("Plot 2: Actual PM2.5 vs Weather-Coupled XGBoost Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 Concentration (µg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_path = os.path.join(MODEL_DIR, "eval_plot2_actual_vs_coupled.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    
    # ------------------ PLOT 3: All Three Comparison ------------------
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, actual, label="Actual PM2.5", color="black", linewidth=2.0)
    plt.plot(timestamps, base_pred, label="Baseline XGBoost", color="#ff7f0e", linestyle="--", linewidth=1.5)
    plt.plot(timestamps, coup_pred, label="Coupled XGBoost", color="#00e5ff", linewidth=1.8)
    plt.axhline(y=250, color="red", linestyle=":", label="Severe Threshold (250 µg/m³)")
    plt.title("Plot 3: Model Comparison — Actual vs Baseline vs Weather-Coupled XGBoost")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 Concentration (µg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(MODEL_DIR, "eval_plot3_all_three.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    
    # ------------------ PLOT 4: High Pollution Episodes (PM2.5 >= 250) ------------------
    severe_mask = actual >= 250.0
    if severe_mask.sum() > 0:
        plt.figure(figsize=(12, 5))
        severe_indices = np.where(severe_mask)[0]
        sample_subset = severe_indices[:200]  # Plot up to first 200 severe points for clarity
        
        plt.scatter(sample_subset, actual[sample_subset], label="Actual PM2.5 (Severe)", color="red", s=30, zorder=3)
        plt.plot(sample_subset, base_pred[sample_subset], label="Baseline Prediction", color="#ff7f0e", linestyle="--")
        plt.plot(sample_subset, coup_pred[sample_subset], label="Coupled Prediction", color="#00e5ff", linewidth=2)
        plt.title("Plot 4: Severe Pollution Episode Focus (Actual PM2.5 ≥ 250 µg/m³)")
        plt.xlabel("Severe Episode Index")
        plt.ylabel("PM2.5 Concentration (µg/m³)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot4_path = os.path.join(MODEL_DIR, "eval_plot4_high_pollution_episodes.png")
        plt.savefig(plot4_path, dpi=150)
        plt.close()
    else:
        plot4_path = None
        
    print(f"Generated 4 evaluation plots saved to '{MODEL_DIR}/'.")
    return [plot1_path, plot2_path, plot3_path, plot4_path]

def inspect_feature_importance():
    """Extracts and displays top 15 coupled model feature importances."""
    model_coupled = xgb.XGBRegressor()
    model_coupled.load_model(COUPLED_MODEL_PATH)
    
    importances = model_coupled.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": COUPLED_FEATURES_FULL,
        "display_name": [FEATURE_NAME_MAP.get(f, f) for f in COUPLED_FEATURES_FULL],
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    return feat_imp

def print_evaluation_summary():
    """Reads saved metrics and predictions and prints Phase 3 evaluation summary."""
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    df_preds = pd.read_csv(PREDICTIONS_PATH)
    
    print("\n" + "="*70)
    print("VAYUSENSE - PHASE 3 MODEL EVALUATION & COMPARISON REPORT")
    print("="*70)
    
    b = metrics["baseline"]
    c = metrics["coupled"]
    imp = metrics["improvements"]
    
    print(f"Forecast Horizon: +{metrics['forecast_horizon_hours']} Hours ahead")
    print("\nMETRICS COMPARISON TABLE:")
    print("-" * 65)
    print(f"{'Metric':<22} | {'Baseline XGB':<14} | {'Coupled XGB':<14} | {'Improvement':<12}")
    print("-" * 65)
    print(f"{'MAE (ug/m3)':<22} | {b['mae']:14.2f} | {c['mae']:14.2f} | {imp['mae_improvement_pct']:+11.2f}%")
    print(f"{'RMSE (ug/m3)':<22} | {b['rmse']:14.2f} | {c['rmse']:14.2f} | {imp['rmse_improvement_pct']:+11.2f}%")
    print(f"{'R2 Score':<22} | {b['r2']:14.4f} | {c['r2']:14.4f} | {imp['r2_diff']:+11.4f}")
    print(f"{'Severe MAE (>=250)':<22} | {b['high_pollution_mae']:14.2f} | {c['high_pollution_mae']:14.2f} | {imp['high_pollution_mae_improvement_pct']:+11.2f}%")
    print("-" * 65)
    
    # Feature Importances
    feat_imp = inspect_feature_importance()
    print("\nTOP 15 COUPLED MODEL FEATURE IMPORTANCES:")
    print("-" * 65)
    for idx, row in feat_imp.head(15).iterrows():
        is_coupling = " [COUPLING TERM]" if row['feature'] in [
            "ventilation_coeff", "pm25_x_humidity", "pm25_x_temp", "pm25_x_wind_speed",
            "pm25_div_pbl", "stagnation_indicator", "inversion_indicator", "lagged_pm25_coupling"
        ] else ""
        print(f" {idx+1:2d}. {row['display_name']:<35} : {row['importance']:.4f}{is_coupling}")
    print("-" * 65)
    
    # Generate Plots
    generate_evaluation_plots(df_preds)
    print("="*70 + "\n")

if __name__ == "__main__":
    print_evaluation_summary()
