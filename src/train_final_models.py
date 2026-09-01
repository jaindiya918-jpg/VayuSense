"""
VayuSense - Phase 3.6 Final Model Refinement & Retraining Pipeline
==================================================================
Removes static seasonal shortcut features (is_winter, month).
Trains Final Baseline XGBoost vs Final Coupled XGBoost and objectively compares
performance against legacy models and coupling-only architecture.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from src.feature_engineering import (
    BASELINE_FEATURES, COUPLED_FEATURES_FULL, COUPLING_FEATURES,
    FINAL_BASELINE_FEATURES, FINAL_COUPLED_FEATURES
)
from src.preprocessing import load_and_preprocess_data
from src.model_trainer import create_chronological_splits, calculate_metrics, perform_data_leakage_check

MODEL_DIR = "models"
FINAL_BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "final_baseline_xgb.json")
FINAL_COUPLED_MODEL_PATH = os.path.join(MODEL_DIR, "final_coupled_xgb.json")
FINAL_METRICS_PATH = os.path.join(MODEL_DIR, "final_model_metrics.json")
FINAL_PREDICTIONS_PATH = "data/processed/final_model_predictions.csv"
FINAL_PLOT_COMPARISON = os.path.join(MODEL_DIR, "final_model_comparison.png")
FINAL_PLOT_HIGH_POLLUTION = os.path.join(MODEL_DIR, "final_high_pollution_comparison.png")

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
    "pm25_roll_std_24h": "24h PM2.5 Std Dev",
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

def generate_final_plots(df_preds: pd.DataFrame):
    timestamps = pd.to_datetime(df_preds["timestamp"])
    actual = df_preds["actual_pm25"].values
    base_pred = df_preds["final_baseline_prediction"].values
    coup_pred = df_preds["final_coupled_prediction"].values
    
    # 1. Final Model Comparison Plot
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, actual, label="Actual PM2.5", color="black", linewidth=2.0)
    plt.plot(timestamps, base_pred, label="Final Baseline XGBoost", color="#ff7f0e", linestyle="--", linewidth=1.5)
    plt.plot(timestamps, coup_pred, label="Final Coupled XGBoost (VayuSense)", color="#00e5ff", linewidth=1.8)
    plt.axhline(y=250, color="red", linestyle=":", label="Severe AQI Threshold (250 ug/m3)")
    plt.title("Phase 3.6 Final Model Comparison — Actual vs Final Baseline vs Final Coupled")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 Concentration (ug/m3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_PLOT_COMPARISON, dpi=150)
    plt.close()
    
    # 2. High Pollution Episode Comparison (PM2.5 >= 250)
    severe_mask = actual >= 250.0
    if severe_mask.sum() > 0:
        plt.figure(figsize=(12, 5))
        severe_indices = np.where(severe_mask)[0]
        subset = severe_indices[:200]
        
        plt.scatter(subset, actual[subset], label="Actual PM2.5 (Severe)", color="red", s=30, zorder=3)
        plt.plot(subset, base_pred[subset], label="Final Baseline Prediction", color="#ff7f0e", linestyle="--")
        plt.plot(subset, coup_pred[subset], label="Final Coupled Prediction", color="#00e5ff", linewidth=2)
        plt.title("Phase 3.6 Severe Episode Focus (Actual PM2.5 >= 250 ug/m3)")
        plt.xlabel("Severe Episode Index")
        plt.ylabel("PM2.5 Concentration (ug/m3)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FINAL_PLOT_HIGH_POLLUTION, dpi=150)
        plt.close()

def run_phase3_6_refinement(forecast_horizon: int = 6):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("="*70)
    print("VAYUSENSE - PHASE 3.6: FINAL MODEL FEATURE REFINEMENT AND RETRAINING")
    print("="*70)
    
    df_features = load_and_preprocess_data(forecast_horizon=forecast_horizon)
    df_train, df_val, df_test = create_chronological_splits(df_features)
    
    perform_data_leakage_check(df_features, FINAL_COUPLED_FEATURES)
    
    y_train = df_train["pm25_target"]
    y_val = df_val["pm25_target"]
    y_test = df_test["pm25_target"]
    
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    
    # 1. Old Baseline (with seasonal shortcuts)
    print("\n[1/5] Training Old Baseline (with seasonal shortcuts)...")
    m_old_base = xgb.XGBRegressor(**params).fit(df_train[BASELINE_FEATURES], y_train, eval_set=[(df_val[BASELINE_FEATURES], y_val)], verbose=False)
    met_old_base = calculate_metrics(y_test.values, m_old_base.predict(df_test[BASELINE_FEATURES]), model_name="Old Baseline")
    
    # 2. Old Coupled (with seasonal shortcuts)
    print("[2/5] Training Old Coupled (with seasonal shortcuts)...")
    m_old_coup = xgb.XGBRegressor(**params).fit(df_train[COUPLED_FEATURES_FULL], y_train, eval_set=[(df_val[COUPLED_FEATURES_FULL], y_val)], verbose=False)
    met_old_coup = calculate_metrics(y_test.values, m_old_coup.predict(df_test[COUPLED_FEATURES_FULL]), model_name="Old Coupled")
    
    # 3. Coupling Features ONLY
    print("[3/5] Training Coupling Features ONLY...")
    m_coup_only = xgb.XGBRegressor(**params).fit(df_train[COUPLING_FEATURES], y_train, eval_set=[(df_val[COUPLING_FEATURES], y_val)], verbose=False)
    met_coup_only = calculate_metrics(y_test.values, m_coup_only.predict(df_test[COUPLING_FEATURES]), model_name="Coupling Only")
    
    # 4. FINAL BASELINE (No is_winter, No month)
    print("[4/5] Training FINAL BASELINE (No seasonal shortcuts)...")
    m_final_base = xgb.XGBRegressor(**params).fit(df_train[FINAL_BASELINE_FEATURES], y_train, eval_set=[(df_val[FINAL_BASELINE_FEATURES], y_val)], verbose=False)
    y_pred_final_base = m_final_base.predict(df_test[FINAL_BASELINE_FEATURES])
    met_final_base = calculate_metrics(y_test.values, y_pred_final_base, model_name="Final Baseline")
    m_final_base.save_model(FINAL_BASELINE_MODEL_PATH)
    
    # 5. FINAL COUPLED (No is_winter, No month + Coupling Features)
    print("[5/5] Training FINAL COUPLED (No seasonal shortcuts + Coupling Features)...")
    m_final_coup = xgb.XGBRegressor(**params).fit(df_train[FINAL_COUPLED_FEATURES], y_train, eval_set=[(df_val[FINAL_COUPLED_FEATURES], y_val)], verbose=False)
    y_pred_final_coup = m_final_coup.predict(df_test[FINAL_COUPLED_FEATURES])
    met_final_coup = calculate_metrics(y_test.values, y_pred_final_coup, model_name="Final Coupled")
    m_final_coup.save_model(FINAL_COUPLED_MODEL_PATH)
    
    # ------------------ 6. FINAL COUPLING IMPACT ------------------
    mae_imp = ((met_final_base['mae'] - met_final_coup['mae']) / met_final_base['mae']) * 100.0
    rmse_imp = ((met_final_base['rmse'] - met_final_coup['rmse']) / met_final_base['rmse']) * 100.0
    r2_diff = met_final_coup['r2'] - met_final_base['r2']
    severe_imp = ((met_final_base['high_pollution_mae'] - met_final_coup['high_pollution_mae']) / met_final_base['high_pollution_mae']) * 100.0
    
    final_summary = {
        "forecast_horizon_hours": forecast_horizon,
        "models_evaluated": {
            "old_baseline": met_old_base,
            "old_coupled": met_old_coup,
            "coupling_only": met_coup_only,
            "final_baseline": met_final_base,
            "final_coupled": met_final_coup
        },
        "final_coupling_impact": {
            "mae_improvement_pct": float(round(mae_imp, 2)),
            "rmse_improvement_pct": float(round(rmse_imp, 2)),
            "r2_improvement": float(round(r2_diff, 4)),
            "severe_mae_improvement_pct": float(round(severe_imp, 2))
        }
    }
    
    with open(FINAL_METRICS_PATH, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\nSaved final metrics to '{FINAL_METRICS_PATH}'")
    
    # ------------------ 7. SAVE FINAL PREDICTIONS ------------------
    df_preds = pd.DataFrame({
        "timestamp": df_test["timestamp"].dt.strftime("%Y-%m-%d %H:00"),
        "station_id": df_test["station_id"],
        "actual_pm25": y_test.values,
        "final_baseline_prediction": y_pred_final_base,
        "final_coupled_prediction": y_pred_final_coup
    })
    df_preds.to_csv(FINAL_PREDICTIONS_PATH, index=False)
    print(f"Saved final predictions to '{FINAL_PREDICTIONS_PATH}'")
    
    # ------------------ 8. GENERATE PLOTS ------------------
    generate_final_plots(df_preds)
    print(f"Saved final plots to '{FINAL_PLOT_COMPARISON}' and '{FINAL_PLOT_HIGH_POLLUTION}'")
    
    # ------------------ 9. FEATURE IMPORTANCE ------------------
    importances = m_final_coup.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": FINAL_COUPLED_FEATURES,
        "display_name": [FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES],
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print("PHASE 3.6 5-MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<20} | {'Features':<18} | {'MAE':<8} | {'RMSE':<8} | {'R2':<7} | {'Severe MAE':<10}")
    print("-" * 75)
    print(f"{'Old Baseline':<20} | {'Weather+Lags+Season':<18} | {met_old_base['mae']:8.2f} | {met_old_base['rmse']:8.2f} | {met_old_base['r2']:7.4f} | {met_old_base['high_pollution_mae']:10.2f}")
    print(f"{'Old Coupled':<20} | {'Base+Coupl+Season':<18} | {met_old_coup['mae']:8.2f} | {met_old_coup['rmse']:8.2f} | {met_old_coup['r2']:7.4f} | {met_old_coup['high_pollution_mae']:10.2f}")
    print(f"{'Coupling Only':<20} | {'Coupling Terms':<18} | {met_coup_only['mae']:8.2f} | {met_coup_only['rmse']:8.2f} | {met_coup_only['r2']:7.4f} | {met_coup_only['high_pollution_mae']:10.2f}")
    print(f"{'Final Baseline':<20} | {'Weather+Lags (NoSeas)':<18} | {met_final_base['mae']:8.2f} | {met_final_base['rmse']:8.2f} | {met_final_base['r2']:7.4f} | {met_final_base['high_pollution_mae']:10.2f}")
    print(f"{'Final Coupled':<20} | {'Coupled (NoSeason)':<18} | {met_final_coup['mae']:8.2f} | {met_final_coup['rmse']:8.2f} | {met_final_coup['r2']:7.4f} | {met_final_coup['high_pollution_mae']:10.2f}")
    print("-" * 75)
    
    print("\nFINAL COUPLED VS FINAL BASELINE IMPACT:")
    print(f"  • MAE Improvement       : {mae_imp:+6.2f}%")
    print(f"  • RMSE Improvement      : {rmse_imp:+6.2f}%")
    print(f"  • R2 Score Improvement  : {r2_diff:+7.4f}")
    print(f"  • Severe MAE Improvement: {severe_imp:+6.2f}%")
    
    print("\nTOP 20 FINAL COUPLED MODEL FEATURE IMPORTANCES:")
    print("-" * 70)
    for idx, row in feat_imp.head(20).iterrows():
        is_coupling = " [COUPLING TERM]" if row['feature'] in COUPLING_FEATURES else ""
        print(f" {idx+1:2d}. {row['display_name']:<35} : {row['importance']:.4f}{is_coupling}")
    print("="*70 + "\n")
    
    return final_summary

if __name__ == "__main__":
    run_phase3_6_refinement()
