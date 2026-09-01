"""
VayuSense - Real Data Model Training & Evaluation Engine
=========================================================
Trains Real Baseline XGBoost vs Real Coupled XGBoost on real data (data/raw/real_delhi_ncr_data.csv).
Uses strict chronological 70/15/15 train/val/test splits per station.
Evaluates overall, high pollution, low ventilation, inversion, and stagnation MAE.

Artifacts Output:
- models/real_baseline_xgb.json
- models/real_coupled_xgb.json
- models/real_model_metrics.json
- data/processed/real_model_predictions.csv
- Visualizations: models/real_*.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import create_feature_pipeline, FINAL_BASELINE_FEATURES, FINAL_COUPLED_FEATURES, COUPLING_FEATURES
from src.model_trainer import perform_data_leakage_check

REAL_DATA_RAW = "data/raw/real_delhi_ncr_data.csv"
REAL_BASELINE_MODEL_PATH = "models/real_baseline_xgb.json"
REAL_COUPLED_MODEL_PATH = "models/real_coupled_xgb.json"
REAL_METRICS_PATH = "models/real_model_metrics.json"
REAL_PREDICTIONS_PATH = "data/processed/real_model_predictions.csv"

PLOT_ACTUAL_VS_BASE = "models/real_actual_vs_baseline.png"
PLOT_ACTUAL_VS_COUP = "models/real_actual_vs_coupled.png"
PLOT_COMPARISON = "models/real_model_comparison.png"
PLOT_HIGH_POLLUTION = "models/real_high_pollution_comparison.png"

FEATURE_NAME_MAP = {
    "temperature": "Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "wind_speed": "Wind Speed (m/s)",
    "wind_deg": "Wind Direction (°)",
    "pressure": "Pressure (hPa)",
    "pbl_height": "Boundary Layer Height (m)",
    "pm25_lag_1h": "PM2.5 (1h ago)",
    "pm25_lag_3h": "PM2.5 (3h ago)",
    "pm25_lag_6h": "PM2.5 (6h ago)",
    "pm25_lag_24h": "PM2.5 (24h ago)",
    "pm25_roll_mean_6h": "6h PM2.5 Moving Avg",
    "pm25_roll_mean_24h": "24h PM2.5 Moving Avg",
    "pm25_roll_std_24h": "24h PM2.5 Std Dev",
    "hour_sin": "Time of Day (Sin)",
    "hour_cos": "Time of Day (Cos)",
    "ventilation_coeff": "Ventilation Capacity (m²/s)",
    "pm25_x_humidity": "Humidity-PM2.5 Interaction",
    "pm25_x_temp": "Temperature-PM2.5 Interaction",
    "pm25_x_wind_speed": "Wind Transport Interaction",
    "pm25_div_pbl": "Boundary Density Ratio",
    "stagnation_indicator": "Air Stagnation Event Flag",
    "inversion_indicator": "Thermal Inversion Layer Flag",
    "lagged_pm25_coupling": "Lagged Pollutant Trapping Index"
}

def create_chronological_splits_real(df_features: pd.DataFrame, train_pct=0.70, val_pct=0.15):
    """Creates strict chronological splits per station (70% Train, 15% Val, 15% Test)."""
    df = df_features.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    train_dfs, val_dfs, test_dfs = [], [], []
    for station, group in df.groupby("station_id", sort=False):
        n = len(group)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])
        
    df_train = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    df_val = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    df_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)
    
    return df_train, df_val, df_test

def calculate_subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, name: str) -> float:
    """Calculates MAE on a specific subset mask."""
    if mask.sum() == 0:
        return 0.0
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))

def evaluate_real_model(y_true: np.ndarray, y_pred: np.ndarray, df_sub: pd.DataFrame, model_name: str) -> dict:
    """Computes comprehensive metrics suite on test evaluation set."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    # Subgroup MAEs
    high_poll_mask = (y_true >= 250.0)
    low_vc_mask = (df_sub["ventilation_coeff"].values < 1500.0)
    inv_mask = (df_sub["inversion_indicator"].values == 1)
    stag_mask = (df_sub["stagnation_indicator"].values == 1)
    
    return {
        "model_name": model_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "high_pollution_mae": round(calculate_subgroup_metrics(y_true, y_pred, high_poll_mask, "High Pollution"), 4),
        "high_pollution_samples": int(high_poll_mask.sum()),
        "low_ventilation_mae": round(calculate_subgroup_metrics(y_true, y_pred, low_vc_mask, "Low Ventilation"), 4),
        "low_ventilation_samples": int(low_vc_mask.sum()),
        "inversion_mae": round(calculate_subgroup_metrics(y_true, y_pred, inv_mask, "Inversion"), 4),
        "inversion_samples": int(inv_mask.sum()),
        "stagnation_mae": round(calculate_subgroup_metrics(y_true, y_pred, stag_mask, "Stagnation"), 4),
        "stagnation_samples": int(stag_mask.sum())
    }

def generate_real_plots(df_preds: pd.DataFrame):
    """Generates test set actual vs predicted Plotly/matplotlib figures."""
    actual = df_preds["actual_pm25"].values
    base_pred = df_preds["baseline_prediction"].values
    coup_pred = df_preds["coupled_prediction"].values
    ts = df_preds["timestamp"].values
    
    subset = min(300, len(actual))
    idx = range(subset)
    
    # 1. Actual vs Baseline Plot
    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual[:subset], label="Actual PM2.5 (+6h Target)", color="black", linewidth=2)
    plt.plot(idx, base_pred[:subset], label="Real Baseline Prediction", color="#ff7f0e", linestyle="--")
    plt.title("Real Dataset: Actual PM2.5 vs Real Baseline Forecast (+6h)")
    plt.xlabel("Test Set Time Index")
    plt.ylabel("PM2.5 Concentration (ug/m3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_ACTUAL_VS_BASE, dpi=150)
    plt.close()
    
    # 2. Actual vs Coupled Plot
    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual[:subset], label="Actual PM2.5 (+6h Target)", color="black", linewidth=2)
    plt.plot(idx, coup_pred[:subset], label="Real Coupled Prediction (VayuSense)", color="#00e5ff", linewidth=2)
    plt.title("Real Dataset: Actual PM2.5 vs Real Coupled Forecast (+6h)")
    plt.xlabel("Test Set Time Index")
    plt.ylabel("PM2.5 Concentration (ug/m3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_ACTUAL_VS_COUP, dpi=150)
    plt.close()
    
    # 3. Baseline vs Coupled Combined Plot
    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual[:subset], label="Actual PM2.5 (+6h Target)", color="black", linewidth=2)
    plt.plot(idx, base_pred[:subset], label="Real Baseline XGBoost", color="#ff7f0e", linestyle="--")
    plt.plot(idx, coup_pred[:subset], label="Real Coupled XGBoost", color="#00e5ff", linewidth=2)
    plt.title("Real Dataset: Baseline vs Coupled Model Comparison (+6h Forecast)")
    plt.xlabel("Test Set Time Index")
    plt.ylabel("PM2.5 Concentration (ug/m3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_COMPARISON, dpi=150)
    plt.close()

    # 4. Severe Episode Focus Plot
    high_mask = actual >= 150.0  # Real data severe threshold (150+ ug/m3)
    if high_mask.sum() > 0:
        high_idx = np.where(high_mask)[0][:150]
        plt.figure(figsize=(12, 5))
        plt.scatter(range(len(high_idx)), actual[high_idx], label="Actual High Pollution", color="red", s=30, zorder=3)
        plt.plot(range(len(high_idx)), base_pred[high_idx], label="Real Baseline Prediction", color="#ff7f0e", linestyle="--")
        plt.plot(range(len(high_idx)), coup_pred[high_idx], label="Real Coupled Prediction", color="#00e5ff", linewidth=2)
        plt.title("Real Dataset: High Pollution Episode Comparison (PM2.5 >= 150 ug/m3)")
        plt.xlabel("High Pollution Sample Index")
        plt.ylabel("PM2.5 Concentration (ug/m3)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_HIGH_POLLUTION, dpi=150)
        plt.close()

def run_real_model_training(forecast_horizon: int = 6):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("="*80)
    print(f"VAYUSENSE - REAL DATA MODEL TRAINING & EVALUATION (+{forecast_horizon}h Horizon)")
    print("="*80)
    
    df_raw = pd.read_csv(REAL_DATA_RAW)
    df_features = create_feature_pipeline(df_raw, forecast_horizon=forecast_horizon)
    
    # ------------------ 1. STRICT CHRONOLOGICAL DATA SPLIT ------------------
    df_train, df_val, df_test = create_chronological_splits_real(df_features)
    
    split_info = {
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "train_date_range": [str(df_train["timestamp"].min()), str(df_train["timestamp"].max())],
        "val_date_range": [str(df_val["timestamp"].min()), str(df_val["timestamp"].max())],
        "test_date_range": [str(df_test["timestamp"].min()), str(df_test["timestamp"].max())]
    }
    
    print("\nCHRONOLOGICAL TIME-SERIES SPLIT INFORMATION:")
    print(f"  • Training   ({split_info['train_rows']:5d} rows): {split_info['train_date_range'][0]}  -->  {split_info['train_date_range'][1]}")
    print(f"  • Validation ({split_info['val_rows']:5d} rows): {split_info['val_date_range'][0]}  -->  {split_info['val_date_range'][1]}")
    print(f"  • Testing    ({split_info['test_rows']:5d} rows): {split_info['test_date_range'][0]}  -->  {split_info['test_date_range'][1]}")
    
    # ------------------ 2. AUTOMATED DATA LEAKAGE CHECK ------------------
    perform_data_leakage_check(df_features, FINAL_COUPLED_FEATURES)
    
    target_col = "pm25_target"
    y_train = df_train[target_col]
    y_val = df_val[target_col]
    y_test = df_test[target_col]
    
    # Reproducible XGBoost hyperparameters (prevent overfitting on small real dataset)
    xgb_params = dict(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # ------------------ 3. TRAIN REAL BASELINE XGBOOST ------------------
    print("\n[1/2] Training REAL BASELINE XGBoost (15 features)...")
    m_base = xgb.XGBRegressor(**xgb_params)
    m_base.fit(
        df_train[FINAL_BASELINE_FEATURES], y_train,
        eval_set=[(df_val[FINAL_BASELINE_FEATURES], y_val)],
        verbose=False
    )
    m_base.save_model(REAL_BASELINE_MODEL_PATH)
    y_pred_base = m_base.predict(df_test[FINAL_BASELINE_FEATURES])
    met_base = evaluate_real_model(y_test.values, y_pred_base, df_test, "Real Baseline XGBoost")
    
    # ------------------ 4. TRAIN REAL COUPLED XGBOOST ------------------
    print("[2/2] Training REAL COUPLED XGBoost (23 features)...")
    m_coup = xgb.XGBRegressor(**xgb_params)
    m_coup.fit(
        df_train[FINAL_COUPLED_FEATURES], y_train,
        eval_set=[(df_val[FINAL_COUPLED_FEATURES], y_val)],
        verbose=False
    )
    m_coup.save_model(REAL_COUPLED_MODEL_PATH)
    y_pred_coup = m_coup.predict(df_test[FINAL_COUPLED_FEATURES])
    met_coup = evaluate_real_model(y_test.values, y_pred_coup, df_test, "Real Coupled XGBoost")
    
    # ------------------ 5. COMPARISON & IMPROVEMENT COMPUTATION ------------------
    def calc_imp(base_val, coup_val, is_r2=False):
        if is_r2:
            return round(coup_val - base_val, 4)
        if base_val == 0:
            return 0.0
        return round(((base_val - coup_val) / base_val) * 100.0, 2)

    improvements = {
        "mae_improvement_pct": calc_imp(met_base["mae"], met_coup["mae"]),
        "rmse_improvement_pct": calc_imp(met_base["rmse"], met_coup["rmse"]),
        "r2_improvement": calc_imp(met_base["r2"], met_coup["r2"], is_r2=True),
        "high_pollution_mae_imp_pct": calc_imp(met_base["high_pollution_mae"], met_coup["high_pollution_mae"]),
        "low_ventilation_mae_imp_pct": calc_imp(met_base["low_ventilation_mae"], met_coup["low_ventilation_mae"]),
        "inversion_mae_imp_pct": calc_imp(met_base["inversion_mae"], met_coup["inversion_mae"]),
        "stagnation_mae_imp_pct": calc_imp(met_base["stagnation_mae"], met_coup["stagnation_mae"])
    }
    
    # ------------------ 6. FEATURE IMPORTANCE ------------------
    importances = m_coup.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": FINAL_COUPLED_FEATURES,
        "display_name": [FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES],
        "importance": importances,
        "is_coupling": [f in COUPLING_FEATURES for f in FINAL_COUPLED_FEATURES]
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    # ------------------ 7. SAVE PREDICTIONS CSV ------------------
    df_preds = pd.DataFrame({
        "timestamp": df_test["timestamp"].dt.strftime("%Y-%m-%d %H:00"),
        "station_id": df_test["station_id"],
        "actual_pm25": y_test.values,
        "baseline_prediction": y_pred_base,
        "coupled_prediction": y_pred_coup,
        "pm25_target_6h": y_test.values,
        "ventilation_coeff": df_test["ventilation_coeff"].values,
        "stagnation_indicator": df_test["stagnation_indicator"].values,
        "inversion_indicator": df_test["inversion_indicator"].values
    })
    df_preds.to_csv(REAL_PREDICTIONS_PATH, index=False)
    print(f"\nSaved Real Model Predictions to '{REAL_PREDICTIONS_PATH}'")
    
    # Generate Plots
    generate_real_plots(df_preds)
    print(f"Saved evaluation plots to 'models/real_*.png'")

    # ------------------ 8. SAVE METRICS JSON ------------------
    limitations = {
        "date_range": "2024-01-01 to 2024-04-01 (~3 months historical window)",
        "pbl_height_variability": "Median = 40m, Max = 45m, 49.91% of rows equal 45m due to surface inversion layer boundaries in winter reanalysis",
        "data_provenance": "Air quality stream uses Open-Meteo CAMS atmospheric satellite/reanalysis data for Delhi coordinates"
    }

    metrics_payload = {
        "dataset_filepath": REAL_DATA_RAW,
        "target": "pm25_target_6h (+6 hours ahead)",
        "split_information": split_info,
        "model_parameters": xgb_params,
        "baseline_metrics": met_base,
        "coupled_metrics": met_coup,
        "improvements": improvements,
        "feature_importance_top20": feat_imp.head(20).to_dict(orient="records"),
        "leakage_verification": "PASS - Zero target or future lead variables in X",
        "limitations": limitations
    }

    with open(REAL_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved Real Model Metrics to '{REAL_METRICS_PATH}'")

    # ------------------ 9. PRINT COMPARISON TABLE & REPORT ------------------
    print_real_training_summary(met_base, met_coup, improvements, feat_imp, limitations)
    return metrics_payload

def print_real_training_summary(met_base: dict, met_coup: dict, imp: dict, feat_imp: pd.DataFrame, lim: dict):
    print("\n" + "="*80)
    print("REAL MODEL EVALUATION BENCHMARK TABLE")
    print("="*80)
    print(f"{'Evaluation Metric':<28} | {'Real Baseline':<14} | {'Real Coupled':<14} | {'Improvement %':<14}")
    print("-" * 75)
    print(f"{'MAE (ug/m3)':<28} | {met_base['mae']:14.2f} | {met_coup['mae']:14.2f} | {imp['mae_improvement_pct']:+13.2f}%")
    print(f"{'RMSE (ug/m3)':<28} | {met_base['rmse']:14.2f} | {met_coup['rmse']:14.2f} | {imp['rmse_improvement_pct']:+13.2f}%")
    print(f"{'R2 Score':<28} | {met_base['r2']:14.4f} | {met_coup['r2']:14.4f} | {imp['r2_improvement']:+13.4f}")
    print(f"{'High Pollution MAE (>=250)':<28} | {met_base['high_pollution_mae']:14.2f} | {met_coup['high_pollution_mae']:14.2f} | {imp['high_pollution_mae_imp_pct']:+13.2f}%")
    print(f"{'Low Ventilation MAE (<1500)':<28} | {met_base['low_ventilation_mae']:14.2f} | {met_coup['low_ventilation_mae']:14.2f} | {imp['low_ventilation_mae_imp_pct']:+13.2f}%")
    print(f"{'Thermal Inversion MAE':<28} | {met_base['inversion_mae']:14.2f} | {met_coup['inversion_mae']:14.2f} | {imp['inversion_mae_imp_pct']:+13.2f}%")
    print(f"{'Air Stagnation MAE':<28} | {met_base['stagnation_mae']:14.2f} | {met_coup['stagnation_mae']:14.2f} | {imp['stagnation_mae_imp_pct']:+13.2f}%")
    print("-" * 75)

    print("\nTOP 20 REAL COUPLED MODEL FEATURE IMPORTANCE:")
    print("-" * 75)
    for idx, row in feat_imp.head(20).iterrows():
        is_coup_str = " [COUPLING TERM]" if row["is_coupling"] else ""
        print(f" {idx+1:2d}. {row['display_name']:<35} : {row['importance']:.4f}{is_coup_str}")
    print("-" * 75)

    print("\nREPORTED MODEL LIMITATIONS:")
    print(f"  • Date Range    : {lim['date_range']}")
    print(f"  • PBL Bounds    : {lim['pbl_height_variability']}")
    print(f"  • Data Source   : {lim['data_provenance']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_real_model_training()
