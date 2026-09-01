"""
VayuSense - SHAP Explainability Engine (Phase 4)
=================================================
Provides global, local, coupling-specific, and high-pollution SHAP model attributions.
Uses TreeExplainer on the trained Final Coupled XGBoost model.

SCIENTIFIC LANGUAGE DIRECTIVE:
- "SHAP identifies which features contributed most to the model's prediction."
- "Physics-informed proxy features"
- "Model attribution by SHAP magnitude" (NOT physical causation).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from src.feature_engineering import FINAL_COUPLED_FEATURES, COUPLING_FEATURES
from src.aqi_calculator import calculate_sub_index, get_aqi_category

MODEL_PATH = "models/final_coupled_xgb.json"
TEST_DATA_PATH = "data/processed/coupled_features.csv"
GLOBAL_RESULTS_PATH = "models/shap_global_results.json"
DEMO_EXPLANATION_PATH = "models/shap_demo_explanation.json"
SHAP_CSV_PATH = "data/processed/shap_test_values.csv"

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
    "ventilation_coeff": "Ventilation Capacity (m²/s)",
    "pm25_x_humidity": "Humidity-PM2.5 Interaction",
    "pm25_x_temp": "Temperature-PM2.5 Interaction",
    "pm25_x_wind_speed": "Wind Transport Interaction",
    "pm25_div_pbl": "Boundary Density Ratio",
    "stagnation_indicator": "Air Stagnation Event Flag",
    "inversion_indicator": "Thermal Inversion Layer Flag",
    "lagged_pm25_coupling": "Lagged Pollutant Trapping Index"
}

class VayuSenseSHAPExplainer:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.feature_names = FINAL_COUPLED_FEATURES
        self.coupling_features = COUPLING_FEATURES
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self, X: pd.DataFrame):
        """
        Computes SHAP Explanation object and raw values for input DataFrame X.
        Ensures column ordering strictly matches training feature order.
        """
        X_ordered = X[self.feature_names]
        shap_explanation = self.explainer(X_ordered)
        return shap_explanation

    def explain_prediction(self, input_row: pd.Series) -> dict:
        """
        Explains single prediction instance using local SHAP value contributions.
        Determines positive vs negative contributors strictly by SHAP sign.
        """
        df_single = pd.DataFrame([input_row.to_dict()])[self.feature_names]
        explanation = self.get_shap_values(df_single)
        
        base_value = float(explanation.base_values[0])
        shap_vals = explanation.values[0]
        raw_vals = df_single.iloc[0].values
        
        pred_pm25 = float(base_value + np.sum(shap_vals))
        
        contributors = []
        for feat, val, raw in zip(self.feature_names, shap_vals, raw_vals):
            contributors.append({
                "feature": feat,
                "display_name": FEATURE_NAME_MAP.get(feat, feat),
                "shap_value": float(round(val, 4)),
                "raw_value": float(round(raw, 4)),
                "is_coupling": feat in self.coupling_features
            })
            
        positive_contributors = [c for c in contributors if c["shap_value"] > 0]
        negative_contributors = [c for c in contributors if c["shap_value"] < 0]
        
        positive_contributors.sort(key=lambda x: x["shap_value"], reverse=True)
        negative_contributors.sort(key=lambda x: x["shap_value"])  # Most negative first
        
        narrative_text = self._generate_human_readable_text(pred_pm25, base_value, positive_contributors, negative_contributors)
        
        return {
            "predicted_pm25": float(round(pred_pm25, 2)),
            "base_value": float(round(base_value, 2)),
            "positive_contributors": positive_contributors[:5],
            "negative_contributors": negative_contributors[:5],
            "narrative": narrative_text
        }

    def _generate_human_readable_text(self, pred_pm25, base_val, pos_contribs, neg_contribs) -> str:
        """Converts local SHAP attribution results into clear narrative text."""
        lines = []
        aqi_sub = calculate_sub_index(pred_pm25, "pm25")
        cat_info = get_aqi_category(aqi_sub)
        
        lines.append(f"VayuSense predicts PM2.5 at **{pred_pm25:.1f} µg/m³** (AQI **{aqi_sub:.0f} - {cat_info['category']}**).")
        lines.append(f"Baseline expected model output across test dataset is `{base_val:.1f} µg/m³`.")
        
        if pos_contribs:
            top_pos = pos_contribs[0]
            lines.append(f"Top factor contributing positively to prediction: **{top_pos['display_name']}** (`+{top_pos['shap_value']:.2f} µg/m³`).")
            
        if neg_contribs:
            top_neg = neg_contribs[0]
            lines.append(f"Top factor suppressing prediction: **{top_neg['display_name']}** (`{top_neg['shap_value']:.2f} µg/m³`).")
            
        coupling_pos = [c for c in pos_contribs if c["is_coupling"]]
        if coupling_pos:
            lines.append("The prediction is also influenced by physics-informed proxy features, including atmospheric dispersion and humidity-particle interactions.")
            
        return "\n".join(lines)

def run_shap_analysis():
    print("="*70)
    print("VAYUSENSE - PHASE 4: SHAP EXPLAINABILITY ENGINE")
    print("="*70)
    
    explainer_obj = VayuSenseSHAPExplainer()
    print(f"SHAP Version: {shap.__version__}")
    
    # Load processed dataset and split test set (15% final chronologically)
    df_all = pd.read_csv(TEST_DATA_PATH)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    test_dfs = []
    for station, group in df_all.groupby("station_id", sort=False):
        n = len(group)
        idx_test = int(n * 0.85)
        test_dfs.append(group.iloc[idx_test:])
    df_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)
    
    n_test = len(df_test)
    print(f"Analyzing {n_test} test observations for SHAP explainability...")
    
    # Compute SHAP values across test dataset
    X_test = df_test[FINAL_COUPLED_FEATURES]
    explanation = explainer_obj.get_shap_values(X_test)
    shap_matrix = explanation.values
    
    # ------------------ 1. GLOBAL EXPLAINABILITY ------------------
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_rank = pd.DataFrame({
        "feature": FINAL_COUPLED_FEATURES,
        "display_name": [FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES],
        "mean_abs_shap": mean_abs_shap,
        "is_coupling": [f in COUPLING_FEATURES for f in FINAL_COUPLED_FEATURES]
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    print("\nTOP 20 GLOBAL SHAP FEATURES (Mean |SHAP Value|):")
    print("-" * 65)
    for idx, row in global_rank.head(20).iterrows():
        is_coup_str = " [COUPLING TERM]" if row["is_coupling"] else ""
        print(f" {idx+1:2d}. {row['display_name']:<35} : {row['mean_abs_shap']:.4f}{is_coup_str}")
    print("-" * 65)
    
    # ------------------ 2. COUPLING VS NON-COUPLING ATTRIBUTION ------------------
    coupling_indices = [FINAL_COUPLED_FEATURES.index(f) for f in COUPLING_FEATURES]
    non_coupling_indices = [i for i in range(len(FINAL_COUPLED_FEATURES)) if i not in coupling_indices]
    
    tot_coupling_shap = float(np.sum(mean_abs_shap[coupling_indices]))
    tot_non_coupling_shap = float(np.sum(mean_abs_shap[non_coupling_indices]))
    tot_shap = tot_coupling_shap + tot_non_coupling_shap
    
    pct_coupling = (tot_coupling_shap / tot_shap) * 100.0
    pct_non_coupling = (tot_non_coupling_shap / tot_shap) * 100.0
    
    print("\nCOUPLING VS NON-COUPLING SHAP ATTRIBUTION SHARE:")
    print("-" * 65)
    print(f"  • Total Coupling SHAP Magnitude    : {tot_coupling_shap:.4f} ({pct_coupling:.2f}% share)")
    print(f"  • Total Non-Coupling SHAP Magnitude: {tot_non_coupling_shap:.4f} ({pct_non_coupling:.2f}% share)")
    print("-" * 65)
    
    # ------------------ 3. HIGH-POLLUTION SHAP ANALYSIS (PM2.5 >= 250) ------------------
    severe_mask = df_test["pm25_target"] >= 250.0
    if severe_mask.sum() > 0:
        shap_severe = shap_matrix[severe_mask.values]
        mean_abs_severe = np.mean(np.abs(shap_severe), axis=0)
        
        severe_rank = pd.DataFrame({
            "feature": FINAL_COUPLED_FEATURES,
            "display_name": [FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES],
            "mean_abs_shap": mean_abs_severe,
            "is_coupling": [f in COUPLING_FEATURES for f in FINAL_COUPLED_FEATURES]
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        
        print(f"\nHIGH-POLLUTION SHAP RANKING (PM2.5 >= 250 ug/m3, N={severe_mask.sum()}):")
        print("-" * 65)
        for idx, row in severe_rank.head(15).iterrows():
            is_coup_str = " [COUPLING TERM]" if row["is_coupling"] else ""
            print(f" {idx+1:2d}. {row['display_name']:<35} : {row['mean_abs_shap']:.4f}{is_coup_str}")
        print("-" * 65)
    else:
        severe_rank = global_rank
        
    # ------------------ 4. DEMO CASE SELECTION & EXPLANATION ------------------
    severe_indices = np.where(severe_mask)[0]
    demo_idx = severe_indices[min(10, len(severe_indices)-1)]
    demo_row = df_test.iloc[demo_idx]
    
    demo_explanation = explainer_obj.explain_prediction(demo_row)
    demo_aqi = calculate_sub_index(demo_explanation["predicted_pm25"], "pm25")
    
    print("\nDEMO SEVERE POLLUTION CASE EXPLANATION:")
    print("-" * 65)
    print(f"  Timestamp    : {demo_row['timestamp']}")
    print(f"  Station      : {demo_row['station_id']}")
    print(f"  Actual PM2.5 : {demo_row['pm25_target']:.1f} ug/m3")
    print(f"  Pred PM2.5   : {demo_explanation['predicted_pm25']:.1f} ug/m3 (AQI {demo_aqi:.0f})")
    print(f"  Base Value   : {demo_explanation['base_value']:.1f} ug/m3")
    print("\n  Top Positive SHAP Contributors (Increasing predicted PM2.5):")
    for item in demo_explanation["positive_contributors"]:
        print(f"   • {item['display_name']:<35}: +{item['shap_value']:.2f} ug/m3 (Value: {item['raw_value']:.2f})")
    print("\n  Top Negative SHAP Contributors (Decreasing predicted PM2.5):")
    for item in demo_explanation["negative_contributors"]:
        print(f"   • {item['display_name']:<35}: {item['shap_value']:.2f} ug/m3 (Value: {item['raw_value']:.2f})")
    print("\n  Natural Language Narrative:")
    print(f"   \"{demo_explanation['narrative']}\"")
    print("-" * 65)
    
    # ------------------ 5. GENERATE PLOTS ------------------
    os.makedirs("models", exist_ok=True)
    
    # Plot 1: Global SHAP Feature Importance
    plt.figure(figsize=(10, 6))
    top_global = global_rank.head(15).iloc[::-1]
    colors = ["#00e5ff" if f in COUPLING_FEATURES else "#1f77b4" for f in top_global["feature"]]
    plt.barh(top_global["display_name"], top_global["mean_abs_shap"], color=colors)
    plt.xlabel("Mean |SHAP Value| (Impact on PM2.5 Forecast in ug/m3)")
    plt.title("Plot 1: Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("models/shap_global_importance.png", dpi=150)
    plt.close()
    
    # Plot 2: SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test, feature_names=[FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES], show=False)
    plt.title("Plot 2: SHAP Summary Beeswarm Plot")
    plt.tight_layout()
    plt.savefig("models/shap_beeswarm.png", dpi=150)
    plt.close()
    
    # Plot 3: Coupling Feature SHAP Importance
    plt.figure(figsize=(10, 5))
    coupling_rank = global_rank[global_rank["is_coupling"]].sort_values(by="mean_abs_shap", ascending=True)
    plt.barh(coupling_rank["display_name"], coupling_rank["mean_abs_shap"], color="#00e5ff")
    plt.xlabel("Mean |SHAP Value| (ug/m3)")
    plt.title("Plot 3: Physics-Informed Coupling Feature SHAP Magnitude")
    plt.tight_layout()
    plt.savefig("models/shap_coupling_importance.png", dpi=150)
    plt.close()
    
    # Plot 4: High-Pollution Episode SHAP Importance
    plt.figure(figsize=(10, 6))
    top_severe = severe_rank.head(15).iloc[::-1]
    colors_sev = ["#ff3366" if f in COUPLING_FEATURES else "#334155" for f in top_severe["feature"]]
    plt.barh(top_severe["display_name"], top_severe["mean_abs_shap"], color=colors_sev)
    plt.xlabel("Mean |SHAP Value| during Severe Smog Episodes (PM2.5 >= 250 ug/m3)")
    plt.title("Plot 4: High-Pollution Episode SHAP Importance")
    plt.tight_layout()
    plt.savefig("models/shap_high_pollution.png", dpi=150)
    plt.close()
    
    # Plot 5: Local Example Waterfall Bar Chart
    plt.figure(figsize=(10, 5))
    top_pos_neg = demo_explanation["positive_contributors"][:4] + demo_explanation["negative_contributors"][:4]
    top_pos_neg.sort(key=lambda x: x["shap_value"])
    
    names = [x["display_name"] for x in top_pos_neg]
    vals = [x["shap_value"] for x in top_pos_neg]
    col_bar = ["#ff3366" if v > 0 else "#00e5ff" for v in vals]
    
    plt.barh(names, vals, color=col_bar)
    plt.xlabel("SHAP Impact on Forecast (ug/m3)")
    plt.title(f"Plot 5: Local Single-Sample SHAP Explanation ({demo_row['station_id']})")
    plt.tight_layout()
    plt.savefig("models/shap_local_example.png", dpi=150)
    plt.close()
    
    # ------------------ 6. SAVE RESULTS JSON & CSV ------------------
    global_results_json = {
        "shap_version": shap.__version__,
        "test_observations_count": n_test,
        "global_ranking": global_rank.to_dict(orient="records"),
        "attribution_share": {
            "total_coupling_shap_magnitude": tot_coupling_shap,
            "total_non_coupling_shap_magnitude": tot_non_coupling_shap,
            "coupling_pct_share": float(round(pct_coupling, 2)),
            "non_coupling_pct_share": float(round(pct_non_coupling, 2))
        },
        "high_pollution_ranking": severe_rank.to_dict(orient="records")
    }
    
    with open(GLOBAL_RESULTS_PATH, "w") as f:
        json.dump(global_results_json, f, indent=2)
    print(f"Saved global SHAP results to '{GLOBAL_RESULTS_PATH}'")
    
    with open(DEMO_EXPLANATION_PATH, "w") as f:
        json.dump(demo_explanation, f, indent=2)
    print(f"Saved demo explanation to '{DEMO_EXPLANATION_PATH}'")
    
    # Save long-format SHAP CSV
    shap_df_list = []
    for i in range(min(500, len(df_test))):
        ts = df_test.iloc[i]["timestamp"]
        st_id = df_test.iloc[i]["station_id"]
        for feat, val in zip(FINAL_COUPLED_FEATURES, shap_matrix[i]):
            shap_df_list.append({
                "timestamp": ts,
                "station_id": st_id,
                "feature": feat,
                "shap_value": float(round(val, 4))
            })
    pd.DataFrame(shap_df_list).to_csv(SHAP_CSV_PATH, index=False)
    print(f"Saved SHAP test values sample CSV to '{SHAP_CSV_PATH}'")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_shap_analysis()
