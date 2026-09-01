"""
VayuSense - Real Data SHAP Explainability Engine
================================================
Computes global and local SHAP model attributions for the Real Coupled XGBoost model (models/real_coupled_xgb.json).
Uses untouched real chronological test dataset (1,312 observations).

Artifacts Output:
- models/real_shap_global_importance.png
- models/real_shap_beeswarm.png
- models/real_shap_coupling_importance.png
- models/real_shap_local_example.png
- models/real_shap_global_results.json
- models/real_shap_demo_explanation.json
"""

import os
import sys
import json
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import create_feature_pipeline, FINAL_COUPLED_FEATURES, COUPLING_FEATURES
from src.aqi_calculator import calculate_sub_index, get_aqi_category
from src.train_real_models import create_chronological_splits_real, FEATURE_NAME_MAP

REAL_MODEL_PATH = "models/real_coupled_xgb.json"
REAL_DATA_RAW = "data/raw/real_delhi_ncr_data.csv"

REAL_GLOBAL_RESULTS_PATH = "models/real_shap_global_results.json"
REAL_DEMO_EXPLANATION_PATH = "models/real_shap_demo_explanation.json"

PLOT_GLOBAL_IMP = "models/real_shap_global_importance.png"
PLOT_BEESWARM = "models/real_shap_beeswarm.png"
PLOT_COUPLING_IMP = "models/real_shap_coupling_importance.png"
PLOT_LOCAL_EX = "models/real_shap_local_example.png"

class VayuSenseRealSHAPExplainer:
    def __init__(self, model_path: str = REAL_MODEL_PATH):
        self.model_path = model_path
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.feature_names = FINAL_COUPLED_FEATURES
        self.coupling_features = COUPLING_FEATURES
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self, X: pd.DataFrame):
        X_ordered = X[self.feature_names]
        return self.explainer(X_ordered)

    def explain_prediction(self, input_row: pd.Series) -> dict:
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
            
        pos_contribs = [c for c in contributors if c["shap_value"] > 0]
        neg_contribs = [c for c in contributors if c["shap_value"] < 0]
        
        pos_contribs.sort(key=lambda x: x["shap_value"], reverse=True)
        neg_contribs.sort(key=lambda x: x["shap_value"])
        
        narrative = self._generate_narrative(pred_pm25, base_value, pos_contribs, neg_contribs)
        
        return {
            "data_mode": "REAL DATA",
            "data_source_note": "Air quality: Open-Meteo CAMS atmospheric stream | Weather: Open-Meteo ERA5",
            "predicted_pm25": float(round(pred_pm25, 2)),
            "base_value": float(round(base_value, 2)),
            "positive_contributors": pos_contribs[:5],
            "negative_contributors": neg_contribs[:5],
            "narrative": narrative
        }

    def _generate_narrative(self, pred_pm25, base_val, pos_contribs, neg_contribs) -> str:
        aqi_sub = calculate_sub_index(pred_pm25, "pm25")
        cat_info = get_aqi_category(aqi_sub)
        
        lines = [
            f"VayuSense predicts PM2.5 at **{pred_pm25:.1f} µg/m³** (CPCB AQI **{aqi_sub:.0f} - {cat_info['category']}**).",
            f"Baseline expected model output across test dataset is `{base_val:.1f} µg/m³`."
        ]
        
        if pos_contribs:
            top_pos = pos_contribs[0]
            lines.append(f"Prediction increased mainly because of: **{top_pos['display_name']}** (`+{top_pos['shap_value']:.2f} µg/m³`).")
            
        if neg_contribs:
            top_neg = neg_contribs[0]
            lines.append(f"Prediction suppressed by: **{top_neg['display_name']}** (`{top_neg['shap_value']:.2f} µg/m³`).")
            
        coupling_pos = [c for c in pos_contribs if c["is_coupling"]]
        if coupling_pos:
            lines.append("The prediction is also influenced by physics-informed proxy features, including boundary density ratio and thermal inversion conditions.")
            
        return "\n".join(lines)

def run_real_shap_analysis():
    print("="*80)
    print("VAYUSENSE - REAL DATA SHAP EXPLAINABILITY ENGINE")
    print("="*80)
    
    explainer_obj = VayuSenseRealSHAPExplainer()
    print(f"SHAP Version: {shap.__version__}")
    
    # 1. Load Real Dataset and extract chronological test set
    df_raw = pd.read_csv(REAL_DATA_RAW)
    df_features = create_feature_pipeline(df_raw, forecast_horizon=6)
    _, _, df_test = create_chronological_splits_real(df_features)
    
    n_test = len(df_test)
    print(f"Analyzing {n_test} real test set observations (Date Range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()})...")
    
    X_test = df_test[FINAL_COUPLED_FEATURES]
    explanation = explainer_obj.get_shap_values(X_test)
    shap_matrix = explanation.values
    
    # ------------------ 2. GLOBAL SHAP RANKING ------------------
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_rank = pd.DataFrame({
        "feature": FINAL_COUPLED_FEATURES,
        "display_name": [FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES],
        "mean_abs_shap": mean_abs_shap,
        "is_coupling": [f in COUPLING_FEATURES for f in FINAL_COUPLED_FEATURES]
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    print("\nTOP 20 REAL COUPLED SHAP FEATURES (Mean |SHAP Value| in ug/m3):")
    print("-" * 75)
    for idx, row in global_rank.head(20).iterrows():
        is_coup_str = " [COUPLING TERM]" if row["is_coupling"] else ""
        print(f" {idx+1:2d}. {row['display_name']:<35} : {row['mean_abs_shap']:.4f}{is_coup_str}")
    print("-" * 75)
    
    # ------------------ 3. COUPLING SHAP ATTRIBUTION SHARE ------------------
    coupling_indices = [FINAL_COUPLED_FEATURES.index(f) for f in COUPLING_FEATURES]
    non_coupling_indices = [i for i in range(len(FINAL_COUPLED_FEATURES)) if i not in coupling_indices]
    
    tot_coupling_shap = float(np.sum(mean_abs_shap[coupling_indices]))
    tot_non_coupling_shap = float(np.sum(mean_abs_shap[non_coupling_indices]))
    tot_shap = tot_coupling_shap + tot_non_coupling_shap
    
    pct_coupling = (tot_coupling_shap / tot_shap) * 100.0
    pct_non_coupling = (tot_non_coupling_shap / tot_shap) * 100.0
    
    print("\nREAL DATA COUPLING SHAP ATTRIBUTION SHARE:")
    print("-" * 75)
    print(f"  • Total Coupling SHAP Magnitude    : {tot_coupling_shap:.4f} ug/m3 ({pct_coupling:.2f}% share)")
    print(f"  • Total Non-Coupling SHAP Magnitude: {tot_non_coupling_shap:.4f} ug/m3 ({pct_non_coupling:.2f}% share)")
    print("-" * 75)
    
    # ------------------ 4. HIGH-RISK LOCAL DEMO SELECTION ------------------
    # Select highest available real test prediction
    preds_test = explainer_obj.model.predict(X_test)
    highest_idx = int(np.argmax(preds_test))
    high_row = df_test.iloc[highest_idx]
    
    local_demo_explanation = explainer_obj.explain_prediction(high_row)
    local_demo_explanation["label"] = "Highest available real test prediction"
    
    pred_pm25_val = local_demo_explanation["predicted_pm25"]
    aqi_val = calculate_sub_index(pred_pm25_val, "pm25")
    aqi_cat_info = get_aqi_category(aqi_val)
    
    print("\nHIGHEST AVAILABLE REAL TEST PREDICTION DEMO CASE:")
    print("-" * 75)
    print(f"  Label        : {local_demo_explanation['label']}")
    print(f"  Timestamp    : {high_row['timestamp']}")
    print(f"  Station      : {high_row['station_id']}")
    print(f"  Actual PM2.5 : {high_row['pm25_target']:.1f} ug/m3")
    print(f"  Pred PM2.5   : {pred_pm25_val:.1f} ug/m3")
    print(f"  AQI Score    : {aqi_val:.0f} ({aqi_cat_info['category']} Category)")
    print(f"  Base Value   : {local_demo_explanation['base_value']:.1f} ug/m3")
    print("\n  Top Positive SHAP Contributors:")
    for item in local_demo_explanation["positive_contributors"]:
        print(f"   • {item['display_name']:<35}: +{item['shap_value']:.2f} ug/m3 (Value: {item['raw_value']:.2f})")
    print("\n  Top Negative SHAP Contributors:")
    for item in local_demo_explanation["negative_contributors"]:
        print(f"   • {item['display_name']:<35}: {item['shap_value']:.2f} ug/m3 (Value: {item['raw_value']:.2f})")
    print(f"\n  Narrative:\n   \"{local_demo_explanation['narrative']}\"")
    print("-" * 75)
    
    # ------------------ 5. GENERATE PLOTS ------------------
    os.makedirs("models", exist_ok=True)
    
    # Plot 1: Real Global SHAP Importance
    plt.figure(figsize=(10, 6))
    top_global = global_rank.head(15).iloc[::-1]
    colors = ["#00e5ff" if f in COUPLING_FEATURES else "#1f77b4" for f in top_global["feature"]]
    plt.barh(top_global["display_name"], top_global["mean_abs_shap"], color=colors)
    plt.xlabel("Real Data Mean |SHAP Value| (Impact in ug/m3)")
    plt.title("Real Dataset: Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_GLOBAL_IMP, dpi=150)
    plt.close()
    
    # Plot 2: Real SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test, feature_names=[FEATURE_NAME_MAP.get(f, f) for f in FINAL_COUPLED_FEATURES], show=False)
    plt.title("Real Dataset: SHAP Summary Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(PLOT_BEESWARM, dpi=150)
    plt.close()
    
    # Plot 3: Real Coupling Feature Importance
    plt.figure(figsize=(10, 5))
    coupling_rank = global_rank[global_rank["is_coupling"]].sort_values(by="mean_abs_shap", ascending=True)
    plt.barh(coupling_rank["display_name"], coupling_rank["mean_abs_shap"], color="#00e5ff")
    plt.xlabel("Real Data Mean |SHAP Value| (ug/m3)")
    plt.title("Real Dataset: Physics-Informed Coupling Terms SHAP Magnitude")
    plt.tight_layout()
    plt.savefig(PLOT_COUPLING_IMP, dpi=150)
    plt.close()
    
    # Plot 4: Local Single-Sample SHAP Explanation
    plt.figure(figsize=(10, 5))
    top_pos_neg = local_demo_explanation["positive_contributors"][:4] + local_demo_explanation["negative_contributors"][:4]
    top_pos_neg.sort(key=lambda x: x["shap_value"])
    
    names = [x["display_name"] for x in top_pos_neg]
    vals = [x["shap_value"] for x in top_pos_neg]
    colors_loc = ["#ff3366" if v > 0 else "#00e5ff" for v in vals]
    
    plt.barh(names, vals, color=colors_loc)
    plt.xlabel("SHAP Impact on Forecast (ug/m3)")
    plt.title(f"Real Dataset: Local SHAP Explanation ({high_row['station_id']}, {high_row['timestamp']})")
    plt.tight_layout()
    plt.savefig(PLOT_LOCAL_EX, dpi=150)
    plt.close()
    
    # ------------------ 6. SAVE RESULTS JSON ------------------
    limitations = {
        "dataset_period": "2024-01-01 to 2024-04-01 (~3 months)",
        "high_pollution_evaluation_note": "Test period contains no PM2.5 >= 250 ug/m3 observations; severe-event SHAP cannot be evaluated statistically.",
        "pbl_variability": "PBL height has limited variability in surface winter reanalysis grid (median 40m, max 45m).",
        "data_provenance": "Air quality: Open-Meteo CAMS atmospheric stream | Weather: Open-Meteo ERA5"
    }
    
    global_results_json = {
        "data_mode": "REAL DATA",
        "shap_version": shap.__version__,
        "test_observations_count": n_test,
        "global_ranking": global_rank.to_dict(orient="records"),
        "attribution_share": {
            "total_coupling_shap_magnitude": tot_coupling_shap,
            "total_non_coupling_shap_magnitude": tot_non_coupling_shap,
            "coupling_pct_share": float(round(pct_coupling, 2)),
            "non_coupling_pct_share": float(round(pct_non_coupling, 2))
        },
        "limitations": limitations
    }
    
    with open(REAL_GLOBAL_RESULTS_PATH, "w") as f:
        json.dump(global_results_json, f, indent=2)
    print(f"Saved real SHAP global results to '{REAL_GLOBAL_RESULTS_PATH}'")
    
    with open(REAL_DEMO_EXPLANATION_PATH, "w") as f:
        json.dump(local_demo_explanation, f, indent=2)
    print(f"Saved real demo explanation to '{REAL_DEMO_EXPLANATION_PATH}'")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_real_shap_analysis()
