"""
VayuSense - Model Performance & Benchmark View
==============================================
Displays comparative evaluation metrics loaded from models/real_model_metrics.json (Real Mode)
or models/final_model_metrics.json (Demo Mode).
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from components.charts import create_pm25_forecast_chart

REAL_METRICS_PATH = "models/real_model_metrics.json"
DEMO_METRICS_PATH = "models/final_model_metrics.json"

def render_model_performance_view(df_preds: pd.DataFrame, station_id: str, data_mode: str = "demo"):
    st.markdown("## ⚖️ Model Performance (Baseline vs Coupled XGBoost)")
    
    if data_mode.lower() == "real":
        # st.markdown(
        #     """
        #     > 🌐 **REAL DATA BENCHMARK EVALUATION**:  
        #     > Models evaluated on untouched real test set (`2024-03-19` to `2024-04-01`, 1,312 observations across 4 Delhi NCR stations).  
        #     > Air Quality Stream: **Open-Meteo CAMS Air Quality Stream** | Weather Stream: **Open-Meteo ERA5**  
        #     > *High pollution (≥ 250 µg/m³) was not present in the spring 2024 real test window.*
        #     """
        # )
        
        if os.path.exists(REAL_METRICS_PATH):
            with open(REAL_METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
                
            b = metrics_data.get("baseline_metrics", {})
            c = metrics_data.get("coupled_metrics", {})
            imp = metrics_data.get("improvements", {})
            
            table_rows = [
                {
                    "Metric": "Overall MAE (µg/m³)",
                    "Real Baseline XGBoost": f"{b.get('mae', 0):.2f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('mae', 0):.2f}",
                    "Improvement": f"{imp.get('mae_improvement_pct', 0):+.2f}%"
                },
                {
                    "Metric": "Overall RMSE (µg/m³)",
                    "Real Baseline XGBoost": f"{b.get('rmse', 0):.2f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('rmse', 0):.2f}",
                    "Improvement": f"{imp.get('rmse_improvement_pct', 0):+.2f}%"
                },
                {
                    "Metric": "R² Score",
                    "Real Baseline XGBoost": f"{b.get('r2', 0):.4f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('r2', 0):.4f}",
                    "Improvement": f"{imp.get('r2_improvement', 0):+.4f} (R² gain)"
                },
                {
                    "Metric": "Low Ventilation MAE (<1500 m²/s)",
                    "Real Baseline XGBoost": f"{b.get('low_ventilation_mae', 0):.2f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('low_ventilation_mae', 0):.2f}",
                    "Improvement": f"{imp.get('low_ventilation_mae_imp_pct', 0):+.2f}%"
                },
                {
                    "Metric": "Thermal Inversion MAE",
                    "Real Baseline XGBoost": f"{b.get('inversion_mae', 0):.2f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('inversion_mae', 0):.2f}",
                    "Improvement": f"{imp.get('inversion_mae_imp_pct', 0):+.2f}%"
                },
                {
                    "Metric": "Air Stagnation MAE",
                    "Real Baseline XGBoost": f"{b.get('stagnation_mae', 0):.2f}",
                    "Real Coupled XGBoost (VayuSense)": f"{c.get('stagnation_mae', 0):.2f}",
                    "Improvement": f"{imp.get('stagnation_mae_imp_pct', 0):+.2f}%"
                }
            ]
            
            st.markdown("### 📊 Real Model Benchmark Comparison Table")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            


        # Show Evaluation Plots if present
        if os.path.exists("models/real_model_comparison.png"):
            st.markdown("### 🖼️ Real Model Evaluation Charts")
            c1, c2 = st.columns(2)
            with c1:
                st.image("models/real_actual_vs_coupled.png", caption="Real Coupled XGBoost Prediction vs Actual Target")
            with c2:
                st.image("models/real_model_comparison.png", caption="Baseline vs Coupled Model Comparison")
    else:
        st.markdown(
            """
            > ⚠️ **DEMO MODE BENCHMARK NOTE**:  
            > Evaluated on synthetic demonstration dataset.  
            """
        )
        if os.path.exists(DEMO_METRICS_PATH):
            with open(DEMO_METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
                
            models_dict = metrics_data.get("models_evaluated", {})
            table_rows = []
            model_labels = {
                "final_baseline": ("Final Baseline", "Weather + Lags (No Season Shortcuts)"),
                "final_coupled": ("Final Coupled (VayuSense)", "Coupled Features (No Season Shortcuts)")
            }
            
            for key, (name, feats) in model_labels.items():
                if key in models_dict:
                    m = models_dict[key]
                    table_rows.append({
                        "Model Architecture": name,
                        "Feature Strategy": feats,
                        "MAE (µg/m³)": f"{m['mae']:.2f}",
                        "RMSE (µg/m³)": f"{m['rmse']:.2f}",
                        "R² Score": f"{m['r2']:.4f}",
                        "Severe MAE (≥250)": f"{m['high_pollution_mae']:.2f}"
                    })
                    
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"### 📈 Station Time-Series Forecast Plot ({station_id})")
    fig_fc = create_pm25_forecast_chart(df_preds, station_id=station_id)
    st.plotly_chart(fig_fc, use_container_width=True)
