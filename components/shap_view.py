"""
VayuSense - SHAP Explainability Dashboard View
================================================
Renders model attribution explanations, local waterfall plots, and coupling SHAP share.
Supports REAL DATA mode vs DEMO mode.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

REAL_GLOBAL_SHAP_PATH = "models/real_shap_global_results.json"
REAL_DEMO_EXPLANATION_PATH = "models/real_shap_demo_explanation.json"

DEMO_GLOBAL_SHAP_PATH = "models/shap_global_results.json"
DEMO_DEMO_EXPLANATION_PATH = "models/shap_demo_explanation.json"

def render_shap_view(explainer_obj, latest_row: pd.Series, data_mode: str = "demo"):
    st.markdown("## 🔍 Why This Forecast? (SHAP Explainability)")
    
    is_real = (data_mode.lower() == "real")
    global_path = REAL_GLOBAL_SHAP_PATH if is_real else DEMO_GLOBAL_SHAP_PATH
    local_path = REAL_DEMO_EXPLANATION_PATH if is_real else DEMO_DEMO_EXPLANATION_PATH
    
    st.markdown(
        f"""
        > 🟢 **LIVE SHAP EXPLANATION ({'LIVE STREAM DATA' if is_real else 'DEMO DATA'})**:  
        > SHAP estimates the positive or negative contribution of each feature to the **model output** relative to the expected model output.  
        > It explains **model decision behavior**, NOT direct physical causality.
        """
    )
    
    try:
        explanation = explainer_obj.explain_prediction(latest_row)
    except Exception as e:
        if is_real:
            st.error("Live SHAP explanation is currently unavailable.")
            st.warning(f"Underlying explainer message: {e}")
            return
        elif os.path.exists(local_path):
            with open(local_path, "r") as f:
                explanation = json.load(f)
        else:
            st.error(f"Error executing explainer: {e}")
            return
            
    c1, c2 = st.columns([1.1, 0.9])
    
    with c1:
        st.markdown("### 💬 Automated Narrative Explanation")
        st.info(explanation.get("narrative", "No narrative available."))
        
        st.markdown("#### **Top Factors Increasing Predicted PM2.5 (+)**")
        for item in explanation.get("positive_contributors", [])[:4]:
            is_coup = " [Physics Proxy]" if item.get("is_coupling", False) else ""
            st.markdown(f"• **{item['display_name']}**{is_coup}: <span style='color:#ef4444; font-weight:700;'>+{item['shap_value']:.2f} µg/m³</span> *(Value: {item['raw_value']:.2f})*", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### **Top Factors Decreasing Predicted PM2.5 (-)**")
        for item in explanation.get("negative_contributors", [])[:4]:
            is_coup = " [Physics Proxy]" if item.get("is_coupling", False) else ""
            st.markdown(f"• **{item['display_name']}**{is_coup}: <span style='color:#00e5ff; font-weight:700;'>{item['shap_value']:.2f} µg/m³</span> *(Value: {item['raw_value']:.2f})*", unsafe_allow_html=True)

    with c2:
        st.markdown("### 📊 Local Feature Attribution Breakdown")
        
        top_pos_neg = explanation.get("positive_contributors", [])[:4] + explanation.get("negative_contributors", [])[:4]
        top_pos_neg.sort(key=lambda x: x["shap_value"])
        
        names = [x["display_name"] for x in top_pos_neg]
        vals = [x["shap_value"] for x in top_pos_neg]
        colors = ["#ef4444" if v > 0 else "#00e5ff" for v in vals]
        
        fig = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f} µg/m³" for v in vals],
            textposition="outside"
        ))
        
        base_v = explanation.get("base_value", 0)
        pred_v = explanation.get("predicted_pm25", 0)
        
        fig.update_layout(
            title=f"SHAP Impact on Forecast (Base: {base_v:.1f} → Pred: {pred_v:.1f} µg/m³)",
            xaxis_title="PM2.5 Impact (µg/m³)",
            yaxis_title="",
            template="plotly_dark",
            height=420,
            margin=dict(l=160, r=60, t=50, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚛️ Contribution of Physics-Informed Proxy Features")
    
    if os.path.exists(global_path):
        with open(global_path, "r") as f:
            global_results = json.load(f)
            
        attr = global_results.get("attribution_share", {})
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(
                f"""
                <div style="background:#1e293b; padding:18px; border-radius:12px; border:1px solid #334155;">
                    <h4 style="margin:0; color:#94a3b8;">PHYSICS-INFORMED PROXY ATTRIBUTION SHARE</h4>
                    <h2 style="margin:6px 0; color:#00e5ff;">{attr.get('coupling_pct_share', 0):.2f}%</h2>
                    <p style="margin:0; font-size:13px; color:#cbd5e1;">Share of model attribution by SHAP magnitude across test data</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_b:
            st.markdown(
                f"""
                <div style="background:#1e293b; padding:18px; border-radius:12px; border:1px solid #334155;">
                    <h4 style="margin:0; color:#94a3b8;">AUTOREGRESSIVE & METEOROLOGICAL SHARE</h4>
                    <h2 style="margin:6px 0; color:#cbd5e1;">{attr.get('non_coupling_pct_share', 0):.2f}%</h2>
                    <p style="margin:0; font-size:13px; color:#cbd5e1;">Time-series lags and raw weather variables</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    if is_real and os.path.exists("models/real_shap_global_importance.png"):
        st.markdown("---")
        st.markdown("### 🖼️ Real Model SHAP Attribution Visualizations")
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            st.image("models/real_shap_global_importance.png", caption="Global SHAP Feature Importance Ranking")
            st.image("models/real_shap_coupling_importance.png", caption="Physics-Informed Proxy Terms SHAP Magnitude")
        with c_i2:
            st.image("models/real_shap_beeswarm.png", caption="SHAP Summary Beeswarm Plot")
            st.image("models/real_shap_local_example.png", caption="Highest Available Real Test Prediction Local SHAP Case")
