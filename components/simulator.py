"""
VayuSense - Weather What-If Simulator Component
===============================================
Performs full physics-informed feature reconstruction before running inference
on the trained Real Coupled XGBoost model (or Demo model).
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from src.coupling_engine import compute_coupling_features
from src.feature_engineering import FINAL_COUPLED_FEATURES
from src.aqi_calculator import calculate_sub_index, get_aqi_category

REAL_MODEL_PATH = "models/real_coupled_xgb.json"
DEMO_MODEL_PATH = "models/final_coupled_xgb.json"

@st.cache_resource
def load_coupled_model_for_sim(data_mode: str = "demo"):
    path = REAL_MODEL_PATH if data_mode.lower() == "real" else DEMO_MODEL_PATH
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

def render_what_if_simulator(base_row: pd.Series, data_mode: str = "demo"):
    is_real = (data_mode.lower() == "real")
    st.markdown("## 🧪 Weather What-If Simulator")
    st.markdown(f"Modify meteorological controls to simulate atmospheric changes and observe real-time predictions from the trained **{'Real' if is_real else 'Demo'} Coupled XGBoost model**.")
    
    model = load_coupled_model_for_sim(data_mode=data_mode)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎛️ Meteorological Controls")
        
        sim_wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.1, max_value=15.0,
            value=float(np.clip(base_row["wind_speed"], 0.1, 15.0)),
            step=0.1,
            help="Higher wind speeds increase horizontal advection and flushing of air pollutants."
        )
        
        sim_pbl_height = st.slider(
            "Planetary Boundary Layer (PBL) Height (m)",
            min_value=10, max_value=2000,
            value=int(np.clip(base_row["pbl_height"], 10, 2000)),
            step=10,
            help="Shallow boundary layers trap emissions near ground, causing high PM2.5."
        )
        
        sim_humidity = st.slider(
            "Relative Humidity (%)",
            min_value=15, max_value=100,
            value=int(np.clip(base_row["humidity"], 15, 100)),
            step=1,
            help="High humidity accelerates hygroscopic aerosol particle growth."
        )
        
        sim_temperature = st.slider(
            "Temperature (°C)",
            min_value=0.0, max_value=48.0,
            value=float(np.clip(base_row["temperature"], 0.0, 48.0)),
            step=0.5
        )

    sim_dict = base_row.to_dict()
    sim_dict["wind_speed"] = sim_wind_speed
    sim_dict["pbl_height"] = sim_pbl_height
    sim_dict["humidity"] = sim_humidity
    sim_dict["temperature"] = sim_temperature
    
    sim_df = pd.DataFrame([sim_dict])
    sim_df = compute_coupling_features(sim_df)
    
    X_sim = sim_df[FINAL_COUPLED_FEATURES]
    scenario_pred_pm25 = float(model.predict(X_sim)[0])
    scenario_aqi = calculate_sub_index(scenario_pred_pm25, "pm25")
    scenario_aqi_info = get_aqi_category(scenario_aqi)
    
    current_pm25 = float(base_row.get("final_coupled_prediction", base_row.get("pm25_target", base_row["pm25"])))
    current_aqi = calculate_sub_index(current_pm25, "pm25")
    current_aqi_info = get_aqi_category(current_aqi)
    
    delta_pm25 = scenario_pred_pm25 - current_pm25
    delta_aqi = scenario_aqi - current_aqi
    
    with col2:
        st.markdown("### 📊 Scenario vs Current Comparison")
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                f"""
                <div style="background:#1e293b; padding:16px; border-radius:12px; border:1px solid #334155; text-align:center;">
                    <h5 style="margin:0; color:#94a3b8;">CURRENT CONDITIONS</h5>
                    <h2 style="margin:4px 0; color:#ffffff;">{current_pm25:.1f} <span style="font-size:14px;">µg/m³</span></h2>
                    <p style="margin:0; font-size:14px; font-weight:700; color:{current_aqi_info['color']};">AQI {current_aqi:.0f} - {current_aqi_info['category']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with c_b:
            st.markdown(
                f"""
                <div style="background:{scenario_aqi_info['color']}; color:#ffffff; padding:16px; border-radius:12px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.2);">
                    <h5 style="margin:0; text-transform:uppercase; font-size:12px; opacity:0.9;">SCENARIO PREDICTION</h5>
                    <h2 style="margin:4px 0; font-size:32px; font-weight:800;">{scenario_pred_pm25:.1f} <span style="font-size:14px;">µg/m³</span></h2>
                    <p style="margin:0; font-size:14px; font-weight:700;">AQI {scenario_aqi:.0f} - {scenario_aqi_info['category']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Predicted Change in PM2.5",
            value=f"{scenario_pred_pm25:.1f} µg/m³",
            delta=f"{delta_pm25:+.1f} µg/m³",
            delta_color="inverse"
        )
        
        sim_vc = sim_wind_speed * sim_pbl_height
        st.write(f"**Scenario Ventilation Capacity ($VC = u \\times h_{{PBL}}$)**: `{sim_vc:.1f} m²/s`")
        
        is_stagnant = (sim_wind_speed < 1.5) and (sim_pbl_height < 400)
        st.write(f"**Air Stagnation Event (Wind < 1.5m/s & PBL < 400m)**: `{'ACTIVE ⚠️' if is_stagnant else 'INACTIVE ✅'}`")
        
        if sim_vc < 1500:
            st.warning("⚠️ **High Atmospheric Trapping Risk**: Scenario ventilation capacity is under 1500 m²/s!")
        else:
            st.success("✅ **Favorable Dispersion**: Scenario ventilation capacity exceeds 1500 m²/s.")
