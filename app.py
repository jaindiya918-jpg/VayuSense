"""
VayuSense - Air Pollution–Weather Coupled Forecasting Dashboard
================================================================
Streamlit Application Entry Point.
Supports:
- REAL DATA MODE: Ingests live near-real-time observations (OpenAQ API / Open-Meteo) for current live model inference.
- HISTORICAL MODEL EVALUATION: Benchmark testing on historical 2024 test split.
- DEMO MODE: Offline testing with pre-generated synthetic dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import truststore
truststore.inject_into_ssl()
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# Configure Page Layout & Modern Styling
st.set_page_config(
    page_title="VayuSense — Air Pollution–Weather Coupled Forecasting",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Dashboard CSS
st.markdown("""
    <style>
    /* Metric/KPI Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Remove default padding from metric container to fit background */
    div[data-testid="stMetric"] > div {
        margin: 0;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 5px;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Custom Styling for Streamlit Sidebar Radio Selection */
    .stRadio div[role="radiogroup"] > label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stRadio div[role="radiogroup"] > label:hover {
        background-color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)


# Insert project root to import modules cleanly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import Custom Dashboard Components
from components.header import render_header
from components.kpi_cards import (
    render_aqi_badge_card,
    render_overview_kpis,
    render_dispersion_cards,
    render_weather_grid
)
from components.charts import (
    create_pm25_forecast_chart,
    create_live_pm25_forecast_chart,
    create_coupling_terms_bar_chart
)
from components.shap_view import render_shap_view
from components.simulator import render_what_if_simulator
from components.map_view import render_station_map
from components.model_perf import render_model_performance_view

from src.aqi_calculator import calculate_overall_aqi, calculate_sub_index, get_aqi_category
from src.real_explainability import VayuSenseRealSHAPExplainer
from src.explainability import VayuSenseSHAPExplainer
from src.feature_engineering import create_feature_pipeline, FINAL_COUPLED_FEATURES

@st.cache_data(ttl=600)
def load_live_dataset(data_mode: str):
    """
    Fetches near-real-time data for live model inference when DATA_MODE=real.
    Uses 10-minute cache TTL with manual refresh option.
    """
    if data_mode.lower() == "real":
        from src.real_data_pipeline import fetch_live_data_pipeline
        return fetch_live_data_pipeline(days_back=3)
    else:
        df_raw = pd.read_csv("data/processed/coupled_features.csv")
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        return df_raw

@st.cache_data
def load_historical_benchmark_data(data_mode: str):
    """
    Loads historical 2024 test predictions for benchmark evaluation view.
    """
    is_real = (data_mode.lower() == "real")
    pred_path = "data/processed/real_model_predictions.csv" if is_real else "data/processed/final_model_predictions.csv"
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])
        return df_pred
    else:
        return pd.DataFrame()

@st.cache_resource
def load_explainer_for_mode(data_mode: str):
    if data_mode.lower() == "real":
        return VayuSenseRealSHAPExplainer()
    else:
        return VayuSenseSHAPExplainer()

def main():
    data_mode = os.getenv("DATA_MODE", "real").lower()
    
    # Calculate timezone-aware current wall-clock system time in Asia/Kolkata (IST)
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    now_ist_str = now_ist.strftime("%d %b %Y, %H:%M IST")
    now_ist_naive = now_ist.replace(tzinfo=None)
    
    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.markdown("## 🌿 VayuSense Controls")
    
    if data_mode == "real":
        st.sidebar.success("🟢 Active Mode: **LIVE / CURRENT DATA**")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("🔄 Refresh Live", help="Fetch latest API observations from OpenAQ & Open-Meteo"):
                st.cache_data.clear()
                st.session_state.pop("selected_ts_str", None)
                st.rerun()
        with c2:
            if st.button("🕐 Latest Obs", help="Reset selected timestamp to latest available real observation"):
                st.session_state.pop("selected_ts_str", None)
                st.rerun()
    else:
        st.sidebar.warning("⚠️ Active Mode: **DEMO / SYNTHETIC**")
        
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "📌 Overview",
            "📈 PM2.5 Forecast",
            "🌦️ Atmospheric Conditions",
            "⚛️ Weather-Pollution Coupling",
            "🧪 Weather What-If Simulator",
            "🗺️ Delhi NCR Station Map",
            "⚖️ Model Performance"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Location & Station Selection")
    
    station_selected = st.sidebar.selectbox(
        "Monitoring Station",
        ["Anand_Vihar", "RK_Puram", "Punjabi_Bagh", "Mandir_Marg"],
        index=0
    )
    
    # Reset selected timestamp when user switches stations
    if st.session_state.get("prev_station") != station_selected:
        st.session_state["prev_station"] = station_selected
        st.session_state.pop("selected_ts_str", None)
        
    # Load dataset & model explainer
    try:
        df_features = load_live_dataset(data_mode)
        df_preds = load_historical_benchmark_data(data_mode)
        explainer_obj = load_explainer_for_mode(data_mode)
    except Exception as e:
        render_header(data_mode=data_mode)
        st.error("⚠️ **Current live data is temporarily unavailable.**")
        st.warning(f"Error fetching live API stream: {e}")
        st.info("Check network connection or OpenAQ/Open-Meteo API availability. To run in synthetic offline mode, set `DATA_MODE=demo` in `.env`.")
        return

    # Filter strictly for selected station, sort chronologically, and deduplicate
    df_st_feat = (
        df_features[df_features["station_id"] == station_selected]
        .sort_values(by="timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    
    if df_st_feat.empty:
        render_header(data_mode=data_mode)
        st.warning(f"No recent observation available for station '{station_selected}'.")
        return

    # Filter dataset for observations at or before current IST time (EXCLUDING FUTURE FORECAST HOURS)
    if data_mode == "real":
        df_st_obs = df_st_feat[df_st_feat["timestamp"] <= now_ist_naive]
        if df_st_obs.empty:
            df_st_obs = df_st_feat
    else:
        df_st_obs = df_st_feat

    # Extract available observation timestamps list (ascending chronological order)
    available_dt = pd.to_datetime(df_st_obs["timestamp"])
    available_obs_ts_strings = available_dt.dt.strftime("%Y-%m-%d %H:00").tolist()
    
    # Absolute latest REAL observation timestamp for selected station (max timestamp <= current IST time)
    latest_obs_ts_str = available_obs_ts_strings[-1]
    latest_obs_display_str = pd.to_datetime(latest_obs_ts_str).strftime("%d %b %Y, %H:00 IST")
    
    # Options list for selectbox: newest actual observations first (options[0] == latest_obs_ts_str!)
    options = available_obs_ts_strings[::-1]

    # Initialize / Default session_state selected timestamp MUST ALWAYS BE latest_obs_ts_str!
    if not st.session_state.get("selected_ts_str") or st.session_state["selected_ts_str"] not in options:
        st.session_state["selected_ts_str"] = latest_obs_ts_str

    default_idx = options.index(st.session_state["selected_ts_str"])

    # Allow user to explore past observations explicitly in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Observation Selector")
    
    selected_from_dropdown = st.sidebar.selectbox(
        "Selected Observation Time (IST)",
        options=options,
        index=default_idx,
        key=f"ts_select_{station_selected}",
        help="Select an observation timestamp. Default is automatically the latest available real observation."
    )
    st.session_state["selected_ts_str"] = selected_from_dropdown

    # Retrieve exact feature row for selected station & timestamp
    selected_feat_df = df_st_obs[df_st_obs["timestamp"].dt.strftime("%Y-%m-%d %H:00") == selected_from_dropdown]
    if not selected_feat_df.empty:
        selected_feat = selected_feat_df.iloc[-1]
    else:
        selected_feat = df_st_obs.iloc[-1]

    # Target Forecast Timestamp (+6 hours ahead from actual observation used)
    obs_ts = selected_feat["timestamp"]
    fc_ts = obs_ts + pd.Timedelta(hours=6)
    fc_ts_str = fc_ts.strftime("%d %b %Y, %H:00 IST")
    obs_ts_str = obs_ts.strftime("%d %b %Y, %H:00 IST")

    # ------------------ TOP HEADER BANNER ------------------
    render_header(
        data_mode=data_mode,
        current_time_str=now_ist_str if data_mode=="real" else None,
        latest_api_str=latest_obs_display_str if data_mode=="real" else None,
        selected_time_str=obs_ts_str if data_mode=="real" else None,
        obs_used_str=obs_ts_str if data_mode=="real" else None
    )
    
    horizon_selected = st.sidebar.select_slider(
        "Forecast Horizon",
        options=["+1 Hour", "+3 Hours", "+6 Hours", "+12 Hours", "+24 Hours"],
        value="+6 Hours"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size:12px; color:#94a3b8; background:#0f172a; padding:12px; border-radius:8px; border:1px solid #334155;">
            <b>🕒 Current Clock (IST):</b> {now_ist_str}<br>
            <b>📡 Latest Available Obs:</b> {latest_obs_display_str}<br>
            <b>📌 Selected Observation:</b> {obs_ts_str}<br>
            <b>🎯 Forecast Target (+6h):</b> {fc_ts_str}<br>
            <b>📍 Active Station:</b> {station_selected}<br>
            <b>Mode:</b> {'Live Stream (OpenAQ/Open-Meteo)' if data_mode=='real' else 'Demo Prototype'}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ------------------ LIVE DYNAMIC MODEL INFERENCE ------------------
    curr_pm25 = float(selected_feat["pm25"])
    curr_aqi_dict = calculate_overall_aqi(selected_feat)
    curr_aqi = curr_aqi_dict["overall_aqi"]
    
    # Construct exact feature vector in strict training feature order
    X_feat = pd.DataFrame([selected_feat.to_dict()])[FINAL_COUPLED_FEATURES]
    pred_pm25 = float(explainer_obj.model.predict(X_feat)[0])
    pred_aqi = calculate_sub_index(pred_pm25, "pm25")
    pred_aqi_info = get_aqi_category(pred_aqi)
    
    risk_level = selected_feat.get("pollution_risk_level", "UNKNOWN")
    vc = float(selected_feat.get("ventilation_coeff", 0))
    
    # ------------------ SEVERE POLLUTION ALERT BANNER ------------------
    if pred_aqi >= 301.0:
        reasons = []
        if vc < 1500: reasons.append("Low ventilation capacity (<1500 m²/s)")
        if selected_feat.get("pbl_height", 1000) < 400: reasons.append("Shallow boundary layer height (<400m)")
        if selected_feat.get("stagnation_indicator", 0) > 0: reasons.append("Air stagnation active (calm winds)")
        if selected_feat.get("inversion_indicator", 0) > 0: reasons.append("Thermal inversion active")
        
        reasons_html = "".join([f"<li>{r}</li>" for r in reasons]) if reasons else "<li>High particulate accumulation</li>"
        
        st.markdown(
            f"""
            <div style="background:#7e0023; color:#ffffff; padding:16px 22px; border-radius:12px; margin-bottom:20px; border-left:6px solid #ff0044;">
                <h3 style="margin:0; font-size:18px;">⚠️ HIGH POLLUTION EVENT EXPECTED IN +6 HOURS</h3>
                <p style="margin:4px 0 8px 0; font-size:14px;">Station: <b>{station_selected}</b> | Real Observation Used: <b>{obs_ts_str}</b> | Target Forecast Time: <b>{fc_ts_str}</b></p>
                <p style="margin:0 0 8px 0; font-size:14px;">Forecast PM2.5 (+6h): <b>{pred_pm25:.1f} µg/m³</b> | Expected AQI: <b>{pred_aqi:.0f} ({pred_aqi_info['category']})</b></p>
                <h5 style="margin:4px 0; font-size:13px; text-transform:uppercase;">Main Contributing Atmospheric Conditions:</h5>
                <ul style="margin:0; padding-left:20px; font-size:13px;">
                    {reasons_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------ PAGE ROUTING ------------------
    
    # --- PAGE 1: OVERVIEW ---
    if page == "📌 Overview":
        render_aqi_badge_card(
            aqi=curr_aqi,
            category=curr_aqi_dict["category"],
            color=curr_aqi_dict["color"],
            pm25_conc=curr_pm25
        )
        render_overview_kpis(
            curr_pm25=curr_pm25,
            curr_aqi=curr_aqi,
            pred_pm25=pred_pm25,
            pred_aqi=pred_aqi,
            risk_level=risk_level,
            ventilation_coeff=vc
        )
        st.markdown("<br>", unsafe_allow_html=True)
        render_dispersion_cards(selected_feat)
        st.markdown("<br>", unsafe_allow_html=True)
        render_weather_grid(selected_feat)

    # --- PAGE 2: PM2.5 FORECAST ---
    elif page == "📈 PM2.5 Forecast":
        st.markdown(f"## 📈 PM2.5 Concentration Forecast (+6h ahead)")
        st.markdown(f"**Station:** `{station_selected}` | **Selected Observation:** `{obs_ts_str}` | **Target Forecast Time (+6h):** `{fc_ts_str}`")
        if data_mode == "real":
            fig_fc = create_live_pm25_forecast_chart(df_st_obs, explainer_obj.model, station_id=station_selected)
            st.plotly_chart(fig_fc, use_container_width=True)
        elif not df_preds.empty:
            fig_fc = create_pm25_forecast_chart(df_preds, station_id=station_selected)
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info(f"Current Observation PM2.5: **{curr_pm25:.1f} µg/m³** (AQI: {curr_aqi:.0f}) → Predicted +6h PM2.5: **{pred_pm25:.1f} µg/m³** (AQI: {pred_aqi:.0f}, {pred_aqi_info['category']}).")

    # --- PAGE 3: ATMOSPHERIC CONDITIONS ---
    elif page == "🌦️ Atmospheric Conditions":
        st.markdown(f"## 🌦️ Current Atmospheric Conditions (`{station_selected}` @ `{obs_ts_str}`)")
        st.markdown(f"Weather Observation Time: `{obs_ts_str}`")
        render_weather_grid(selected_feat)
        st.markdown("<br>", unsafe_allow_html=True)
        render_dispersion_cards(selected_feat)

    # --- PAGE 4: WEATHER-POLLUTION COUPLING ---
    elif page == "⚛️ Weather-Pollution Coupling":
        st.markdown(f"## ⚛️ Weather-Pollution Coupling Engine (`{station_selected}` @ `{obs_ts_str}`)")
        st.markdown(
            """
            Traditional AQI forecasting treats weather and pollution history as separate independent variables.  
            **VayuSense explicitly models physics-informed proxy interactions** between particulate concentration and boundary layer dynamics.
            """
        )
        
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            fig_bar = create_coupling_terms_bar_chart(selected_feat)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.markdown("### 🔄 Physical Interaction Flow")
            st.markdown(
                """
                ```
                WEATHER (Wind, Temp, Humidity, Pressure, PBL)
                                   ↓
                       ATMOSPHERIC STABILITY
                                   ↓
                       POLLUTANT DISPERSION (VC)
                                   ↓
                           PM2.5 DENSITY
                                   ↓
                         CPCB AQI SUB-INDEX
                ```
                """
            )
            
        st.markdown("---")
        st.markdown(
            "#### **Coupling Proxy Definitions**",
            help="**Ventilation Capacity ($VC = u \\times h_{PBL}$)**: Simplified ventilation capacity ($m^2/s$). Lower VC ($<1500$ m²/s) indicates weaker atmospheric ventilation.\n\n**$PM_{2.5} \\times RH$**: Interaction proxy for humidity-dependent particulate behavior.\n\n**$PM_{2.5} / h_{PBL}$**: Concentration density ratio inside the mixed boundary layer volume."
        )


    # --- PAGE 6: WHAT-IF SIMULATOR ---
    elif page == "🧪 Weather What-If Simulator":
        selected_feat_dict = selected_feat.to_dict()
        selected_feat_dict["final_coupled_prediction"] = pred_pm25
        render_what_if_simulator(pd.Series(selected_feat_dict), data_mode=data_mode)

    # --- PAGE 7: DELHI NCR STATION MAP ---
    elif page == "🗺️ Delhi NCR Station Map":
        render_station_map(df_features, station_selected=station_selected, selected_ts_str=obs_ts.strftime("%Y-%m-%d %H:00"))

    # --- PAGE 8: MODEL PERFORMANCE ---
    elif page == "⚖️ Model Performance":
        st.markdown("## 📊 HISTORICAL MODEL EVALUATION (2024 Test Set Benchmark)")
        render_model_performance_view(df_preds, station_id=station_selected, data_mode=data_mode)


if __name__ == "__main__":
    main()
