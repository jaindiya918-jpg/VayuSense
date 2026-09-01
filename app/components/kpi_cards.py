"""
VayuSense - KPI & Status Cards Component
=========================================
Displays CPCB AQI status badges, weather cards, and atmospheric risk indicators.
"""

import streamlit as st

def render_aqi_badge_card(aqi: float, category: str, color: str, pm25_conc: float):
    """
    Renders large CPCB AQI card clearly distinguishing between:
    - PM2.5 Concentration (µg/m³)
    - AQI (Unitless 0-500+ scale)
    """
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: #ffffff;
            padding: 22px 28px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            margin-bottom: 24px;
        ">
            <h4 style="margin: 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.95;">
                Current Air Quality Index (AQI)
            </h4>
            <h1 style="margin: 6px 0; font-size: 52px; font-weight: 900; line-height: 1.0;">
                {aqi:.0f}
            </h1>
            <div style="display: flex; justify-content: center; gap: 12px; align-items: center; margin-top: 8px;">
                <span style="background: rgba(0,0,0,0.25); font-size: 18px; font-weight: 700; padding: 4px 18px; border-radius: 20px;">
                    {category} Category
                </span>
                <span style="background: rgba(255,255,255,0.25); font-size: 15px; font-weight: 600; padding: 4px 14px; border-radius: 20px;">
                    PM2.5: {pm25_conc:.1f} µg/m³
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_overview_kpis(
    curr_pm25: float,
    curr_aqi: float,
    pred_pm25: float,
    pred_aqi: float,
    risk_level: str,
    ventilation_coeff: float
):
    """Renders 5 top KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Current PM2.5", value=f"{curr_pm25:.1f} µg/m³")
    with col2:
        st.metric(label="Current AQI", value=f"{curr_aqi:.0f}")
    with col3:
        diff_pm = pred_pm25 - curr_pm25
        st.metric(
            label="Forecast PM2.5 (+6h)",
            value=f"{pred_pm25:.1f} µg/m³",
            delta=f"{diff_pm:+.1f} µg/m³",
            delta_color="inverse"
        )
    with col4:
        diff_aqi = pred_aqi - curr_aqi
        st.metric(
            label="Forecast AQI (+6h)",
            value=f"{pred_aqi:.0f}",
            delta=f"{diff_aqi:+.0f}",
            delta_color="inverse"
        )
    with col5:
        risk_color = "red" if risk_level in ["HIGH", "SEVERE"] else "green"
        st.metric(
            label="Pollution Risk",
            value=risk_level,
            delta=f"VC: {ventilation_coeff:.0f} m²/s",
            delta_color="inverse" if ventilation_coeff < 1500 else "normal"
        )

def render_weather_grid(row):
    """Renders meteorological metrics grid."""
    st.markdown("### 🌦️ Current Meteorological Observations")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    with c1:
        st.metric("Temperature", f"{row['temperature']:.1f} °C")
    with c2:
        st.metric("Humidity", f"{row['humidity']:.1f} %")
    with c3:
        st.metric("Wind Speed", f"{row['wind_speed']:.1f} m/s")
    with c4:
        st.metric("Wind Direction", f"{row['wind_deg']:.0f} °")
    with c5:
        st.metric("Pressure", f"{row['pressure']:.1f} hPa")
    with c6:
        st.metric("PBL Height", f"{row['pbl_height']:.0f} m")
    with c7:
        st.metric("Rainfall", f"{row['rainfall']:.1f} mm")

def render_dispersion_cards(row):
    """Renders atmospheric dispersion status section."""
    st.markdown("### 🌪️ Atmospheric Dispersion & Stability")
    
    vc = float(row["ventilation_coeff"])
    disp_status = row.get("dispersion_status", "UNKNOWN")
    is_stag = bool(row.get("stagnation_indicator", 0) > 0)
    is_inv = bool(row.get("inversion_indicator", 0) > 0)
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:16px; border-radius:10px; border:1px solid #334155;">
                <h5 style="margin:0; color:#94a3b8; font-size:12px;">VENTILATION CAPACITY</h5>
                <h2 style="margin:4px 0; color:#00e5ff;">{vc:.0f} <span style="font-size:16px;">m²/s</span></h2>
                <p style="margin:0; font-size:13px; color:#cbd5e1;">Flushing rate (u × PBL)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c2:
        disp_color = "#ef4444" if "POOR" in disp_status or "SEVERE" in disp_status else "#22c55e"
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:16px; border-radius:10px; border:1px solid #334155;">
                <h5 style="margin:0; color:#94a3b8; font-size:12px;">DISPERSION STATUS</h5>
                <h3 style="margin:4px 0; color:{disp_color};">{disp_status}</h3>
                <p style="margin:0; font-size:13px; color:#cbd5e1;">Atmospheric Dilution</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c3:
        stag_str = "ACTIVE ⚠️" if is_stag else "INACTIVE ✅"
        stag_color = "#ef4444" if is_stag else "#22c55e"
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:16px; border-radius:10px; border:1px solid #334155;">
                <h5 style="margin:0; color:#94a3b8; font-size:12px;">AIR STAGNATION</h5>
                <h3 style="margin:4px 0; color:{stag_color};">{stag_str}</h3>
                <p style="margin:0; font-size:13px; color:#cbd5e1;">Calm Wind (<1.5m/s) & Low PBL (<400m)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c4:
        inv_str = "DETECTED 🌡️" if is_inv else "NONE ✅"
        inv_color = "#f59e0b" if is_inv else "#22c55e"
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:16px; border-radius:10px; border:1px solid #334155;">
                <h5 style="margin:0; color:#94a3b8; font-size:12px;">THERMAL INVERSION</h5>
                <h3 style="margin:4px 0; color:{inv_color};">{inv_str}</h3>
                <p style="margin:0; font-size:13px; color:#cbd5e1;">Positive Gradient (dT/dz > 0)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
