"""
VayuSense - KPI & Status Cards Component
=========================================
Displays CPCB AQI status badges, weather cards, and atmospheric risk indicators.
"""

import streamlit as st

import plotly.graph_objects as go

def render_aqi_badge_card(aqi: float, category: str, color: str, pm25_conc: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number={'suffix': "", 'font': {'size': 60, 'color': '#f8fafc', 'family': 'Inter'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Air Quality Index (AQI)", 'font': {'size': 18, 'color': '#94a3b8', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "#334155", 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1e293b",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.08)'},
                {'range': [51, 100], 'color': 'rgba(234, 179, 8, 0.08)'},
                {'range': [101, 200], 'color': 'rgba(249, 115, 22, 0.08)'},
                {'range': [201, 300], 'color': 'rgba(239, 68, 68, 0.08)'},
                {'range': [301, 400], 'color': 'rgba(185, 28, 28, 0.08)'},
                {'range': [401, 500], 'color': 'rgba(126, 34, 206, 0.08)'}
            ]
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter"}
    )
    
    st.markdown(
        """
        <style>
        .aqi-stats {
            display: flex; justify-content: center; gap: 24px;
            margin-top: -30px; margin-bottom: 24px; z-index: 10; position: relative;
        }
        .aqi-pill {
            background: #1e293b; border: 1px solid #334155;
            padding: 8px 24px; border-radius: 20px;
            font-weight: 600; font-size: 15px; color: #f8fafc;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
        f"""
        <div class="aqi-stats">
            <span class="aqi-pill" style="border-top: 3px solid {color};">{category} Category</span>
            <span class="aqi-pill">PM2.5: {pm25_conc:.1f} µg/m³</span>
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
        st.metric(
            label="Pollution Risk",
            value=risk_level,
            delta=f"VC: {ventilation_coeff:.0f} m²/s",
            delta_color="inverse" if ventilation_coeff < 1500 else "normal"
        )

def render_weather_grid(row):
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
