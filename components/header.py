"""
VayuSense - Dashboard Header & Banner Component
================================================
Renders REAL DATA MODE vs DEMO MODE banners and title header cleanly using st.html only.
Supports explicit display of:
1. Current System Time (IST)
2. Latest Available Real Observation (OpenAQ / Open-Meteo)
3. Currently Selected / Requested Time
"""

import streamlit as st

def render_header(
    data_mode: str = "demo",
    current_time_str: str = None,
    latest_api_str: str = None,
    selected_time_str: str = None,
    obs_used_str: str = None
):
    is_real = (data_mode.lower() == "real")
    
    if is_real:
        badge = (
            '<span style="background: linear-gradient(90deg, #10b981, #059669); color: #ffffff; '
            'padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; '
            'letter-spacing: 0.8px; box-shadow: 0 2px 8px rgba(16,185,129,0.4);">🟢 LIVE / CURRENT DATA</span>'
        )
        
        c_str = f" | <b>Current Time:</b> {current_time_str}" if current_time_str else ""
        l_str = f" | <b>Latest Available Observation:</b> {latest_api_str}" if latest_api_str else ""
        s_str = f" | <span style='color:#00e5ff; font-weight:700;'>Selected Time: {selected_time_str}</span>" if selected_time_str else ""
        o_str = f" | <b>Real Observation Used:</b> {obs_used_str}" if (obs_used_str and obs_used_str != selected_time_str) else ""
        
        notice = (
            f'🌐 <b>Data Source:</b> OpenAQ / Open-Meteo Stream{c_str}{l_str}{s_str}{o_str} | <b>Forecast Horizon:</b> +6 hours<br>'
            '<i>Note: Air-quality data may include OpenAQ observations and/or CAMS atmospheric-composition model/reanalysis data. Model inference uses the latest actual observation at or before the selected time.</i>'
        )
    else:
        badge = (
            '<span style="background-color: #ef4444; color: #ffffff; padding: 6px 14px; '
            'border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.8px; '
            'box-shadow: 0 2px 8px rgba(239,68,68,0.4);">⚠️ DEMO MODE — SYNTHETIC DATA</span>'
        )
        notice = (
            'ℹ️ <b>Notice:</b> Current prototype uses synthetic demonstration dataset. '
            'Switch `DATA_MODE=real` in `.env` to enable live streaming data.'
        )

    header_html = (
        '<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 20px 24px; border-radius: 14px; border: 1px solid #334155; margin-bottom: 20px;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
        '<div>'
        '<h1 style="margin:0; font-size: 32px; font-weight: 800; background: linear-gradient(90deg, #00e5ff, #7c4dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🌿 VayuSense</h1>'
        '<p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 15px; font-weight: 500;">AI-Powered Weather–Pollution Coupled Forecasting System (Delhi NCR Focus)</p>'
        '</div>'
        f'<div style="text-align: right; margin-top: 8px;">{badge}</div>'
        '</div>'
        f'<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #334155; color: #cbd5e1; font-size: 12px;">{notice}</div>'
        '</div>'
    )
    
    st.html(header_html)
