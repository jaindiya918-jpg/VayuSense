"""
VayuSense - Dashboard Header & Banner Component
================================================
Renders REAL DATA MODE vs DEMO MODE banners and title header cleanly using st.html only.
"""

import streamlit as st

def render_header(data_mode: str = "demo"):
    is_real = (data_mode.lower() == "real")
    
    if is_real:
        badge = (
            '<span style="background: linear-gradient(90deg, #10b981, #059669); color: #ffffff; '
            'padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; '
            'letter-spacing: 0.8px; box-shadow: 0 2px 8px rgba(16,185,129,0.4);">🌐 REAL DATA MODE</span>'
        )
        notice = (
            '🌐 <b>Real Data Provenance:</b> Air Quality Stream: <b>Open-Meteo CAMS Air Quality Stream</b> | '
            'Weather Stream: <b>Open-Meteo ERA5 Reanalysis</b> | Data Period: <b>2024-01-01 to 2024-04-01</b><br>'
            '<i>Note: Air quality observations represent CAMS satellite/atmospheric reanalysis for Delhi coordinates.</i>'
        )
    else:
        badge = (
            '<span style="background-color: #ef4444; color: #ffffff; padding: 6px 14px; '
            'border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.8px; '
            'box-shadow: 0 2px 8px rgba(239,68,68,0.4);">⚠️ DEMO MODE — SYNTHETIC DATA</span>'
        )
        notice = (
            'ℹ️ <b>Notice:</b> Current prototype uses synthetic data for demonstration. '
            'Real-time CPCB/meteorological data integration is planned for deployment.'
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
