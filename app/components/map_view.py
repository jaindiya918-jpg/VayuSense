"""
VayuSense - Delhi NCR Station Map Component
============================================
Renders interactive Plotly scatter map of Delhi NCR monitoring stations.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.aqi_calculator import calculate_overall_aqi, get_aqi_category

STATION_COORDS = {
    "Anand_Vihar": {"lat": 28.6469, "lon": 77.3162, "name": "Anand Vihar"},
    "RK_Puram": {"lat": 28.5644, "lon": 77.1724, "name": "RK Puram"},
    "Punjabi_Bagh": {"lat": 28.6683, "lon": 77.1247, "name": "Punjabi Bagh"},
    "Mandir_Marg": {"lat": 28.6364, "lon": 77.2011, "name": "Mandir Marg"}
}

def render_station_map(df_latest: pd.DataFrame):
    st.markdown("## 🗺️ Delhi NCR Monitoring Station Map")
    st.markdown("Interactive station map showing real-time location status across Delhi NCR:")
    
    map_data = []
    for station_id, coords in STATION_COORDS.items():
        st_row = df_latest[df_latest["station_id"] == station_id]
        if not st_row.empty:
            row = st_row.iloc[0]
            aqi_dict = calculate_overall_aqi(row)
            pm25_val = float(row["pm25"])
            risk_val = row.get("pollution_risk_level", "UNKNOWN")
            
            map_data.append({
                "station_id": station_id,
                "name": coords["name"],
                "lat": coords["lat"],
                "lon": coords["lon"],
                "pm25": pm25_val,
                "aqi": aqi_dict["overall_aqi"],
                "category": aqi_dict["category"],
                "color": aqi_dict["color"],
                "risk": risk_val
            })
            
    df_map = pd.DataFrame(map_data)
    
    fig = go.Figure()
    
    for idx, row in df_map.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[row["lat"]],
            lon=[row["lon"]],
            mode="markers+text",
            text=[f"<b>{row['name']}</b><br>AQI {row['aqi']:.0f} ({row['category']})"],
            textposition="top center",
            marker=dict(
                size=22,
                color=row["color"],
                line=dict(width=2, color="white")
            ),
            hoverinfo="text",
            hovertext=f"Station: {row['name']}<br>PM2.5: {row['pm25']:.1f} µg/m³<br>AQI: {row['aqi']:.0f} ({row['category']})<br>Risk: {row['risk']}",
            name=row["name"]
        ))
        
    fig.update_layout(
        geo=dict(
            scope="asia",
            center=dict(lat=28.62, lon=77.21),
            projection_scale=120,
            showland=True,
            landcolor="#1e293b",
            showocean=True,
            oceancolor="#0f172a",
            showlakes=True,
            lakecolor="#0f172a",
            subunitcolor="#334155"
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
