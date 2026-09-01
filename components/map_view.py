"""
VayuSense - Delhi NCR Monitoring Station Map Component
======================================================
Renders an interactive Plotly map of Delhi NCR monitoring stations over an open geographic basemap.
Uses carto-darkmatter / open-street-map tile styles (requires ZERO Mapbox tokens).
Features:
- Dynamic map centering & optimal zoom based on station coordinates
- AQI-color-coded markers with proportional sizing
- Active station selection highlighting
- Rich interactive hovercards with live observation timestamps
- Missing station data handling
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.aqi_calculator import calculate_overall_aqi, get_aqi_category

STATION_COORDS = {
    "Anand_Vihar": {"lat": 28.6469, "lon": 77.3162, "name": "Anand Vihar"},
    "RK_Puram": {"lat": 28.5644, "lon": 77.1724, "name": "RK Puram"},
    "Punjabi_Bagh": {"lat": 28.6683, "lon": 77.1247, "name": "Punjabi Bagh"},
    "Mandir_Marg": {"lat": 28.6364, "lon": 77.2011, "name": "Mandir Marg"}
}

def _normalize_station_name(name: str) -> str:
    if not name:
        return ""
    return str(name).lower().replace("_", " ").strip()

def render_station_map(df_latest: pd.DataFrame, station_selected: str = None, selected_ts_str: str = None):
    st.markdown("## 🗺️ Delhi NCR Monitoring Station Map")
    st.markdown("Latest available air-quality status across selected Delhi NCR monitoring stations:")
    
    if df_latest is None or df_latest.empty:
        st.warning("⚠️ Station location data is currently unavailable.")
        return
        
    map_data = []
    norm_selected = _normalize_station_name(station_selected)
    
    for station_id, coords in STATION_COORDS.items():
        lat, lon = coords["lat"], coords["lon"]
        
        # Coordinate boundary validation (Delhi NCR bounding box)
        if not (28.0 <= lat <= 29.5 and 76.5 <= lon <= 78.0):
            continue
            
        st_row = df_latest[df_latest["station_id"] == station_id]
        
        if not st_row.empty:
            # If selected timestamp provided, try to find matching row for station
            if selected_ts_str:
                st_ts = st_row[pd.to_datetime(st_row["timestamp"]).dt.strftime("%Y-%m-%d %H:00") == selected_ts_str]
                if not st_ts.empty:
                    row = st_ts.iloc[-1]
                else:
                    row = st_row.sort_values(by="timestamp").iloc[-1]
            else:
                row = st_row.sort_values(by="timestamp").iloc[-1]
                
            aqi_dict = calculate_overall_aqi(row)
            pm25_val = float(row.get("pm25", np.nan))
            risk_val = row.get("pollution_risk_level", "UNKNOWN")
            
            ts_val = row.get("timestamp")
            if isinstance(ts_val, pd.Timestamp):
                ts_str = ts_val.strftime("%d %b %Y, %H:00 IST")
            else:
                ts_str = str(ts_val)
                
            has_data = True
            aqi_val = aqi_dict["overall_aqi"]
            category_str = aqi_dict["category"]
            color_str = aqi_dict["color"]
        else:
            has_data = False
            pm25_val = np.nan
            aqi_val = 0
            category_str = "No recent data"
            color_str = "#64748b"  # Neutral slate gray
            risk_val = "UNAVAILABLE"
            ts_str = "No recent observation available"

        # Calculate proportional marker size (range 20 to 36)
        if has_data and not np.isnan(aqi_val):
            marker_size = max(20, min(36, int(18 + (aqi_val / 20.0))))
        else:
            marker_size = 18
            
        is_selected = False
        if norm_selected:
            is_selected = (_normalize_station_name(station_id) == norm_selected) or (_normalize_station_name(coords["name"]) == norm_selected)

        map_data.append({
            "station_id": station_id,
            "name": coords["name"],
            "lat": lat,
            "lon": lon,
            "pm25": pm25_val,
            "aqi": aqi_val,
            "category": category_str,
            "color": color_str,
            "risk": risk_val,
            "has_data": has_data,
            "timestamp": ts_str,
            "marker_size": marker_size,
            "is_selected": is_selected
        })
        
    if not map_data:
        st.error("No valid station coordinates found to display.")
        return

    df_map = pd.DataFrame(map_data)
    
    # Calculate dynamic geographic center
    center_lat = float(df_map["lat"].mean())
    center_lon = float(df_map["lon"].mean())
    
    fig = go.Figure()
    
    # Render station markers
    for idx, row in df_map.iterrows():
        is_sel = row["is_selected"]
        
        # Border outline: Cyan highlight for active selected station, white for others
        line_color = "#00e5ff" if is_sel else "#ffffff"
        line_width = 3.5 if is_sel else 1.8
        
        if row["has_data"]:
            hover_html = (
                f"<b>Station: {row['name']}</b><br>"
                f"<b>Status:</b> LIVE<br>"
                f"<b>AQI:</b> {row['aqi']:.0f} ({row['category']})<br>"
                f"<b>PM2.5:</b> {row['pm25']:.1f} µg/m³<br>"
                f"<b>Risk Level:</b> {row['risk']}<br>"
                f"<b>Observation Time:</b> {row['timestamp']}"
            )
            marker_label = f"<b>{row['name']}</b> ({row['aqi']:.0f})"
        else:
            hover_html = (
                f"<b>Station: {row['name']}</b><br>"
                f"<b>Status:</b> No recent data available<br>"
                f"<b>Observation Time:</b> {row['timestamp']}"
            )
            marker_label = f"<b>{row['name']}</b> (N/A)"

        # Use Scattermap in Plotly 7.0 / 6.0 with fallback to Scattermapbox
        try:
            scatter_cls = go.Scattermap
        except AttributeError:
            scatter_cls = go.Scattermapbox
            
        fig.add_trace(scatter_cls(
            lat=[row["lat"]],
            lon=[row["lon"]],
            mode="markers+text",
            text=[marker_label],
            textposition="top right",
            textfont=dict(size=12, color="#ffffff", family="Inter, sans-serif"),
            marker=dict(
                size=row["marker_size"] + (6 if is_sel else 0),
                color=row["color"],
                opacity=1.0 if row["has_data"] else 0.6
            ),
            hoverinfo="text",
            hovertext=hover_html,
            name=row["name"]
        ))

    # Map layout configuration using token-free carto-darkmatter tiles
    map_config = dict(
        style="carto-darkmatter",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=10.8
    )
    
    try:
        fig.update_layout(
            map=map_config,
            template="plotly_dark",
            height=580,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    except Exception:
        # Fallback layout key for older Plotly versions
        fig.update_layout(
            mapbox=map_config,
            template="plotly_dark",
            height=580,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False
        )
        
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend summary cards below map
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🟢 **Good / Satisfactory**: AQI 0–100")
    with col2:
        st.markdown("🟡 **Moderate**: AQI 101–200")
    with col3:
        st.markdown("🟠 **Poor**: AQI 201–300")
    with col4:
        st.markdown("🔴 **Very Poor / Severe**: AQI 301+")
