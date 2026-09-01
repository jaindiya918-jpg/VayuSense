"""
VayuSense - Plotly Chart Components
===================================
Renders Plotly time-series forecast charts dynamically supporting both
Real Data predictions (baseline_prediction / coupled_prediction)
and Demo Data predictions (final_baseline_prediction / final_coupled_prediction).
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_pm25_forecast_chart(df_preds: pd.DataFrame, station_id: str, forecast_horizon: int = 6) -> go.Figure:
    df_st = df_preds[df_preds["station_id"] == station_id].copy()
    df_st["timestamp"] = pd.to_datetime(df_st["timestamp"])
    
    # Dynamically resolve actual target and prediction column names for Real vs Demo mode
    actual_col = "actual_pm25" if "actual_pm25" in df_st.columns else ("pm25_target_6h" if "pm25_target_6h" in df_st.columns else "pm25_target")
    base_col = "baseline_prediction" if "baseline_prediction" in df_st.columns else "final_baseline_prediction"
    coup_col = "coupled_prediction" if "coupled_prediction" in df_st.columns else "final_coupled_prediction"
    
    base_name = "Real Baseline XGBoost" if "baseline_prediction" in df_st.columns else "Final Baseline XGBoost"
    coup_name = "Real Coupled XGBoost (VayuSense)" if "coupled_prediction" in df_st.columns else "Final Coupled XGBoost (VayuSense)"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_st["timestamp"],
        y=df_st[actual_col],
        mode="lines",
        name=f"Actual PM2.5 (+{forecast_horizon}h Target)",
        line=dict(color="#ffffff", width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_st["timestamp"],
        y=df_st[base_col],
        mode="lines",
        name=base_name,
        line=dict(color="#ff9900", width=2, dash="dash")
    ))
    
    fig.add_trace(go.Scatter(
        x=df_st["timestamp"],
        y=df_st[coup_col],
        mode="lines",
        name=coup_name,
        line=dict(color="#00e5ff", width=2.5)
    ))
    
    fig.add_hline(
        y=250,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Severe AQI Threshold (250 µg/m³)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"PM2.5 Forecast Comparison (+{forecast_horizon}h ahead) — Station: {station_id}",
        xaxis_title="Target Forecast Timestamp (+6h)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_live_pm25_forecast_chart(df_st_feat: pd.DataFrame, model, station_id: str, forecast_horizon: int = 6) -> go.Figure:
    """
    Renders live PM2.5 observation time-series and model +6h forecast predictions for real-time mode.
    """
    from src.feature_engineering import FINAL_COUPLED_FEATURES
    df_st = df_st_feat.copy().sort_values(by="timestamp").reset_index(drop=True)
    df_st["timestamp"] = pd.to_datetime(df_st["timestamp"])
    
    # Compute +6h predictions for all valid feature rows
    X_matrix = df_st[FINAL_COUPLED_FEATURES]
    preds = model.predict(X_matrix)
    df_st["pred_pm25_6h"] = preds
    df_st["target_timestamp_6h"] = df_st["timestamp"] + pd.Timedelta(hours=forecast_horizon)
    
    fig = go.Figure()
    
    # 1. Observed Real-Time PM2.5
    fig.add_trace(go.Scatter(
        x=df_st["timestamp"],
        y=df_st["pm25"],
        mode="lines+markers",
        name="Observed PM2.5 (API Stream)",
        line=dict(color="#00e5ff", width=2.5),
        marker=dict(size=4)
    ))
    
    # 2. Model +6h Forecast Stream
    fig.add_trace(go.Scatter(
        x=df_st["target_timestamp_6h"],
        y=df_st["pred_pm25_6h"],
        mode="lines",
        name=f"VayuSense Forecast (+{forecast_horizon}h Target)",
        line=dict(color="#7c4dff", width=2, dash="dash")
    ))
    
    # 3. Latest Forecast Target Point Highlight
    latest_row = df_st.iloc[-1]
    fc_ts = latest_row["target_timestamp_6h"]
    fc_val = latest_row["pred_pm25_6h"]
    
    fig.add_trace(go.Scatter(
        x=[fc_ts],
        y=[fc_val],
        mode="markers+text",
        name="Latest +6h Forecast Target",
        text=[f" <b>Forecast Target ({fc_ts.strftime('%H:00')}): {fc_val:.1f} µg/m³</b>"],
        textposition="top right",
        textfont=dict(color="#00e5ff", size=12),
        marker=dict(color="#ff0055", size=14, symbol="star", line=dict(color="#ffffff", width=2))
    ))
    
    fig.add_hline(
        y=250,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Severe AQI Threshold (250 µg/m³)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Live PM2.5 Stream & +{forecast_horizon}h Coupled Forecast — Station: {station_id}",
        xaxis_title="Timestamp (Asia/Kolkata)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_coupling_terms_bar_chart(row: pd.Series) -> go.Figure:
    terms = [
        ("PM2.5 × Humidity", float(row.get("pm25_x_humidity", 0))),
        ("PM2.5 × Temperature", float(row.get("pm25_x_temp", 0))),
        ("PM2.5 × Wind Speed", float(row.get("pm25_x_wind_speed", 0))),
        ("Ventilation Capacity (m²/s)", float(row.get("ventilation_coeff", 0)))
    ]
    
    labels = [t[0] for t in terms]
    vals = [t[1] for t in terms]
    
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color="#00e5ff",
        text=[f"{v:.1f}" for v in vals],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Physics-Informed Proxy Interaction Values",
        xaxis_title="Calculated Value",
        yaxis_title="",
        template="plotly_dark",
        height=320,
        margin=dict(l=180, r=50, t=50, b=40)
    )
    
    return fig
