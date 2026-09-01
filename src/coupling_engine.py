"""
VayuSense - Weather-Pollution Coupling Engine
======================================================
NOTE & SCIENTIFIC DISCLAIMER:
The features engineered in this module are "physics-informed proxy features"
and interaction terms designed to provide intuitive atmospheric transport and trapping signals
to tree-based machine learning models (XGBoost).

These simplified mathematical formulas do NOT represent a full atmospheric chemistry
transport model (such as WRF-Chem or CMAQ), nor do they claim to model exact aerosol radiative
forcing, chemical transformation kinetics, or universal atmospheric thresholds.
They serve as interpretable prototype proxies for hackathon demonstration.
"""

import numpy as np
import pandas as pd

# Default Configurable Thresholds (Prototype / Demo Risk Indicators)
DEFAULT_STAGNATION_WIND_THRESHOLD = 1.5  # m/s
DEFAULT_STAGNATION_PBL_THRESHOLD = 400.0  # m
DEFAULT_EPSILON = 1e-5

# Configurable Ventilation Capacity Thresholds (m²/s)
VC_THRESHOLDS = {
    "SEVERE_TRAPPING": 800.0,
    "POOR_DISPERSION": 1500.0,
    "MODERATE_DISPERSION": 2000.0
}

# Configurable PM2.5 Risk Concentration Thresholds (µg/m³)
PM25_RISK_THRESHOLDS = {
    "LOW": 60.0,
    "MODERATE": 120.0,
    "HIGH": 250.0
}

def classify_dispersion_status(
    vc: float,
    thresholds: dict = VC_THRESHOLDS
) -> str:
    """
    Categorizes atmospheric dispersion capacity into human-readable labels.
    - SEVERE TRAPPING: VC < 800 m²/s
    - POOR DISPERSION: 800 <= VC < 1500 m²/s
    - MODERATE DISPERSION: 1500 <= VC < 2000 m²/s
    - GOOD DISPERSION: VC >= 2000 m²/s
    """
    if np.isnan(vc):
        return "UNKNOWN"
    if vc < thresholds["SEVERE_TRAPPING"]:
        return "SEVERE TRAPPING"
    elif vc < thresholds["POOR_DISPERSION"]:
        return "POOR DISPERSION"
    elif vc < thresholds["MODERATE_DISPERSION"]:
        return "MODERATE DISPERSION"
    else:
        return "GOOD DISPERSION"

def classify_pollution_risk(
    pm25: float,
    thresholds: dict = PM25_RISK_THRESHOLDS
) -> str:
    """
    Categorizes PM2.5 pollution risk level into human-readable labels.
    - LOW: PM2.5 < 60 µg/m³
    - MODERATE: 60 <= PM2.5 < 120 µg/m³
    - HIGH: 120 <= PM2.5 < 250 µg/m³
    - SEVERE: PM2.5 >= 250 µg/m³
    """
    if np.isnan(pm25):
        return "UNKNOWN"
    if pm25 < thresholds["LOW"]:
        return "LOW"
    elif pm25 < thresholds["MODERATE"]:
        return "MODERATE"
    elif pm25 < thresholds["HIGH"]:
        return "HIGH"
    else:
        return "SEVERE"

def compute_coupling_features(
    df: pd.DataFrame,
    stagnation_wind_threshold: float = DEFAULT_STAGNATION_WIND_THRESHOLD,
    stagnation_pbl_threshold: float = DEFAULT_STAGNATION_PBL_THRESHOLD,
    epsilon: float = DEFAULT_EPSILON
) -> pd.DataFrame:
    """
    Computes physics-informed weather-pollution interaction proxy features.
    
    Proxy Interactions:
    A. pm25_x_humidity: Interaction proxy between particulate concentration and relative humidity.
    B. pm25_x_temp: Temperature-pollution interaction proxy.
    C. pm25_x_wind_speed: Interaction proxy for particulate concentration and horizontal wind transport.
    D. pm25_div_pbl: Pollutant trapping ratio (concentration density in boundary layer volume).
    E. ventilation_coeff: Atmospheric dispersion capacity (wind_speed * pbl_height in m²/s).
    F. stagnation_indicator: Binary flag for calm wind AND shallow mixing height.
    G. inversion_indicator: Binary flag for positive temperature gradient (dT/dz > 0 °C/100m).
    H. lagged_pm25_coupling: Ratio of prior hour's PM2.5 to current atmospheric dispersion capacity.
    
    Risk Indicators:
    - dispersion_status: Categorical dispersion label.
    - pollution_risk_level: Categorical risk label.
    """
    df_out = df.copy()
    
    # Handle safe numeric conversion and fill NaNs in raw input if any
    wind_speed = df_out["wind_speed"].clip(lower=0.0) if "wind_speed" in df_out.columns else 0.0
    pbl_height = df_out["pbl_height"].clip(lower=0.0) if "pbl_height" in df_out.columns else 0.0
    pm25 = df_out["pm25"].clip(lower=0.0) if "pm25" in df_out.columns else 0.0
    humidity = df_out["humidity"] if "humidity" in df_out.columns else 0.0
    temperature = df_out["temperature"] if "temperature" in df_out.columns else 0.0
    temp_gradient = df_out["temp_gradient"] if "temp_gradient" in df_out.columns else 0.0
    
    # A. PM2.5 x Humidity (Physics-informed interaction proxy)
    df_out["pm25_x_humidity"] = pm25 * humidity
    
    # B. PM2.5 x Temperature
    df_out["pm25_x_temp"] = pm25 * temperature
    
    # C. PM2.5 x Wind Speed
    df_out["pm25_x_wind_speed"] = pm25 * wind_speed
    
    # D. Pollutant Trapping Ratio (PM2.5 / (PBL + epsilon))
    df_out["pm25_div_pbl"] = pm25 / (pbl_height + epsilon)
    
    # E. Ventilation Coefficient (m²/s)
    df_out["ventilation_coeff"] = wind_speed * pbl_height
    
    # F. Stagnation Indicator (Configurable wind & PBL thresholds)
    df_out["stagnation_indicator"] = (
        (wind_speed < stagnation_wind_threshold) & (pbl_height < stagnation_pbl_threshold)
    ).astype(int)
    
    # G. Inversion Indicator
    # positive temperature gradient (dT/dz > 0 °C/100m) represents temperature inversion
    df_out["inversion_indicator"] = (temp_gradient > 0.0).astype(int)
    
    # H. Lagged PM2.5 Coupling
    if "pm25_lag_1h" in df_out.columns:
        pm25_lag = df_out["pm25_lag_1h"].clip(lower=0.0)
    else:
        pm25_lag = pm25
    df_out["lagged_pm25_coupling"] = pm25_lag / (df_out["ventilation_coeff"] + epsilon)
    
    # Risk Indicators
    df_out["dispersion_status"] = df_out["ventilation_coeff"].apply(classify_dispersion_status)
    df_out["pollution_risk_level"] = pm25.apply(classify_pollution_risk)
    
    return df_out
