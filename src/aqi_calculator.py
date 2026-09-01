"""
VayuSense - CPCB Indian National Air Quality Index (AQI) Calculator
====================================================================
METHODOLOGY & SCIENTIFIC DOCUMENTATION:
This module implements the official Central Pollution Control Board (CPCB) Indian National
Air Quality Index (AQI) breakpoint interpolation methodology.

CRITICAL DISTINCTION:
- Pollutant Concentration (e.g. PM2.5 in µg/m³) is the physical mass density of particles.
- AQI (Air Quality Index) is a unitless normalized risk scale (0–500+) computed via pollutant sub-indices.
  AQI != PM2.5 Concentration!

CPCB Linear Interpolation Formula:
    I = I_low + [(I_high - I_low) / (C_high - C_low)] * (C - C_low)
    where:
    - C = Observed Pollutant Concentration
    - C_low, C_high = Concentration Breakpoint Range
    - I_low, I_high = AQI Sub-index Breakpoint Range

CPCB AQI Categories & Color Palette:
    0 – 50   : Good          (#00e400)
    51 – 100  : Satisfactory    (#7bb31a)
    101 – 200 : Moderate        (#ff7e00)
    201 – 300 : Poor            (#ff0000)
    301 – 400 : Very Poor       (#99004c)
    401 – 500+: Severe          (#7e0023)
"""

import numpy as np
import pandas as pd

# Official CPCB Breakpoints: List of tuples (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 800.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 800.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
}

def calculate_sub_index(conc: float, pollutant: str) -> float:
    """
    Calculates the CPCB AQI sub-index for a single pollutant concentration using linear interpolation.
    Handles edge cases: zero, negative, NaN, and concentrations exceeding upper breakpoints.
    """
    if pd.isna(conc) or conc is None or conc < 0:
        return 0.0
    
    pol_key = pollutant.lower().replace(".", "")
    if pol_key not in CPCB_BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant '{pollutant}'. Supported: {list(CPCB_BREAKPOINTS.keys())}")
        
    breakpoints = CPCB_BREAKPOINTS[pol_key]
    
    # Below lowest breakpoint boundary
    if conc <= breakpoints[0][0]:
        return float(breakpoints[0][2])
        
    # Within defined breakpoint ranges
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc <= c_high:
            sub_index = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
            return float(round(sub_index, 1))
            
    # Exceeding highest defined breakpoint range (linear extrapolation capped at 999)
    c_low, c_high, i_low, i_high = breakpoints[-1]
    sub_index = i_high + ((i_high - i_low) / (c_high - c_low)) * (conc - c_high)
    return float(round(min(sub_index, 999.0), 1))

def get_aqi_category(aqi: float) -> dict:
    """
    Returns official CPCB AQI category name, hex color code, and severity rank.
    """
    if pd.isna(aqi) or aqi < 0:
        return {"category": "Unknown", "color": "#94a3b8", "severity": 0}
    elif aqi <= 50:
        return {"category": "Good", "color": "#00e400", "severity": 1}
    elif aqi <= 100:
        return {"category": "Satisfactory", "color": "#7bb31a", "severity": 2}
    elif aqi <= 200:
        return {"category": "Moderate", "color": "#ff7e00", "severity": 3}
    elif aqi <= 300:
        return {"category": "Poor", "color": "#ff0000", "severity": 4}
    elif aqi <= 400:
        return {"category": "Very Poor", "color": "#99004c", "severity": 5}
    else:
        return {"category": "Severe", "color": "#7e0023", "severity": 6}

def calculate_overall_aqi(row: pd.Series) -> dict:
    """
    Calculates sub-indices for all available pollutants in a dataset row.
    Returns overall AQI (maximum sub-index), dominant pollutant, and category info.
    If pollutants are missing, calculates over available pollutants and lists missing parameters.
    """
    sub_indices = {}
    missing_pollutants = []
    
    for pol in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if pol in row and not pd.isna(row[pol]):
            sub_indices[pol] = calculate_sub_index(row[pol], pol)
        else:
            missing_pollutants.append(pol.upper())
            
    if not sub_indices:
        return {
            "overall_aqi": 0.0,
            "dominant_pollutant": "NONE",
            "category": "Unknown",
            "color": "#94a3b8",
            "severity": 0,
            "sub_indices": {},
            "missing_pollutants": missing_pollutants,
            "calculation_note": "No pollutant concentration data available."
        }
        
    overall_aqi = max(sub_indices.values())
    dominant_pol = max(sub_indices, key=sub_indices.get).upper()
    cat_info = get_aqi_category(overall_aqi)
    
    note = "Full CPCB AQI computed."
    if missing_pollutants:
        note = f"AQI computed from available pollutants. Missing: {', '.join(missing_pollutants)}."
        
    return {
        "overall_aqi": overall_aqi,
        "dominant_pollutant": dominant_pol,
        "category": cat_info["category"],
        "color": cat_info["color"],
        "severity": cat_info["severity"],
        "sub_indices": sub_indices,
        "missing_pollutants": missing_pollutants,
        "calculation_note": note
    }
