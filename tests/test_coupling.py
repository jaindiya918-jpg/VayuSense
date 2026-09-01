"""
VayuSense - Phase 2 Unit Tests (Coupling Engine & AQI Calculator)
===================================================================
Tests physics-informed coupling formulas, risk indicators, edge cases,
and CPCB AQI sub-index breakpoint calculations.
"""

import unittest
import numpy as np
import pandas as pd
from src.coupling_engine import compute_coupling_features, classify_dispersion_status, classify_pollution_risk
from src.aqi_calculator import calculate_sub_index, calculate_overall_aqi, get_aqi_category

class TestCouplingEngine(unittest.TestCase):
    
    def setUp(self):
        self.sample_data = pd.DataFrame([{
            "pm25": 100.0,
            "humidity": 80.0,
            "temperature": 25.0,
            "wind_speed": 1.2,
            "pbl_height": 300.0,
            "temp_gradient": 1.5,
            "pm25_lag_1h": 90.0
        }])

    def test_01_ventilation_coefficient(self):
        """Test 1: ventilation_coeff = wind_speed * pbl_height"""
        df = compute_coupling_features(self.sample_data)
        expected = 1.2 * 300.0  # 360.0 m²/s
        self.assertAlmostEqual(df.iloc[0]["ventilation_coeff"], expected, places=4)

    def test_02_pm25_div_pbl(self):
        """Test 2: pm25_div_pbl = pm25 / (pbl_height + eps)"""
        df = compute_coupling_features(self.sample_data, epsilon=1e-5)
        expected = 100.0 / (300.0 + 1e-5)
        self.assertAlmostEqual(df.iloc[0]["pm25_div_pbl"], expected, places=4)

    def test_03_pm25_x_humidity(self):
        """Test 3: pm25_x_humidity = pm25 * humidity"""
        df = compute_coupling_features(self.sample_data)
        expected = 100.0 * 80.0  # 8000.0
        self.assertAlmostEqual(df.iloc[0]["pm25_x_humidity"], expected, places=4)

    def test_04_pm25_x_temp(self):
        """Test 4: pm25_x_temp = pm25 * temperature"""
        df = compute_coupling_features(self.sample_data)
        expected = 100.0 * 25.0  # 2500.0
        self.assertAlmostEqual(df.iloc[0]["pm25_x_temp"], expected, places=4)

    def test_05_pm25_x_wind_speed(self):
        """Test 5: pm25_x_wind_speed = pm25 * wind_speed"""
        df = compute_coupling_features(self.sample_data)
        expected = 100.0 * 1.2  # 120.0
        self.assertAlmostEqual(df.iloc[0]["pm25_x_wind_speed"], expected, places=4)

    def test_06_stagnation_indicator(self):
        """Test 6: Stagnation indicator flag (wind < 1.5 AND PBL < 400)"""
        # Stagnant case (1.2 m/s, 300m PBL)
        df1 = compute_coupling_features(self.sample_data)
        self.assertEqual(df1.iloc[0]["stagnation_indicator"], 1)
        
        # Non-stagnant case (wind = 2.5 m/s)
        data_non_stag = self.sample_data.copy()
        data_non_stag["wind_speed"] = 2.5
        df2 = compute_coupling_features(data_non_stag)
        self.assertEqual(df2.iloc[0]["stagnation_indicator"], 0)

    def test_07_inversion_indicator(self):
        """Test 7: Inversion indicator flag (temp_gradient > 0)"""
        df1 = compute_coupling_features(self.sample_data)
        self.assertEqual(df1.iloc[0]["inversion_indicator"], 1)
        
        data_no_inv = self.sample_data.copy()
        data_no_inv["temp_gradient"] = -0.5
        df2 = compute_coupling_features(data_no_inv)
        self.assertEqual(df2.iloc[0]["inversion_indicator"], 0)

    def test_08_lagged_coupling(self):
        """Test 8: lagged_pm25_coupling = pm25_lag_1h / (ventilation_coeff + eps)"""
        df = compute_coupling_features(self.sample_data, epsilon=1e-5)
        vc = 1.2 * 300.0
        expected = 90.0 / (vc + 1e-5)
        self.assertAlmostEqual(df.iloc[0]["lagged_pm25_coupling"], expected, places=4)

    def test_09_edge_cases_zero_pbl_and_wind(self):
        """Test edge cases: zero PBL height and zero wind speed (div by zero prevention)"""
        edge_data = pd.DataFrame([{
            "pm25": 450.0,
            "humidity": 90.0,
            "temperature": 10.0,
            "wind_speed": 0.0,
            "pbl_height": 0.0,
            "temp_gradient": 2.0,
            "pm25_lag_1h": 400.0
        }])
        df = compute_coupling_features(edge_data, epsilon=1e-5)
        self.assertFalse(np.isnan(df.iloc[0]["pm25_div_pbl"]))
        self.assertFalse(np.isinf(df.iloc[0]["pm25_div_pbl"]))
        self.assertEqual(df.iloc[0]["ventilation_coeff"], 0.0)
        self.assertEqual(df.iloc[0]["dispersion_status"], "SEVERE TRAPPING")

    def test_10_aqi_sub_index_calculations(self):
        """Test 9 & 10: AQI sub-index linear interpolation and exact boundary values"""
        # Exact breakpoint boundaries for PM2.5
        self.assertEqual(calculate_sub_index(0.0, "pm25"), 0.0)
        self.assertEqual(calculate_sub_index(30.0, "pm25"), 50.0)
        self.assertEqual(calculate_sub_index(60.0, "pm25"), 100.0)
        self.assertEqual(calculate_sub_index(90.0, "pm25"), 200.0)
        self.assertEqual(calculate_sub_index(120.0, "pm25"), 300.0)
        self.assertEqual(calculate_sub_index(250.0, "pm25"), 400.0)
        self.assertEqual(calculate_sub_index(500.0, "pm25"), 500.0)
        
        # Midpoint interpolation (e.g. PM2.5 = 45 -> midpoint of 30.1-60.0 maps to AQI ~75.5)
        sub_idx_45 = calculate_sub_index(45.0, "pm25")
        self.assertTrue(51.0 <= sub_idx_45 <= 100.0)
        
        # Very high concentration exceeding 500
        sub_idx_high = calculate_sub_index(600.0, "pm25")
        self.assertGreater(sub_idx_high, 500.0)
        
        # AQI Category Boundaries
        self.assertEqual(get_aqi_category(35)["category"], "Good")
        self.assertEqual(get_aqi_category(75)["category"], "Satisfactory")
        self.assertEqual(get_aqi_category(150)["category"], "Moderate")
        self.assertEqual(get_aqi_category(250)["category"], "Poor")
        self.assertEqual(get_aqi_category(350)["category"], "Very Poor")
        self.assertEqual(get_aqi_category(450)["category"], "Severe")

    def test_11_aqi_missing_pollutants(self):
        """Test AQI overall calculation when some pollutants are missing"""
        partial_row = pd.Series({"pm25": 100.0, "pm10": 200.0})  # NO2, SO2, CO, O3 missing
        res = calculate_overall_aqi(partial_row)
        self.assertIn("missing_pollutants", res)
        self.assertGreater(len(res["missing_pollutants"]), 0)
        self.assertGreater(res["overall_aqi"], 0)

if __name__ == "__main__":
    unittest.main()
