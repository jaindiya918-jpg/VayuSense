"""
VayuSense - Real Data Pipeline & Pre-Training Validation Unit Tests
====================================================================
Validates OpenAQ & Open-Meteo data providers, .env configuration switch,
real dataset schema compatibility, multi-horizon targets, audit, and pre-training JSON payload.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.data_providers.openmeteo_provider import OpenMeteoProvider
from src.data_providers.openaq_provider import OpenAQProvider
from src.audit_real_data import perform_real_data_audit
from src.pre_training_validation import run_pre_training_validation, PRE_TRAIN_JSON

REAL_DATA_PATH = "data/raw/real_delhi_ncr_data.csv"

class TestRealDataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads real data initializers."""
        cls.openmeteo = OpenMeteoProvider()
        cls.openaq = OpenAQProvider()

    def test_01_openmeteo_provider_returns_df(self):
        """Test 1: Validates Open-Meteo weather provider API call."""
        df = self.openmeteo.fetch_weather_data(28.6469, 77.3162, "2024-01-01", "2024-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        expected_cols = ["timestamp", "temperature", "humidity", "wind_speed", "wind_deg", "pressure", "rainfall", "pbl_height"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_02_openaq_provider_returns_df(self):
        """Test 2: Validates OpenAQ provider API call and fallback."""
        df = self.openaq.fetch_air_quality_data(28.6469, 77.3162, "2024-01-01", "2024-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        expected_cols = ["timestamp", "pm25", "pm10", "no2", "so2", "co", "o3"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_03_real_dataset_csv_exists_and_valid(self):
        """Test 3: Validates saved real_delhi_ncr_data.csv file and schema."""
        self.assertTrue(os.path.exists(REAL_DATA_PATH), f"Missing {REAL_DATA_PATH}")
        df_real = pd.read_csv(REAL_DATA_PATH)
        self.assertEqual(len(df_real), 8832)
        self.assertEqual(df_real["is_demo"].sum(), 0)
        
        stations = df_real["station_id"].unique()
        self.assertEqual(set(stations), {"Anand_Vihar", "RK_Puram", "Punjabi_Bagh", "Mandir_Marg"})

    def test_04_real_data_audit_passes(self):
        """Test 4: Validates that quality audit passes without leakage or unphysical values."""
        audit_report = perform_real_data_audit(REAL_DATA_PATH)
        self.assertFalse(audit_report["future_data_leakage_detected"])
        self.assertFalse(audit_report["station_leakage_detected"])
        self.assertEqual(audit_report["duplicate_row_count"], 0)
        self.assertEqual(len(audit_report["suspicious_physical_values"]), 0)

    def test_05_pre_training_validation_json(self):
        """Test 5: Validates pre-training validation payload generation."""
        val_data = run_pre_training_validation(REAL_DATA_PATH)
        self.assertTrue(os.path.exists(PRE_TRAIN_JSON))
        self.assertEqual(val_data["wind_speed_audit"]["final_model_unit"], "m/s")
        self.assertTrue(val_data["confirmations"]["target_excluded_from_X"])
        self.assertTrue(val_data["confirmations"]["future_variables_excluded"])

if __name__ == "__main__":
    unittest.main()
