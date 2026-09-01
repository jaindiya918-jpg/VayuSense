import os
import unittest
import pandas as pd
import numpy as np
from src.data_generator import generate_and_save_dataset
from src.preprocessing import load_and_preprocess_data
from src.feature_engineering import BASELINE_FEATURES, COUPLING_FEATURES

RAW_DATA_PATH = "data/raw/delhi_ncr_aqi_weather_demo.csv"
PROCESSED_DATA_PATH = "data/processed/coupled_features.csv"

class TestPhase1(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generates raw synthetic demo data and processes features."""
        cls.df_raw = generate_and_save_dataset(output_path=RAW_DATA_PATH, days=15)
        cls.df_processed = load_and_preprocess_data(
            raw_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH,
            forecast_horizon=6,
            force_regenerate=True
        )

    def test_01_raw_data_schema(self):
        """Validates schema, column existence, and DEMO tag of generated dataset."""
        required_cols = [
            "data_source", "timestamp", "station_id", "temperature", "humidity",
            "wind_speed", "pbl_height", "temp_gradient", "pm25", "pm10", "no2", "so2", "co", "o3"
        ]
        for col in required_cols:
            self.assertIn(col, self.df_raw.columns, f"Missing required raw column: {col}")
            
        # Verify DEMO dataset tag
        self.assertTrue((self.df_raw["data_source"] == "DEMO_SYNTHETIC").all())
        
        # Verify non-empty dataset
        self.assertGreater(len(self.df_raw), 0)
        
    def test_02_missing_values_report(self):
        """Validates that there are 0 missing values in both raw and processed datasets."""
        raw_nulls = self.df_raw.isnull().sum().sum()
        proc_nulls = self.df_processed.isnull().sum().sum()
        
        self.assertEqual(raw_nulls, 0, f"Raw data contains {raw_nulls} missing values.")
        self.assertEqual(proc_nulls, 0, f"Processed data contains {proc_nulls} missing values.")

    def test_03_chronological_ordering(self):
        """Validates strict per-station chronological timestamp ordering."""
        for station, group in self.df_processed.groupby("station_id"):
            timestamps = pd.to_datetime(group["timestamp"])
            is_sorted = timestamps.is_monotonic_increasing
            self.assertTrue(is_sorted, f"Timestamps for station {station} are not strictly chronological!")

    def test_04_coupling_formulas(self):
        """Validates exact math for physics-informed coupling proxy terms."""
        row = self.df_processed.iloc[0]
        
        # Ventilation Coefficient = wind_speed * pbl_height
        expected_vc = row["wind_speed"] * row["pbl_height"]
        self.assertAlmostEqual(row["ventilation_coeff"], expected_vc, places=3)
        
        # Stagnation Indicator
        expected_stag = 1 if (row["wind_speed"] < 1.5 and row["pbl_height"] < 400.0) else 0
        self.assertEqual(row["stagnation_indicator"], expected_stag)
        
        # Inversion Indicator
        expected_inv = 1 if (row["temp_gradient"] > 0.0) else 0
        self.assertEqual(row["inversion_indicator"], expected_inv)

    def test_05_feature_set_existence(self):
        """Validates that all Baseline and Coupling features exist in processed data."""
        for feat in BASELINE_FEATURES:
            self.assertIn(feat, self.df_processed.columns, f"Missing baseline feature: {feat}")
        for feat in COUPLING_FEATURES:
            self.assertIn(feat, self.df_processed.columns, f"Missing coupling feature: {feat}")

if __name__ == "__main__":
    unittest.main()
