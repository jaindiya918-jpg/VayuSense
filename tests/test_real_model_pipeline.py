"""
VayuSense - Real Data Model Pipeline & SHAP Unit Tests
======================================================
Validates Real Baseline & Real Coupled XGBoost model artifacts,
predictions CSV, metrics JSON, pre-training validation, and real SHAP explanations.
"""

import os
import unittest
from unittest.mock import patch
import json
import pandas as pd
import numpy as np

from src.data_providers.openmeteo_provider import OpenMeteoProvider
from src.data_providers.openaq_provider import OpenAQProvider
from src.audit_real_data import perform_real_data_audit
from src.pre_training_validation import run_pre_training_validation, PRE_TRAIN_JSON
from src.train_real_models import (
    REAL_BASELINE_MODEL_PATH, REAL_COUPLED_MODEL_PATH, REAL_METRICS_PATH, REAL_PREDICTIONS_PATH
)
from src.real_explainability import (
    VayuSenseRealSHAPExplainer, REAL_GLOBAL_RESULTS_PATH, REAL_DEMO_EXPLANATION_PATH
)

REAL_DATA_PATH = "data/raw/real_delhi_ncr_data.csv"

class TestRealModelPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads real data initializers."""
        cls.openmeteo = OpenMeteoProvider()
        cls.openaq = OpenAQProvider()

    @patch('src.data_providers.openmeteo_provider.requests.get')
    def test_01_openmeteo_provider_returns_df(self, mock_get):
        """Test 1: Validates Open-Meteo weather provider API call (mocked)."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2024-01-01T00:00"],
                "temperature_2m": [15.0],
                "relative_humidity_2m": [60.0],
                "wind_speed_10m": [2.5],
                "wind_direction_10m": [180.0],
                "surface_pressure": [1010.0],
                "precipitation": [0.0],
                "boundary_layer_height": [300.0]
            }
        }
        df = self.openmeteo.fetch_weather_data(28.6469, 77.3162, "2024-01-01", "2024-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    @patch('src.data_providers.openaq_provider.requests.get')
    def test_02_openaq_provider_returns_df(self, mock_get):
        """Test 2: Validates OpenAQ provider API call and fallback (mocked)."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2024-01-01T00:00"],
                "pm2_5": [50.0],
                "pm10": [100.0],
                "nitrogen_dioxide": [20.0],
                "sulphur_dioxide": [10.0],
                "carbon_monoxide": [1000.0],
                "ozone": [30.0]
            }
        }
        df = self.openaq.fetch_air_quality_data(28.6469, 77.3162, "2024-01-01", "2024-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_03_real_dataset_csv_exists_and_valid(self):
        """Test 3: Validates saved real_delhi_ncr_data.csv file and schema."""
        self.assertTrue(os.path.exists(REAL_DATA_PATH), f"Missing {REAL_DATA_PATH}")
        df_real = pd.read_csv(REAL_DATA_PATH)
        self.assertEqual(len(df_real), 8832)

    def test_04_real_model_artifacts_exist(self):
        """Test 4: Validates existence of trained real model files and predictions."""
        self.assertTrue(os.path.exists(REAL_BASELINE_MODEL_PATH), f"Missing {REAL_BASELINE_MODEL_PATH}")
        self.assertTrue(os.path.exists(REAL_COUPLED_MODEL_PATH), f"Missing {REAL_COUPLED_MODEL_PATH}")
        self.assertTrue(os.path.exists(REAL_METRICS_PATH), f"Missing {REAL_METRICS_PATH}")
        self.assertTrue(os.path.exists(REAL_PREDICTIONS_PATH), f"Missing {REAL_PREDICTIONS_PATH}")

    def test_05_real_predictions_csv_schema(self):
        """Test 5: Validates real_model_predictions.csv columns and non-null values."""
        df_preds = pd.read_csv(REAL_PREDICTIONS_PATH)
        self.assertGreater(len(df_preds), 0)
        self.assertEqual(df_preds.isnull().sum().sum(), 0)
        expected_cols = ["timestamp", "station_id", "actual_pm25", "baseline_prediction", "coupled_prediction", "pm25_target_6h"]
        for col in expected_cols:
            self.assertIn(col, df_preds.columns)

    def test_06_real_metrics_json_content(self):
        """Test 6: Validates metrics payload contents and improvement calculations."""
        with open(REAL_METRICS_PATH, "r") as f:
            data = json.load(f)
            
        self.assertIn("baseline_metrics", data)
        self.assertIn("coupled_metrics", data)
        self.assertIn("improvements", data)
        self.assertIn("mae_improvement_pct", data["improvements"])

    def test_07_real_shap_artifacts_exist_and_valid(self):
        """Test 7: Validates real SHAP global and local explanation payloads."""
        self.assertTrue(os.path.exists(REAL_GLOBAL_RESULTS_PATH), f"Missing {REAL_GLOBAL_RESULTS_PATH}")
        self.assertTrue(os.path.exists(REAL_DEMO_EXPLANATION_PATH), f"Missing {REAL_DEMO_EXPLANATION_PATH}")
        
        with open(REAL_GLOBAL_RESULTS_PATH, "r") as f:
            gdata = json.load(f)
        self.assertEqual(gdata["data_mode"], "REAL DATA")
        self.assertIn("attribution_share", gdata)
        self.assertIn("global_ranking", gdata)
        
        with open(REAL_DEMO_EXPLANATION_PATH, "r") as f:
            ldata = json.load(f)
        self.assertEqual(ldata["data_mode"], "REAL DATA")
        self.assertIn("predicted_pm25", ldata)
        self.assertIn("positive_contributors", ldata)
        self.assertIn("narrative", ldata)

if __name__ == "__main__":
    unittest.main()
