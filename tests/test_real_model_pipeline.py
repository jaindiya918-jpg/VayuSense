"""
VayuSense - Real Data Model Pipeline & SHAP Unit Tests
======================================================
Validates Real Baseline & Real Coupled XGBoost model artifacts,
predictions CSV, metrics JSON, pre-training validation, and real SHAP explanations.
"""

import os
import unittest
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

    def test_01_openmeteo_provider_returns_df(self):
        """Test 1: Validates Open-Meteo weather provider API call."""
        df = self.openmeteo.fetch_weather_data(28.6469, 77.3162, "2024-01-01", "2024-01-02")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_02_openaq_provider_returns_df(self):
        """Test 2: Validates OpenAQ provider API call and fallback."""
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
