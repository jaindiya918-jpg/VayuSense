"""
VayuSense - Phase 3.6 Unit Tests (Final Model Pipeline)
========================================================
Validates final feature selection (no seasonal shortcuts), leakage checks,
final model artifact existence, and prediction files.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    FINAL_BASELINE_FEATURES, FINAL_COUPLED_FEATURES, COUPLING_FEATURES
)
from src.model_trainer import perform_data_leakage_check, create_chronological_splits

FINAL_BASELINE_PATH = "models/final_baseline_xgb.json"
FINAL_COUPLED_PATH = "models/final_coupled_xgb.json"
FINAL_METRICS_PATH = "models/final_model_metrics.json"
FINAL_PREDICTIONS_PATH = "data/processed/final_model_predictions.csv"
PROCESSED_PATH = "data/processed/coupled_features.csv"

class TestFinalModelPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads processed dataset for testing."""
        cls.df_features = pd.read_csv(PROCESSED_PATH)

    def test_01_no_seasonal_shortcuts_in_final_features(self):
        """Validates that is_winter and month are REMOVED from final feature sets."""
        self.assertNotIn("is_winter", FINAL_BASELINE_FEATURES)
        self.assertNotIn("month", FINAL_BASELINE_FEATURES)
        self.assertNotIn("is_winter", FINAL_COUPLED_FEATURES)
        self.assertNotIn("month", FINAL_COUPLED_FEATURES)

    def test_02_hour_encodings_preserved(self):
        """Validates that hour_sin and hour_cos ARE preserved in final feature sets."""
        self.assertIn("hour_sin", FINAL_BASELINE_FEATURES)
        self.assertIn("hour_cos", FINAL_BASELINE_FEATURES)
        self.assertIn("hour_sin", FINAL_COUPLED_FEATURES)
        self.assertIn("hour_cos", FINAL_COUPLED_FEATURES)

    def test_03_leakage_check_passes(self):
        """Validates that explicit data leakage checks pass cleanly."""
        self.assertTrue(perform_data_leakage_check(self.df_features, FINAL_BASELINE_FEATURES))
        self.assertTrue(perform_data_leakage_check(self.df_features, FINAL_COUPLED_FEATURES))

    def test_04_final_artifacts_exist(self):
        """Validates that all Phase 3.6 final model artifacts exist on disk."""
        self.assertTrue(os.path.exists(FINAL_BASELINE_PATH), f"Missing {FINAL_BASELINE_PATH}")
        self.assertTrue(os.path.exists(FINAL_COUPLED_PATH), f"Missing {FINAL_COUPLED_PATH}")
        self.assertTrue(os.path.exists(FINAL_METRICS_PATH), f"Missing {FINAL_METRICS_PATH}")
        self.assertTrue(os.path.exists(FINAL_PREDICTIONS_PATH), f"Missing {FINAL_PREDICTIONS_PATH}")

    def test_05_final_predictions_file(self):
        """Validates final_model_predictions.csv schema and zero nulls."""
        df_preds = pd.read_csv(FINAL_PREDICTIONS_PATH)
        required_cols = ["timestamp", "station_id", "actual_pm25", "final_baseline_prediction", "final_coupled_prediction"]
        for col in required_cols:
            self.assertIn(col, df_preds.columns)
        self.assertEqual(df_preds.isnull().sum().sum(), 0)

if __name__ == "__main__":
    unittest.main()
