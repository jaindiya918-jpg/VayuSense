"""
VayuSense - Phase 4 Unit Tests (SHAP Explainability Engine)
===========================================================
Validates SHAP model loading, feature order alignment, dimension shapes,
base value + SHAP sum reconstruction tolerance, and sign logic.
"""

import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import FINAL_COUPLED_FEATURES
from src.explainability import VayuSenseSHAPExplainer

PROCESSED_PATH = "data/processed/coupled_features.csv"

class TestExplainabilityEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads dataset and initializes SHAP explainer."""
        cls.df_data = pd.read_csv(PROCESSED_PATH)
        cls.explainer_obj = VayuSenseSHAPExplainer()

    def test_01_shap_model_loads(self):
        """Test 1: Validates SHAP model loading and TreeExplainer initialization."""
        self.assertIsNotNone(self.explainer_obj.model)
        self.assertIsNotNone(self.explainer_obj.explainer)

    def test_02_feature_order_matches(self):
        """Test 2: Validates that feature names strictly match training feature order."""
        self.assertEqual(self.explainer_obj.feature_names, FINAL_COUPLED_FEATURES)

    def test_03_shap_values_dimensions(self):
        """Test 3: Validates SHAP matrix shape matches input row count and feature count."""
        sample_df = self.df_data.head(10)[FINAL_COUPLED_FEATURES]
        exp = self.explainer_obj.get_shap_values(sample_df)
        self.assertEqual(exp.values.shape, (10, len(FINAL_COUPLED_FEATURES)))

    def test_04_reconstruction_tolerance(self):
        """Test 4: Validates prediction == base_value + sum(shap_values) within 1e-3."""
        sample_row = self.df_data.iloc[0]
        sample_df = pd.DataFrame([sample_row.to_dict()])[FINAL_COUPLED_FEATURES]
        
        exp = self.explainer_obj.get_shap_values(sample_df)
        base_val = exp.base_values[0]
        shap_sum = np.sum(exp.values[0])
        reconstructed_pred = base_val + shap_sum
        
        actual_pred = self.explainer_obj.model.predict(sample_df)[0]
        self.assertAlmostEqual(reconstructed_pred, actual_pred, places=3)

    def test_05_positive_negative_sign_logic(self):
        """Test 5: Validates positive SHAP increases prediction and negative SHAP decreases prediction."""
        sample_row = self.df_data.iloc[0]
        result = self.explainer_obj.explain_prediction(sample_row)
        
        for item in result["positive_contributors"]:
            self.assertGreater(item["shap_value"], 0.0)
        for item in result["negative_contributors"]:
            self.assertLess(item["shap_value"], 0.0)

    def test_06_demo_explanation_generated(self):
        """Test 6: Validates that local explanation output contains required keys and text."""
        sample_row = self.df_data.iloc[5]
        result = self.explainer_obj.explain_prediction(sample_row)
        
        self.assertIn("predicted_pm25", result)
        self.assertIn("base_value", result)
        self.assertIn("positive_contributors", result)
        self.assertIn("negative_contributors", result)
        self.assertIn("narrative", result)
        self.assertIsInstance(result["narrative"], str)
        self.assertGreater(len(result["narrative"]), 20)

if __name__ == "__main__":
    unittest.main()
