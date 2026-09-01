"""
VayuSense - Phase 5 Unit Tests (Dashboard Integration & Smoke Tests)
====================================================================
Validates prediction data loading, SHAP explainer integration,
what-if simulator feature reconstruction, and AQI card math.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import FINAL_COUPLED_FEATURES
from src.coupling_engine import compute_coupling_features
from src.aqi_calculator import calculate_overall_aqi, calculate_sub_index
from src.explainability import VayuSenseSHAPExplainer

PREDICTIONS_PATH = "data/processed/final_model_predictions.csv"
PROCESSED_PATH = "data/processed/coupled_features.csv"

class TestDashboardIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads data and initializers."""
        cls.df_preds = pd.read_csv(PREDICTIONS_PATH)
        cls.df_feats = pd.read_csv(PROCESSED_PATH)
        cls.explainer = VayuSenseSHAPExplainer()

    def test_01_predictions_data_exists(self):
        """Test 1: Validates that final model predictions exist and have 0 nulls."""
        self.assertTrue(os.path.exists(PREDICTIONS_PATH))
        self.assertGreater(len(self.df_preds), 0)
        self.assertEqual(self.df_preds.isnull().sum().sum(), 0)

    def test_02_all_stations_present(self):
        """Test 2: Validates all 4 Delhi NCR stations are present in dataset."""
        stations = self.df_preds["station_id"].unique()
        expected = ["Anand_Vihar", "RK_Puram", "Punjabi_Bagh", "Mandir_Marg"]
        for st in expected:
            self.assertIn(st, stations, f"Missing station: {st}")

    def test_03_simulator_feature_reconstruction(self):
        """Test 3: Validates that Weather What-If simulator reconstructs coupling terms cleanly."""
        base_row = self.df_feats.iloc[0]
        sim_dict = base_row.to_dict()
        
        # Modify wind speed to 5.0 m/s and PBL height to 1200m
        sim_dict["wind_speed"] = 5.0
        sim_dict["pbl_height"] = 1200.0
        
        sim_df = pd.DataFrame([sim_dict])
        sim_df = compute_coupling_features(sim_df)
        
        # Check ventilation coeff = 5.0 * 1200 = 6000 m2/s
        self.assertAlmostEqual(sim_df.iloc[0]["ventilation_coeff"], 6000.0)
        self.assertEqual(sim_df.iloc[0]["stagnation_indicator"], 0)
        self.assertEqual(sim_df.iloc[0]["dispersion_status"], "GOOD DISPERSION")
        
        # Verify model can run prediction on reconstructed feature vector
        X_sim = sim_df[FINAL_COUPLED_FEATURES]
        pred = self.explainer.model.predict(X_sim)[0]
        self.assertFalse(np.isnan(pred))

    def test_04_aqi_sub_index_distinction(self):
        """Test 4: Validates distinction between PM2.5 concentration and AQI score."""
        pm25_conc = 100.0
        aqi_sub = calculate_sub_index(pm25_conc, "pm25")
        
        self.assertNotEqual(pm25_conc, aqi_sub)
        self.assertEqual(aqi_sub, 233.8)  # CPCB linear interpolation for 100.0 ug/m3 is 233.8

if __name__ == "__main__":
    unittest.main()
