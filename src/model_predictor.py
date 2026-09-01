"""
VayuSense - Model Predictor Module
===================================
Handles independent inference for Baseline and Coupled XGBoost models.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import xgboost as xgb
from src.feature_engineering import BASELINE_FEATURES, COUPLED_FEATURES_FULL

BASELINE_MODEL_PATH = "models/baseline_xgb.json"
COUPLED_MODEL_PATH = "models/coupled_xgb.json"

class VayuSensePredictor:
    def __init__(
        self,
        baseline_path: str = BASELINE_MODEL_PATH,
        coupled_path: str = COUPLED_MODEL_PATH
    ):
        self.baseline_model = xgb.XGBRegressor()
        self.baseline_model.load_model(baseline_path)
        
        self.coupled_model = xgb.XGBRegressor()
        self.coupled_model.load_model(coupled_path)
        
        self.baseline_features = BASELINE_FEATURES
        self.coupled_features = COUPLED_FEATURES_FULL

    def predict_row(self, row: pd.Series) -> dict:
        """
        Accepts a single feature pandas Series and returns predictions from both models.
        """
        df_row = pd.DataFrame([row.to_dict()])
        return self.predict_dataframe(df_row)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> list:
        """
        Accepts a feature DataFrame and returns a list of prediction dictionaries.
        """
        X_base = df[self.baseline_features]
        X_coup = df[self.coupled_features]
        
        preds_base = self.baseline_model.predict(X_base)
        preds_coup = self.coupled_model.predict(X_coup)
        
        results = []
        for b_pred, c_pred in zip(preds_base, preds_coup):
            results.append({
                "baseline_prediction": float(round(b_pred, 2)),
                "coupled_prediction": float(round(c_pred, 2)),
                "coupling_delta": float(round(c_pred - b_pred, 2))
            })
            
        return results

if __name__ == "__main__":
    predictor = VayuSensePredictor()
    print("VayuSensePredictor successfully initialized and loaded trained models.")
