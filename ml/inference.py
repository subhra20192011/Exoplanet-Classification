"""
ML Inference engine for the Kepler Exoplanet pipeline.
Handles model loading, input validation, feature ordering, probability extraction,
and feature contribution analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ml.feature_definitions import (
    FEATURE_ORDER,
    CLASS_MAPPING,
    CLASS_DISPLAY,
    CLASS_BADGE,
    CLASS_DESCRIPTION,
    FEATURE_METADATA
)

class KeplerInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default model path resolution
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "kepler_pipeline_v3.pkl")
            if not os.path.exists(model_path):
                # Fallback to root if models/ not used
                alt_path = os.path.join(base_dir, "kepler_pipeline_v3.pkl")
                if os.path.exists(alt_path):
                    model_path = alt_path

        self.model_path = model_path
        self.model = None
        self.feature_importances_dict: Dict[str, float] = {}
        self.load_model()

    def load_model(self):
        """Loads the trained sklearn Pipeline model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Extract feature importances if available
        try:
            if hasattr(self.model, "named_steps") and "classifier" in self.model.named_steps:
                classifier = self.model.named_steps["classifier"]
                if hasattr(classifier, "feature_importances_"):
                    importances = classifier.feature_importances_
                    for idx, feat in enumerate(FEATURE_ORDER):
                        if idx < len(importances):
                            self.feature_importances_dict[feat] = float(importances[idx])
        except Exception as e:
            print(f"Warning: Could not extract feature importances: {e}")

    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Takes raw dictionary input and aligns it strictly to the expected 36-feature DataFrame.
        Fills missing fields with median defaults.
        """
        row_dict = {}
        for feat in FEATURE_ORDER:
            val = input_data.get(feat, None)
            if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
                row_dict[feat] = FEATURE_METADATA.get(feat, {}).get("default", np.nan)
            else:
                try:
                    row_dict[feat] = float(val)
                except (ValueError, TypeError):
                    row_dict[feat] = FEATURE_METADATA.get(feat, {}).get("default", np.nan)
                    
        return pd.DataFrame([row_dict], columns=FEATURE_ORDER)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes prediction on input dictionary and returns rich classification results.
        """
        if self.model is None:
            self.load_model()

        df_input = self.prepare_features(input_data)
        
        # Predict class index and probabilities
        pred_idx = int(self.model.predict(df_input)[0])
        probabilities = self.model.predict_proba(df_input)[0]

        pred_class = CLASS_MAPPING.get(pred_idx, "UNKNOWN")
        display_name = CLASS_DISPLAY.get(pred_class, pred_class)
        badge = CLASS_BADGE.get(pred_class, "⚪")
        description = CLASS_DESCRIPTION.get(pred_class, "")

        prob_breakdown = {
            "CONFIRMED": float(probabilities[0]),
            "FALSE POSITIVE": float(probabilities[1]),
            "CANDIDATE": float(probabilities[2])
        }

        winning_probability = float(probabilities[pred_idx])

        # Confidence level tier
        if winning_probability >= 0.80:
            confidence_level = "High Confidence"
        elif winning_probability >= 0.50:
            confidence_level = "Moderate Confidence"
        else:
            confidence_level = "Marginal Confidence"

        # Top influential features for this model
        top_features = self.get_top_influential_features(df_input, top_n=5)

        return {
            "predicted_class": pred_class,
            "display_name": display_name,
            "badge": badge,
            "class_index": pred_idx,
            "probability": round(winning_probability, 4),
            "confidence_level": confidence_level,
            "probabilities": {k: round(v, 4) for k, v in prob_breakdown.items()},
            "description": description,
            "model_name": "XGBoost Classifier (Median Imputer Pipeline)",
            "top_features": top_features
        }

    def get_top_influential_features(self, df_input: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Returns top N global model feature importances with user's input values."""
        if not self.feature_importances_dict:
            return []

        sorted_feats = sorted(
            self.feature_importances_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        results = []
        for feat_key, importance in sorted_feats:
            meta = FEATURE_METADATA.get(feat_key, {})
            val = float(df_input[feat_key].iloc[0])
            results.append({
                "feature": feat_key,
                "name": meta.get("name", feat_key),
                "unit": meta.get("unit", ""),
                "value": round(val, 4) if not np.isnan(val) else None,
                "importance_score": round(importance, 4),
                "tooltip": meta.get("tooltip", "")
            })
        return results

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns comprehensive metadata about the deployed model."""
        classifier_params = {}
        if hasattr(self.model, "named_steps") and "classifier" in self.model.named_steps:
            classifier_params = self.model.named_steps["classifier"].get_params()

        sorted_importance = [
            {
                "feature": k,
                "name": FEATURE_METADATA.get(k, {}).get("name", k),
                "unit": FEATURE_METADATA.get(k, {}).get("unit", ""),
                "importance": round(v, 4)
            }
            for k, v in sorted(self.feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "model_type": "XGBoost Classifier within Scikit-Learn Pipeline",
            "pipeline_steps": ["SimpleImputer(strategy='median')", "XGBClassifier"],
            "features_count": len(FEATURE_ORDER),
            "classes": ["CONFIRMED (0)", "FALSE POSITIVE (1)", "CANDIDATE (2)"],
            "test_accuracy": 0.7865,
            "cv_accuracy": 0.7854,
            "macro_f1": 0.74,
            "hyperparameters": {
                "n_estimators": classifier_params.get("n_estimators", 200),
                "learning_rate": classifier_params.get("learning_rate", 0.05),
                "max_depth": classifier_params.get("max_depth", 5),
                "subsample": classifier_params.get("subsample", 0.8),
                "colsample_bytree": classifier_params.get("colsample_bytree", 0.8),
                "random_state": classifier_params.get("random_state", 42)
            },
            "feature_importances": sorted_importance
        }
