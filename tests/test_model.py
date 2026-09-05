import os
import pytest
import numpy as np
import pandas as pd
from ml.inference import KeplerInferenceEngine
from ml.feature_definitions import FEATURE_ORDER, CLASS_MAPPING

def test_model_initialization():
    """Verify that the model loads successfully from artifacts."""
    engine = KeplerInferenceEngine()
    assert engine.model is not None
    assert len(engine.feature_importances_dict) == len(FEATURE_ORDER)

def test_feature_preparation():
    """Verify that feature preparation produces exactly 36 ordered columns."""
    engine = KeplerInferenceEngine()
    dummy_input = {"koi_period": 10.5, "koi_prad": 2.1}
    df_prepared = engine.prepare_features(dummy_input)
    
    assert list(df_prepared.columns) == FEATURE_ORDER
    assert df_prepared.shape == (1, 36)
    assert df_prepared["koi_period"].iloc[0] == 10.5
    assert df_prepared["koi_prad"].iloc[0] == 2.1
    # Check that missing features received non-null defaults
    assert not np.isnan(df_prepared["koi_steff"].iloc[0])

def test_prediction_output_structure():
    """Verify that predict returns the full structured response dictionary."""
    engine = KeplerInferenceEngine()
    sample_input = {
        "koi_period": 9.488,
        "koi_duration": 3.79,
        "koi_depth": 424.0,
        "koi_prad": 2.39,
        "koi_steff": 5757.0
    }
    result = engine.predict(sample_input)

    assert "predicted_class" in result
    assert result["predicted_class"] in ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    
    # Probabilities should sum to approximately 1.0
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01

def test_model_metadata():
    """Verify model metadata returns expected notebook metrics and parameters."""
    engine = KeplerInferenceEngine()
    meta = engine.get_model_metadata()

    assert meta["test_accuracy"] == 0.7865
    assert meta["cv_accuracy"] == 0.7854
    assert meta["features_count"] == 36
    assert len(meta["feature_importances"]) == 36
