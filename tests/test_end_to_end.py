import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_end_to_end_sample_candidate_prediction():
    """
    End-to-end test simulating a user selecting a preset sample,
    sending it to FastAPI /predict, and getting back the expected classification.
    """
    # 1. Fetch sample candidates
    samples_resp = client.get("/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()
    assert len(samples) >= 3

    # Find the Kepler-227 b confirmed exoplanet sample
    k227b = next((s for s in samples if "Kepler-227 b" in s["name"]), samples[0])
    
    # 2. Submit candidate features to prediction endpoint
    pred_resp = client.post("/predict", json=k227b["features"])
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()

    # 3. Assert prediction matches expected structure and values
    assert pred_data["predicted_class"] == "CONFIRMED"
    assert pred_data["display_name"] == "Confirmed Exoplanet"
    assert pred_data["probability"] > 0.70
    assert pred_data["probabilities"]["CONFIRMED"] > 0.70
    assert len(pred_data["top_features"]) > 0

def test_end_to_end_false_positive_sample():
    """End-to-end test for a false positive astrophysical signal."""
    samples_resp = client.get("/samples")
    samples = samples_resp.json()
    
    fp_sample = next((s for s in samples if s["true_disposition"] == "FALSE POSITIVE"), None)
    if fp_sample:
        pred_resp = client.post("/predict", json=fp_sample["features"])
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert pred_data["predicted_class"] == "FALSE POSITIVE"
        assert pred_data["probabilities"]["FALSE POSITIVE"] > 0.60
