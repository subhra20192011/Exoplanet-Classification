import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["health"] == "/health"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "hyperparameters" in data
    assert "notebook_comparison" in data
    assert len(data["notebook_comparison"]) == 4

def test_samples_endpoint():
    response = client.get("/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    assert "id" in samples[0]
    assert "features" in samples[0]

def test_dataset_stats_endpoint():
    response = client.get("/dataset-stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] > 0
    assert "class_distribution" in data
    assert "CONFIRMED" in data["class_distribution"]

def test_predict_endpoint_valid_payload():
    payload = {
        "koi_period": 9.488,
        "koi_duration": 3.79,
        "koi_depth": 424.0,
        "koi_prad": 2.39,
        "koi_teq": 898.0,
        "koi_steff": 5757.0,
        "koi_srad": 0.99
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
    assert "probability" in data
    assert "probabilities" in data
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_endpoint_empty_payload():
    # Empty payload should use default median values and succeed
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
