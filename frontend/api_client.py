import os
import requests
from typing import Dict, Any, List, Optional

# Read backend URL from environment or default to local FastAPI port 8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

class ExoplanetAPIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = 5.0

    def check_health(self) -> Dict[str, Any]:
        """Queries the FastAPI /health endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "offline", "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Queries /model-info endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Error fetching model info from API: {e}")
        
        # Fallback to local inference module if backend is temporarily offline
        try:
            from ml.inference import KeplerInferenceEngine
            from ml.feature_definitions import MODEL_METRICS_COMPARISON, METRIC_DEFINITIONS
            engine = KeplerInferenceEngine()
            meta = engine.get_model_metadata()
            meta["notebook_comparison"] = MODEL_METRICS_COMPARISON
            meta["metric_definitions"] = METRIC_DEFINITIONS
            return meta
        except Exception:
            return {}

    def get_samples(self) -> List[Dict[str, Any]]:
        """Queries /samples endpoint for curated candidate examples."""
        try:
            resp = requests.get(f"{self.base_url}/samples", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Error fetching samples from API: {e}")

        # Fallback to loading sample_candidates.json
        import json
        samples_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "sample_candidates.json"
        )
        if os.path.exists(samples_file):
            try:
                with open(samples_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Queries /dataset-stats endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/dataset-stats", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Error fetching dataset stats from API: {e}")

        return {
            "total_records": 9564,
            "total_features": 36,
            "class_distribution": {"FALSE POSITIVE": 5023, "CONFIRMED": 2293, "CANDIDATE": 2248},
            "features_summary": {}
        }

    def predict(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calls POST /predict on the FastAPI backend."""
        try:
            resp = requests.post(
                f"{self.base_url}/predict",
                json=candidate_data,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            else:
                return {
                    "success": False,
                    "error": f"API Error ({resp.status_code}): {resp.text}"
                }
        except requests.exceptions.RequestException as e:
            # Fallback to local ML inference if FastAPI server is unreachable
            try:
                from ml.inference import KeplerInferenceEngine
                engine = KeplerInferenceEngine()
                res = engine.predict(candidate_data)
                return {"success": True, "data": res, "is_fallback": True}
            except Exception as local_err:
                return {
                    "success": False,
                    "error": f"Connection failed to {self.base_url} and fallback failed: {local_err}"
                }

api_client = ExoplanetAPIClient()
