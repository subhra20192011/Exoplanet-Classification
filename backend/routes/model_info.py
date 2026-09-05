from fastapi import APIRouter
from typing import Dict, Any
from backend.services.model_service import model_service
from ml.feature_definitions import MODEL_METRICS_COMPARISON, METRIC_DEFINITIONS

router = APIRouter(tags=["Model Info"])

@router.get("/model-info", summary="Get Model Architecture & Metrics")
def get_model_info() -> Dict[str, Any]:
    """
    Returns comprehensive metadata regarding the deployed model,
    its hyperparameters, notebook evaluation metrics comparison, and feature importance rankings.
    """
    metadata = model_service.get_metadata()
    metadata["notebook_comparison"] = MODEL_METRICS_COMPARISON
    metadata["metric_definitions"] = METRIC_DEFINITIONS
    return metadata
