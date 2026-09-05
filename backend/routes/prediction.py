from fastapi import APIRouter, HTTPException
from backend.schemas import CandidateInput, PredictionResponse
from backend.services.model_service import model_service

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse, summary="Classify Exoplanet Candidate")
def predict_candidate(payload: CandidateInput) -> PredictionResponse:
    """
    Accepts Kepler astrophysical and photometric transit observations,
    executes preprocessing and median imputation, feeds features into the
    trained XGBoost classifier pipeline, and returns exoplanet classification
    probabilities along with feature importance insights.
    """
    try:
        input_data = payload.model_dump()
        result = model_service.predict(input_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during candidate classification: {str(e)}"
        )
