from fastapi import APIRouter
from backend.schemas import HealthResponse
from backend.services.model_service import model_service
from backend.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse, summary="API Health Check")
def health_check():
    """Returns the operational status of the Kepler Exoplanet API and ML pipeline."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=model_service.is_loaded()
    )
