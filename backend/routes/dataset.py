from fastapi import APIRouter
from typing import List, Dict, Any
from backend.schemas import SampleCandidate, DatasetStatsResponse
from backend.services.dataset_service import dataset_service

router = APIRouter(tags=["Dataset"])

@router.get("/samples", response_model=List[SampleCandidate], summary="Get Curated Sample Candidates")
def get_sample_candidates() -> List[SampleCandidate]:
    """
    Returns real, verified candidate objects from the Kepler dataset
    (e.g., Kepler-227 b, Kepler-20 b, Eclipsing Binaries, Candidates)
    suitable for 1-click demonstration.
    """
    return dataset_service.get_samples()

@router.get("/dataset-stats", response_model=DatasetStatsResponse, summary="Get Kepler Dataset Statistics")
def get_dataset_statistics() -> DatasetStatsResponse:
    """
    Returns dataset distributions, class counts, and summary statistics.
    """
    stats = dataset_service.get_dataset_stats()
    return DatasetStatsResponse(**stats)
