from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class CandidateInput(BaseModel):
    # Transit Characteristics
    koi_period: Optional[float] = Field(9.488, description="Orbital Period (days)", examples=[9.488])
    koi_period_err1: Optional[float] = Field(0.00005, description="+1-sigma period uncertainty (days)", examples=[0.00005])
    koi_period_err2: Optional[float] = Field(-0.00005, description="-1-sigma period uncertainty (days)", examples=[-0.00005])
    koi_time0bk: Optional[float] = Field(170.53, description="Transit Epoch (BKJD days)", examples=[170.53])
    koi_time0bk_err1: Optional[float] = Field(0.003, description="+1-sigma epoch uncertainty (days)", examples=[0.003])
    koi_time0bk_err2: Optional[float] = Field(-0.003, description="-1-sigma epoch uncertainty (days)", examples=[-0.003])
    koi_impact: Optional[float] = Field(0.537, description="Impact Parameter ratio", examples=[0.537])
    koi_impact_err1: Optional[float] = Field(0.05, description="+1-sigma impact uncertainty", examples=[0.05])
    koi_impact_err2: Optional[float] = Field(-0.05, description="-1-sigma impact uncertainty", examples=[-0.05])
    koi_duration: Optional[float] = Field(3.79, description="Transit Duration (hours)", examples=[3.79])
    koi_duration_err1: Optional[float] = Field(0.08, description="+1-sigma duration uncertainty (hours)", examples=[0.08])
    koi_duration_err2: Optional[float] = Field(-0.08, description="-1-sigma duration uncertainty (hours)", examples=[-0.08])
    koi_depth: Optional[float] = Field(424.0, description="Transit Depth (ppm)", examples=[424.0])
    koi_depth_err1: Optional[float] = Field(10.0, description="+1-sigma depth uncertainty (ppm)", examples=[10.0])
    koi_depth_err2: Optional[float] = Field(-10.0, description="-1-sigma depth uncertainty (ppm)", examples=[-10.0])
    koi_prad: Optional[float] = Field(2.39, description="Planetary Radius (Earth radii)", examples=[2.39])
    koi_prad_err1: Optional[float] = Field(0.15, description="+1-sigma planet radius uncertainty", examples=[0.15])
    koi_prad_err2: Optional[float] = Field(-0.15, description="-1-sigma planet radius uncertainty", examples=[-0.15])
    koi_teq: Optional[float] = Field(898.0, description="Equilibrium Temperature (Kelvin)", examples=[898.0])
    koi_insol: Optional[float] = Field(149.0, description="Insolation Flux (Earth flux)", examples=[149.0])
    koi_insol_err1: Optional[float] = Field(15.0, description="+1-sigma insolation flux uncertainty", examples=[15.0])
    koi_insol_err2: Optional[float] = Field(-15.0, description="-1-sigma insolation flux uncertainty", examples=[-15.0])
    koi_model_snr: Optional[float] = Field(23.3, description="Transit Signal-to-Noise Ratio (SNR)", examples=[23.3])
    koi_tce_plnt_num: Optional[float] = Field(1.0, description="TCE Planet Number in system", examples=[1.0])
    
    # Stellar Properties
    koi_steff: Optional[float] = Field(5757.0, description="Stellar Effective Temperature (K)", examples=[5757.0])
    koi_steff_err1: Optional[float] = Field(100.0, description="+1-sigma stellar temperature uncertainty", examples=[100.0])
    koi_steff_err2: Optional[float] = Field(-100.0, description="-1-sigma stellar temperature uncertainty", examples=[-100.0])
    koi_slogg: Optional[float] = Field(4.44, description="Stellar Surface Gravity log10(cm/s²)", examples=[4.44])
    koi_slogg_err1: Optional[float] = Field(0.05, description="+1-sigma surface gravity uncertainty", examples=[0.05])
    koi_slogg_err2: Optional[float] = Field(-0.05, description="-1-sigma surface gravity uncertainty", examples=[-0.05])
    koi_srad: Optional[float] = Field(0.99, description="Stellar Radius (Solar radii)", examples=[0.99])
    koi_srad_err1: Optional[float] = Field(0.06, description="+1-sigma stellar radius uncertainty", examples=[0.06])
    koi_srad_err2: Optional[float] = Field(-0.06, description="-1-sigma stellar radius uncertainty", examples=[-0.06])
    
    # Coordinates & Magnitude
    ra: Optional[float] = Field(292.68, description="Right Ascension (degrees)", examples=[292.68])
    dec: Optional[float] = Field(43.83, description="Declination (degrees)", examples=[43.83])
    koi_kepmag: Optional[float] = Field(14.53, description="Kepler Band Magnitude", examples=[14.53])

class FeatureImportanceItem(BaseModel):
    feature: str
    name: str
    unit: str
    value: Optional[float] = None
    importance_score: float
    tooltip: str

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Target class: CONFIRMED, FALSE POSITIVE, or CANDIDATE")
    display_name: str = Field(..., description="User-friendly classification title")
    badge: str = Field(..., description="Visual indicator emoji badge")
    class_index: int = Field(..., description="Encoded integer class index (0, 1, or 2)")
    probability: float = Field(..., description="Probability of the winning class (0.0 to 1.0)")
    confidence_level: str = Field(..., description="High, Moderate, or Marginal confidence")
    probabilities: Dict[str, float] = Field(..., description="Probability breakdown across all 3 classes")
    description: str = Field(..., description="Astronomical explanation of the prediction result")
    model_name: str = Field(..., description="Name of the model utilized")
    top_features: List[FeatureImportanceItem] = Field(default_factory=list, description="Top influential features")

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = True

class SampleCandidate(BaseModel):
    id: str
    name: str
    true_disposition: str
    category: str
    description: str
    features: Dict[str, Optional[float]]

class DatasetStatsResponse(BaseModel):
    total_records: int
    total_features: int
    class_distribution: Dict[str, int]
    features_summary: Dict[str, Dict[str, float]]
