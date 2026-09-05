import os

class Settings:
    PROJECT_NAME: str = "Kepler Exoplanet Classifier API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Host & Port
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Base paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "kepler_pipeline_v3.pkl")
    DATASET_PATH: str = os.path.join(BASE_DIR, "data", "cumulative.csv")
    SAMPLES_PATH: str = os.path.join(BASE_DIR, "data", "sample_candidates.json")
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]

settings = Settings()
