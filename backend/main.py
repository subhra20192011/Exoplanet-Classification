from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routes import health, model_info, prediction, dataset

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    ## 🔭 Kepler Exoplanet Candidate Classification API
    
    This API serves machine-learning predictions using a pipeline trained on NASA Kepler Mission photometric data.
    
    ### Key Features:
    * **3-Class Classification**: Confirmed Exoplanet, False Positive, or Candidate
    * **Preprocessed Pipeline**: Includes robust median imputation and tuned XGBoost classifier
    * **Explainability**: Returns class probability distributions and key feature importance scores
    * **Real Astronomical Presets**: Curated sample records from `cumulative.csv`
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route handlers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(prediction.router)
app.include_router(dataset.router)

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to the Kepler Exoplanet Classifier API",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "samples": "/samples"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
