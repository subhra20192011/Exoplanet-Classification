# 🔭 Kepler Exoplanet Candidate Classification Web Application

A full-stack, machine-learning-powered web application for detecting and classifying exoplanetary transit candidates from NASA's Kepler Space Telescope. Built with a **Streamlit** frontend, **FastAPI** backend, and a production-grade **XGBoost Pipeline**.

---

## 🌟 Overview

The **Kepler Space Telescope** observed over 150,000 stars to detect exoplanetary transits (periodic dips in stellar brightness). However, astrophysical false positives (such as eclipsing binary star systems and background star blends) frequently mimic planetary signals.

This application provides an interactive, beginner-friendly interface backed by a high-performance machine learning inference engine that classifies candidate observations into three categories:
1. 🟢 **Confirmed Exoplanet**
2. 🟡 **Exoplanet Candidate**
3. 🔴 **False Positive (Non-Planet)**

---

## 🏗️ Architecture & Technology Stack

```
                                 HTTP REST API
  ┌───────────────────────┐   (JSON /predict)   ┌────────────────────────┐
  │   Streamlit Frontend  │ ──────────────────> │    FastAPI Backend     │
  │  (Interactive UI/UX)  │ <────────────────── │   (REST / OpenAPI)     │
  └───────────────────────┘    Probabilities    └───────────┬────────────┘
                                                            │
                                                            ▼
                                                ┌────────────────────────┐
                                                │   ML Pipeline (v3)     │
                                                │ 1. SimpleImputer       │
                                                │ 2. XGBoost Classifier  │
                                                └────────────────────────┘
```

* **Frontend**: Streamlit with custom CSS styling, interactive charts, metric explainers, and session history.
* **Backend**: FastAPI, Uvicorn, Pydantic with automated OpenAPI Swagger docs (`/docs`).
* **ML Model**: Scikit-Learn Pipeline (`kepler_pipeline_v3.pkl`) containing a `SimpleImputer` (median strategy) and `XGBClassifier` (200 estimators, max depth 5, learning rate 0.05).
* **Dataset**: NASA Kepler Cumulative KOI Archive (`cumulative.csv`, 9,564 records).

---

## 🏆 Model Benchmark Comparison

Trained and cross-validated using **Group 5-Fold Cross-Validation** (grouped by star ID `kepid` to prevent stellar data leakage across train/test splits):

| Rank | Model | Test Accuracy | 5-Fold CV Accuracy | Macro F1 | Deployment Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ⭐ **Best** | **XGBoost Classifier** | **78.65%** | **78.54%** | **0.74** | **Deployed Pipeline** |
| Runner-up | Random Forest | 78.02% | 77.20% | 0.73 | Evaluated in Notebook |
| 3 | Decision Tree | 67.98% | 70.37% | 0.66 | Evaluated in Notebook |
| 4 | Logistic Regression | 66.04% | 68.00% | 0.64 | Evaluated in Notebook |

---

## 📁 Project Structure

```
exoplanet classification/
│
├── frontend/
│   ├── streamlit_app.py       # Main Streamlit application
│   ├── api_client.py          # FastAPI HTTP client with graceful fallback
│   ├── components.py          # Reusable UI widgets, cards, & charts
│   └── style.css              # Custom styling for dark & light mode
│
├── backend/
│   ├── main.py                # FastAPI entry point & CORS configuration
│   ├── config.py              # Environment settings & path configuration
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── routes/
│   │   ├── health.py          # GET /health
│   │   ├── model_info.py      # GET /model-info
│   │   ├── prediction.py      # POST /predict
│   │   └── dataset.py         # GET /samples & GET /dataset-stats
│   └── services/
│       ├── model_service.py   # ML pipeline wrapper & inference logic
│       └── dataset_service.py # Dataset querying & statistics
│
├── ml/
│   ├── inference.py           # Core ML engine with feature alignment
│   └── feature_definitions.py # 36 Kepler features metadata, tooltips, & defaults
│
├── models/
│   └── kepler_pipeline_v3.pkl # Pre-trained deployable XGBoost pipeline
│
├── data/
│   ├── cumulative.csv         # Full NASA Kepler dataset
│   └── sample_candidates.json # Curated 1-click real candidate presets
│
├── notebooks/
│   └── Exoplanent_Detection_final.ipynb # Original research notebook
│
├── tests/
│   ├── test_model.py          # Model unit tests
│   ├── test_backend_api.py    # FastAPI endpoint test suite
│   └── test_end_to_end.py     # End-to-end integration tests
│
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── .env                       # Local environment configuration
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- `pip`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Backend
In your first terminal window:
```bash
uvicorn backend.main:app --reload --port 8000
```
* **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative API Documentation (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 4. Run Streamlit Frontend
In a second terminal window:
```bash
streamlit run frontend/streamlit_app.py
```
* **Web UI**: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Automated Tests

Run the complete test suite:
```bash
pytest tests/ -v
```

---

## 💡 Key Features for Users

1. **🧪 1-Click Sample Candidates**: Instantly load real Kepler objects (e.g., **Kepler-227 b**, **Kepler-20 b**, Eclipsing Binary false alarms).
2. **🔍 Explainable Predictions**: View top influential astrophysical features for each classification.
3. **📊 Probability Breakdown**: Transparent probability distribution across Confirmed, Candidate, and False Positive classes.
4. **📜 Session History & CSV Export**: Track predictions during your session and export full records with timestamps.
5. **📖 Educational Guide**: Learn the Kepler mission basics, transit photometry physics, and machine learning principles.

---

## ⚠️ Scientific Disclaimer

The predictions generated by this platform are statistical outputs derived from machine-learning models trained on Kepler photometric data. They do not constitute scientific confirmation of an exoplanet without independent follow-up observations (e.g., radial velocity spectroscopy or high-resolution imaging).
