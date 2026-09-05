"""
Feature definitions, metadata, units, tooltips, groupings, and defaults
for the Kepler Exoplanet classification pipeline.
"""

from typing import Dict, Any, List

FEATURE_ORDER: List[str] = [
    "koi_period", "koi_period_err1", "koi_period_err2",
    "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2",
    "koi_impact", "koi_impact_err1", "koi_impact_err2",
    "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_prad", "koi_prad_err1", "koi_prad_err2",
    "koi_teq",
    "koi_insol", "koi_insol_err1", "koi_insol_err2",
    "koi_model_snr", "koi_tce_plnt_num",
    "koi_steff", "koi_steff_err1", "koi_steff_err2",
    "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad", "koi_srad_err1", "koi_srad_err2",
    "ra", "dec", "koi_kepmag"
]

CLASS_MAPPING: Dict[int, str] = {
    0: "CONFIRMED",
    1: "FALSE POSITIVE",
    2: "CANDIDATE"
}

CLASS_DISPLAY: Dict[str, str] = {
    "CONFIRMED": "Confirmed Exoplanet",
    "FALSE POSITIVE": "False Positive (Non-Planet)",
    "CANDIDATE": "Exoplanet Candidate"
}

CLASS_BADGE: Dict[str, str] = {
    "CONFIRMED": "🟢",
    "FALSE POSITIVE": "🔴",
    "CANDIDATE": "🟡"
}

CLASS_DESCRIPTION: Dict[str, str] = {
    "CONFIRMED": "The transit signatures strongly match the physical properties of a genuine exoplanet orbiting its host star.",
    "FALSE POSITIVE": "The observed signal is consistent with non-planetary phenomena, such as background eclipsing binaries or stellar variability.",
    "CANDIDATE": "The signal exhibits planetary transit characteristics but requires further observational follow-up or radial-velocity validation."
}

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    # --- TRANSIT PARAMETERS ---
    "koi_period": {
        "name": "Orbital Period",
        "unit": "days",
        "category": "transit",
        "tooltip": "The time taken by the candidate exoplanet to complete one full orbit around its host star.",
        "default": 9.488,
        "min": 0.2,
        "max": 1000.0,
        "step": 0.1
    },
    "koi_time0bk": {
        "name": "Transit Center Epoch",
        "unit": "BKJD (days)",
        "category": "transit",
        "tooltip": "The time of transit center in Barycentric Kepler Julian Day (BJD - 2454833.0).",
        "default": 170.53,
        "min": 0.0,
        "max": 1000.0,
        "step": 0.5
    },
    "koi_impact": {
        "name": "Impact Parameter",
        "unit": "ratio",
        "category": "transit",
        "tooltip": "The sky-projected distance between the center of the stellar disk and the center of the planet disk at transit midpoint (0 = central transit, 1 = grazing transit).",
        "default": 0.537,
        "min": 0.0,
        "max": 2.0,
        "step": 0.05
    },
    "koi_duration": {
        "name": "Transit Duration",
        "unit": "hours",
        "category": "transit",
        "tooltip": "The duration of the observed transit from first contact to fourth contact as the planet crosses the star.",
        "default": 3.79,
        "min": 0.1,
        "max": 100.0,
        "step": 0.1
    },
    "koi_depth": {
        "name": "Transit Depth",
        "unit": "ppm",
        "category": "transit",
        "tooltip": "The fraction of stellar light blocked during transit in parts per million (ppm). 10,000 ppm = 1% drop in stellar brightness.",
        "default": 424.0,
        "min": 1.0,
        "max": 200000.0,
        "step": 10.0
    },
    "koi_prad": {
        "name": "Planetary Radius",
        "unit": "R⊕ (Earth radii)",
        "category": "transit",
        "tooltip": "Estimated radius of the candidate planet in Earth radii. Earth = 1.0 R⊕, Jupiter ≈ 11.2 R⊕.",
        "default": 2.39,
        "min": 0.1,
        "max": 100.0,
        "step": 0.1
    },
    "koi_model_snr": {
        "name": "Transit Signal-to-Noise (SNR)",
        "unit": "ratio",
        "category": "transit",
        "tooltip": "Transit depth divided by the root-mean-square noise of the Kepler light curve. Higher values indicate clearer detection.",
        "default": 23.3,
        "min": 0.0,
        "max": 5000.0,
        "step": 1.0
    },

    # --- ENVIRONMENT & ENERGY ---
    "koi_teq": {
        "name": "Equilibrium Temperature",
        "unit": "Kelvin (K)",
        "category": "environment",
        "tooltip": "Theoretical surface temperature of the planet assuming thermal equilibrium without greenhouse atmosphere. (Earth ≈ 255 K, Liquid water zone ≈ 200 - 320 K).",
        "default": 898.0,
        "min": 50.0,
        "max": 5000.0,
        "step": 10.0
    },
    "koi_insol": {
        "name": "Insolation Flux",
        "unit": "F⊕ (Earth units)",
        "category": "environment",
        "tooltip": "Amount of stellar energy received per unit area compared to Earth (Earth = 1.0 F⊕).",
        "default": 149.0,
        "min": 0.0,
        "max": 100000.0,
        "step": 5.0
    },

    # --- STELLAR PROPERTIES ---
    "koi_steff": {
        "name": "Stellar Effective Temperature",
        "unit": "Kelvin (K)",
        "category": "stellar",
        "tooltip": "Surface temperature of the host star. Sun = 5,778 K. Red dwarfs are ~3,000-4,000 K; hot blue stars are >7,000 K.",
        "default": 5757.0,
        "min": 2500.0,
        "max": 15000.0,
        "step": 50.0
    },
    "koi_slogg": {
        "name": "Stellar Surface Gravity (log g)",
        "unit": "log10(cm/s²)",
        "category": "stellar",
        "tooltip": "Base-10 logarithm of gravitational acceleration at the stellar surface. Sun = 4.44. Lower values (<4.0) suggest evolved giant stars.",
        "default": 4.44,
        "min": 0.0,
        "max": 6.0,
        "step": 0.05
    },
    "koi_srad": {
        "name": "Stellar Radius",
        "unit": "R☉ (Solar radii)",
        "category": "stellar",
        "tooltip": "Radius of the host star measured in Solar radii (Sun = 1.0 R☉).",
        "default": 0.99,
        "min": 0.1,
        "max": 50.0,
        "step": 0.05
    },
    "koi_kepmag": {
        "name": "Kepler Band Magnitude",
        "unit": "mag",
        "category": "stellar",
        "tooltip": "Apparent brightness of the star in the Kepler optical filter. Lower numbers mean brighter stars in the telescope field.",
        "default": 14.53,
        "min": 5.0,
        "max": 21.0,
        "step": 0.1
    },

    # --- COORDINATES & SYSTEM ---
    "ra": {
        "name": "Right Ascension (RA)",
        "unit": "degrees",
        "category": "coordinates",
        "tooltip": "Celestial longitude coordinate in the J2000 equinox coordinate system.",
        "default": 292.68,
        "min": 0.0,
        "max": 360.0,
        "step": 0.1
    },
    "dec": {
        "name": "Declination (Dec)",
        "unit": "degrees",
        "category": "coordinates",
        "tooltip": "Celestial latitude coordinate north (+) or south (-) of the celestial equator.",
        "default": 43.83,
        "min": -90.0,
        "max": 90.0,
        "step": 0.1
    },
    "koi_tce_plnt_num": {
        "name": "TCE Planet Number",
        "unit": "count",
        "category": "coordinates",
        "tooltip": "Sequence number of the candidate signal detected around this specific host star (1 for first planet, 2 for second, etc.).",
        "default": 1.0,
        "min": 1.0,
        "max": 10.0,
        "step": 1.0
    },

    # --- UNCERTAINTIES (ERROR COLUMNS) ---
    "koi_period_err1": {"name": "Period Upper Uncertainty", "unit": "days", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in orbital period.", "default": 0.00005, "min": 0.0, "max": 10.0, "step": 0.00001},
    "koi_period_err2": {"name": "Period Lower Uncertainty", "unit": "days", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in orbital period.", "default": -0.00005, "min": -10.0, "max": 0.0, "step": 0.00001},
    "koi_time0bk_err1": {"name": "Epoch Upper Uncertainty", "unit": "days", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in transit epoch.", "default": 0.003, "min": 0.0, "max": 5.0, "step": 0.001},
    "koi_time0bk_err2": {"name": "Epoch Lower Uncertainty", "unit": "days", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in transit epoch.", "default": -0.003, "min": -5.0, "max": 0.0, "step": 0.001},
    "koi_impact_err1": {"name": "Impact Upper Uncertainty", "unit": "ratio", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in impact parameter.", "default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01},
    "koi_impact_err2": {"name": "Impact Lower Uncertainty", "unit": "ratio", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in impact parameter.", "default": -0.05, "min": -5.0, "max": 0.0, "step": 0.01},
    "koi_duration_err1": {"name": "Duration Upper Uncertainty", "unit": "hours", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in transit duration.", "default": 0.08, "min": 0.0, "max": 10.0, "step": 0.01},
    "koi_duration_err2": {"name": "Duration Lower Uncertainty", "unit": "hours", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in transit duration.", "default": -0.08, "min": -10.0, "max": 0.0, "step": 0.01},
    "koi_depth_err1": {"name": "Depth Upper Uncertainty", "unit": "ppm", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in transit depth.", "default": 10.0, "min": 0.0, "max": 1000.0, "step": 1.0},
    "koi_depth_err2": {"name": "Depth Lower Uncertainty", "unit": "ppm", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in transit depth.", "default": -10.0, "min": -1000.0, "max": 0.0, "step": 1.0},
    "koi_prad_err1": {"name": "Planet Radius Upper Uncertainty", "unit": "R⊕", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in planetary radius.", "default": 0.15, "min": 0.0, "max": 20.0, "step": 0.05},
    "koi_prad_err2": {"name": "Planet Radius Lower Uncertainty", "unit": "R⊕", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in planetary radius.", "default": -0.15, "min": -20.0, "max": 0.0, "step": 0.05},
    "koi_insol_err1": {"name": "Insolation Upper Uncertainty", "unit": "F⊕", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in insolation flux.", "default": 15.0, "min": 0.0, "max": 5000.0, "step": 1.0},
    "koi_insol_err2": {"name": "Insolation Lower Uncertainty", "unit": "F⊕", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in insolation flux.", "default": -15.0, "min": -5000.0, "max": 0.0, "step": 1.0},
    "koi_steff_err1": {"name": "Stellar Temp Upper Uncertainty", "unit": "K", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in stellar temperature.", "default": 100.0, "min": 0.0, "max": 2000.0, "step": 10.0},
    "koi_steff_err2": {"name": "Stellar Temp Lower Uncertainty", "unit": "K", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in stellar temperature.", "default": -100.0, "min": -2000.0, "max": 0.0, "step": 10.0},
    "koi_slogg_err1": {"name": "Surface Gravity Upper Uncertainty", "unit": "dex", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in log(g).", "default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01},
    "koi_slogg_err2": {"name": "Surface Gravity Lower Uncertainty", "unit": "dex", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in log(g).", "default": -0.05, "min": -2.0, "max": 0.0, "step": 0.01},
    "koi_srad_err1": {"name": "Stellar Radius Upper Uncertainty", "unit": "R☉", "category": "uncertainty", "tooltip": "+1-sigma uncertainty in stellar radius.", "default": 0.06, "min": 0.0, "max": 10.0, "step": 0.01},
    "koi_srad_err2": {"name": "Stellar Radius Lower Uncertainty", "unit": "R☉", "category": "uncertainty", "tooltip": "-1-sigma uncertainty in stellar radius.", "default": -0.06, "min": -10.0, "max": 0.0, "step": 0.01}
}

FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {
    "transit": {
        "title": "🌑 Transit Characteristics",
        "description": "Photometric properties of the candidate passing in front of its star.",
        "features": ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_impact", "koi_time0bk", "koi_model_snr"]
    },
    "stellar": {
        "title": "☀️ Host Star Properties",
        "description": "Physical parameters of the parent star observed by the Kepler photometer.",
        "features": ["koi_steff", "koi_srad", "koi_slogg", "koi_kepmag"]
    },
    "environment": {
        "title": "🌌 Energy & Temperature",
        "description": "Estimated stellar radiation and surface equilibrium temperature.",
        "features": ["koi_teq", "koi_insol"]
    },
    "coordinates": {
        "title": "📍 Celestial Coordinates & System",
        "description": "Sky location on the celestial sphere and multi-planet index.",
        "features": ["ra", "dec", "koi_tce_plnt_num"]
    },
    "uncertainties": {
        "title": "📐 Measurement Uncertainties (±1σ Errors)",
        "description": "Observational uncertainties for orbital, stellar, and physical parameters.",
        "features": [
            "koi_period_err1", "koi_period_err2",
            "koi_time0bk_err1", "koi_time0bk_err2",
            "koi_impact_err1", "koi_impact_err2",
            "koi_duration_err1", "koi_duration_err2",
            "koi_depth_err1", "koi_depth_err2",
            "koi_prad_err1", "koi_prad_err2",
            "koi_insol_err1", "koi_insol_err2",
            "koi_steff_err1", "koi_steff_err2",
            "koi_slogg_err1", "koi_slogg_err2",
            "koi_srad_err1", "koi_srad_err2"
        ]
    }
}

MODEL_METRICS_COMPARISON = [
    {
        "rank": "⭐ Best",
        "model": "XGBoost",
        "test_accuracy": 0.7865,
        "cv_accuracy": 0.7854,
        "macro_f1": 0.74,
        "precision_confirmed": 0.76,
        "recall_confirmed": 0.86,
        "f1_confirmed": 0.81,
        "precision_fp": 0.85,
        "recall_fp": 0.88,
        "f1_fp": 0.86,
        "precision_candidate": 0.64,
        "recall_candidate": 0.49,
        "f1_candidate": 0.55,
        "status": "Deployed Model"
    },
    {
        "rank": "Runner-up",
        "model": "Random Forest",
        "test_accuracy": 0.7802,
        "cv_accuracy": 0.7720,
        "macro_f1": 0.73,
        "precision_confirmed": 0.75,
        "recall_confirmed": 0.84,
        "f1_confirmed": 0.79,
        "precision_fp": 0.83,
        "recall_fp": 0.90,
        "f1_fp": 0.86,
        "precision_candidate": 0.65,
        "recall_candidate": 0.45,
        "f1_candidate": 0.53,
        "status": "Evaluated"
    },
    {
        "rank": "3",
        "model": "Decision Tree",
        "test_accuracy": 0.6798,
        "cv_accuracy": 0.7037,
        "macro_f1": 0.66,
        "precision_confirmed": 0.69,
        "recall_confirmed": 0.80,
        "f1_confirmed": 0.74,
        "precision_fp": 0.87,
        "recall_fp": 0.67,
        "f1_fp": 0.76,
        "precision_candidate": 0.42,
        "recall_candidate": 0.58,
        "f1_candidate": 0.49,
        "status": "Evaluated"
    },
    {
        "rank": "4",
        "model": "Logistic Regression",
        "test_accuracy": 0.6604,
        "cv_accuracy": 0.6800,
        "macro_f1": 0.64,
        "precision_confirmed": 0.63,
        "recall_confirmed": 0.86,
        "f1_confirmed": 0.73,
        "precision_fp": 0.89,
        "recall_fp": 0.63,
        "f1_fp": 0.74,
        "precision_candidate": 0.40,
        "recall_candidate": 0.52,
        "f1_candidate": 0.45,
        "status": "Evaluated"
    }
]

METRIC_DEFINITIONS = {
    "Accuracy": {
        "title": "Classification Accuracy",
        "desc": "The percentage of all predictions (across Confirmed, False Positive, and Candidate) that the model predicted correctly."
    },
    "Precision": {
        "title": "Precision",
        "desc": "Out of all candidates predicted as a specific class, what fraction actually belonged to that class? High precision means few false alarms."
    },
    "Recall": {
        "title": "Recall (Sensitivity)",
        "desc": "Out of all true exoplanets in the test set, what fraction did the model successfully identify? High recall means few missed discoveries."
    },
    "F1 Score": {
        "title": "F1 Score",
        "desc": "The harmonic mean of Precision and Recall. It balances both false alarms and missed detections into a single score."
    },
    "Cross-Validation": {
        "title": "Group 5-Fold Cross-Validation",
        "desc": "Validation across 5 grouped folds keeping all planets around the same star (kepid) in the same fold to prevent stellar data leakage."
    }
}
