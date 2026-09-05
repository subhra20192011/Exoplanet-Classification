import os
import sys
import json
import time
import datetime
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.api_client import api_client
from frontend.components import (
    render_hero_banner,
    render_api_status_badge,
    render_prediction_result,
    render_feature_importance_breakdown,
    render_model_comparison_table
)
from ml.feature_definitions import (
    FEATURE_ORDER,
    FEATURE_METADATA,
    FEATURE_GROUPS,
    MODEL_METRICS_COMPARISON,
    METRIC_DEFINITIONS,
    CLASS_MAPPING,
    CLASS_DISPLAY
)

# Set page configuration
st.set_page_config(
    page_title="Kepler Exoplanet Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_file = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state["history"] = []

if "current_inputs" not in st.session_state:
    st.session_state["current_inputs"] = {
        feat: FEATURE_METADATA.get(feat, {}).get("default", 0.0)
        for feat in FEATURE_ORDER
    }

if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Home"

# Sidebar Navigation
st.sidebar.title("🔭 Kepler Classifier")
nav_options = [
    "🏠 Home",
    "🔭 Predict",
    "🤖 Model & Metrics",
    "📊 Dataset Explorer",
    "📖 Learn Astronomy",
    "ℹ️ About & FAQ"
]

# Health check
health_status = api_client.check_health()
render_api_status_badge(health_status)

st.sidebar.markdown("---")
selected_nav = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0,
    key="nav_radio"
)
st.session_state["nav_page"] = selected_nav

# Sidebar quick information
with st.sidebar.expander("🌌 Quick Mission Facts", expanded=False):
    st.markdown("""
    * **Telescope**: NASA Kepler Space Telescope
    * **Target Field**: Cygnus & Lyra Constellations
    * **Stars Monitored**: >150,000 Main-Sequence Stars
    * **Confirmed Planets Found**: >2,600 Exoplanets
    * **Model Deployed**: XGBoost Pipeline
    """)

# =========================================================================
# PAGE 1: HOME
# =========================================================================
if selected_nav == "🏠 Home":
    render_hero_banner()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🪐 Discover Alien Worlds with Machine Learning")
        st.write("""
The universe contains billions of stars, and many of them are believed to host planets orbiting around them. Discovering these distant worlds, known as **exoplanets**, is one of the greatest achievements of modern astronomy. Every confirmed exoplanet helps scientists understand how planetary systems form, evolve, and whether conditions suitable for life may exist beyond our Solar System.

One of NASA's most successful space missions, the **Kepler Space Telescope**, was designed specifically to search for these hidden worlds. Kepler continuously observed the brightness of more than **150,000 stars** over several years, collecting millions of measurements and identifying thousands of potential planetary candidates from subtle changes in starlight.

The primary method used by Kepler is called the **planetary transit method**. When a planet passes directly between its host star and the telescope, it blocks a tiny fraction of the star's light, creating a small and periodic dip in brightness. By analyzing the depth, duration, and frequency of these transits, astronomers can estimate important properties such as a planet's size, orbital period, and distance from its star.

However, detecting real exoplanets is far from straightforward. Many transit-like signals are actually caused by **stellar activity, eclipsing binary stars, background celestial objects, or instrumental noise**. These false positives often resemble genuine planetary transits, making manual analysis both time-consuming and scientifically challenging.

This application uses an advanced **Machine Learning pipeline powered by XGBoost** to distinguish confirmed exoplanets from false positive signals. The model is trained on NASA's Kepler candidate dataset, automatically preprocesses astronomical features, and performs intelligent classification with an overall **78.65% accuracy** on unseen data.

Beyond making predictions, this platform allows users to explore astronomical features, understand how AI interprets Kepler observations, and experience how data science is transforming space exploration. It demonstrates the powerful combination of **Astronomy, Machine Learning, and Artificial Intelligence** in the search for new worlds across our galaxy.
""")
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔭 Start Prediction", type="primary", use_container_width=True):
                st.session_state["nav_page"] = "🔭 Predict"
                st.rerun()
        with c_btn2:
            if st.button("📖 Learn How It Works", use_container_width=True):
                st.session_state["nav_page"] = "📖 Learn Astronomy"
                st.rerun()

    with col2:
        st.markdown("### 🔬 How Does The Pipeline Work?")
        st.markdown("""
        ```
        🔭 Raw Transit Signal (Light Curve)
                       ↓
        📊 Extract 36 Astrophysical Features
                       ↓
        ⚙️ SimpleImputer (Median Imputation)
                       ↓
        🤖 Trained XGBoost Multi-Class Classifier
                       ↓
        🎯 Probability Breakdown & Classification
           [Confirmed • Candidate • False Positive]
        ```
        """)
        st.info("💡 **Ready to test?** Head over to the **Predict** page and try 1-click real candidate presets like **Kepler-227 b** or test your own custom measurements!", icon="🚀")

# =========================================================================
# PAGE 2: PREDICTION WORKFLOW
# =========================================================================
elif selected_nav == "🔭 Predict":
    st.markdown("## 🔭 Exoplanet Candidate Classification")
    st.write("Enter transit and stellar measurements below, or load a verified sample candidate to test the model.")

    # 1-Click Sample Candidate Loader
    samples = api_client.get_samples()
    sample_options = ["Custom Values / Manual Entry"] + [s["name"] for s in samples]

    col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
    with col_s1:
        selected_sample_name = st.selectbox(
            "🧪 Try a Sample Candidate (Preloaded from NASA Kepler Data)",
            sample_options,
            help="Select an actual candidate record from the Kepler mission dataset to instantly populate the form."
        )
    
    with col_s2:
        st.write("")
        st.write("")
        load_clicked = st.button("📥 Load Sample", use_container_width=True)

    with col_s3:
        st.write("")
        st.write("")
        reset_clicked = st.button("🔄 Reset Form", use_container_width=True)

    if load_clicked and selected_sample_name != "Custom Values / Manual Entry":
        matched_sample = next((s for s in samples if s["name"] == selected_sample_name), None)
        if matched_sample:
            for k, v in matched_sample["features"].items():
                if v is not None:
                    st.session_state["current_inputs"][k] = float(v)
            st.session_state["latest_prediction"] = None
            st.success(f"✅ Loaded {matched_sample['name']}! True Archive Disposition: **{matched_sample['true_disposition']}**")
            st.caption(f"ℹ️ {matched_sample.get('description', '')}")

    if reset_clicked:
        st.session_state["current_inputs"] = {
            feat: FEATURE_METADATA.get(feat, {}).get("default", 0.0)
            for feat in FEATURE_ORDER
        }
        st.session_state["latest_prediction"] = None
        st.info("🔄 Form reset to standard median defaults.")
        st.rerun()

    st.markdown("---")

    # Form Fields in Logical Tabs/Sections
    st.markdown("### 📝 Candidate Observation Parameters")
    
    form_tab1, form_tab2, form_tab3, form_tab4, form_tab5 = st.tabs([
        "🌑 Transit Characteristics",
        "☀️ Host Star Properties",
        "🌌 Energy & Temperature",
        "📍 Celestial Coordinates",
        "📐 Measurement Uncertainties (±1σ)"
    ])

    inputs_to_send = {}

    with form_tab1:
        st.caption("Key photometric properties measured during the planetary transit.")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(FEATURE_GROUPS["transit"]["features"]):
            meta = FEATURE_METADATA[feat]
            curr_val = float(st.session_state["current_inputs"].get(feat, meta["default"]))
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                label = f"{meta['name']} ({meta['unit']})" if meta['unit'] else meta['name']
                val = st.number_input(
                    label=label,
                    value=curr_val,
                    step=float(meta.get("step", 0.1)),
                    help=meta["tooltip"],
                    key=f"input_{feat}"
                )
                inputs_to_send[feat] = val
                st.session_state["current_inputs"][feat] = val

    with form_tab2:
        st.caption("Physical characteristics of the host star.")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(FEATURE_GROUPS["stellar"]["features"]):
            meta = FEATURE_METADATA[feat]
            curr_val = float(st.session_state["current_inputs"].get(feat, meta["default"]))
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                label = f"{meta['name']} ({meta['unit']})" if meta['unit'] else meta['name']
                val = st.number_input(
                    label=label,
                    value=curr_val,
                    step=float(meta.get("step", 0.1)),
                    help=meta["tooltip"],
                    key=f"input_{feat}"
                )
                inputs_to_send[feat] = val
                st.session_state["current_inputs"][feat] = val

    with form_tab3:
        st.caption("Planetary equilibrium temperature and radiation flux.")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(FEATURE_GROUPS["environment"]["features"]):
            meta = FEATURE_METADATA[feat]
            curr_val = float(st.session_state["current_inputs"].get(feat, meta["default"]))
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                label = f"{meta['name']} ({meta['unit']})" if meta['unit'] else meta['name']
                val = st.number_input(
                    label=label,
                    value=curr_val,
                    step=float(meta.get("step", 0.1)),
                    help=meta["tooltip"],
                    key=f"input_{feat}"
                )
                inputs_to_send[feat] = val
                st.session_state["current_inputs"][feat] = val

    with form_tab4:
        st.caption("Sky coordinates and planetary system index.")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(FEATURE_GROUPS["coordinates"]["features"]):
            meta = FEATURE_METADATA[feat]
            curr_val = float(st.session_state["current_inputs"].get(feat, meta["default"]))
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                label = f"{meta['name']} ({meta['unit']})" if meta['unit'] else meta['name']
                val = st.number_input(
                    label=label,
                    value=curr_val,
                    step=float(meta.get("step", 0.1)),
                    help=meta["tooltip"],
                    key=f"input_{feat}"
                )
                inputs_to_send[feat] = val
                st.session_state["current_inputs"][feat] = val

    with form_tab5:
        st.caption("Observational errors and measurement uncertainties. (Advanced / Pre-filled with realistic defaults)")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(FEATURE_GROUPS["uncertainties"]["features"]):
            meta = FEATURE_METADATA[feat]
            curr_val = float(st.session_state["current_inputs"].get(feat, meta["default"]))
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                label = f"{meta['name']} ({meta['unit']})" if meta['unit'] else meta['name']
                val = st.number_input(
                    label=label,
                    value=curr_val,
                    step=float(meta.get("step", 0.001)),
                    help=meta["tooltip"],
                    key=f"input_{feat}"
                )
                inputs_to_send[feat] = val
                st.session_state["current_inputs"][feat] = val

    # Prediction Action Button
    st.markdown("---")
    col_pred_btn, col_blank = st.columns([2, 3])
    with col_pred_btn:
        predict_button = st.button("🔭 Classify Candidate", type="primary", use_container_width=True)

    if predict_button:
        with st.spinner("🔬 Analyzing candidate photometric signature via XGBoost pipeline..."):
            time.sleep(0.3) # Responsive micro-transition
            resp = api_client.predict(inputs_to_send)

            if resp.get("success"):
                pred_data = resp["data"]
                st.session_state["latest_prediction"] = pred_data
                
                # Add to session history
                history_entry = {
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "prediction": pred_data.get("display_name"),
                    "class": pred_data.get("predicted_class"),
                    "probability": f"{pred_data.get('probability', 0.0)*100:.1f}%",
                    "period_days": inputs_to_send.get("koi_period"),
                    "radius_earth": inputs_to_send.get("koi_prad"),
                    "snr": inputs_to_send.get("koi_model_snr")
                }
                st.session_state["history"].insert(0, history_entry)
                st.toast("✅ Analysis complete!", icon="🔭")
            else:
                st.error(f"❌ Prediction failed: {resp.get('error')}")

    # Display Prediction Results
    if st.session_state["latest_prediction"]:
        st.markdown("### 🎯 Classification Result")
        render_prediction_result(st.session_state["latest_prediction"])
        
        # Feature Importance Explanation
        top_feats = st.session_state["latest_prediction"].get("top_features", [])
        render_feature_importance_breakdown(top_feats)

        # Download Result CSV
        res_dict = st.session_state["latest_prediction"]
        download_record = {
            "prediction": res_dict.get("predicted_class"),
            "display_name": res_dict.get("display_name"),
            "winning_probability": res_dict.get("probability"),
            "prob_confirmed": res_dict.get("probabilities", {}).get("CONFIRMED"),
            "prob_false_positive": res_dict.get("probabilities", {}).get("FALSE POSITIVE"),
            "prob_candidate": res_dict.get("probabilities", {}).get("CANDIDATE"),
            "timestamp": datetime.datetime.now().isoformat(),
            **inputs_to_send
        }
        df_download = pd.DataFrame([download_record])
        csv_data = df_download.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="⬇️ Download Classification Result (CSV)",
            data=csv_data,
            file_name=f"kepler_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Session Prediction History
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("### 📜 Current Session Prediction History")
        df_hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()

# =========================================================================
# PAGE 3: MODEL & METRICS COMPARISON
# =========================================================================
elif selected_nav == "🤖 Model & Metrics":
    st.markdown("## 🤖 Machine Learning Model Architecture & Performance")
    st.write("Overview of the trained ML models, group cross-validation benchmarks, and feature importance rankings.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🏆 Notebook Model Comparison Benchmark")
        st.write("All candidate models were trained with **Group 5-Fold Cross-Validation** (grouped by star ID `kepid` to prevent stellar data leakage across train/test splits).")
        render_model_comparison_table(MODEL_METRICS_COMPARISON)

    with col2:
        st.markdown("### ⚙️ Deployed Pipeline Details")
        st.markdown("""
        * **Pipeline Model**: `kepler_pipeline_v3.pkl`
        * **Preprocessing Step**: `SimpleImputer(strategy='median')`
        * **Classifier Step**: `XGBClassifier`
        * **Number of Input Features**: 36
        * **Number of Output Classes**: 3 (`CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`)
        * **Hyperparameters**:
          * `n_estimators`: 200
          * `max_depth`: 5
          * `learning_rate`: 0.05
          * `subsample`: 0.8
          * `colsample_bytree`: 0.8
        """)

    st.markdown("---")
    st.markdown("### 📚 Metric Definitions & Terminology")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        with st.expander("ⓘ What is Classification Accuracy?", expanded=True):
            st.write(METRIC_DEFINITIONS["Accuracy"]["desc"])
        with st.expander("ⓘ What is Precision?", expanded=False):
            st.write(METRIC_DEFINITIONS["Precision"]["desc"])
    with m_col2:
        with st.expander("ⓘ What is Recall (Sensitivity)?", expanded=True):
            st.write(METRIC_DEFINITIONS["Recall"]["desc"])
        with st.expander("ⓘ What is the F1 Score?", expanded=False):
            st.write(METRIC_DEFINITIONS["F1 Score"]["desc"])

    # Feature Importance Chart
    st.markdown("---")
    st.markdown("### 📊 Global XGBoost Feature Importance Ranking")
    st.write("Features that contribute most significantly to decision boundaries across all three classes:")
    
    model_info = api_client.get_model_info()
    feat_imps = model_info.get("feature_importances", [])
    if feat_imps:
        df_imp = pd.DataFrame(feat_imps).head(12)
        df_imp = df_imp.rename(columns={"name": "Feature Name", "importance": "Relative Importance Score"})
        st.bar_chart(data=df_imp.set_index("Feature Name")["Relative Importance Score"])

# =========================================================================
# PAGE 4: DATASET EXPLORER
# =========================================================================
elif selected_nav == "📊 Dataset Explorer":
    st.markdown("## 📊 NASA Kepler Mission Dataset Explorer")
    st.write("Summary statistics and distributions from the original Kepler Cumulative Dataset (`cumulative.csv`).")

    stats = api_client.get_dataset_stats()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{stats.get('total_records', 9564):,}")
    m2.metric("Total Features", stats.get('total_features', 36))
    m3.metric("Confirmed Exoplanets", f"{stats.get('class_distribution', {}).get('CONFIRMED', 2293):,}")
    m4.metric("False Positives", f"{stats.get('class_distribution', {}).get('FALSE POSITIVE', 5023):,}")

    st.markdown("---")
    st.markdown("### 🎯 Class Balance Distribution")
    disp_df = pd.DataFrame(list(stats.get("class_distribution", {}).items()), columns=["Disposition Class", "Count"])
    st.bar_chart(data=disp_df.set_index("Disposition Class")["Count"])

    # Sample preview
    st.markdown("### 🔬 Kepler Dataset Sample Preview")
    csv_path = os.path.join(PROJECT_ROOT, "data", "cumulative.csv")
    if os.path.exists(csv_path):
        df_preview = pd.read_csv(csv_path, nrows=100)
        display_cols = ["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_prad", "koi_teq", "koi_steff"]
        valid_cols = [c for c in display_cols if c in df_preview.columns]
        st.dataframe(df_preview[valid_cols].head(15), use_container_width=True)

# =========================================================================
# PAGE 5: LEARN ASTRONOMY
# =========================================================================
elif selected_nav == "📖 Learn Astronomy":
    st.markdown("## 📖 Exoplanetary Astronomy & Detection Guide")
    st.write("Learn the core scientific principles behind exoplanet discovery and transit photometry.")

    with st.expander("🌌 1. What is an Exoplanet?", expanded=True):
        st.markdown("""
        An **exoplanet** (extrasolar planet) is a planet located outside our Solar System that orbits another star. 
        The first confirmed detection of an exoplanet orbiting a Sun-like star occurred in 1995 (51 Pegasi b). 
        Since then, thousands of exoplanets have been discovered, ranging from rocky terrestrial worlds to gas giants larger than Jupiter.
        """)

    with st.expander("🔭 2. What is the NASA Kepler Mission?", expanded=True):
        st.markdown("""
        Launched in 2009, the **Kepler Space Telescope** was a specialized space observatory equipped with a 0.95-meter photometer. 
        It stared continuously at a fixed field of over 150,000 stars in the constellations Cygnus and Lyra, measuring tiny variations in stellar brightness with unprecedented photometric precision.
        """)

    with st.expander("🌑 3. How Does the Transit Method Work?", expanded=False):
        st.markdown("""
        When an exoplanet's orbit is aligned such that it crosses the line of sight between Earth and its host star, it periodically blocks a tiny fraction of the star's light.
        * **Transit Depth ($\Delta F / F$)**: Proportional to the ratio of the planet's cross-sectional area to the star's area ($R_p^2 / R_*^2$).
        * **Transit Period ($P$)**: The recurring interval between consecutive transits, corresponding to the planet's orbital year.
        * **Transit Duration ($T$)**: The duration of the transit chord as the planet moves across the stellar disk.
        """)

    with st.expander("🤖 4. How Does Machine Learning Help?", expanded=False):
        st.markdown("""
        Kepler produced millions of light curve data points. Automated pipeline algorithms flagged tens of thousands of potential signals (Threshold Crossing Events or TCEs). 
        However, many signals are caused by **astrophysical false positives** (e.g., grazing eclipsing binary stars or background star blends) or instrumental artifacts. 
        Machine learning models rapidly evaluate 36 astrophysical parameters simultaneously, classifying candidates with high consistency.
        """)

    with st.expander("⚠️ 5. What are the Scientific Limitations?", expanded=False):
        st.markdown("""
        A machine-learning prediction is a **statistical classification** based on photometric observations. 
        Final scientific confirmation of an exoplanet typically requires independent follow-up observations, such as:
        1. **High-Precision Radial Velocity**: Measuring the gravitational wobble of the host star to determine planetary mass.
        2. **High-Resolution Adaptive Optics Imaging**: Ruling out nearby background stars that could blend with the target star.
        """)

# =========================================================================
# PAGE 6: ABOUT & FAQ
# =========================================================================
elif selected_nav == "ℹ️ About & FAQ":
    st.markdown("## ℹ️ About & Frequently Asked Questions")

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### ❓ Frequently Asked Questions")
        with st.expander("Does a 'Confirmed Exoplanet' prediction mean a new planet was discovered?", expanded=True):
            st.write("No. The classifier indicates that the input measurements match the physical signatures of confirmed planets in the Kepler archive. Genuine astronomical discovery requires multi-instrument validation and peer-reviewed confirmation.")
        
        with st.expander("What is the difference between Candidate and Confirmed?"):
            st.write("A **Candidate** is a signal with planetary transit characteristics that has not yet been fully validated by radial velocity or imaging. A **Confirmed Exoplanet** has passed all statistical vetting tests and independent follow-ups.")
        
        with st.expander("Why do models use 36 features instead of just Transit Depth?"):
            st.write("Transit depth alone only tells you how much light was blocked. You also need orbital period, transit duration, stellar temperature, stellar radius, impact parameter, and signal-to-noise ratio to rule out eclipsing binary stars.")

        with st.expander("Which model gives the best performance on this dataset?"):
            st.write("According to group cross-validation experiments in the research notebook, **XGBoost achieved the highest accuracy (78.65%)**, closely followed by **Random Forest (78.02%)**, outperforming Decision Trees and Logistic Regression.")

    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        * **Frontend**: Streamlit
        * **Backend**: FastAPI (REST API with OpenAPI / Swagger)
        * **ML Pipeline**: Scikit-Learn & XGBoost
        * **Dataset**: NASA Kepler Cumulative KOI Archive
        * **Communication**: HTTP REST (`POST /predict`, `GET /model-info`, `GET /samples`)
        """)
        
        st.markdown("### 🏛️ Academic Citation & Data Credit")
        st.caption("""
        Data provided by NASA Exoplanet Archive and Kepler Science Operations Center. 
        Developed as an academic machine-learning web application for exoplanet transit classification.
        """)
