import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def render_hero_banner():
    """Renders the top welcoming hero banner."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔭 Kepler Exoplanet Classifier</div>
        <div class="hero-subtitle">
            A machine-learning platform trained on high-precision NASA Kepler Mission space telescope observations to detect and classify transit signals as <strong>Confirmed Exoplanets</strong>, <strong>Planetary Candidates</strong>, or <strong>False Positives</strong>.
        </div>
        <div>
            <span class="badge-tag">NASA Kepler Mission</span>
            <span class="badge-tag">XGBoost Classifier</span>
            <span class="badge-tag">Transit Photometry</span>
            <span class="badge-tag">78.65% Benchmark Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_badge(health: Dict[str, Any]):
    """Renders a small status indicator in the sidebar."""
    if health.get("status") == "healthy":
        st.sidebar.markdown(
            '<div class="status-pill status-online">🟢 FastAPI Backend Online (v1.0.0)</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="status-pill status-offline">🔴 FastAPI Offline (Local Mode Active)</div>',
            unsafe_allow_html=True
        )
        st.sidebar.caption("Run `uvicorn backend.main:app --reload` to start the backend.")

def render_prediction_result(result: Dict[str, Any]):
    """Renders an intuitive, high-contrast result card."""
    pred_class = result.get("predicted_class", "UNKNOWN")
    display_name = result.get("display_name", pred_class)
    badge = result.get("badge", "⚪")
    prob = result.get("probability", 0.0) * 100
    confidence = result.get("confidence_level", "Moderate")
    description = result.get("description", "")
    probs = result.get("probabilities", {})

    card_class = "result-card-confirmed"
    if pred_class == "FALSE POSITIVE":
        card_class = "result-card-false-positive"
    elif pred_class == "CANDIDATE":
        card_class = "result-card-candidate"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="result-header">
            <span>{badge}</span>
            <span>{display_name}</span>
        </div>
        <div class="result-probability">
            {prob:.1f}% Model Probability
        </div>
        <div style="font-size: 0.95rem; opacity: 0.92; margin-bottom: 12px;">
            Confidence: <strong>{confidence}</strong> | Evaluator: <em>{result.get('model_name', 'XGBoost')}</em>
        </div>
        <div style="font-size: 0.92rem; line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("#### 📊 Probability Distribution Across Classes")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        p_conf = probs.get("CONFIRMED", 0.0) * 100
        st.metric(label="🟢 Confirmed Exoplanet", value=f"{p_conf:.1f}%")
        st.progress(float(probs.get("CONFIRMED", 0.0)))
        
    with c2:
        p_cand = probs.get("CANDIDATE", 0.0) * 100
        st.metric(label="🟡 Exoplanet Candidate", value=f"{p_cand:.1f}%")
        st.progress(float(probs.get("CANDIDATE", 0.0)))
        
    with c3:
        p_fp = probs.get("FALSE POSITIVE", 0.0) * 100
        st.metric(label="🔴 False Positive", value=f"{p_fp:.1f}%")
        st.progress(float(probs.get("FALSE POSITIVE", 0.0)))

    st.info("ℹ️ **Scientific Disclaimer**: Model probabilities reflect statistical patterns identified by machine learning on Kepler photometric measurements. They do not constitute immediate scientific confirmation without spectroscopic radial velocity verification.", icon="🔬")

def render_feature_importance_breakdown(top_features: List[Dict[str, Any]]):
    """Renders the top contributing feature cards."""
    if not top_features:
        return

    with st.expander("🔍 Why did the model make this prediction? (Top Influential Features)", expanded=False):
        st.write("The XGBoost model weighs specific astrophysical signatures heavily when scoring transit signals:")
        
        for item in top_features:
            feat_name = item.get("name", item.get("feature"))
            unit = f" ({item.get('unit')})" if item.get("unit") else ""
            val = item.get("value")
            score = item.get("importance_score", 0.0) * 100
            tooltip = item.get("tooltip", "")

            st.markdown(f"""
            <div class="feature-item-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>{feat_name}{unit}</strong>
                    <span style="color: #6e8efb; font-weight: 600; font-size: 0.88rem;">Relative Weight: {score:.1f}%</span>
                </div>
                <div style="font-size: 0.88rem; color: #a0aec0; margin-bottom: 4px;">
                    Current Candidate Value: <strong>{val if val is not None else 'N/A'}</strong>
                </div>
                <div style="font-size: 0.82rem; color: #718096;">
                    ℹ️ {tooltip}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_model_comparison_table(models_data: List[Dict[str, Any]]):
    """Renders the model evaluation comparison table from the notebook."""
    if not models_data:
        return

    df_comp = pd.DataFrame(models_data)
    df_display = pd.DataFrame({
        "Rank": df_comp["rank"],
        "Model": df_comp["model"],
        "Test Accuracy": df_comp["test_accuracy"].apply(lambda x: f"{x*100:.2f}%"),
        "5-Fold CV Accuracy": df_comp["cv_accuracy"].apply(lambda x: f"{x*100:.2f}%"),
        "Macro F1 Score": df_comp["macro_f1"].apply(lambda x: f"{x*100:.1f}%"),
        "Deployment Status": df_comp["status"]
    })
    st.table(df_display)
