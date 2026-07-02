import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# ===============================
# LOAD MODEL AND SCALER
# ===============================
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')  # Load PCA to match training pipeline
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# ===============================
# FEATURE DEFINITIONS
# ===============================
feature_names = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
    'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP',
    'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle',
    'Family_History', 'Chronic_Stress', 'Gender', 'Age'
]

feature_descriptions = {
    'Chest_Pain': "Presence of chest discomfort or pain.",
    'Shortness_of_Breath': "Difficulty breathing.",
    'Fatigue': "Feeling unusually tired or weak.",
    'Palpitations': "Rapid, strong, or irregular heartbeat.",
    'Dizziness': "Feeling lightheaded or faint.",
    'Swelling': "Swelling in legs, ankles, or feet.",
    'Pain_Arms_Jaw_Back': "Pain radiating to arms, jaw, or back.",
    'Cold_Sweats_Nausea': "Cold sweating or nausea episodes.",
    'High_BP': "High blood pressure.",
    'High_Cholesterol': "Elevated cholesterol levels.",
    'Diabetes': "Diabetic or prediabetic condition.",
    'Smoking': "Current or recent smoking habit.",
    'Obesity': "BMI ≥ 30 or medically obese.",
    'Sedentary_Lifestyle': "Low physical activity.",
    'Family_History': "Family history of heart disease.",
    'Chronic_Stress': "Persistent stress affecting heart health.",
    'Gender': "Male = 0, Female = 1.",
    'Age': "Age in years."
}

# Define PCA feature names for prediction
pca_feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]  # e.g., ['PC1', 'PC2', ..., 'PC17']

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(
    page_title="🩺 Heart Disease Dashboard",
    layout="wide",
    page_icon="🩺"
)

# ===============================
# STYLES FOR MEDICAL DASHBOARD
# ===============================
st.markdown("""
    <style>
    /* Card style */
    .card {
        background-color: #f9f9f9;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Risk bar container */
    .bar-container {
        background-color: #e6f2ff;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 10px;
    }
    .glow-bar {
        height: 35px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 35px;
        animation: glow 1.5s infinite;
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px currentColor; }
        50% { box-shadow: 0 0 20px currentColor; }
        100% { box-shadow: 0 0 5px currentColor; }
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# HEADER (Plain)
# ===============================
st.title("🩺 Heart Disease Risk Predictor")
st.write("Adjust Age and see the risk update dynamically.")
st.markdown("---")

# ===============================
# TWO-COLUMN LAYOUT
# ===============================
left_col, right_col = st.columns([1.2, 1])

# -------------------------------
# LEFT PANEL - INPUTS
# -------------------------------
with left_col:
    st.subheader("🩺 Patient Inputs")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_data = {}

    for feature in feature_names:
        desc = feature_descriptions.get(feature, "")
        if feature == 'Gender':
            selected = st.selectbox(
                f"🧍 {feature}",
                options=["Male", "Female"],
                index=0,
                help=desc,
                key=feature
            )
            input_data[feature] = 0 if selected == "Male" else 1
        elif feature == 'Age':
            input_data[feature] = st.slider(
                f"🎂 {feature}",
                min_value=0.0, max_value=120.0, value=30.0, step=1.0,
                help=desc,
                key=feature
            )
        else:
            selected = st.selectbox(
                f"⚙️ {feature}",
                options=["No", "Yes"],
                index=0,
                help=desc,
                key=feature
            )
            input_data[feature] = 0 if selected == "No" else 1
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT PANEL - PREDICTION
# -------------------------------
with right_col:
    st.subheader("🔍 Prediction Results")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names).astype(float)
    
    # Preprocess: scale and PCA transform the input
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    input_pca_df = pd.DataFrame(input_pca, columns=pca_feature_names)

    # Predict
    prediction = model.predict(input_pca_df)[0]
    probability = model.predict_proba(input_pca_df)[0][1] * 100

    # Determine risk level & color
    if probability < 30:
        risk_level = "Low"
        bar_color = "#28a745"  # Green
        emoji = "✅"
    elif probability < 70:
        risk_level = "Moderate"
        bar_color = "#ffc107"  # Orange
        emoji = "⚠️"
    else:
        risk_level = "High"
        bar_color = "#dc3545"  # Red
        emoji = "🚨"

    # Risk bar
    st.markdown(f"### Risk Level: {emoji} **{risk_level}**")
    st.markdown(f"""
        <div class="bar-container">
            <div class="glow-bar" style="width:{probability:.1f}%; background-color:{bar_color}; color:white;">
                {probability:.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Risk Probability", f"{probability:.2f}%")
    col2.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")

    # Input summary
    with st.expander("📝 Patient Input Summary"):
        st.dataframe(input_df.style.set_properties(**{'font-size': '16px', 'color':'#004080'}))

    # Downloadable report
    csv = input_df.copy()
    csv["Prediction"] = "High Risk" if prediction == 1 else "Low Risk"
    csv["Probability (%)"] = probability
    st.download_button(
        "💾 Download Full Report (CSV)",
        csv.to_csv(index=False),
        "heart_disease_medical_dashboard_report.csv",
        "text/csv",
        use_container_width=True
    )

    st.warning("⚠️ AI prediction. Always consult a healthcare professional for accurate diagnosis.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#004080;'>Developed with 🩺 using Streamlit & CatBoost | Educational / Presentation use</p>", unsafe_allow_html=True)