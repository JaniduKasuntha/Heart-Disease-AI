import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load saved components
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Define all original feature names (for user input and scaling)
feature_names = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
    'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP',
    'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle',
    'Family_History', 'Chronic_Stress', 'Gender', 'Age'
]

# Define PCA feature names (for prediction after transformation)
pca_feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]  # e.g., ['PC1', 'PC2', ..., 'PC17']

# Streamlit app
st.title("Heart Disease Risk Predictor")
st.write("Enter patient details below for a risk assessment. Use 'Yes' (1) or 'No' (0) for binary features and actual values for Age.")

# Collect user inputs for all features with Yes/No options
input_data = {}
for feature in feature_names:
    if feature == 'Age':  # Special handling for Age (continuous)
        input_data[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0, value=30.0)
    else:  # Binary features with Yes/No selection
        input_data[feature] = st.selectbox(f"{feature}:", options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]

# Predict button
if st.button("Predict Risk"):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure column order matches training data
    input_df = input_df[feature_names]
    
    # Preprocess: scale and PCA transform the input
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    
    # Convert PCA-transformed data to DataFrame with PCA feature names (CatBoost requires this)
    input_pca_df = pd.DataFrame(input_pca, columns=pca_feature_names)
    
    # Predict
    try:
        prediction = model.predict(input_pca_df)[0]
        probability = model.predict_proba(input_pca_df)[0][1] * 100  # Percentage for positive class
        # Output
        risk_label = "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"
        st.success(f"Predicted Risk: {risk_label}")
        st.info(f"Risk Probability: {probability:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
    
    # Optional: Add explanations or warnings
    st.warning("This is a model prediction—consult a doctor for real advice. Potential biases in training data may affect results.")