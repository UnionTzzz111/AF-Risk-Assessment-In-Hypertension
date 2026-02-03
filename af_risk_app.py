import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import traceback # For detailed error logging
import warnings

# --- Global Settings ---
warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature definitions (Must be consistent with model training)
continuous_features = ['LVEDd', 'RVEDd', 'IVS', 'PAD', 'LAD', 'RAD', 'Age', 'AOD', 'LVPW']
categorical_features = ['Gender'] # Assuming 'Gender' is the only categorical feature for simplicity
target = 'AF'
all_features_for_models = continuous_features + categorical_features # Order matters for prediction

# Display names for features (optional, for better UI)
feature_name_display_map = {
    'LVEDd': 'LVEDd (mm) - Left Ventricular End-Diastolic Diameter',
    'RVEDd': 'RVEDd (mm) - Right Ventricular End-Diastolic Diameter',
    'IVS': 'IVS (mm) - Interventricular Septum Thickness',
    'PAD': 'PAD (mm) - Pulmonary Artery Diameter',
    'LAD': 'LAD (mm) - Left Atrial Diameter',
    'RAD': 'RAD (mm) - Right Atrial Diameter',
    'Age': 'Age (Years)',
    'AOD': 'AOD (Years) - Age of Disease Onset',
    'LVPW': 'LVPW (mm) - Left Ventricular Posterior Wall Thickness',
    'Gender': 'Gender',
    'AF': 'Atrial Fibrillation (AF)'
}


# --- Utility Functions ---
@st.cache_resource(show_spinner="⏳ Loading pre-trained model and scaler...")
def load_pretrained_model_and_params():
    st.info("🔄 **Step 1/1**: Loading `af_risk_model_and_params.pkl`...")
    try:
        with open('af_risk_model_and_params.pkl', 'rb') as f:
            # Load the dictionary containing all saved objects
            saved_objects = pickle.load(f)

        xgboost_classifier_model = saved_objects['model']
        feature_scaler = saved_objects['scaler']
        xgboost_optimal_threshold = saved_objects['optimal_threshold']
        feature_stats = saved_objects['feature_stats'] # Dictionary for slider min/max/mean

        st.success("✓ Pre-trained model loaded successfully.") # Changed message
        return xgboost_classifier_model, feature_scaler, xgboost_optimal_threshold, feature_stats
    except FileNotFoundError:
        st.error(
            "❌ **Error**: `af_risk_model_and_params.pkl` not found. Please ensure it's in the same directory as `af_risk_app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error**: Failed to load `af_risk_model_and_params.pkl`. Error message: {e}")
        st.code(traceback.format_exc())
        st.stop()
    return None, None, None, None


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="AF Risk Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and parameters at the start
xgboost_classifier_model, feature_scaler, xgboost_optimal_threshold, feature_stats = load_pretrained_model_and_params()

st.title("AF Risk Prediction App")
st.markdown("""
This application helps assess the risk of Atrial Fibrillation (AF) based on patient features.
Please input the patient's data in the sidebar to get a personalized risk assessment.
""") # Modified opening description

# ===========================
# Streamlit Sidebar: User Input Interface
# ===========================
st.sidebar.header("AF Risk Assessment") # Changed title
st.sidebar.markdown("Please enter the patient's feature values below for Atrial Fibrillation (AF) risk assessment.") # Modified sidebar description

user_patient_input = {}

st.sidebar.subheader("Continuous Features")
for feature_short_name in continuous_features:
    display_name = feature_name_display_map.get(feature_short_name, feature_short_name)
    stats = feature_stats.get(feature_short_name, {'min': 0, 'max': 100, 'mean': 50}) # Fallback for safety

    # Determine step size for the slider (e.g., 0.1 for floats, 1 for integers)
    if isinstance(stats['mean'], float) and stats['mean'] != int(stats['mean']):
        step = 0.1
        format_str = "%.1f"
    else:
        step = 1
        format_str = "%d"

    user_patient_input[feature_short_name] = st.sidebar.slider(
        label=display_name,
        min_value=float(stats['min']),
        max_value=float(stats['max']),
        value=float(stats['mean']),
        step=step,
        format=format_str,
        key=f"slider_{feature_short_name}"
    )

st.sidebar.subheader("Categorical Features")
user_patient_input['Gender'] = st.sidebar.selectbox(
    label=feature_name_display_map.get('Gender', 'Gender'),
    options=['Male', 'Female'], # Assuming these are the expected categories
    index=0, # Default to Male
    key="select_gender"
)

# Convert categorical features to numerical (e.g., one-hot or label encoding consistent with training)
# For 'Gender', we'll assume Male=0, Female=1 if that's how it was encoded during training.
# This part is crucial to match the model's expectation.
# For simplicity, let's assume a direct mapping based on common practice.
user_patient_input_processed = user_patient_input.copy()
user_patient_input_processed['Gender'] = 0 if user_patient_input_processed['Gender'] == 'Male' else 1


st.sidebar.markdown("---")
if st.sidebar.button("Assess AF Risk", type="primary"): # Changed button text from "Predict" to "Assess"
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_patient_input_processed])

    # Ensure the order of columns matches the training data (important for prediction)
    input_df = input_df[all_features_for_models]

    # Scale the input features using the loaded scaler
    scaled_input = feature_scaler.transform(input_df)

    # Make prediction (probability)
    prediction_proba = xgboost_classifier_model.predict_proba(scaled_input)[:, 1][0] # Probability of AF (class 1)

    # Apply the optimal threshold
    prediction_class = 1 if prediction_proba >= xgboost_optimal_threshold else 0

    st.subheader("Assessment Results") # Changed subheader
    st.markdown(f"**Patient AF Risk Probability:** `{prediction_proba:.4f}`")
    st.markdown(f"**Optimal Classification Threshold:** `{xgboost_optimal_threshold:.4f}`")

    if prediction_class == 1:
        st.error("⚠️ **Assessment:** High Risk of Atrial Fibrillation (AF)") # Changed "Prediction" to "Assessment"
        st.write("Based on the input features, the model indicates a high risk of Atrial Fibrillation.") # Changed "predicts" to "indicates"
    else:
        st.success("✅ **Assessment:** Low Risk of Atrial Fibrillation (AF)") # Changed "Prediction" to "Assessment"
        st.write("Based on the input features, the model indicates a low risk of Atrial Fibrillation.") # Changed "predicts" to "indicates"

    st.markdown("""
    ---
    **Disclaimer:** This is an automated risk assessment for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.
    """) # Modified disclaimer to remove "predictive model"

st.markdown("---") # This markdown separator is now standalone, as all content below it has been removed.

# ===========================
# "About This Application" SECTION HAS BEEN COMPLETELY REMOVED
# ===========================

