import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
import warnings
import traceback
import pickle # 导入 pickle 库用于加载模型和 scaler

# ===========================
# Streamlit Page Configuration
# ===========================
st.set_page_config(layout="wide", page_title="Atrial Fibrillation Risk Prediction and Model Analysis")

st.title("💖 Atrial Fibrillation Risk Assessor")
st.markdown("An interactive platform for machine learning-based Atrial Fibrillation (AF) risk assessment.")
st.markdown("---")

# ===========================
# Global Settings
# ===========================
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define internal feature and target names (should match your Excel columns)
continuous_features = ['LVEDd', 'RVEDd', 'IVS', 'PAD', 'LAD', 'RAD', 'Age', 'AOD', 'LVPW']
categorical_features = ['Gender']
target = 'AF'

all_features_for_models = continuous_features + categorical_features

# Mapping for displaying full English names in the UI
feature_name_display_map = {
    'LVEDd': 'Left Ventricular End-Diastolic Diameter',
    'RVEDd': 'Right Ventricular End-Diastolic Diameter',
    'IVS': 'Interventricular Septum Thickness',
    'PAD': 'Pulmonary Artery Diameter',
    'LAD': 'Left Atrial Diameter',
    'RAD': 'Right Atrial Diameter',
    'Age': 'Patient Age',
    'AOD': 'Aortic Diameter',
    'LVPW': 'Left Ventricular Posterior Wall Thickness',
    'Gender': 'Patient Gender',
    'AF': 'Atrial Fibrillation'  # Also map target variable for display if needed
}

# ===========================
# Utility Functions (部分不再使用但保留以防将来扩展)
# ===========================
# diagnose_data 函数不再直接从文件中读取，因此不会在部署的应用中调用
# 但如果你想在本地运行时分析数据，可以手动调用。
def diagnose_data(features_train_original, labels_train_original, features_validation_original,
                  labels_validation_original):
    """
    Diagnoses basic statistical properties of training and validation datasets, and displays them using Streamlit.
    This function receives raw (unstandardized/unencoded) feature and label data.
    """
    st.subheader("📊 Data Diagnostics")
    st.info("The following provides basic statistics for the loaded raw data on the training and validation sets.")

    # ... (diagnose_data 函数内容保持不变，但不会在部署的应用中被调用) ...
    # 1. Sample Size Check
    st.write(f"**【Sample Size】**")
    st.write(f"  - Training Samples:   `{len(labels_train_original)}`")
    st.write(f"  - Validation Samples:   `{len(labels_validation_original)}`")

    # 2. Class Distribution
    st.write(
        f"**【Class Distribution】** ({feature_name_display_map.get(target, target)} = 1 is positive class)")  # Using mapped name for display
    training_af_rate = labels_train_original.mean()
    validation_af_rate = labels_validation_original.mean()
    training_positive_count = labels_train_original.sum()
    training_negative_count = len(labels_train_original) - training_positive_count
    validation_positive_count = labels_validation_original.sum()
    validation_negative_count = len(labels_validation_original) - validation_positive_count

    st.write(f"  - **Training Set:**")
    st.write(
        f"    - {feature_name_display_map.get(target, target)}=1: `{training_positive_count}` (`{training_af_rate:.2%}`)")  # Using mapped name for display
    st.write(
        f"    - {feature_name_display_map.get(target, target)}=0: `{training_negative_count}` (`{1 - training_af_rate:.2%}`)")  # Using mapped name for display
    st.write(f"  - **Validation Set:**")
    st.write(
        f"    - {feature_name_display_map.get(target, target)}=1: `{validation_positive_count}` (`{validation_af_rate:.2%}`)")  # Using mapped name for display
    st.write(
        f"    - {feature_name_display_map.get(target, target)}=0: `{validation_negative_count}` (`{1 - validation_af_rate:.2%}`)")  # Using mapped name for display

    # Validation set sample size warning
    if validation_positive_count < 30 or validation_negative_count < 30:
        st.warning(
            f"⚠️ **Warning**: Insufficient positive/negative samples in the validation set may lead to unstable evaluation metrics!")

    # 3. Feature Distribution Differences (Drift Detection)
    st.write(f"**【Feature Distribution Difference (KS Test)】** (p < 0.05 indicates a significant difference)")
    drift_detected = False
    for i, column_name in enumerate(features_train_original.columns):
        if len(np.unique(features_train_original.iloc[:, i])) > 1 and len(
                np.unique(features_validation_original.iloc[:, i])) > 1:
            statistic, p_value = ks_2samp(features_train_original.iloc[:, i], features_validation_original.iloc[:, i])
            if p_value < 0.05:
                # Display full name for the feature in the warning
                display_name = feature_name_display_map.get(column_name, column_name)
                st.warning(
                    f"  - ⚠️  Feature `{display_name}`: p-value=`{p_value:.4f}` (**Distribution drift detected!**)")
                drift_detected = True

    if drift_detected:
        st.error(
            f"**⚠️  Note**: Significant distribution differences detected in some features between the training and validation sets, which may affect model performance on the validation set.")
    else:
        st.success(f"**✓ All features have consistent distributions between training and validation sets.**")

    # 4. Missing Values Check
    st.write(f"**【Missing Values】**")
    training_missing_count = features_train_original.isnull().sum().sum()
    validation_missing_count = features_validation_original.isnull().sum().sum()
    st.write(f"  - Training Set:   `{training_missing_count}` missing values")
    st.write(f"  - Validation Set:   `{validation_missing_count}` missing values")

    if training_missing_count > 0 or validation_missing_count > 0:
        st.warning(
            f"⚠️ **Warning**: Missing values detected! The model automatically handled them before training (StandardScaler does not handle NaNs), but please check the raw data.")

    # 5. Feature Statistics Summary
    st.write(f"**【Feature Statistics Summary (Training Set)】** (Mean, Standard Deviation, Min, Max)")
    # Map column names for display in the dataframe header
    display_df = features_train_original.describe().loc[['mean', 'std', 'min', 'max']].T
    display_df.index = display_df.index.map(lambda x: feature_name_display_map.get(x, x))
    st.dataframe(display_df)

    st.markdown("---")


def find_youden_threshold(true_labels, predicted_scores):
    """Finds the threshold that maximizes Youden's Index."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, predicted_scores)
    youden_index = true_positive_rate - false_positive_rate
    best_index = np.argmax(youden_index)
    best_threshold = thresholds[best_index]
    best_youden = youden_index[best_index]
    return best_threshold, best_youden


def bootstrap_auc(true_labels, predicted_scores, num_bootstrap_samples=1000, model_index=0):
    """
    Bootstraps AUC to calculate 95% Confidence Interval.
    """
    random_seed_for_bootstrap = RANDOM_SEED + model_index * 1000
    rng = np.random.default_rng(random_seed_for_bootstrap)
    auc_bootstrap_samples = []
    true_labels_np = np.asarray(true_labels)
    predicted_scores_np = np.asarray(predicted_scores)

    for _ in range(num_bootstrap_samples):
        indices = rng.choice(len(true_labels_np), size=len(true_labels_np), replace=True)
        bootstrap_true_labels = true_labels_np[indices]
        bootstrap_predicted_scores = predicted_scores_np[indices]

        if len(np.unique(bootstrap_true_labels)) < 2:
            continue

        try:
            auc = roc_auc_score(bootstrap_true_labels, bootstrap_predicted_scores)
            auc_bootstrap_samples.append(auc)
        except ValueError:
            continue

    if not auc_bootstrap_samples:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_successful': 0
        }

    return {
        'mean': np.mean(auc_bootstrap_samples),
        'ci_lower': np.percentile(auc_bootstrap_samples, 2.5),
        'ci_upper': np.percentile(auc_bootstrap_samples, 97.5),
        'n_successful': len(auc_bootstrap_samples)
    }


def calculate_performance_metrics(true_labels, predicted_scores, decision_threshold):
    """Calculates performance metrics based on a given threshold (without Bootstrap)."""
    predicted_classes = (predicted_scores >= decision_threshold).astype(int)
    confusion_matrix_values = confusion_matrix(true_labels, predicted_classes)

    if confusion_matrix_values.shape == (2, 2):
        true_negative, false_positive, false_negative, true_positive = confusion_matrix_values.ravel()
    elif confusion_matrix_values.shape == (1, 1):
        if true_labels.sum() == 0:
            true_negative, false_positive, false_negative, true_positive = confusion_matrix_values[0, 0], 0, 0, 0
        else:
            true_negative, false_positive, false_negative, true_positive = 0, 0, 0, confusion_matrix_values[0, 0]
    else:
        true_negative, false_positive, false_negative, true_positive = 0, 0, 0, 0

    sensitivity_score = true_positive / (true_positive + false_negative) if (
                                                                                        true_positive + false_negative) > 0 else 0.000
    specificity_score = true_negative / (true_negative + false_positive) if (
                                                                                        true_negative + false_positive) > 0 else 0.000
    positive_predictive_value = true_positive / (true_positive + false_positive) if (
                                                                                                true_positive + false_positive) > 0 else 0.000
    negative_predictive_value = true_negative / (true_negative + false_negative) if (
                                                                                                true_negative + false_negative) > 0 else 0.000
    f1_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (
                                                                                                          2 * true_positive + false_positive + false_negative) > 0 else 0.000

    return {
        'Sensitivity': sensitivity_score,
        'Specificity': specificity_score,
        'Positive Predictive Value': positive_predictive_value,
        'Negative Predictive Value': negative_predictive_value,
        'F1 Score': f1_score
    }


def risk_stratification_by_youden(true_labels, predicted_probabilities, precalculated_best_threshold):
    """
    Calculates statistics for high-risk and low-risk populations based on a pre-calculated Youden's Index optimal threshold.
    """
    optimal_threshold = precalculated_best_threshold

    high_risk_mask = predicted_probabilities >= optimal_threshold
    low_risk_mask = predicted_probabilities < optimal_threshold

    high_risk_n_count = high_risk_mask.sum()
    high_risk_af_count = true_labels[high_risk_mask].sum()
    high_risk_af_rate = high_risk_af_count / high_risk_n_count if high_risk_n_count > 0 else 0

    low_risk_n_count = low_risk_mask.sum()
    low_risk_af_count = true_labels[low_risk_mask].sum()
    low_risk_af_rate = low_risk_af_count / low_risk_n_count if low_risk_n_count > 0 else 0

    return {
        'best_threshold': optimal_threshold,
        'high_risk_n': int(high_risk_n_count),
        'high_risk_af': int(high_risk_af_count),
        'high_risk_rate': high_risk_af_rate,
        'low_risk_n': int(low_risk_n_count),
        'low_risk_af': int(low_risk_af_count),
        'low_risk_rate': low_risk_af_rate
    }


def smooth_curve(false_positive_rate, true_positive_rate, factor=10):
    """Smooths the ROC curve."""
    false_positive_rate = np.asarray(false_positive_rate)
    true_positive_rate = np.asarray(true_positive_rate)

    unique_fpr, unique_indices = np.unique(false_positive_rate, return_index=True)
    unique_tpr = true_positive_rate[unique_indices]

    if len(unique_fpr) < 2:
        return false_positive_rate, true_positive_rate

    sort_idx = np.argsort(unique_fpr)
    unique_fpr = unique_fpr[sort_idx]
    unique_tpr = unique_tpr[sort_idx]

    interpolator_function = interp1d(
        unique_fpr,
        unique_tpr,
        kind='linear',
        bounds_error=False,
        fill_value=(unique_tpr[0], unique_tpr[-1])
    )

    new_false_positive_rate = np.linspace(unique_fpr.min(), unique_fpr.max(),
                                          int(len(unique_fpr) * factor))
    new_true_positive_rate = interpolator_function(new_false_positive_rate)

    return new_false_positive_rate, new_true_positive_rate


# ===========================
# Core Function for Loading Pre-trained Model and Scaler (Cached)
# ===========================
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

        st.success("✓ Pre-trained model, scaler, optimal threshold, and feature stats loaded successfully.")
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


# ===========================
# Call Cached Function on Streamlit App Startup
# ===========================
xgboost_classifier_model, feature_scaler, xgboost_optimal_threshold, feature_stats = load_pretrained_model_and_params()

if xgboost_classifier_model is None:
    st.error(
        "Application initialization failed. Please ensure the model file is correctly generated and placed.")
    st.stop()

# ===========================
# Streamlit Sidebar: User Input Interface
# ===========================
st.sidebar.header("Patient Information Input")
st.sidebar.markdown("Please enter the patient's feature values below for Atrial Fibrillation (AF) risk prediction.")

user_patient_input = {}

st.sidebar.subheader("Continuous Features")
for feature_short_name in continuous_features:  # Iterate using short names
    # Get stats from the loaded feature_stats dictionary
    stats = feature_stats.get(feature_short_name, {'min': 0.0, 'max': 100.0, 'mean': 50.0}) # Default if not found
    min_feature_value = stats['min']
    max_feature_value = stats['max']
    mean_feature_value = stats['mean']

    # Use full name for display
    display_name = feature_name_display_map.get(feature_short_name, feature_short_name)
    user_patient_input[feature_short_name] = st.sidebar.slider(
        f"{display_name} (Range: `{min_feature_value:.2f}` - `{max_feature_value:.2f}`)",
        min_value=float(min_feature_value),
        max_value=float(max_feature_value),
        value=float(mean_feature_value),
        step=(max_feature_value - min_feature_value) / 100 if (max_feature_value - min_feature_value) > 0 else 0.1
    )

st.sidebar.subheader("Categorical Features")
# Use full name for display for Gender
gender_display_name = feature_name_display_map.get('Gender', 'Gender')
selected_gender_string = st.sidebar.radio(
    f"{gender_display_name}:",
    options=['Male', 'Female'],
    index=0  # Default 'Male'
)
# Store using internal short name
user_patient_input['Gender'] = 1 if selected_gender_string == 'Female' else 0

# ===========================
# Main Content Area: Interactive Prediction Results
# ===========================
st.header("🔬 Interactive Risk Assessment Results")

if st.button("🚀 AF Risk Assessment"):
    # Input data uses internal short names
    ordered_input_data = pd.DataFrame([user_patient_input], columns=all_features_for_models)

    scaled_input_data = feature_scaler.transform(ordered_input_data)
    final_processed_input_dataframe = pd.DataFrame(scaled_input_data, columns=all_features_for_models)

    predicted_probability = xgboost_classifier_model.predict_proba(final_processed_input_dataframe)[:, 1][0]
    predicted_class = 1 if predicted_probability >= xgboost_optimal_threshold else 0

    st.write("### Prediction Results:")
    # Use mapped name for AF display
    st.markdown(
        f"**Probability of {feature_name_display_map.get('AF', 'AF')} (P({feature_name_display_map.get('AF', 'AF')}=1)):** `<span style='font-size: 1.5em; color: #4CAF50;'>{predicted_probability:.4f}</span>`",
        unsafe_allow_html=True)

    if predicted_class == 1:
        st.error(
            f"**Predicted Class:** **High Risk / {feature_name_display_map.get('AF', 'Atrial Fibrillation (AF)')}**")
    else:
        st.success(
            f"**Predicted Class:** **Low Risk / Non-{feature_name_display_map.get('AF', 'Atrial Fibrillation (Non-AF)')}**")

    st.markdown("---")
    st.subheader("Patient Input Data Overview:")
    # Create a display dataframe with full names as index
    displayed_user_input = pd.DataFrame(user_patient_input, index=["User Input"]).T
    displayed_user_input.index = displayed_user_input.index.map(lambda x: feature_name_display_map.get(x, x))

    # Convert 0/1 back to 'Male'/'Female' for intuitive display, using mapped name
    if feature_name_display_map.get('Gender') in displayed_user_input.index:
        displayed_user_input.loc[feature_name_display_map['Gender']] = "Female" if user_patient_input[
                                                                                       'Gender'] == 1 else "Male"
    st.dataframe(displayed_user_input)

st.markdown("---")

# ===========================
# Main Content Area: Model Configuration Details
# ===========================
st.header("⚙️ Model Configuration Details")
st.write("The following parameters were used in the loaded XGBoost model:")
# Filter out parameters with None values
filtered_xgboost_params = {k: v for k, v in xgboost_classifier_model.get_params().items() if v is not None}
st.json(filtered_xgboost_params)

st.write(f"Model's optimal classification threshold (Youden's Index): `{xgboost_optimal_threshold:.4f}`")

# Display feature lists using full names
continuous_features_display = [feature_name_display_map.get(f, f) for f in continuous_features]
categorical_features_display = [feature_name_display_map.get(f, f) for f in categorical_features]
target_display = feature_name_display_map.get(target, target)

st.write(f"Continuous features used for training: `{continuous_features_display}`")
st.write(
    f"Categorical features used for training: `{categorical_features_display}` (standardized as numerical features during training)")
st.write(f"Target variable: `{target_display}`")

st.markdown("---")


