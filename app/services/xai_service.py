import math

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.exceptions import NotFittedError

from app.services.health_utils import get_bp_category

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Load models
models = {
    "stroke": joblib.load(MODEL_DIR / "stroke_v1.pkl"),
    "hypertension": joblib.load(MODEL_DIR / "hypertension_v1.pkl"),
    "diabetes": joblib.load(MODEL_DIR / "diabetes_v1.pkl")
}

MODEL_FEATURES = {
    "diabetes": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ],
    "stroke": [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "gender_Male",
        "gender_Other",
        "ever_married_Yes",
        "work_type_Never_worked",
        "work_type_Private",
        "work_type_Self-employed",
        "work_type_children",
        "Residence_type_Urban",
        "smoking_status_formerly smoked",
        "smoking_status_never smoked",
        "smoking_status_smokes",
    ],
}


def _model_features(model_key, model):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and len(feature_names) > 0:
        return list(feature_names)
    return MODEL_FEATURES.get(model_key, [])


def _feature_value(feature_set, name):
    if name in feature_set:
        return feature_set[name]

    gender = str(feature_set.get("gender", "")).strip().lower()

    defaults = {
        "Pregnancies": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "DiabetesPedigreeFunction": 0.5,
        "Salt_Intake": 5,
        "Stress_Score": 5,
        "Sleep_Duration": 7,
        "heart_disease": 0,
        "gender_Male": 1 if gender == "male" else 0,
        "gender_Other": 1 if gender == "other" else 0,
        "ever_married_Yes": 1 if feature_set.get("age", 0) >= 18 else 0,
        "work_type_Never_worked": 0,
        "work_type_Private": 1,
        "work_type_Self-employed": 0,
        "work_type_children": 0,
        "Residence_type_Urban": 1,
        "smoking_status_formerly smoked": 0,
        "smoking_status_never smoked": 0 if feature_set.get("is_smoker") else 1,
        "smoking_status_smokes": 1 if feature_set.get("is_smoker") else 0,
    }
    return defaults.get(name, 0)


def _build_model_frame(model_key, model, feature_set):
    features = _model_features(model_key, model)
    values = {name: _feature_value(feature_set, name) for name in features}
    return pd.DataFrame([values])


def _baseline_value(name):
    baselines = {
        "age": 45,
        "Age": 45,
        "avg_glucose_level": 100,
        "Glucose": 100,
        "BloodPressure": 80,
        "bmi": 25,
        "BMI": 25,
        "hypertension": 0,
        "heart_disease": 0,
        "Pregnancies": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "DiabetesPedigreeFunction": 0.5,
        "Salt_Intake": 5,
        "Stress_Score": 5,
        "Sleep_Duration": 7,
        "is_smoker": 0,
        "gender_Male": 0,
        "gender_Other": 0,
        "ever_married_Yes": 1,
        "work_type_Never_worked": 0,
        "work_type_Private": 1,
        "work_type_Self-employed": 0,
        "work_type_children": 0,
        "Residence_type_Urban": 1,
        "smoking_status_formerly smoked": 0,
        "smoking_status_never smoked": 1,
        "smoking_status_smokes": 0,
    }
    return baselines.get(name, 0)


def _build_background_frame(df):
    return pd.DataFrame([{name: _baseline_value(name) for name in df.columns}])


def _safe_number(value, default=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(numeric) or math.isinf(numeric):
        return default

    return numeric


def _coerce_shap_vector(raw_values, expected_length):
    values = raw_values

    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    array = np.asarray(values)

    if array.ndim == 3:
        if array.shape[0] == 1 and array.shape[2] > 1:
            array = array[0, :, 1]
        elif array.shape[0] > 1 and array.shape[1] == 1:
            array = array[1, 0, :]
        else:
            array = array.reshape(-1)
    elif array.ndim == 2:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[1] == 2 and array.shape[0] == expected_length:
            array = array[:, 1]
        elif array.shape[0] > 1 and array.shape[1] == expected_length:
            array = array[1]
        else:
            array = array.reshape(-1)

    array = np.asarray(array).reshape(-1)

    if array.size != expected_length:
        raise ValueError(f"Expected {expected_length} SHAP values, got {array.size}")

    return array


def _calculate_shap_values(model, df):
    background = _build_background_frame(df)
    explainers = [
        lambda: shap.Explainer(model, background),
        lambda: shap.LinearExplainer(model, background),
        lambda: shap.TreeExplainer(model),
    ]

    last_error = None
    for build_explainer in explainers:
        try:
            explanation = build_explainer()(df)
            raw_values = getattr(explanation, "values", explanation)
            return _coerce_shap_vector(raw_values, len(df.columns))
        except Exception as error:
            last_error = error

    raise ValueError(f"Unable to compute SHAP values: {last_error}")


def _top_factors_from_shap(model, df):
    shap_values = _calculate_shap_values(model, df)
    influences = []

    for i, name in enumerate(df.columns):
        influence = _safe_number(shap_values[i])
        value = _safe_number(df.iloc[0, i]) if pd.api.types.is_numeric_dtype(df[name]) else df.iloc[0, i]
        influences.append({
            "feature": name.replace("_", " ").replace("avg ", "").title(),
            "influence": influence,
            "value": value,
        })

    return sorted(influences, key=lambda x: abs(x["influence"]), reverse=True)


def _top_factors_from_coefficients(model, df):
    coef = getattr(model, "coef_", None)
    if coef is None:
        return []

    weights = np.asarray(coef)[0]
    influences = []
    for i, name in enumerate(df.columns):
        value = float(df.iloc[0, i])
        baseline = _baseline_value(name)
        influences.append({
            "feature": name.replace("_", " ").replace("avg ", "").title(),
            "influence": _safe_number(weights[i] * (value - baseline)),
            "value": value,
        })
    return sorted(influences, key=lambda x: abs(x["influence"]), reverse=True)


def _split_factor_directions(factors, threshold=0.05):
    risk_factors = [factor for factor in factors if factor.get("influence", 0) > threshold]
    protective_factors = [factor for factor in factors if factor.get("influence", 0) < -threshold]
    return risk_factors[:3], protective_factors[:2]


def _hypertension_assessment(feature_set):
    systolic = feature_set.get("systolic_bp", 0)
    diastolic = feature_set.get("diastolic_bp", 0)
    bp_status = feature_set.get("bp_status") or get_bp_category(systolic, diastolic)

    probability = {
        "Normal": 0.08,
        "Elevated": 0.22,
        "Hypertension Stage 1": 0.48,
        "Hypertension Stage 2": 0.78,
        "Hypertensive Crisis": 0.97,
    }.get(bp_status, 0.2)

    bmi = _safe_number(feature_set.get("bmi", 0))
    salt_score = _safe_number(feature_set.get("Salt_Intake", 5))
    stress_score = _safe_number(feature_set.get("Stress_Score", 5))
    sleep_duration = _safe_number(feature_set.get("Sleep_Duration", 7))
    activity_score = _safe_number(feature_set.get("physical_activity_score", 5))

    if feature_set.get("family_history_hypertension"):
        probability += 0.05
    if feature_set.get("has_hypertension_history"):
        probability += 0.08
    if feature_set.get("is_smoker"):
        probability += 0.04
    if bmi >= 30:
        probability += 0.06
    elif bmi >= 25:
        probability += 0.03
    if salt_score >= 8:
        probability += 0.05
    elif salt_score <= 3:
        probability -= 0.03
    if stress_score >= 8:
        probability += 0.04
    elif stress_score <= 3:
        probability -= 0.02
    if sleep_duration < 6:
        probability += 0.03
    elif sleep_duration >= 8:
        probability -= 0.02
    if activity_score >= 8:
        probability -= 0.03
    elif activity_score <= 3:
        probability += 0.02

    probability = max(0.02, min(0.99, probability))

    all_factors = [
        {"feature": "Systolic Blood Pressure", "influence": _safe_number(round((systolic - 118) / 35, 3)), "value": _safe_number(systolic)},
        {"feature": "Diastolic Blood Pressure", "influence": _safe_number(round((diastolic - 76) / 20, 3)), "value": _safe_number(diastolic)},
        {"feature": "BMI", "influence": _safe_number(round((bmi - 25) / 10, 3)), "value": bmi},
        {"feature": "Salt Intake", "influence": _safe_number(round((salt_score - 5) / 10, 3)), "value": salt_score},
        {"feature": "Stress Score", "influence": _safe_number(round((stress_score - 5) / 10, 3)), "value": stress_score},
        {"feature": "Sleep Duration", "influence": _safe_number(round((7 - sleep_duration) / 8, 3)), "value": sleep_duration},
        {"feature": "Physical Activity", "influence": _safe_number(round((5 - activity_score) / 10, 3)), "value": activity_score},
    ]
    if feature_set.get("family_history_hypertension"):
        all_factors.append({"feature": "Family History Hypertension", "influence": 0.18, "value": 1})
    if feature_set.get("has_hypertension_history"):
        all_factors.append({"feature": "Previous Hypertension", "influence": 0.22, "value": 1})
    if feature_set.get("is_smoker"):
        all_factors.append({"feature": "Smoking Status Smokes", "influence": 0.14, "value": 1})

    ordered_factors = sorted(all_factors, key=lambda x: abs(x["influence"]), reverse=True)
    top_factors, protective_factors = _split_factor_directions(ordered_factors, threshold=0.04)

    return {
        "probability": probability,
        "classification": bp_status,
        "justification": get_risk_justification("hypertension", probability, top_factors, protective_factors),
        "top_factors": top_factors or ordered_factors[:3],
        "protective_factors": protective_factors,
    }


def get_risk_justification(model_key, prob, top_factors, protective_factors):
    """Generates a natural language explanation based on SHAP impact."""

    reasons = [f["feature"].lower() for f in top_factors if f["influence"] > 0.05]
    protective_reasons = [f["feature"].lower() for f in protective_factors if f["influence"] < -0.05]

    if prob < 0.25:
        if protective_reasons:
            if len(protective_reasons) > 1:
                protective_text = ", ".join(protective_reasons[:-1]) + ", and " + protective_reasons[-1]
            else:
                protective_text = protective_reasons[0]
            return (
                f"Your {model_key} risk is low. Supportive factors such as {protective_text} "
                f"are helping keep this risk down."
            )
        return f"Your {model_key} risk is low. Your vitals are within a healthy range."

    base_text = f"Your {model_key} risk is elevated"
    if reasons:
        if len(reasons) > 1:
            reason_str = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
        else:
            reason_str = reasons[0]
        return f"{base_text} primarily due to {reason_str}."

    return f"{base_text} due to a combination of physiological factors."

def get_comprehensive_report(feature_set):
    report = {}

    for key, model in models.items():
        if key == "hypertension":
            report[key] = _hypertension_assessment(feature_set)
            continue

        try:
            df = _build_model_frame(key, model, feature_set)
            if df.empty:
                raise NotFittedError(f"{key} model has no fitted feature metadata")

            prob = _safe_number(model.predict_proba(df)[0][1])
            try:
                all_factors = _top_factors_from_shap(model, df)
            except ValueError:
                all_factors = _top_factors_from_coefficients(model, df)
            top_factors, protective_factors = _split_factor_directions(all_factors)
            justification = get_risk_justification(key, prob, top_factors, protective_factors)
        except NotFittedError:
            raise
        
        report[key] = {
            "probability": round(float(prob), 2),
            "justification": justification,
            "top_factors": top_factors or all_factors[:3],
            "protective_factors": protective_factors,
        }
    
    return report
