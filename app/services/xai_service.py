import math

import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.exceptions import NotFittedError

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

    defaults = {
        "Pregnancies": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "DiabetesPedigreeFunction": 0.5,
        "Salt_Intake": 5,
        "Stress_Score": 5,
        "Sleep_Duration": 7,
        "heart_disease": 0,
        "gender_Male": 1 if feature_set.get("gender") == 1 else 0,
        "gender_Other": 0,
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

    return sorted(influences, key=lambda x: abs(x["influence"]), reverse=True)[:3]


def _top_factors_from_coefficients(model, df):
    coef = getattr(model, "coef_", None)
    if coef is None:
        return []

    weights = np.asarray(coef)[0]
    influences = []
    for i, name in enumerate(df.columns):
        value = float(df.iloc[0, i])
        influences.append({
            "feature": name.replace("_", " ").replace("avg ", "").title(),
            "influence": _safe_number(weights[i] * value),
            "value": value,
        })
    return sorted(influences, key=lambda x: abs(x["influence"]), reverse=True)[:3]


def _hypertension_fallback(feature_set):
    systolic = feature_set.get("systolic_bp", 0)
    diastolic = feature_set.get("diastolic_bp", 0)
    if systolic >= 140 or diastolic >= 90:
        probability = 0.8
    elif systolic >= 130 or diastolic >= 80:
        probability = 0.6
    elif systolic >= 120:
        probability = 0.35
    else:
        probability = 0.15

    top_factors = [
        {"feature": "Systolic Blood Pressure", "influence": _safe_number(round(systolic / 180, 2)), "value": _safe_number(systolic)},
        {"feature": "Diastolic Blood Pressure", "influence": _safe_number(round(diastolic / 120, 2)), "value": _safe_number(diastolic)},
        {"feature": "BMI", "influence": _safe_number(round(feature_set.get("bmi", 0) / 50, 2)), "value": _safe_number(feature_set.get("bmi", 0))},
    ]

    return {
        "probability": probability,
        "justification": get_risk_justification("hypertension", probability, top_factors),
        "top_factors": top_factors,
    }

def get_risk_justification(model_key, prob, top_factors):
    """Generates a natural language explanation based on SHAP impact."""
    
    # Filter factors that significantly pushed the risk UP
    reasons = [f['feature'].lower() for f in top_factors if f['influence'] > 0.05]
    
    if prob < 0.25:
        return f"Your {model_key} risk is low. Your vitals are within a healthy range."
    
    # Constructing the sentence
    base_text = f"Your {model_key} risk is elevated"
    if reasons:
        # Formats as: "due to high glucose level, advanced age, and smoking status"
        if len(reasons) > 1:
            reason_str = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
        else:
            reason_str = reasons[0]
        return f"{base_text} primarily due to {reason_str}."
    
    return f"{base_text} due to a combination of physiological factors."

def get_comprehensive_report(feature_set):
    report = {}

    for key, model in models.items():
        try:
            df = _build_model_frame(key, model, feature_set)
            if df.empty:
                raise NotFittedError(f"{key} model has no fitted feature metadata")

            prob = _safe_number(model.predict_proba(df)[0][1])
            try:
                top_factors = _top_factors_from_shap(model, df)
            except ValueError:
                top_factors = _top_factors_from_coefficients(model, df)
            justification = get_risk_justification(key, prob, top_factors)
        except NotFittedError:
            if key == "hypertension":
                report[key] = _hypertension_fallback(feature_set)
                continue
            raise
        
        report[key] = {
            "probability": round(float(prob), 2),
            "justification": justification,
            "top_factors": top_factors
        }
    
    return report
