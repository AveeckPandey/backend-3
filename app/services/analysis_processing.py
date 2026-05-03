from __future__ import annotations

from typing import Any

from app.services.health_utils import calculate_bmi, get_bp_category, is_hypertensive

REFERENCE_DEFAULTS = {
    "name": "Patient",
    "age": 45,
    "gender": "Unknown",
    "glucose": 100.0,
    "systolic_bp": 118,
    "diastolic_bp": 76,
    "sleep_duration": 7.0,
    "stress_score": 5,
    "salt_intake_level": "Moderate",
    "physical_activity_level": "Moderate",
    "bmi": 25.0,
}

_BOOLEAN_TRUE = {"1", "true", "yes", "y", "on"}
_BOOLEAN_FALSE = {"0", "false", "no", "n", "off"}
_CHOICE_VALUES = {
    "gender": {"male": "Male", "female": "Female", "other": "Other", "unknown": "Unknown"},
    "physical_activity_level": {"low": "Low", "moderate": "Moderate", "high": "High"},
    "salt_intake_level": {"low": "Low", "moderate": "Moderate", "high": "High"},
}


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _to_bool(value: Any, default: bool = False) -> bool:
    value = _blank_to_none(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in _BOOLEAN_TRUE:
        return True
    if normalized in _BOOLEAN_FALSE:
        return False
    return default


def _normalize_number(
    *,
    value: Any,
    label: str,
    notes: list[str],
    minimum: float,
    maximum: float,
    default: float,
    integer: bool = False,
) -> float | int:
    value = _blank_to_none(value)
    if value is None:
        notes.append(f"{label} was missing. Used reference value {default}.")
        return int(default) if integer else float(default)

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        notes.append(f"{label} was invalid. Used reference value {default}.")
        return int(default) if integer else float(default)

    if numeric < minimum or numeric > maximum:
        notes.append(
            f"{label} was outside the supported range ({minimum}-{maximum}). "
            f"Used reference value {default}."
        )
        return int(default) if integer else float(default)

    return int(round(numeric)) if integer else round(numeric, 2)


def _normalize_optional_measurement(
    *,
    value: Any,
    label: str,
    notes: list[str],
    minimum: float,
    maximum: float,
) -> float | None:
    value = _blank_to_none(value)
    if value is None:
        notes.append(f"{label} was not provided.")
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        notes.append(f"{label} was invalid and was excluded from BMI calculation.")
        return None

    if numeric < minimum or numeric > maximum:
        notes.append(
            f"{label} was outside the supported range ({minimum}-{maximum}) and was excluded from BMI calculation."
        )
        return None

    return round(numeric, 2)


def _normalize_choice(value: Any, field_name: str, label: str, notes: list[str]) -> str:
    value = _blank_to_none(value)
    choices = _CHOICE_VALUES[field_name]
    if value is None:
        default = REFERENCE_DEFAULTS[field_name]
        notes.append(f"{label} was missing. Used default value {default}.")
        return default

    normalized = str(value).strip().lower()
    if normalized in choices:
        return choices[normalized]

    default = REFERENCE_DEFAULTS[field_name]
    notes.append(f"{label} value '{value}' was not recognized. Used default value {default}.")
    return default


def _salt_intake_score(level: str) -> int:
    normalized = level.strip().lower()
    if normalized == "low":
        return 3
    if normalized == "high":
        return 8
    return 5


def _activity_score(level: str) -> int:
    normalized = level.strip().lower()
    if normalized == "low":
        return 3
    if normalized == "high":
        return 8
    return 5


def normalize_analysis_inputs(raw: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []

    raw_name = _blank_to_none(raw.get("name"))
    name = str(raw_name).strip() if raw_name is not None else REFERENCE_DEFAULTS["name"]
    if raw_name is None:
        notes.append("Patient name was missing. Using 'Patient' in the generated report.")

    age = _normalize_number(
        value=raw.get("age"),
        label="Age",
        notes=notes,
        minimum=0,
        maximum=120,
        default=REFERENCE_DEFAULTS["age"],
        integer=True,
    )
    gender = _normalize_choice(raw.get("gender"), "gender", "Gender", notes)
    glucose = _normalize_number(
        value=raw.get("glucose"),
        label="Glucose",
        notes=notes,
        minimum=40,
        maximum=600,
        default=REFERENCE_DEFAULTS["glucose"],
    )

    weight_kg = _normalize_optional_measurement(
        value=raw.get("weight_kg"),
        label="Weight",
        notes=notes,
        minimum=20,
        maximum=400,
    )
    height_cm = _normalize_optional_measurement(
        value=raw.get("height_cm"),
        label="Height",
        notes=notes,
        minimum=100,
        maximum=250,
    )
    if weight_kg is not None and height_cm is not None:
        bmi = calculate_bmi(weight_kg, height_cm)
        bmi_source = "calculated"
    else:
        bmi = REFERENCE_DEFAULTS["bmi"]
        bmi_source = "reference_default"
        notes.append(
            f"BMI used reference value {REFERENCE_DEFAULTS['bmi']} because weight and height were incomplete."
        )

    systolic_bp = _normalize_number(
        value=raw.get("systolic_bp"),
        label="Systolic blood pressure",
        notes=notes,
        minimum=70,
        maximum=250,
        default=REFERENCE_DEFAULTS["systolic_bp"],
        integer=True,
    )
    diastolic_bp = _normalize_number(
        value=raw.get("diastolic_bp"),
        label="Diastolic blood pressure",
        notes=notes,
        minimum=40,
        maximum=150,
        default=REFERENCE_DEFAULTS["diastolic_bp"],
        integer=True,
    )
    bp_status = get_bp_category(systolic_bp, diastolic_bp)

    physical_activity_level = _normalize_choice(
        raw.get("physical_activity_level"),
        "physical_activity_level",
        "Physical activity level",
        notes,
    )
    salt_intake_level = _normalize_choice(
        raw.get("salt_intake_level"),
        "salt_intake_level",
        "Salt intake level",
        notes,
    )
    sleep_duration = _normalize_number(
        value=raw.get("sleep_duration"),
        label="Sleep duration",
        notes=notes,
        minimum=0,
        maximum=24,
        default=REFERENCE_DEFAULTS["sleep_duration"],
    )
    stress_score = _normalize_number(
        value=raw.get("stress_score"),
        label="Stress score",
        notes=notes,
        minimum=1,
        maximum=10,
        default=REFERENCE_DEFAULTS["stress_score"],
        integer=True,
    )

    normalized = {
        "name": name,
        "age": age,
        "gender": gender,
        "glucose": glucose,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
        "bmi_source": bmi_source,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bp_status": bp_status,
        "is_smoker": _to_bool(raw.get("is_smoker")),
        "had_stroke_history": _to_bool(raw.get("had_stroke_history")),
        "has_heart_disease": _to_bool(raw.get("has_heart_disease")),
        "has_diabetes_history": _to_bool(raw.get("has_diabetes_history")),
        "has_hypertension_history": _to_bool(raw.get("has_hypertension_history")),
        "family_history_diabetes": _to_bool(raw.get("family_history_diabetes")),
        "family_history_hypertension": _to_bool(raw.get("family_history_hypertension")),
        "family_history_stroke": _to_bool(raw.get("family_history_stroke")),
        "physical_activity_level": physical_activity_level,
        "physical_activity_score": _activity_score(physical_activity_level),
        "sleep_duration": sleep_duration,
        "stress_score": stress_score,
        "salt_intake_level": salt_intake_level,
        "salt_intake_score": _salt_intake_score(salt_intake_level),
        "processing_notes": notes,
        "input_summary": {
            "name": name,
            "age": age,
            "gender": gender,
            "glucose_mg_dl": glucose,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": bmi,
            "bmi_source": bmi_source,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "bp_status": bp_status,
            "blood_pressure_reading": f"{systolic_bp}/{diastolic_bp} mmHg",
            "is_smoker": _to_bool(raw.get("is_smoker")),
            "physical_activity_level": physical_activity_level,
            "sleep_duration": sleep_duration,
            "stress_score": stress_score,
            "salt_intake_level": salt_intake_level,
            "family_history_diabetes": _to_bool(raw.get("family_history_diabetes")),
            "family_history_hypertension": _to_bool(raw.get("family_history_hypertension")),
            "family_history_stroke": _to_bool(raw.get("family_history_stroke")),
            "has_heart_disease": _to_bool(raw.get("has_heart_disease")),
            "has_diabetes_history": _to_bool(raw.get("has_diabetes_history")),
            "has_hypertension_history": _to_bool(raw.get("has_hypertension_history")),
            "had_stroke_history": _to_bool(raw.get("had_stroke_history")),
            "hypertension_flag": is_hypertensive(bp_status),
        },
    }
    return normalized
