from pydantic import BaseModel, Field, field_validator
from typing import Any

class AnalysisRequest(BaseModel):
    # Extracted from PDF
    name: str | None = None
    age: int | None = None
    gender: str | None = None
    glucose: float | None = None
    
    # Manual User Inputs
    weight_kg: float | None = None
    height_cm: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    is_smoker: bool | None = False
    had_stroke_history: bool = False
    has_heart_disease: bool = False
    has_diabetes_history: bool = False
    has_hypertension_history: bool = False
    family_history_diabetes: bool = False
    family_history_hypertension: bool = False
    family_history_stroke: bool = False
    physical_activity_level: str = "Moderate"
    sleep_duration: float | None = 7
    stress_score: int | None = 5
    salt_intake_level: str = "Moderate"

    @field_validator("*", mode="before")
    @classmethod
    def blank_strings_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value
    
class AnalysisResponse(BaseModel):
    name: str
    bmi: float
    bp_status: str
    diabetes_risk: str
    stroke_probability: float
    ai_recommendation: str
    # SHAP Data (Features that impacted the score)
    top_risk_factors: list[dict]
    risk_factors: list[dict]
    input_summary: dict[str, Any]
    processing_notes: list[str]
    results: dict[str, Any]


class ReportRequest(BaseModel):
    name: str | None = None
    age: int | None = None
    gender: str | None = None
    glucose: float | None = None
    bmi: float
    bp_status: str | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    physical_activity_level: str | None = None
    salt_intake_level: str | None = None
    sleep_duration: float | None = None
    stress_score: int | None = None
    input_summary: dict[str, Any] | None = None
    processing_notes: list[str] = Field(default_factory=list)
    results: dict[str, Any]
    ai_recommendation: str

    @field_validator("*", mode="before")
    @classmethod
    def blank_report_strings_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ReportResponse(BaseModel):
    file_name: str
    download_path: str


class ChatRequest(BaseModel):
    message: str
    patient_name: str | None = None
    glucose: float | None = None
    bp_status: str | None = None
    bmi: float | None = None
    risk_summary: str | None = None


class ChatResponse(BaseModel):
    reply: str
