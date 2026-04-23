from pydantic import BaseModel
from typing import Any

class AnalysisRequest(BaseModel):
    # Extracted from PDF
    name: str
    age: int
    gender: str
    glucose: float
    
    # Manual User Inputs
    weight_kg: float
    height_cm: float
    systolic_bp: int
    diastolic_bp: int
    is_smoker: bool
    had_stroke_history: bool = False
    has_heart_disease: bool = False
    has_diabetes_history: bool = False
    has_hypertension_history: bool = False
    family_history_diabetes: bool = False
    family_history_hypertension: bool = False
    family_history_stroke: bool = False
    physical_activity_level: str = "Moderate"
    sleep_duration: float = 7
    stress_score: int = 5
    salt_intake_level: str = "Moderate"
    
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
    results: dict[str, Any]


class ReportRequest(BaseModel):
    name: str
    age: int | None = None
    gender: str | None = None
    glucose: float | None = None
    bmi: float
    bp_status: str | None = None
    results: dict[str, Any]
    ai_recommendation: str


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
