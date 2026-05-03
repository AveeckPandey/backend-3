import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    ReportRequest,
    ReportResponse,
)
from app.services.analysis_processing import normalize_analysis_inputs
from app.services.ai_advice import get_ai_medical_summary
from app.services.chat_service import chat_about_health
from app.services.gemini_utils import TokenLimitError
from app.services.health_utils import get_bp_category, is_hypertensive
from app.services.image_extraction_service import extract_from_image
from app.services.ocr_service import extract_from_pdf
from app.services.report_generator import REPORT_DIR, generate_pdf_report
from app.services.xai_service import get_comprehensive_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def _build_feature_set(normalized: dict) -> dict:
    return {
        "age": normalized["age"],
        "avg_glucose_level": normalized["glucose"],
        "Glucose": normalized["glucose"],
        "Age": normalized["age"],
        "BloodPressure": normalized["systolic_bp"],
        "bmi": normalized["bmi"],
        "BMI": normalized["bmi"],
        "gender": normalized["gender"],
        "smoking_status": 1 if normalized["is_smoker"] else 0,
        "hypertension": 1 if is_hypertensive(normalized["bp_status"]) else 0,
        "heart_disease": 1 if normalized["has_heart_disease"] else 0,
        "systolic_bp": normalized["systolic_bp"],
        "diastolic_bp": normalized["diastolic_bp"],
        "bp_status": normalized["bp_status"],
        "is_smoker": normalized["is_smoker"],
        "Salt_Intake": normalized["salt_intake_score"],
        "Stress_Score": normalized["stress_score"],
        "Sleep_Duration": normalized["sleep_duration"],
        "physical_activity_score": normalized["physical_activity_score"],
        "family_history_diabetes": normalized["family_history_diabetes"],
        "family_history_hypertension": normalized["family_history_hypertension"],
        "family_history_stroke": normalized["family_history_stroke"],
        "physical_activity_level": normalized["physical_activity_level"],
        "salt_intake_level": normalized["salt_intake_level"],
        "has_diabetes_history": normalized["has_diabetes_history"],
        "has_hypertension_history": normalized["has_hypertension_history"],
        "had_stroke_history": normalized["had_stroke_history"],
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/extract-report")
async def extract_report(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload a PDF report.")

    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extracted = extract_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "name": extracted.get("name") or extracted.get("Name") or "",
        "age": extracted.get("age"),
        "gender": extracted.get("gender"),
        "glucose": extracted.get("glucose"),
        "file_name": file.filename,
    }


@app.post("/extract-image-report")
async def extract_image_report(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG report image.")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extracted = extract_from_image(tmp_path)
    except TokenLimitError as error:
        raise HTTPException(status_code=429, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to read report image. {error}")
    finally:
        os.unlink(tmp_path)

    return {
        "name": extracted.get("name") or "",
        "age": extracted.get("age"),
        "gender": extracted.get("gender"),
        "glucose": extracted.get("glucose"),
        "file_name": file.filename,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_health(data: AnalysisRequest):
    normalized = normalize_analysis_inputs(data.model_dump())
    features = _build_feature_set(normalized)
    report = get_comprehensive_report(features)
    ai_summary = get_ai_medical_summary(normalized["name"], report, normalized["bmi"])

    stroke = report.get("stroke", {})
    diabetes = report.get("diabetes", {})

    return {
        "name": normalized["name"],
        "bmi": normalized["bmi"],
        "bp_status": normalized["bp_status"],
        "diabetes_risk": "High" if diabetes.get("probability", 0) >= 0.5 else "Normal",
        "stroke_probability": stroke.get("probability", 0.0),
        "physical_activity_level": normalized["physical_activity_level"],
        "salt_intake_level": normalized["salt_intake_level"],
        "sleep_duration": normalized["sleep_duration"],
        "stress_score": normalized["stress_score"],
        "had_stroke_history": normalized["had_stroke_history"],
        "has_heart_disease": normalized["has_heart_disease"],
        "has_diabetes_history": normalized["has_diabetes_history"],
        "has_hypertension_history": normalized["has_hypertension_history"],
        "family_history_diabetes": normalized["family_history_diabetes"],
        "family_history_hypertension": normalized["family_history_hypertension"],
        "family_history_stroke": normalized["family_history_stroke"],
        "ai_recommendation": ai_summary,
        "top_risk_factors": stroke.get("top_factors", [])[:4],
        "risk_factors": stroke.get("top_factors", [])[:4],
        "input_summary": normalized["input_summary"],
        "processing_notes": normalized["processing_notes"],
        "results": report,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(data: ChatRequest):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Please enter a message.")

    try:
        reply = chat_about_health(data)
    except TokenLimitError as error:
        raise HTTPException(status_code=429, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Chat assistant is unavailable right now. Please try again in a moment.",
        )

    return {"reply": reply}


@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(data: ReportRequest):
    payload = data.model_dump()
    if not payload.get("bp_status") and payload.get("systolic_bp") is not None and payload.get("diastolic_bp") is not None:
        payload["bp_status"] = get_bp_category(payload.get("systolic_bp"), payload.get("diastolic_bp"))
    path = generate_pdf_report(payload)
    return {
        "file_name": path.name,
        "download_path": f"/reports/{path.name}",
    }


@app.get("/reports/{file_name}")
async def download_report(file_name: str):
    path = (REPORT_DIR / file_name).resolve()
    report_root = REPORT_DIR.resolve()

    if report_root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=file_name,
    )


# Allow your Expo app to connect securely
