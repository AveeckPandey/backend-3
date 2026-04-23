from app.schemas.analysis import ChatRequest
from app.services.gemini_utils import get_gemini_model, normalize_gemini_error


def _build_context(data: ChatRequest) -> str:
    context_parts: list[str] = []

    if data.patient_name:
        context_parts.append(f"Patient name: {data.patient_name}")
    if data.glucose is not None:
        context_parts.append(f"Glucose: {data.glucose} mg/dL")
    if data.bp_status:
        context_parts.append(f"Blood pressure status: {data.bp_status}")
    if data.bmi is not None:
        context_parts.append(f"BMI: {data.bmi}")
    if data.risk_summary:
        context_parts.append(f"Risk summary: {data.risk_summary}")

    return "\n".join(context_parts) if context_parts else "No patient context provided."


def chat_about_health(data: ChatRequest) -> str:
    model = get_gemini_model()
    prompt = f"""
You are NeuroRiskXAI's in-app health assistant.

Rules:
- Be calm, practical, and concise.
- Do not claim to diagnose disease.
- If the user mentions severe symptoms or very high risk, advise medical review promptly.
- Answer in plain language suitable for a patient.
- Keep the reply under 120 words.

Available context:
{_build_context(data)}

User message:
{data.message.strip()}
"""

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if not text:
            return "I could not generate a useful reply just now. Please try asking again."
        return text
    except Exception as error:
        raise normalize_gemini_error(error)
