import os


def _fallback_summary(patient_name, results, bmi):
    risk_lines = []
    high_risk_models = []

    for model_key, report in results.items():
        probability = float(report.get("probability", 0))
        risk_lines.append(f"{model_key}: {round(probability * 100)}%")
        if probability >= 0.5:
            high_risk_models.append(model_key)

    urgent_note = (
        f" The elevated {', '.join(high_risk_models)} result should be reviewed with a doctor immediately."
        if high_risk_models
        else " No model crossed the high-risk threshold, but the results should still be reviewed with a clinician."
    )

    return (
        f"{patient_name}'s BMI is {bmi}, with model risk estimates of {', '.join(risk_lines)}."
        f"{urgent_note} Focus on the listed risk factors, maintain follow-up care, and avoid changing medication without medical advice."
    )


def _build_prompt(patient_name, results, bmi):
    risk_lines = []
    for model_key, report in results.items():
        risk_lines.append(
            f"- {model_key.title()} Risk: {float(report.get('probability', 0)) * 100:.0f}% "
            f"SHAP justification: {report.get('justification', 'No justification available.')}"
        )

    return f"""
You are a cautious clinical assistant summarizing a risk-screening result for {patient_name}.

Use ONLY the provided SHAP justifications and risk percentages. Do not invent medical history. If the risk is high, recommend seeing a doctor immediately.

Provided data:
- BMI: {bmi}
{chr(10).join(risk_lines)}

Write exactly 3 concise sentences:
1. Overall risk status grounded in the provided percentages.
2. Most urgent concern, if any, using only the listed SHAP justifications.
3. Practical lifestyle or follow-up recommendation, with a doctor-immediately warning if any risk is high.
"""


def get_ai_medical_summary(patient_name, results, bmi):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return _fallback_summary(patient_name, results, bmi)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(_build_prompt(patient_name, results, bmi))
        text = getattr(response, "text", "").strip()
        return text if len(text) >= 50 else _fallback_summary(patient_name, results, bmi)
    except Exception:
        return _fallback_summary(patient_name, results, bmi)
