import json
import re

from PIL import Image

from app.services.gemini_utils import get_gemini_model, normalize_gemini_error


def _extract_json(raw_text: str) -> dict:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    json_block = fenced_match.group(1) if fenced_match else raw_text.strip()

    try:
        parsed = json.loads(json_block)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_from_image(image_path: str) -> dict:
    model = get_gemini_model()
    prompt = """
Read this lab report image and extract only the following fields as JSON:
{
  "name": string | null,
  "age": number | null,
  "gender": string | null,
  "glucose": number | null
}

Rules:
- Use the random blood sugar or glucose value if present.
- Return only JSON.
- If a field is missing, use null.
"""

    try:
        with Image.open(image_path) as image:
            response = model.generate_content([prompt, image])
        text = getattr(response, "text", "").strip()
        parsed = _extract_json(text)
        return {
            "name": parsed.get("name"),
            "age": parsed.get("age"),
            "gender": parsed.get("gender"),
            "glucose": parsed.get("glucose"),
        }
    except Exception as error:
        raise normalize_gemini_error(error)
