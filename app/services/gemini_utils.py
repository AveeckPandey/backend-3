import os


class TokenLimitError(Exception):
    pass


def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def normalize_gemini_error(error: Exception) -> Exception:
    message = str(error).lower()
    token_markers = (
        "resource has been exhausted",
        "quota",
        "rate limit",
        "429",
        "token",
        "exceeded",
    )

    if any(marker in message for marker in token_markers):
        return TokenLimitError(
            "AI usage limit reached. Please wait a bit and try again after the Gemini token quota resets."
        )

    return error
