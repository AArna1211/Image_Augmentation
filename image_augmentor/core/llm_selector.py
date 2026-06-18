import os
import google.generativeai as genai

_client = None

def _get_client():
    global _client
    if _client is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _client = genai.GenerativeModel("gemini-1.5-flash")
    return _client

def llm_suggest_augmentations(class_name: str, available_ops: list[str], top_k: int = 3) -> list[str]:
    prompt = (
        f"You are an image processing expert.\n"
        f"Image category: {class_name}\n"
        f"Available operations: {', '.join(available_ops)}\n"
        f"Reply with ONLY a comma-separated list of the {top_k} most suitable operations "
        f"from the available ones. No explanation."
    )
    try:
        model = _get_client()
        response = model.generate_content(prompt)
        raw = response.text.strip()
        picks = [s.strip() for s in raw.split(",") if s.strip() in available_ops]
        return picks[:top_k] if picks else available_ops[:top_k]  # fallback to first N
    except Exception:
        return available_ops[:top_k]  # always return something valid