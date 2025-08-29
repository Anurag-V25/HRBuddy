# agents/workflow.py
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

POLICY_HINTS = [
    "policy","leave","vacation","holiday","attendance","dress",
    "payroll","benefit","paternity","maternity","salary","loan","gratuity",
    "work from home","remote"
]
SITE_HINTS = ["website","hoonartek","about","culture","mission","vision"]

def _keyword_guess(message: str) -> str:
    m = (message or "").lower()
    if any(k in m for k in SITE_HINTS): return "website"
    if any(k in m for k in POLICY_HINTS): return "policy"
    return "policy"

def route_intent(message: str) -> str:
    try:
        prompt = (
            "Classify the user message into exactly one of: `policy` or `website`.\n"
            f"Message: {message!r}\nOnly return one label."
        )
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        intent = (resp.text or "").strip().lower()
        return intent if intent in {"policy","website"} else _keyword_guess(message)
    except Exception:
        return _keyword_guess(message)