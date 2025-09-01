# main.py
# Full Flask app with clean UI formatting, small-talk handling, onboarding + RAG

import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple
from pathlib import Path
from enum import Enum, auto

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import jwt

# --- Your agents/modules ---
from agents.onboarding_agent import OnboardingAgent
from agents.rag_agent import RagAgent
from enhanced_advanced_dashboard import create_enhanced_dashboard
from flask import Flask, request, jsonify
from chatbot_backend import get_chatbot_response, health

# ===========================
# Config
# ===========================
load_dotenv()

APP_SECRET = os.getenv("SECRET_KEY", "dev-secret-change-me")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-jwt-change-me")
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "120"))

EXCEL_PATH = os.getenv("EMPLOYEE_VERIFICATION_PATH", "data/Employee Verification.xlsx")

LEADERSHIP_ROLES = {"HR-VP", "HR Admin", "CSO", "Manager", "HR", "HR Manager"}

REQUIRED_COLUMNS = {
    "Employee ID",
    "Employee Name",
    "Employee Email ID",
    "Password",
    "Date of Joining",
    "Designation",
}

# UI options
SHOW_SOURCES = False

# Static/UI directory
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# Helpers: data / auth
# ===========================
def load_employee_sheet() -> pd.DataFrame:
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Employee Verification.xlsx not found at: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def parse_date(value) -> Optional[datetime]:
    if pd.isna(value) or value == "":
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None

def compute_days_since(doj: Optional[datetime]) -> Optional[int]:
    if not doj:
        return None
    try:
        return (datetime.now().date() - doj.date()).days
    except Exception:
        return None

def decide_next_route(designation: str, days_since_doj: Optional[int]) -> str:
    """
    Routing:
      - Leadership roles -> dashboard
      - Employee with <30 days -> onboarding
      - Otherwise -> employee_chat
    """
    desig_norm = (designation or "").strip().lower()
    leadership_roles = {role.lower() for role in LEADERSHIP_ROLES}
    if desig_norm in leadership_roles:
        return "dashboard"
    if desig_norm == "employee":
        if days_since_doj is not None and days_since_doj < 30:
            return "onboarding"
        return "employee_chat"
    return "employee_chat"



# ===========================
# Helpers: JWT
# ===========================
def create_token(claims: Dict) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=TOKEN_TTL_MINUTES)
    payload = {
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        **claims,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_login_from_excel(email: str, password: str) -> Optional[Dict]:
    df = load_employee_sheet()
    df_local = df.copy()
    df_local["__email_lc__"] = df_local["Employee Email ID"].astype(str).str.strip().str.lower()

    email_lc = (email or "").strip().lower()
    row = df_local.loc[df_local["__email_lc__"] == email_lc]
    if row.empty:
        return None
    row = row.iloc[0]

    stored_pw = safe_str(row["Password"])
    if safe_str(password) != stored_pw:
        return None

    emp_id = safe_str(row["Employee ID"])
    full_name = safe_str(row["Employee Name"])
    designation = safe_str(row["Designation"])
    doj_dt = parse_date(row["Date of Joining"])
    days_since_doj = compute_days_since(doj_dt)

    return {
        "emp_id": emp_id,
        "email": email_lc,
        "full_name": full_name,
        "designation": designation,
        "doj": doj_dt.date().isoformat() if doj_dt else None,
        "days_since_doj": days_since_doj,
    }

def decode_bearer_token() -> Tuple[Optional[Dict], Optional[str]]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, "Missing or invalid Authorization header."
    token = auth.split(" ", 1)[1]
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return claims, None
    except jwt.ExpiredSignatureError:
        return None, "Token expired."
    except Exception as e:
        return None, f"Invalid token: {e}"

# ===========================
# Helpers: UI formatting + small talk
# ===========================

def _strip_trailing_json(text: str) -> str:
    """Remove a single trailing {...} JSON block if present and valid."""
    if not text:
        return text
    last = text.rfind("{")
    if last == -1:
        return text
    tail = text[last:]
    try:
        json.loads(tail)
        return text[:last].rstrip()
    except Exception:
        return text

def _strip_inline_json_lines(text: str) -> str:
    """
    Remove any lines that look like standalone JSON (the model sometimes prints a metadata object).
    Safe heuristic: if a trimmed line starts with '{' and ends with '}', and json.loads succeeds -> drop it.
    """
    if not text:
        return text
    cleaned = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                json.loads(s)
                continue  # drop this JSON line
            except Exception:
                pass
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def _strip_raw_source_lines(text: str) -> str:
    """
    Remove lines like 'SOURCES:', 'Source 1: ...', 'doc_0 (relevance ...)' printed by the model.
    We'll optionally add our own curated Sources section from meta.
    """
    if not text:
        return text
    cleaned = []
    for ln in text.splitlines():
        lns = ln.strip()
        up = lns.upper()
        if up.startswith("SOURCES"):
            continue
        if up.startswith("SOURCE "):
            continue
        if up.startswith("DOC_") and "RELEVANCE" in up:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def _render_sources_from_meta(meta: dict) -> str:
    if not SHOW_SOURCES:
        return ""
    if not isinstance(meta, dict):
        return ""
    top = meta.get("top_sources") or []
    if not top:
        return ""
    lines = ["", "**Sources**"]
    for s in top[:5]:
        sid = str(s.get("id", "")).strip()
        score = s.get("score", 0)
        if not sid:
            continue
        lines.append(f"- {sid} (relevance {score:.2f})")
    return "\n".join(lines) if len(lines) > 1 else ""

def _format_for_ui(raw_text: str, meta: dict) -> str:
    """
    Final, user-facing Markdown:
      1) Remove any trailing JSON block.
      2) Remove standalone JSON lines.
      3) Remove model-printed 'Source...' noise.
      4) Optionally append curated Sources from meta (SHOW_SOURCES=true).
    """
    text = (raw_text or "").strip()
    text = _strip_trailing_json(text)
    text = _strip_inline_json_lines(text)
    text = _strip_raw_source_lines(text)
    # collapse excessive blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    # curated Sources (optional)
    src = _render_sources_from_meta(meta or {})
    return (text + ("\n\n" + src if src else "")).strip()

_SMALL_TALK_TRIGGERS = {
    "hi","hello","hey","hey there","hiya","morning","good morning","gm",
    "good afternoon","good evening","evening","thanks","thank you","ty","yo"
}

def _is_small_talk(msg: str) -> bool:
    if not msg:
        return False
    m = msg.strip().lower()
    if m in _SMALL_TALK_TRIGGERS:
        return True
    for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        if m.startswith(g):
            if "?" not in m and len(m.split()) <= 5:
                return True
    return False

def _small_talk_reply(name: str = "") -> str:
    if name:
        return f"Hi {name}! How can I help today? I can assist with promotions, referrals, leave, dress code, and more."
    return "Hi! How can I help today? I can assist with promotions, referrals, leave, dress code, and more."

# ===========================
# App
# ===========================
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["SECRET_KEY"] = APP_SECRET
CORS(app, resources={r"/*": {"origins": "*"}})

# Agents
onboarding = OnboardingAgent(EXCEL_PATH)
rag = RagAgent()  # Initialize the RAG agent

# Dashboard
create_enhanced_dashboard(app)

# ===========================
# HTML routes
# ===========================
@app.get("/")
def home():
    from flask import render_template
    return render_template("login.html")

@app.get("/chat")
def chat_page():
    from flask import render_template
    return render_template("chat.html")

@app.get("/attrition")
def attrition_page():
    return app.send_static_file("index.html")

@app.route("/chat/attrition")
def chat_attrition():
    from flask import render_template
    return render_template("attration.html")

# ===========================
# Health
# ===========================

# ===========================
# Auth
# ===========================
@app.post("/auth/login")
def login():
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    email = safe_str(data.get("email", ""))
    password = safe_str(data.get("password", ""))
    if not email or not password:
        return jsonify({"error": "Both 'email' and 'password' are required."}), 400

    try:
        claims = verify_login_from_excel(email, password)
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500
    except ValueError as ve:
        return jsonify({"error": f"Excel schema error: {ve}"}), 500
    except Exception as e:
        return jsonify({"error": f"Login failed: {e}"}), 500

    if not claims:
        return jsonify({"error": "Invalid email or password."}), 401

    next_route = decide_next_route(claims.get("designation", ""), claims.get("days_since_doj"))
    token = create_token(claims)

    if next_route == "dashboard":
        route_url = "/dashboard/"
    elif next_route == "onboarding":
        route_url = "/chat?mode=onboarding"
    else:
        route_url = "/chat"

    return jsonify({
        "token": token,
        **claims,
        "next_route": next_route,
        "route_url": route_url
    }), 200

# ===========================
# “Who am I”
# ===========================
@app.get("/me")
def me():
    claims, err = decode_bearer_token()
    if err:
        return jsonify({"error": err}), 401
    return jsonify({"user": claims}), 200

@app.get("/api/whoami")
def whoami():
    return me()

# ===========================
# Chat APIs (RAG + Onboarding)
# ===========================
@app.post("/api/chat")
def api_chat():
    claims, err = decode_bearer_token()
    if err:
        return jsonify({"error": err}), 401

    data = request.get_json(force=True, silent=True) or {}
    text = safe_str(data.get("message", ""))
    mode = safe_str(data.get("mode", ""))  # "onboarding" or ""
    employee_id = claims.get("emp_id")
    session_id = employee_id or claims.get("email")

    profile = {
        "emp_id": employee_id,
        "email": claims.get("email"),
        "full_name": claims.get("full_name"),
        "designation": claims.get("designation"),
        "doj": claims.get("doj"),
        "days_since_doj": claims.get("days_since_doj"),
    }

    # Small talk short-circuit (no RAG/no sources)
    if _is_small_talk(text):
        pretty = _small_talk_reply(profile.get("full_name") or "")
        return jsonify({"reply": pretty, "meta": {}, "agent": "smalltalk"}), 200

    # Build a (currently unused) UserProfile for future personalization

    # Explicit onboarding mode (button/URL controlled)
    # Explicit onboarding mode (button/URL controlled)
    if mode == "onboarding":
        action = safe_str((data.get("action") or ""))  # <— NEW: catch action from buttons hitting /api/chat
        if action:
            reply, meta = onboarding.handle_button(session_id, action, data.get("value"), profile, extra=(data.get("extra") or {}), rag=rag)
        elif text in ("__init__", ""):
            reply, meta = onboarding.start(session_id, profile)
        else:
            reply, meta = onboarding.handle_turn(session_id, text, profile, rag=rag)
        pretty = _format_for_ui(reply, meta if isinstance(meta, dict) else {})
        return jsonify({"reply": pretty, **(meta if isinstance(meta, dict) and SHOW_SOURCES else {})}), 200


    # New joiners/freshers → onboarding automatically
    desig = (profile.get("designation") or "").strip().lower()
    days = profile.get("days_since_doj")
    should_onboard = (desig in {"freshers", "fresher"}) or (desig == "employee" and (days is not None and days < 30))
    if should_onboard:
        if text in ("__init__", ""):
            reply, meta = onboarding.start(session_id, profile)
        else:
            reply, meta = onboarding.handle_turn(session_id, text, profile, rag=rag)
        pretty = _format_for_ui(reply, meta if isinstance(meta, dict) else {})
        return jsonify({"reply": pretty, **(meta if isinstance(meta, dict) and SHOW_SOURCES else {})}), 200

    # ========== RAG flow ==========
    if text.strip() == "__init__":
        emp_name = profile.get("full_name", "")
        reply = (f"👋 Hi {emp_name}!\n"
                 "Welcome to HR Buddy 🙂\n"
                 "Ask me anything about Hoonartek policies, benefits, referrals, promotions, or onboarding.")
        return jsonify({"reply": reply, "agent": "rag", "meta": {}}), 200

    feedback = safe_str(data.get("feedback", ""))

    # derive a deterministic RAG thread per employee and call using only supported kwargs
    rag_session_id = f"{employee_id}:rag" if employee_id else session_id
    answer, meta = rag.answer_any(
        text,
    )

    pretty = _format_for_ui(answer, meta)
    return jsonify({"reply": pretty, "meta": (meta if SHOW_SOURCES else {}), "agent": "rag"}), 200

# ===========================
# Legacy/aux endpoints
# ===========================
@app.route("/api/attrition/predict", methods=["POST"])
def api_predict_single():
    try:
        data = request.get_json(force=True) or {}
        employee_id = int(data.get("employee_id"))
        horizon_days = int(data.get("horizon_days", 60))
        if horizon_days not in {30, 40, 50, 60, 90, 180}:
            return jsonify({"error": "Invalid horizon_days; use 30,40,50,60,90,180"}), 400
        from app import api_forecast
        result = api_forecast(employee_id, horizon_days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to predict: {e}"}), 500

@app.route("/api/attrition/predict/bulk", methods=["POST"])
def api_predict_bulk():
    try:
        data = request.get_json(force=True) or {}
        filters = data.get("filters", {}) or {}
        horizon_days = int(data.get("horizon_days", 60))
        limit = int(data.get("limit", 100))
        if horizon_days not in {30, 40, 50, 60, 90, 180}:
            return jsonify({"error": "Invalid horizon_days; use 30,40,50,60,90,180"}), 400
        from app import api_bulk_forecast
        results = api_bulk_forecast(filters, horizon_days, limit)
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": f"Failed to bulk predict: {e}"}), 500

@app.route("/api/attrition/action-plan", methods=["POST"])
def api_action_plan():
    try:
        data = request.get_json(force=True) or {}
        employee_id = int(data.get("employee_id"))
        actions = data.get("actions", [])
        if not isinstance(actions, list) or not actions:
            return jsonify({"error": "actions must be a non-empty list"}), 400
        from app import api_create_action_plan
        result = api_create_action_plan(employee_id, actions)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to create plan: {e}"}), 500

@app.route("/healthz", methods=["GET"])
def healthz():
    status = {"status": "ok", "excel_path": EXCEL_PATH, "has_excel": os.path.exists(EXCEL_PATH)}
    return jsonify(health()), 200

@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    try:
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        session_id = payload.get("session_id", "default")
        profile = payload.get("profile", {})
        user_name = payload.get("user_name", "there")

        # Default welcome message if nothing typed
        if not message:
            welcome = (
                f"👋 Hi {profile.get('full_name', user_name or 'there').split()[0]}!\n"
                "Welcome to HR Buddy 🙂\n"
                "I can help you with HR metrics and Employee attrition analysis."
            )
            return jsonify({
                "message": welcome,
                "response_type": "text",
                "data": {},
                "confidence": 1.0
            }), 200

        # Get response from backend
        out = get_chatbot_response(message, session_id=session_id, user_name=user_name, profile=profile)

        return jsonify({
            "message": out.get("message", ""),
            "response_type": out.get("type", "text"),
            "data": {},  # no table data anymore
            "confidence": float(out.get("confidence", 0.9))
        }), 200

    except Exception as e:
        return jsonify({
            "message": "⚠️ Sorry, I’m having trouble right now. Please try again.",
            "response_type": "text",
            "data": {"error": str(e)},
            "confidence": 0.0
        }), 200

# ===========================
# File upload (onboarding doc marker)
# ===========================
@app.post("/api/upload")
def api_upload():
    claims, err = decode_bearer_token()
    if err:
        return jsonify({"error": err}), 401
    data = request.get_json(force=True, silent=True) or {}
    kind = safe_str(data.get("kind", "")).lower()
    session_id = claims.get("emp_id") or claims.get("email")
    onboarding.mark_document_uploaded(session_id, kind)
    return jsonify({"ok": True}), 200

@app.post("/api/onboarding/action")
def api_onboarding_action():
    # Auth
    claims, err = decode_bearer_token()
    if err:
        return jsonify({"error": err}), 401

    data = request.get_json(force=True, silent=True) or {}
    action = safe_str(data.get("action", ""))   # e.g., "start", "next", "back", "select", "upload", "rag"
    value  = data.get("value")                  # optional payload from button (e.g., selected option)
    extra  = data.get("extra") or {}            # optional dict for richer payloads

    if not action:
        return jsonify({"error": "Missing 'action' in body."}), 400

    # session & profile
    employee_id = claims.get("emp_id")
    session_id  = employee_id or claims.get("email")
    profile = {
        "emp_id": employee_id,
        "email": claims.get("email"),
        "full_name": claims.get("full_name"),
        "designation": claims.get("designation"),
        "doj": claims.get("doj"),
        "days_since_doj": claims.get("days_since_doj"),
    }

    # delegate to onboarding agent (works even if text is empty)
    reply, meta = onboarding.handle_button(session_id, action, value, profile, extra=extra, rag=rag)

    # format for UI (no JSON, no raw “Source …” lines)
    pretty = _format_for_ui(reply, meta if isinstance(meta, dict) else {})
    return jsonify({"reply": pretty, **(meta if isinstance(meta, dict) and SHOW_SOURCES else {})}), 200


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)