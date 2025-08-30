# agents/onboarding_agent.py
# --------------------------------------------------------------------------------------
# New Onboarding (Agentic) — STRICT routing:
#   ✅ SerpAPI ONLY for:
#      • 🏠 Stay locations (PGs/Hostels/Accommodation)
#      • 🍽️ Nearby food courts (Restaurants/Cafes/Eateries)
#   ✅ Default answers for:
#      • 📋 Project allocation
#      • 📍 Project location
#   ✅ RAG integration:
#      • All organization-policy queries go to your RAG agent
#        (for BOTH New Onboarding and Pre-Onboarding routes)
#
# Pre-Onboarding flow — UNCHANGED except we don’t intercept inside it; we route
# org-policy questions at the public agent layer so both flows benefit without edits.
# --------------------------------------------------------------------------------------

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd

# =========================
# Shared configuration
# =========================
DEFAULT_EXCEL = os.getenv("EMPLOYEE_EXCEL", "data/Employee Verification.xlsx")

COL_EMPLOYEE_ID = "Employee ID"
COL_EMPLOYEE_EMAIL = "Employee Email ID"
COL_EMPLOYEE_NAME = "Employee Name"
COL_DESIGNATION = "Designation"
COL_DOJ = "Date of Joining"
COL_BUDDY = "Buddy"
SHAREPOINT_UPLOAD_URL = "https://hoonartekhub.sharepoint.com/_layouts/15/sharepoint.aspx"

DEMO_OTP = os.getenv("PREONBOARDING_DEMO_OTP", "123456")
QUICK_FAQ = ["Reporting location", "Dress code", "Working hours", "Whom do I report to?"]

# Local / APIs
MAGARPATTA_COORDINATES = "@18.515961,73.926331,14z"
SERP_API_KEY = os.getenv("SERP_API_KEY", "")          # required for SerpAPI use
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")      # optional (tone + intent refinement)

# SerpAPI import (safe)
try:
    from serpapi import GoogleSearch  # pip install google-search-results
    _SERPAPI_AVAILABLE = True
except Exception:
    GoogleSearch = None  # type: ignore
    _SERPAPI_AVAILABLE = False

# Gemini import (optional, graceful fallback)
try:
    import google.generativeai as genai  # pip install google-generativeai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _GEMINI_AVAILABLE = True
    else:
        _GEMINI_AVAILABLE = False
except Exception:
    _GEMINI_AVAILABLE = False


# =========================
# Utils
# =========================
def _safe_str(v: Any) -> str:
    return "" if v is None else str(v).strip()

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", _safe_str(s))

def _parse_date_maybe(v) -> Optional[datetime]:
    if v is None or pd.isna(v):
        return None
    if isinstance(v, datetime):
        return v
    try:
        return pd.to_datetime(v, errors="coerce").to_pydatetime()
    except Exception:
        return None

def _is_fresher(desig: str) -> bool:
    return "fresh" in (desig or "").strip().lower()

def _looks_like_email(s: str) -> bool:
    s = s.strip().lower()
    return "@" in s and "." in s and " " not in s

def _maps_url_from(title: str, address: str) -> str:
    q = quote_plus(f"{title} {address} Magarpatta City Pune")
    return f"https://www.google.com/maps/search/?api=1&query={q}"


# =========================
# Intent Agent (rules + optional Gemini refinement)
# =========================
class IntentAgent:
    """
    Returns exactly one of:
      'stay', 'food', 'project_allocation', 'project_location', 'org_policy', 'other'
    """
    STAY_KWS = ['stay', 'pg', 'hostel', 'accommodation', 'room', 'rent', 'flat', 'apartment', 'co-living', 'coliving']
    FOOD_KWS = ['food', 'restaurant', 'cafe', 'eatery', 'dining', 'breakfast', 'lunch', 'dinner', 'canteen']
    ALLOC_KWS = ['project allocation', 'which project', 'assigned project', 'my project']
    LOC_KWS = ['project location', 'office location', 'workplace', 'office address', 'project site']

    # Org-policy examples (RAG-eligible)
    POLICY_KWS = [
        'policy', 'policies', 'leave', 'holiday', 'reimbursement', 'travel policy',
        'expense', 'wfh', 'work from home', 'remote', 'probation', 'notice period',
        'benefit', 'grievance', 'security', 'device', 'laptop', 'email policy',
        'code of conduct', 'it policy', 'hr policy', 'overtime', 'timesheet'
    ]

    def classify(self, user_input: str) -> str:
        text = (user_input or "").lower()

        # 1) rule-based — strict and fast
        if any(k in text for k in self.ALLOC_KWS):
            return "project_allocation"
        if any(k in text for k in self.LOC_KWS):
            return "project_location"
        if any(k in text for k in self.STAY_KWS):
            return "stay"
        if any(k in text for k in self.FOOD_KWS):
            return "food"
        if any(k in text for k in self.POLICY_KWS):
            return "org_policy"

        # 2) optional Gemini refinement
        if _GEMINI_AVAILABLE:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "Classify the user message into exactly one label: "
                    "'stay', 'food', 'project_allocation', 'project_location', 'org_policy', or 'other'. "
                    "Return only the label.\n\n"
                    f"Message: {user_input!r}"
                )
                out = model.generate_content(prompt)
                label = (out.text or "").strip().lower()
                if label in {"stay","food","project_allocation","project_location","org_policy","other"}:
                    return label
            except Exception:
                pass

        return "other"


class ResponseStyler:
    """Optional Gemini polish; returns raw text if Gemini not available."""
    def style(self, md_text: str) -> str:
        if not _GEMINI_AVAILABLE:
            return md_text
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Polish the message to be friendly and concise. Keep markdown, emojis, URLs.\n\n"
                f"{md_text}"
            )
            out = model.generate_content(prompt)
            return (out.text or "").strip() or md_text
        except Exception:
            return md_text


# =========================
# SerpAPI agent limited to Stay/Food
# =========================
class StayFoodAgent:
    """
    This agent is called ONLY when intent in {'stay','food'}.
    It will never be called for other intents.
    """
    def __init__(self, serp_api_key: str, ll: str = MAGARPATTA_COORDINATES):
        self.serp_api_key = serp_api_key
        self.ll = ll

    def _search(self, query: str) -> Dict:
        if not _SERPAPI_AVAILABLE or not self.serp_api_key:
            return {}
        params = {
            "engine": "google_maps",
            "q": query,
            "ll": self.ll,
            "api_key": self.serp_api_key
        }
        try:
            search = GoogleSearch(params)  # type: ignore
            return search.get_dict() or {}
        except Exception:
            return {}

    def _format(self, items: List[Dict], header: str) -> Tuple[str, List[Dict]]:
        if not items:
            return ("Sorry, I couldn’t find results right now. Please try broader keywords.", [])
        lines = [header, ""]
        enriched = []
        for i, it in enumerate(items[:10], 1):
            title = it.get("title", "N/A")
            addr = it.get("address", "Address not available")
            phone = it.get("phone", "Phone not available")
            rating = it.get("rating", "No rating")
            reviews = it.get("reviews", 0)
            hours = it.get("hours", "Hours not available")
            maps_url = it.get("gps_coordinates", None)
            # Build a robust maps URL; if serp doesn’t give link, compose a search URL.
            url = it.get("link") or _maps_url_from(title, addr)

            lines.append(f"**{i}. {title}**")
            if rating != "No rating":
                lines.append(f"⭐ {rating}/5 ({reviews} reviews)")
            lines.append(f"📍 {addr}")
            lines.append(f"📞 {phone}")
            lines.append(f"🕒 {hours}")
            lines.append(f"🗺️ Google Maps: {url}")
            lines.append("—" * 40)

            enriched.append({
                "rank": i,
                "title": title,
                "address": addr,
                "phone": phone,
                "rating": rating,
                "reviews": reviews,
                "hours": hours,
                "google_maps_url": url
            })
        return "\n".join(lines), enriched

    def handle(self, intent: str, user_text: str) -> Tuple[str, Dict[str, Any]]:
        if intent == "stay":
            q = f"{user_text} PG hostel accommodation near Magarpatta City Pune"
            header = "🏠 **Stay options near Magarpatta City:**"
        else:
            q = f"{user_text} restaurant cafe food court near Magarpatta City Pune"
            header = "🍽️ **Food options near Magarpatta City:**"

        raw = self._search(q)
        local = raw.get("local_results") or []
        md, rows = self._format(local, header)
        return md, {"intent": intent, "used_serp": True, "results": rows}


# =========================
# HR info agent (defaults only; no Serp)
# =========================
class HRInfoAgent:
    RESPONSES = {
        "project_allocation": (
            "📋 **Project Allocation**\n\n"
            "HR will share the project allocation details with you soon.\n"
            "💡 For any urgency, please reach out to HR."
        ),
        "project_location": (
            "📍 **Project Location**\n\n"
            "Your project location and project details will be set with you soon.\n"
            "💡 HR will provide the exact workplace details shortly."
        )
    }
    def handle(self, intent: str) -> Tuple[str, Dict[str, Any]]:
        return self.RESPONSES[intent], {"intent": intent, "used_serp": False}


# =========================
# Fallback agent (NO Serp)
# =========================
class FallbackAgent:
    def handle(self) -> Tuple[str, Dict[str, Any]]:
        msg = (
            "I’m enabled for just these right now:\n"
            "• 🏠 Stay locations (PGs/Hostels/Accommodation)\n"
            "• 🍽️ Nearby food courts (Restaurants/Cafes)\n"
            "• 📋 Project allocation (default info)\n"
            "• 📍 Project location (default info)\n\n"
            "For other queries like **nearest school/college/hospital** I’m not enabled yet. "
            "Please use Google Maps or contact HR."
        )
        return msg, {"intent": "other", "used_serp": False}


# =========================
# New Onboarding Orchestrator
# =========================
@dataclass
class _AgenticSession:
    started: bool = False
    progress: int = 0

class _NewOnboardingAgentic:
    """
    Orchestrates intents. Critically:
    - ONLY 'stay' and 'food' call SerpAPI.
    - Project intents → default HR answers.
    - 'org_policy' → RAG (if provided) else a polite note.
    - Everything else → fallback (NO Serp).
    """
    def __init__(self):
        self._sessions: Dict[str, _AgenticSession] = {}
        self.intent = IntentAgent()
        self.serp = StayFoodAgent(SERP_API_KEY, MAGARPATTA_COORDINATES)
        self.hr = HRInfoAgent()
        self.fallback = FallbackAgent()
        self.styler = ResponseStyler()

    def _buttons(self):
        return [
            {"label": "🏠 Stay near Magarpatta", "send": "stay"},
            {"label": "🍽️ Food near Magarpatta", "send": "food"},
            {"label": "📋 Project allocation", "send": "project allocation"},
            {"label": "📍 Project location", "send": "project location"},
            {"label": "❓ Help / Menu", "send": "menu"},
        ]

    def start(self, session_id: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        st = self._sessions.get(session_id) or _AgenticSession()
        st.started, st.progress = True, 10
        self._sessions[session_id] = st
        first = (profile.get("full_name") or "there").split()[0]
        welcome = (
            f"Welcome to Hoonartek, {first}! 🎉\n"
            "I can help with:\n"
            "• Stay locations (PGs/Hostels/Accommodation)\n"
            "• Nearby food courts (Restaurants/Cafes)\n"
            "• Project allocation\n"
            "• Project location"
        )
        return welcome, {"agent": "new", "progress": st.progress, "actions": self._buttons()}

    def handle_turn(self, session_id: str, message: str, profile: Dict[str, Any], rag=None) -> Tuple[str, Dict[str, Any]]:
        st = self._sessions.get(session_id)
        if not st or not st.started:
            return self.start(session_id, profile)

        m = (message or "").strip()
        ml = m.lower()

        if ml in {"help", "menu", "options"}:
            help_text = (
                "🔍 **What I can do**\n"
                "• Type **stay** or describe an accommodation need.\n"
                "• Type **food** or describe cuisine/preferences.\n"
                "• Ask about **project allocation** or **project location**.\n"
                "• Organisation policy queries go to our RAG knowledge base."
            )
            return help_text, {"agent": "new", "progress": st.progress, "actions": self._buttons()}

        label = self.intent.classify(m)

        # STRICT GATE: Only stay/food -> SerpAPI
        if label in {"stay", "food"}:
            md, meta = self.serp.handle(label, m)
            return self.styler.style(md), {**meta, "agent": "new", "actions": self._buttons()}

        # Project info defaults
        if label in {"project_allocation", "project_location"}:
            md, meta = self.hr.handle(label)
            return self.styler.style(md), {**meta, "agent": "new", "actions": self._buttons()}

        # Org policy → RAG (if provided), else short note
        if label == "org_policy":
            if rag is not None:
                ans, rmeta = rag.answer_any(m)
                # Ensure consistent structure
                rmeta = rmeta or {}
                rmeta.update({"agent": "new", "intent": "org_policy", "used_serp": False})
                return ans, rmeta
            else:
                note = (
                    "I can route org-policy queries to our knowledge base, but it’s not connected right now. "
                    "Please attach the RAG agent, then ask again."
                )
                return note, {"agent": "new", "intent": "org_policy", "used_serp": False}

        # Everything else → fallback (NO Serp)
        md, meta = self.fallback.handle()
        return self.styler.style(md), {**meta, "agent": "new", "actions": self._buttons()}


# =========================
# PRE-ONBOARDING FLOW — UNCHANGED
# =========================
@dataclass
class PreState:
    step: int = 1
    progress: int = 0
    verified: bool = False
    who: Dict[str, Any] = field(default_factory=dict)
    phone: Optional[str] = None
    email_verified: bool = False
    otp_sent_sms: bool = False
    otp_sent_email: bool = False
    uploads: Dict[str, bool] = field(default_factory=lambda: {"pan": False, "aadhaar": False, "education": False})
    faq_open: bool = False
    feedback: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    rated: bool = False

class _PreOnboardingFlow:
    def _mk_actions_meta(self, actions):
        return {"actions": actions} if actions else {}
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self._df = self._load_excel()
        self._sessions: Dict[str, PreState] = {}

    def _load_excel(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.excel_path)
        except Exception:
            return pd.DataFrame()
        df.columns = [str(c).strip() for c in df.columns]
        for need in [COL_EMPLOYEE_ID, COL_EMPLOYEE_EMAIL, COL_DESIGNATION]:
            if need not in df.columns:
                return pd.DataFrame()
        df[COL_EMPLOYEE_EMAIL] = df[COL_EMPLOYEE_EMAIL].astype(str).str.lower().str.strip()
        df[COL_EMPLOYEE_ID] = df[COL_EMPLOYEE_ID].apply(_safe_str)
        return df

    def _lookup_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        if self._df.empty:
            return None
        rows = self._df[self._df[COL_EMPLOYEE_EMAIL] == (email or "").lower().strip()]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def start(self, session_id: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        st = self._sessions.get(session_id) or PreState()
        self._sessions[session_id] = st
        st.step = 1
        st.progress = 5
        first = (profile.get("full_name") or "there").split()[0]
        msg = (
            f"Welcome to Hoonartek, {first}! 🎉\n"
            "Let’s quickly verify your contact details."
            " Please share your **mobile number**."
        )
        return msg, {"progress": st.progress, "agent": "pre"}

    def handle_turn(self, session_id: str, message: str, profile: Dict[str, Any], rag=None) -> Tuple[str, Dict[str, Any]]:
        st = self._sessions.get(session_id)
        if not st:
            return self.start(session_id, profile)

        m = (message or "").strip()
        ml = m.lower()

        # ===== Step 1: Phone -> SMS OTP =====
        if st.step == 1:
            if st.phone is None:
                nums = re.findall(r"\d{10,}", ml)
                if not nums:
                    return "Please send your mobile number (e.g., `my mobile is 9876543210`).", {"progress": st.progress, "agent": "pre"}
                st.phone = nums[0]
                st.otp_sent_sms = True
                st.progress = 12
                return (f"OTP sent to **{st.phone}** (demo OTP: {DEMO_OTP}). Please reply `OTP 123456`.",
                        {"progress": st.progress, "agent": "pre"})

            # Verify SMS OTP
            if st.otp_sent_sms:
                code = None
                if ml.startswith("otp"):
                    code = _digits_only(ml.replace("otp", ""))
                elif ml.isdigit() and len(ml) in (4, 5, 6):
                    code = ml

                if code is None:
                    return "Please reply with the SMS OTP (e.g., `OTP 123456`).", {"progress": st.progress, "agent": "pre"}

                if _digits_only(code) == _digits_only(DEMO_OTP):
                    st.progress = 18
                    st.step = 2
                    return ("✅ Phone verified. Now share your **registered email** (e.g., `my email is name@domain.com`).",
                            {"progress": st.progress, "agent": "pre"})
                else:
                    return "That OTP doesn’t match. Please try again.", {"progress": st.progress, "agent": "pre"}

        # ===== Step 2: Email -> Email OTP =====
        if st.step == 2:
            if not st.otp_sent_email:
                if ml.startswith("my email is "):
                    email = _safe_str(m[len("my email is "):]).lower()
                elif _looks_like_email(m):
                    email = m.strip().lower()
                else:
                    email = None

                if not email:
                    return "Please share your **registered email** (e.g., `my email is name@domain.com`).", {"progress": st.progress, "agent": "pre"}

                st.otp_sent_email = True
                st.progress = 24
                st.who = self._lookup_by_email(email) or {
                    COL_EMPLOYEE_EMAIL: email,
                    COL_EMPLOYEE_NAME: profile.get("full_name") or "",
                    COL_DESIGNATION: "Freshers",
                }
                return (f"OTP sent to **{email}** (demo OTP: {DEMO_OTP}). Please reply `EMAIL OTP 123456`.",
                        {"progress": st.progress, "agent": "pre"})

            if st.otp_sent_email and not st.email_verified:
                code = None
                if ml.startswith("email otp"):
                    code = _digits_only(ml.replace("email otp", ""))
                elif ml.startswith("otp"):
                    code = _digits_only(ml.replace("otp", ""))
                elif ml.isdigit() and len(ml) in (4, 5, 6):
                    code = ml

                if code is None:
                    return "Please reply with the email OTP (e.g., `EMAIL OTP 123456`).", {"progress": st.progress, "agent": "pre"}

                if _digits_only(code) == _digits_only(DEMO_OTP):
                    st.email_verified = True
                    st.step = 4
                    st.progress = 75
                    msg = (
                        "✅ Email verified.\n\n"
                        f"Now please upload these 3 documents at {SHAREPOINT_UPLOAD_URL}:\n"
                        "• PAN\n"
                        "• Aadhaar\n"
                        "• Educational Certificates"
                    )
                    quick = [{"label": q, "send": q} for q in QUICK_FAQ]
                    quick.append({"label": "Move to Feedback", "send": "move to feedback"})
                    return msg, {"progress": st.progress, "actions": quick, "agent": "pre"}
                else:
                    return "That email OTP doesn’t match. Please try again.", {"progress": st.progress, "agent": "pre"}

        # ===== Step 3: (skipped uploads UI – link provided above) =====
        if st.step == 3:
            st.step = 4
            st.progress = 75
            msg = (
                f"Please upload your documents at {SHAREPOINT_UPLOAD_URL}:\n"
                "• PAN\n• Aadhaar\n• Educational Certificates\n\n"
                "Once done, you can browse the FAQs below or move to feedback."
            )
            quick = [{"label": q, "send": q} for q in QUICK_FAQ]
            quick.append({"label": "Move to Feedback", "send": "move to feedback"})
            return msg, {"progress": st.progress, "actions": quick, "agent": "pre"}

        # ===== Step 4: FAQs =====
        if st.step == 4:
            if ml in ("move to feedback", "feedback", "rate"):
                st.step = 5
                st.progress = 90
                return ("Before we wrap up, could you rate this pre-onboarding experience? (1–5)\n"
                        "You can also add a short comment after rating.",
                        {"progress": st.progress, "actions": [
                            {"label": "Rate 5 ⭐", "send": "rating 5"},
                            {"label": "Rate 4 ⭐", "send": "rating 4"},
                            {"label": "Rate 3 ⭐", "send": "rating 3"},
                        ], "agent": "pre"})
            faq_map = {
                "reporting location": "Your reporting location will be shared in your offer letter / joining mail. If unsure, please check with HR.",
                "dress code": "Smart casuals and business formals are acceptable, provided they are neat and professional in appearance. Business formal attire is required during client visits.",
                "working hours": "Standard working hours are typically 9 hours including breaks. Your manager may share team-specific timings.",
                "whom do i report to?": "You’ll receive your reporting manager details in your joining mail. If missing, HR will confirm before Day 1.",
            }
            for key, ans in faq_map.items():
                if key in ml:
                    return ans, {"progress": st.progress, "agent": "pre"}

            quick = [{"label": q, "send": q} for q in QUICK_FAQ]
            quick.append({"label": "Move to Feedback", "send": "move to feedback"})
            return ("Ask me about reporting location, dress code, working hours, or manager.",
                    {"progress": st.progress, "actions": quick, "agent": "pre"})

        # ===== Step 5: Feedback =====
        if st.step == 5:
            stars = None
            if ml.startswith("rating"):
                parts = ml.split()
                if len(parts) > 1 and parts[1].isdigit():
                    stars = int(parts[1])
            elif ml.isdigit():
                stars = int(ml)

            if stars is not None and 1 <= stars <= 5:
                st.feedback["rating"] = stars
                st.rated = True
                return "Thanks! Any short comment you’d like to add?", {"progress": st.progress, "agent": "pre"}

            if m:
                st.feedback["comment"] = m

            st.finished = True
            st.progress = 100
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/preonboarding_feedback.jsonl", "a", encoding="utf-8") as f:
                    rec = {"session": session_id, "who": st.who, "feedback": st.feedback, "ts": int(time.time())}
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

            reply = "🎉 Thank you! Pre-onboarding is complete. We’ll see you on Day 1."
            actions = [] if st.rated else [
                {"label": "Rate 5 ⭐", "send": "rating 5"},
                {"label": "Rate 4 ⭐", "send": "rating 4"},
                {"label": "Rate 3 ⭐", "send": "rating 3"},
            ]
            return reply, {"progress": st.progress, "done": True, "actions": actions, "agent": "pre"}

        return "Pre-onboarding complete.", {"progress": 100, "done": True, "agent": "pre"}


# =========================
# PUBLIC MERGED AGENT (RAG hook for BOTH routes)
# =========================
class OnboardingAgent:
    """
    Router:
      - If message is an organization policy query → send to RAG (both flows).
      - Else if designation contains 'fresh' → Pre-Onboarding
      - Else → New Onboarding (Agentic)
    """
    def __init__(self, excel_path: str = DEFAULT_EXCEL):
        self._pre = _PreOnboardingFlow(excel_path)
        self._new = _NewOnboardingAgentic()
        self._intent_for_router = IntentAgent()  # for global RAG routing

    def session_count(self) -> int:
        return len(self._pre._sessions) + len(self._new._sessions)

    def is_active(self, session_id: str) -> bool:
        return (session_id in self._pre._sessions) or (session_id in self._new._sessions)

    def _route(self, profile: Dict[str, Any]) -> str:
        desig = (profile.get("designation") or "").strip().lower()
        return "pre" if _is_fresher(desig) else "new"

    # ---- Conversation API ----
    def start(self, session_id: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if self._route(profile) == "pre":
            return self._pre.start(session_id, profile)
        return self._new.start(session_id, profile)

    def handle_turn(self, session_id: str, message: str, profile: Dict[str, Any], rag=None) -> Tuple[str, Dict[str, Any]]:
        # Global RAG routing for org-policy (works for BOTH pre and new)
        label = self._intent_for_router.classify(message or "")
        if label == "org_policy":
            if rag is not None:
                ans, meta = rag.answer_any(message)
                meta = meta or {}
                meta.update({"agent": self._route(profile), "intent": "org_policy", "used_serp": False})
                return ans, meta
            else:
                note = "Org-policy RAG is not connected. Please attach your RAG_Agent and ask again."
                return note, {"agent": self._route(profile), "intent": "org_policy", "used_serp": False}

        # Otherwise use normal routing
        if self._route(profile) == "pre":
            return self._pre.handle_turn(session_id, message, profile, rag=rag)
        return self._new.handle_turn(session_id, message, profile, rag=rag)

    def handle_button(self, session_id: str, action: str, value=None, profile=None, extra=None, rag=None):
        """
        UI button entrypoint.
        action: {"start","restart","select","upload","rag","menu"}
        """
        action = (action or "").strip().lower()
        if action in {"start", "restart"}:
            return self.start(session_id, profile or {})

        if action == "menu":
            return self.handle_turn(session_id, "menu", profile or {}, rag=rag)

        if action == "select":
            return self.handle_turn(session_id, str(value or ""), profile or {}, rag=rag)

        if action == "upload":
            # Pre-onboarding upload acknowledgement (kept for compat)
            kind = (extra or {}).get("doc_kind") or (value or "")
            return (f"✅ Noted. '{kind or 'document'}' marked as uploaded. Anything else?",
                    {"status": "ok", "doc": kind})

        if action == "rag":
            question = (value or "").strip() or "I have a question about HR/Org policy."
            if rag is not None:
                return rag.answer_any(question)
            return "RAG is not connected right now.", {"error": "rag_unavailable"}

        # default: treat as user turn
        return self.handle_turn(session_id, str(value or ""), profile or {}, rag=rag)