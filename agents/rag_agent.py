# agents/rag_agent.py
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz

# --- LLM (Gemini) ------------------------------------------------------------
import google.generativeai as genai

# --- Vector search -----------------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from textblob import TextBlob
import re

# ===========================
# Configuration / Constants
# ===========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

INDEX_DIR      = Path(os.getenv("HR_INDEX_DIR", "data/index")).resolve()
EMBED_MODEL    = os.getenv("HR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K          = int(os.getenv("RAG_TOP_K", "6"))          # per-variant fetch
N_VARIANTS     = int(os.getenv("RAG_QUERY_VARIANTS", "3")) # multi-query size
RRF_K          = int(os.getenv("RAG_RRF_K", "60"))         # RRF smoothing
MIN_COS_OK     = float(os.getenv("RAG_MIN_COS", "0.35"))   # cosine gate
MIN_LEN_OK     = int(os.getenv("RAG_MIN_LEN", "100"))      # ignore tiny chunks
MAX_CTX_CHARS  = int(os.getenv("RAG_MAX_CTX", "20000"))

FALLBACK = (
    "Sorry, I couldn't find that information in the provided documents.\n"
    "For more details, visit: https://hoonartek.com/ and https://hoonartek.keka.com/#/home/dashboard"
)

HR_PERSONA = (
    "You are ARIA, Hoonartek’s HR assistant. Your user is an IT employee. "
    "Be clear, friendly, and strictly policy‑accurate. Only use the given context."
    "Always try to answer HR-related queries first, especially if they mention policies like leave, dress code, attendance, probation, promotion, referral, travel, or WFH."
    "Only say 'not related' if the query is clearly outside HR (e.g., politics, sports, weather)."
    "Always provide a complete response that finishes sentences properly.\n"
    "Never copy raw paragraphs directly. Summarize in 3–4 clear, complete sentences.\n"
    "Do not stop midway. If content is long, summarize into short sentences.\n"
    "Respond in short, professional sentences (no bullets, no headings)."
    "If info is missing, say: 'Not specified in the retrieved documents.'"
)

def _maybe_init_gemini() -> bool:
    """Configure Gemini if the key is present. Return True if usable."""
    if not GEMINI_API_KEY:
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # quick smoke call that doesn't bill tokens (model construct)
        _ = genai.GenerativeModel(GEMINI_MODEL)
        return True
    except Exception:
        return False


# ===========================
# Helper utilities
# ===========================
def _to_cos_from_l2(d2: float) -> float:
    """If embeddings are normalized, FAISS L2 ~ 2(1-cos)."""
    try:
        return max(-1.0, min(1.0, 1.0 - (float(d2) / 2.0)))
    except Exception:
        return -1.0

def _dedupe_keep_order(docs: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for d in docs:
        key = (d.page_content[:160], str(d.metadata))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def _extractive_answer(question: str, context: str) -> str:
    """
    Tiny, dependency‑free fallback if Gemini is unavailable.
    We surface the most relevant paragraph by keyword overlap.
    """
    q_terms = {t.lower() for t in question.split() if len(t) > 2}
    best, best_score = "", 0
    for para in context.split("\n\n"):
        score = sum(1 for t in para.lower().split() if t in q_terms)
        if score > best_score:
            best, best_score = para.strip(), score
    return best or FALLBACK

def _summarize_with_gemini(question: str, context: str, gemini_ok: bool) -> str:
    if not gemini_ok:
        return _extractive_answer(question, context)

    prompt = (
        f"{HR_PERSONA}\n\n"
        "Answer the user’s question using only the CONTEXT below.\n"
        "If the answer is not clearly present, respond exactly with this line:\n"
        f"{FALLBACK}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text or FALLBACK
    except Exception:
        return FALLBACK


# ===========================
# RAG Agent
# ===========================
class RagAgent:
    """
    Advanced semantic RAG over HR PDFs only:
      - FAISS + normalized sentence‑transformers embeddings
      - Multi‑query expansion with Gemini (if available)
      - RRF fusion across variants
      - Cosine confidence gate (proactive fallback)
    """

    def __init__(self, index_dir: str = str(INDEX_DIR), embed_model: str = EMBED_MODEL):
        self.index_dir = Path(index_dir).resolve()
        self.embed_model = embed_model
        self.gemini_ok = _maybe_init_gemini()

        # embeddings must match the build_index.py config (normalized=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_dir}.\n"
                f"Build it with: python agents/build_index.py --source 'HR Documents' --out 'data/index'"
            )

        self.vdb = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # ---------- public surface ----------
    def health(self) -> Dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "embed_model": self.embed_model,
            "gemini": self.gemini_ok,
        }

    def answer_any(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Single entry point used by the backend; answers from HR docs."""
        return self._answer_hr(query)

    def answer_policy(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Kept for compatibility; same behavior as answer_any."""
        return self._answer_hr(query)

    # ---------- internals ----------
    def _expand_queries(self, q: str, n: int) -> List[str]:
        """Paraphrase query to widen recall; degrade gracefully if Gemini missing."""
        if not self.gemini_ok or n <= 1:
            return [q]

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            prompt = (
                "Rewrite the user's HR policy question into diverse but equivalent queries "
                f"(return exactly {n} lines; one query per line). Question:\n{q}"
            )
            resp = model.generate_content(prompt)
            lines = [ln.strip("-• ").strip() for ln in (resp.text or "").splitlines() if ln.strip()]
            out = []
            for s in lines:
                if s and s.lower() != q.lower():
                    out.append(s)
                if len(out) >= n:
                    break
            return [q] + out if out else [q]
        except Exception:
            return [q]

    def _rrf_fuse(self, ranked_lists: List[List[Tuple[Any, float]]], k: int) -> List[Tuple[Any, float]]:
        rrf_scores = defaultdict(float)
        id_to_doc, id_to_best_cos = {}, {}

        for rlist in ranked_lists:
            for rank, (doc, cos) in enumerate(rlist):
                did = id(doc)
                rrf_scores[did] += 1.0 / (RRF_K + rank)
                id_to_doc[did] = doc
                id_to_best_cos[did] = max(id_to_best_cos.get(did, -1), cos)

        fused = [(id_to_doc[did], rrf_scores[did], id_to_best_cos[did]) for did in rrf_scores]
        fused.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(doc, cos) for (doc, _score, cos) in fused[:k]]

    def _retrieve(self, query: str) -> List[Tuple[Any, float]]:
        variants = self._expand_queries(query, N_VARIANTS)
        per_variant: List[List[Tuple[Any, float]]] = []

        for q in variants:
            docs_scores = self.vdb.similarity_search_with_score(q, k=TOP_K)
            scored = []
            for doc, d2 in docs_scores:
                cos = _to_cos_from_l2(d2)
                scored.append((doc, cos))
            per_variant.append(scored)

        fused = self._rrf_fuse(per_variant, k=TOP_K * max(1, len(variants)))
        docs = _dedupe_keep_order([d for d, _ in fused])

        # keep best cosine for each doc id
        best_cos: Dict[int, float] = {}
        for rlist in per_variant:
            for d, cos in rlist:
                did = id(d)
                best_cos[did] = max(best_cos.get(did, -1), cos)

        return [(d, best_cos.get(id(d), -1)) for d in docs]

    COMMON_MISSPELLINGS = {
        "passrwd": "password",
        "intall": "install",
        "emial": "email",
        "onbording": "onboarding",
        "benifits": "benefits",
        "attandance": "attendance",
        "referal": "referral",
        "promtion": "promotion",
        "leavs": "leaves",
        # Add more as needed
    }

    def correct_common_typos(self, text: str) -> str:
        words = text.split()
        corrected = [self.COMMON_MISSPELLINGS.get(w.lower(), w) for w in words]
        return " ".join(corrected)

    def _answer_hr(self, query: str) -> Tuple[str, Dict[str, Any]]:
        # First, correct common HR/IT typos
        query = self.correct_common_typos(query)
        # Then, autocorrect with TextBlob for general spelling
        corrected_query = str(TextBlob(query).correct())
        q = (corrected_query or "").strip()
        if not q:
            return "Please type your question.", {}

        # Quick replies handling
        if q.lower() in {"yes", "y", "ask another", "continue"}:
            return "Great — go ahead and ask your next question.", {"awaiting_question": True}
        if q.lower() in {"move to feedback", "feedback"}:
            return "Sure, opening the feedback card…", {"handoff": "feedback"}

        # Small talk should NOT show follow‑up buttons
        small = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        if q.lower() in small:
            return ("Hello! 👋 I'm HR Buddy. Please type your question about Hoonartek, "
                    "our HR policies, or company culture."), {}

        # First time init (from UI)
        if q == "__init__":
            greet = (
                "👋 Welcome to HR Buddy!\n\n"
                "I'm here to help you with any questions about Hoonartek, our company culture, or HR policies.\n\n"
                "Type your question below to get started!"
            )
            return greet, {}

        # Detect non-HR queries and return a polite fallback
        if self._detect_policy_topic(q) is None:
            return ("It seems your question isn't related to Hoonartek's HR policies or procedures. "
                    "If you have a question about our company, policies, or benefits, feel free to ask!"), {"followup_offered": False}

        # Retrieve and confidence‑gate
        hits = self._retrieve(q)
        strong = [
            (d, cos) for d, cos in hits
            if cos >= MIN_COS_OK and len((d.page_content or "").strip()) >= MIN_LEN_OK
        ]
        if not strong:
            return FALLBACK, {"followup_offered": False}

        # Build context (budgeted)
        parts, total = [], 0
        for d, _cos in strong:
            txt = (d.page_content or "").strip()
            if not txt:
                continue
            if total + len(txt) > MAX_CTX_CHARS:
                break
            parts.append(txt)
            total += len(txt)

        if not parts:
            return FALLBACK, {"followup_offered": False}

        context = "\n\n".join(parts)
        answer = _summarize_with_gemini(q, context, self.gemini_ok)

        # Remove fallback if real answer is present
        if answer.strip() != FALLBACK.strip() and FALLBACK.strip() in answer:
            answer = answer.replace(FALLBACK.strip(), "").strip()

        # Only append follow‑up when we returned a *proper* answer (not fallback)
        if answer.strip() != FALLBACK.strip():
            answer = f"{answer}\n\nSource: https://hoonartek.keka.com/#/org/documents/org/folder/414"
            return answer, {"followup_offered": True}

        return answer, {"followup_offered": False}

    def _detect_policy_topic(self, user_query: str) -> Optional[str]:
        q = (user_query or "").lower()

        topics = {
            "leave": ["leave", "vacation", "holiday", "paid time off", "pto", "earned leave", "sick leave", "casual leave", "annual leave"],
            "dress_code": ["dress code", "attire", "clothing", "uniform", "grooming", "appearance", "business casual", "formal wear", "what to wear"],
            "attendance": ["attendance", "timesheet", "absent", "working hours", "office hours", "shift timing", "punch in", "punch out"],
            "probation": ["probation", "confirmation", "probation period"],
            "promotion": ["promotion", "promote", "career growth", "progression"],
            "referral": ["referral", "employee referral", "referral bonus", "refer a friend"],
            "wfh": ["wfh", "work from home", "remote work", "hybrid work", "telecommute", "flexible work"],
            "travel_expense": ["travel", "expense", "reimbursement", "business trip", "claim", "travel allowance", "ta da", "lodging", "conveyance"],
        }

        # Exact/regex match first
        for key, kws in topics.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", q):
                    return key

        # Fuzzy match fallback
        for key, kws in topics.items():
            for kw in kws:
                if fuzz.partial_ratio(q, kw) > 80:
                    return key

        return None
