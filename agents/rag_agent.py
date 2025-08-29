# agents/rag_agent.py
# RAG agent with:
# - Structured policy formatting
# - Smart "detail mode" (no Policy Summary for narrow questions)
# - Single Keka Source link appended to every answer
# - Clean context (no "Document N / Relevance ...")
# - Output compaction to reduce visual gaps

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# LLM / Embeddings / Vector DB
# ---------------------------
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------
# Config / Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EnhancedRAG")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

INDEX_DIR = Path(os.getenv("HR_INDEX_DIR", "data/index")).resolve()
EMBED_MODEL = os.getenv("HR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("RAG_TOP_K", "8"))
MIN_COS_OK = float(os.getenv("RAG_MIN_COS", "0.35"))
MIN_LEN_OK = int(os.getenv("RAG_MIN_LEN", "100"))
MAX_CTX_CHARS = int(os.getenv("RAG_MAX_CTX", "24000"))

HR_DOCS_LINK = "https://hoonartek.keka.com/#/org/documents/org"

# ---------------------------
# Types
# ---------------------------
class QueryType(Enum):
    POLICY_INQUIRY = "policy_inquiry"
    PROCEDURE_REQUEST = "procedure_request"
    BENEFITS_INFO = "benefits_info"
    COMPLIANCE_CHECK = "compliance_check"
    TECHNICAL_POLICY = "technical_policy"
    CAREER_DEVELOPMENT = "career_development"
    GENERAL_HR = "general_hr"

class UserRole(Enum):
    JUNIOR_DEVELOPER = "junior_developer"
    SENIOR_DEVELOPER = "senior_developer"
    TEAM_LEAD = "team_lead"
    ARCHITECT = "architect"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    PRODUCT_MANAGER = "product_manager"
    INTERN = "intern"

@dataclass
class UserProfile:
    user_id: str = "anonymous"
    role: UserRole = UserRole.JUNIOR_DEVELOPER
    department: str = "IT"
    experience_years: int = 1
    seniority_level: str = "junior"
    specializations: List[str] = None
    current_projects: List[str] = None
    learning_goals: List[str] = None
    preferred_communication_style: str = "concise"

@dataclass
class ConversationContext:
    session_id: str
    conversation_history: List[Dict[str, Any]]
    current_topic: Optional[str]
    query_intent: Optional[QueryType]
    unresolved_issues: List[str]
    satisfaction_score: Optional[float]
    timestamp: datetime

# ---------------------------
# Prompt Manager
# ---------------------------
class RoleBasedPromptManager:
    def __init__(self):
        self.templates = self._templates()

    def _templates(self) -> Dict[QueryType, str]:
        return {
            QueryType.POLICY_INQUIRY: "You are ARIA, Hoonartek's HR Policy assistant.",
            QueryType.PROCEDURE_REQUEST: "You are ARIA, a precise process guide.",
            QueryType.BENEFITS_INFO: "You are ARIA, a benefits specialist.",
            QueryType.COMPLIANCE_CHECK: "You are ARIA, a compliance advisor.",
            QueryType.TECHNICAL_POLICY: "You are ARIA, a technical policy advisor.",
            QueryType.CAREER_DEVELOPMENT: "You are ARIA, a career development advisor.",
            QueryType.GENERAL_HR: "You are ARIA, a friendly HR assistant.",
        }

    # ------ Topic / detail detection ------
    def _detect_policy_topic(self, user_query: str) -> Optional[str]:
        q = (user_query or "").lower()
        topics = {
            "leave": ["leave", "leaves", "vacation", "pto"],
            "dress_code": ["dress code", "attire", "clothing"],
            "attendance": ["attendance", "late", "absent", "timesheet"],
            "probation": ["probation", "confirmation"],
            "promotion": ["promotion", "promote", "nomination"],
            "referral": ["referral", "refer", "recommendation program"],
            "wfh": ["wfh", "work from home", "remote work", "hybrid"],
            "travel_expense": ["travel", "expense", "reimbursement", "ta", "da"],
        }
        for key, kws in topics.items():
            if any(kw in q for kw in kws):
                return key
        return None

    def _detect_detail_slot(self, user_query: str) -> Optional[str]:
        """Return a specific policy slot if the question is narrow, else None."""
        q = (user_query or "").lower()

        tests = [
            ("accrual", r"\b(accrue|accrual|per\s*month|monthly|1\.75|\d+(\.\d+)?\s*days\s*/\s*month)\b"),
            ("eligibility", r"\b(eligib|new\s+joiners?|after\s+\d+\s+(months?|days?)|probation)\b"),
            ("entitlement", r"\b(how\s+many|entitled|days\s+per\s+year|max(?:imum)?\s+\d+|21\s+paid)\b"),
            ("approval", r"\b(approve|approval|workflow|who\s+approves|manager\s+approval|hr\s+approval|sla)\b"),
            ("carry_forward", r"\b(carry[-\s]?forward|carryover|lapse|encash(?:ment)?)\b"),
            ("how_to_apply", r"\b(how\s+to\s+apply|apply\s+in\s+keka|keka\s+steps|application\s+steps)\b"),
        ]
        for slot, rx in tests:
            if re.search(rx, q):
                return slot
        return None

    def _slot_heading(self, topic: Optional[str], slot: str) -> str:
        mapping = {
            "entitlement": "### Entitlement (days/year)",
            "eligibility": "### Eligibility (new joiners / tenure)",
            "accrual": "### Accrual (rate, proration)",
            "approval": "### Approval Workflow (who approves, SLA)",
            "carry_forward": "### Lapse / Carry Forward / Encashment",
            "how_to_apply": "### How to Apply (Keka steps)",
        }
        return mapping.get(slot, "### Details")

    # ------ Structured templates (full vs detail) ------
    def _structured_template_for(self, topic: Optional[str], detail_slot: Optional[str] = None) -> str:
        """
        Returns a section template.
        - If detail_slot is set, we only show that ONE section (no Policy Summary).
        - Otherwise we return the full policy layout for that topic.
        Any missing facts must be written as: 'Not specified in the retrieved documents.'
        """
        # Detail mode → single section only, concise
        if detail_slot:
            return (
                "Answer concisely (3–6 lines) for the specific question.\n"
                "Do NOT include a 'Policy Summary' or unrelated sections.\n"
                f"Include ONLY this heading:\n{self._slot_heading(topic, detail_slot)}\n"
                "If some detail is missing in the references, write: 'Not specified in the retrieved documents.'"
            )

        # Full layouts (topic-specific)
        if topic == "leave":
            return (
                "FORMAT YOUR ANSWER USING THESE HEADINGS (Markdown, no extra text outside sections):\n"
                "### Policy Summary\n"
                "### Entitlement (days/year)\n"
                "### Eligibility (new joiners / tenure)\n"
                "### Accrual (rate, proration)\n"
                "### Usage Rules (blackout, minimum usage, carry-forward)\n"
                "### Approval Workflow (who approves, SLA)\n"
                "### Lapse / Carry Forward / Encashment\n"
                "### How to Apply (Keka steps)\n"
                "### Notes / Exceptions\n"
                "For any item not present in the context, write: 'Not specified in the retrieved documents.'"
            )
        if topic == "dress_code":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Acceptable Attire\n"
                "### Not Allowed / Restrictions\n"
                "### Client Site Guidance\n"
                "### Enforcement / Consequences\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "attendance":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Working Hours / Shifts\n"
                "### Late / Absence Rules\n"
                "### Attendance Marking (system/process)\n"
                "### Escalation / Exceptions\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "probation":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Probation Duration\n"
                "### Confirmation Criteria\n"
                "### Extension Rules\n"
                "### Process / Approvals\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "promotion":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Eligibility & Nomination\n"
                "### Assessment (technical, HR)\n"
                "### Decision & Communication\n"
                "### Timelines\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "referral":
            return (
                "FORMAT (Markdown):\n"
                "### Program Summary\n"
                "### Who Can Refer\n"
                "### How to Refer (steps)\n"
                "### Rewards / Payouts\n"
                "### Rules / Restrictions\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "wfh":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Eligibility\n"
                "### Request / Approval Flow\n"
                "### Expectations (availability, security)\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )
        if topic == "travel_expense":
            return (
                "FORMAT (Markdown):\n"
                "### Policy Summary\n"
                "### Eligible Expenses\n"
                "### Limits / Approvals\n"
                "### Claim Process (steps)\n"
                "### Timlines & Documentation\n"
                "### Notes\n"
                "Use 'Not specified in the retrieved documents.' where applicable."
            )

        # Generic structure fallback
        return (
            "If the content appears to describe a policy, structure it using clear Markdown headings and short bullet points.\n"
            "If it is general guidance, a brief paragraph followed by concise bullets is fine."
        )

    # ------ Prompt builder ------
    def create_prompt(
        self,
        query_type: QueryType,
        user_profile: Dict[str, Any],
        memory: Dict[str, Any],
        context_docs: str,
        user_query: str,
    ) -> str:
        role_obj = user_profile.get("role", "junior_developer")
        if hasattr(role_obj, "value"):
            role_obj = role_obj.value
        role_str = str(role_obj).replace("_", " ").title()

        recent = memory.get("recent_messages", [])
        topic = self._detect_policy_topic(user_query)
        detail_slot = self._detect_detail_slot(user_query)
        structure_rules = self._structured_template_for(topic, detail_slot)

        sys_rules = (
            "SYSTEM:\n"
            "- You are ARIA, Hoonartek's HR assistant.\n"
            "- Write a clear, professional answer in Markdown.\n"
            "- If the question is narrow, answer ONLY that section (no Policy Summary, no unrelated sections).\n"
            "- NEVER copy or list raw reference headers or scores from the context.\n"
            "- Do NOT print lines like 'Document 1', 'Relevance', or 'SOURCE [id]'.\n"
            "- Do not output any JSON. Do not invent sources.\n"
            "- If something isn't in the docs, explicitly say: 'Not specified in the retrieved documents.'\n"
        )

        recent_section = ""
        if recent:
            last = recent[-6:]
            recent_section = "RECENT_TURNS (for grounding; do not copy):\n" + "\n".join(
                f"{m.get('role','user')}: {m.get('content','')[:300].replace('\n', ' ')}" for m in last
            ) + "\n"

        template = self.templates.get(query_type, self.templates[QueryType.GENERAL_HR])

        # Detail-mode banner (strong nudge)
        detail_banner = ""
        if detail_slot:
            detail_banner = (
                "\nDETAIL MODE:\n"
                "- The user asked a specific sub-question.\n"
                "- Write a short, direct answer with ONLY the single required heading.\n"
                "- Do NOT include an overview or other sections.\n"
            )

        prompt = "\n\n".join([
            sys_rules,
            template,
            f"USER_ROLE: {role_str}",
            recent_section.strip(),
            "REFERENCE MATERIAL (do not copy headers/scores):\n" + (context_docs[:MAX_CTX_CHARS] if context_docs else "None"),
            f"\nUSER_QUESTION:\n{user_query}",
            detail_banner,
            "\nANSWER SHAPE REQUIREMENTS:\n" + structure_rules,
            "\nINSTRUCTIONS:\n"
            "- Write only the final answer (no extra disclaimers, no JSON).\n"
            "- Do NOT list reference items or relevance scores.\n"
        ])
        return prompt

# ---------------------------
# Enhanced RAG Agent
# ---------------------------
class EnhancedRagAgent:
    def __init__(self, index_dir: str = str(INDEX_DIR), embed_model: str = EMBED_MODEL):
        self.index_dir = Path(index_dir).resolve()
        if not self.index_dir.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_dir}")

        # embeddings + vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vdb = FAISS.load_local(str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True)

        # LLM for answering
        self.gemini_ok = False
        self._init_llm()

        # prompt manager + session memory
        self.prompt = RoleBasedPromptManager()
        self.active_sessions: Dict[str, ConversationContext] = {}

    def _init_llm(self):
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_ok = True
            except Exception as e:
                log.error(f"Gemini init failed: {e}")
                self.gemini_ok = False
        else:
            log.info("No GEMINI_API_KEY provided; LLM disabled.")

    # ---- Public API ----
    def answer_with_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        response_format: str = "markdown",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Canonical entrypoint used by main.py.
        """
        session_id = session_id or f"anon:{int(datetime.utcnow().timestamp())}"
        ctx = self._get_or_create_session(session_id)

        # Retrieve relevant docs
        resolved_query = query
        retrieved = self._retrieve(resolved_query)
        if not retrieved:
            fallback = self._fallback_response(resolved_query)
            final = self._finalize_for_ui(fallback)
            self._append_history(ctx, "user", query)
            self._append_history(ctx, "assistant", final)
            return final, {"top_sources": [], "memory_used": True}

        memory_block = {"recent_messages": ctx.conversation_history[-10:]}
        context_block = self._format_context(retrieved, memory_block)

        # Create strong prompt (query type set to POLICY_INQUIRY; template logic handles narrow/detail cases)
        user_profile = asdict(UserProfile())
        user_profile["role"] = "employee"
        prompt = self.prompt.create_prompt(QueryType.POLICY_INQUIRY, user_profile, memory_block, context_block, resolved_query)

        # Generate with Gemini
        response_text = self._generate_gemini(prompt)

        # Append the HR docs link at the very end (exactly one line)
        response_text = response_text.rstrip() + f"\n\nSource: {HR_DOCS_LINK}"

        # Tidy text for UI (gap reduction included)
        clean = self._finalize_for_ui(response_text)

        # remember turn
        self._append_history(ctx, "user", query)
        self._append_history(ctx, "assistant", clean)

        meta = {"top_sources": [], "memory_used": True}
        if response_format == "json":
            return json.dumps({"answer": clean, "meta": meta}, indent=2), meta
        return clean, meta

    # Backwards-compat shim
    def answer_any(
        self,
        text: str,
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        response_format: str = "markdown",
    ) -> Tuple[str, Dict[str, Any]]:
        return self.answer_with_context(text, session_id=session_id, feedback=feedback, response_format=response_format)

    # ---- Retrieval helpers ----
    def _retrieve(self, query: str) -> List[Tuple[Any, float]]:
        try:
            docs_scores = self.vdb.similarity_search_with_score(query, k=TOP_K)
        except Exception as e:
            log.warning(f"FAISS similarity_search failed: {e}")
            return []
        out = []
        for doc, d2 in docs_scores:
            content = getattr(doc, "page_content", "") or ""
            if len(content) < MIN_LEN_OK:
                continue
            # Convert L2 distance proxy to ~cosine-ish score
            try:
                cos = 1.0 - (float(d2) / 2.0)
            except Exception:
                cos = 0.0
            if cos >= MIN_COS_OK:
                out.append((doc, cos))
        return out

    def _format_context(self, retrieved_docs: List[Tuple[Any, float]], memory_block: Dict[str, Any]) -> str:
        parts = []
        # recent conversation (not to be copied)
        recent = memory_block.get("recent_messages") or []
        if recent:
            parts.append("=== RECENT TURNS (for grounding; do not copy) ===")
            for m in recent[-6:]:
                parts.append(f"{(m.get('role') or 'user').upper()}: {m.get('content','')[:300].replace('\n',' ')}")

        parts.append("\n=== REFERENCES (do not copy headers) ===")
        total = 0
        for i, (doc, _score) in enumerate(retrieved_docs):
            meta = getattr(doc, "metadata", {}) or {}
            doc_id = meta.get("id") or meta.get("source") or f"doc_{i+1}"
            content = getattr(doc, "page_content", "") or ""
            header = f"\n--- SOURCE [{doc_id}] ---\n"  # no numbering, no scores
            chunk = header + content
            if total + len(chunk) > MAX_CTX_CHARS:
                remain = MAX_CTX_CHARS - total
                if remain > 200:
                    chunk = header + content[: remain - 50] + "..."
                else:
                    break
            parts.append(chunk)
            total += len(chunk)
        return "\n".join(parts)

    # ---- LLM wrapper ----
    def _generate_gemini(self, prompt: str) -> str:
        if not self.gemini_ok:
            return "Sorry, my language model backend is unavailable right now."
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            res = model.generate_content(prompt)
            return (res.text or "").strip()
        except Exception as e:
            log.error(f"Gemini generation failed: {e}")
            return "I hit an error while generating the response. Please try again."

    # ---- Utilities / Output hygiene ----
    def _strip_trailing_json(self, text: str) -> str:
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

    def _strip_inline_json_lines(self, text: str) -> str:
        """Remove any lines that look like standalone JSON objects."""
        if not text:
            return text
        cleaned = []
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    json.loads(s)
                    continue
                except Exception:
                    pass
            cleaned.append(ln)
        return "\n".join(cleaned).strip()

    def _strip_raw_source_lines(self, text: str) -> str:
        """Remove stray 'Sources:' / 'Source 1:' / 'Document 1, Relevance:' lines the model might echo."""
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
            if re.match(r"^\s*\d+\.\s*Document\s+\d+\s*,?\s*Relevance\s*:", lns, flags=re.IGNORECASE):
                continue
            if re.match(r"^\s*Document\s+\d+\s*[(:]", lns, flags=re.IGNORECASE):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned).strip()

    def _compact_newlines(self, text: str) -> str:
        """
        Reduce visual gaps:
        - Collapse 3+ newlines to 2
        - Remove blank line immediately after a heading
        - Keep lists tight (no extra blank lines between bullets)
        """
        if not text:
            return text

        # 1) collapse 3+ consecutive newlines to exactly 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 2) remove blank line directly after a Markdown heading
        #    e.g., "### Title\n\n- item" -> "### Title\n- item"
        text = re.sub(r"(?m)^(#{1,6}\s.+?)\n\s*\n", r"\1\n", text)

        # 3) avoid blank lines between list items accidentally produced
        #    e.g., "- a\n\n- b" -> "- a\n- b"
        text = re.sub(r"(?m)^(\s*[-*]\s.+)\n\s*\n(\s*[-*]\s)", r"\1\n\2", text)

        return text.strip()

    def _finalize_for_ui(self, raw_text: str) -> str:
        """Final user-facing Markdown cleanup (also reduces spacing)."""
        text = (raw_text or "").strip()
        text = self._strip_trailing_json(text)
        text = self._strip_inline_json_lines(text)
        text = self._strip_raw_source_lines(text)
        text = self._compact_newlines(text)
        return text.strip()

    def _fallback_response(self, query: str) -> str:
        msg = (
            "I couldn't find this in our HR documents. "
            "Please contact HR or your BU HR for confirmation."
        )
        return msg + f"\n\nSource: {HR_DOCS_LINK}"

    def _append_history(self, ctx: ConversationContext, role: str, content: str):
        ctx.conversation_history.append({
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat()
        })
        ctx.timestamp = datetime.utcnow()

    def _get_or_create_session(self, session_id: str) -> ConversationContext:
        ctx = self.active_sessions.get(session_id)
        if ctx:
            return ctx
        ctx = ConversationContext(
            session_id=session_id,
            conversation_history=[],
            current_topic=None,
            query_intent=None,
            unresolved_issues=[],
            satisfaction_score=None,
            timestamp=datetime.utcnow(),
        )
        self.active_sessions[session_id] = ctx
        return ctx

# ---------------------------
# Factory
# ---------------------------
def create_enhanced_rag_agent() -> EnhancedRagAgent:
    return EnhancedRagAgent()
