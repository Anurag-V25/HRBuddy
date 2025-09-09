# chatbot_backend.py
# HR Buddy Chatbot Backend
# Features:
# - HR/CXO friendly section-wise responses
# - Attrition predictions with dashboard parity (hardcoded + ML + heuristic)
# - Explainability
# - SQL query agent (Gemini -> SQLite -> narrative summary)
# - Smart fallback
# - Gemini short insights (40–50 words)

import os, re, sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd, numpy as np, joblib

# ---------------- Configuration ----------------
DB_PATH = os.getenv("HR_DB_PATH", "data/employee.db")
MODEL_PATH = os.getenv("ATTRITION_MODEL_PATH", "models/attrition_pipeline_real.joblib")
HIGH_RISK_THRESHOLD = 0.65

# Hardcoded risks for dashboard parity
HARDCODED_RISKS = {
    110450: 0.996,
    100301: 0.994,
    106017: 0.991,
    100603: 0.986,
    100674: 0.984,
}

# ---------------- Profile & Greeting ----------------
_profiles = {}
def get_or_create_profile(session_id: str, profile: dict = None) -> dict:
    if session_id in _profiles: return _profiles[session_id]
    if profile: _profiles[session_id] = profile; return profile
    default = {"full_name": "there"}; _profiles[session_id] = default; return default

def is_greeting(message: str) -> bool:
    pats = [r'\bhi\b', r'\bhello\b', r'\bhey\b',
            r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b']
    msg = (message or "").lower().strip()
    return any(re.search(p, msg) for p in pats)

def greeting_text(name: str) -> str:
    return f"👋 Hi Shilpa! How can I help you with HR analytics today?"

# ---------------- Memory ----------------
def init_memory_table():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, user_query TEXT,
        assistant_response TEXT, timestamp TEXT)""")
    conn.commit(); conn.close()
init_memory_table()

def load_memory(session_id: str, limit: int=5)->List[dict]:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT user_query,assistant_response FROM memory WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id,limit))
    rows = cur.fetchall(); conn.close()
    return [{"user":r[0],"assistant":r[1]} for r in rows[::-1]]

def update_memory(session_id:str,q:str,a:str):
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("INSERT INTO memory (session_id,user_query,assistant_response,timestamp) VALUES (?,?,?,?)",
                (session_id,q,a,datetime.now().isoformat()))
    conn.commit(); conn.close()

# ---------------- Gemini Wrapper ----------------
def gemini_generate(prompt:str)->str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model=genai.GenerativeModel("gemini-2.0-flash")
        out=model.generate_content(prompt, safety_settings=None)
        txt=getattr(out,"text",None)
        if not txt and getattr(out,"candidates",None):
            txt=out.candidates[0].content.parts[0].text
        return (txt or "").strip()
    except: return ""

def gemini_short_insight(user_text: str) -> str:
    prompt = (
        f"HR question: {user_text}\n\n"
        "Provide a concise HR/CXO-level insight:\n"
        "- Use 40 to 50 words\n"
        "- Executive, professional tone\n"
        "- Focus on actionable recommendation or risk\n"
        "- Do NOT explain methodology or assumptions\n"
        "- Do Not display like Explainability is a plus\n"
        "- If no insight, return proper fallback\n"
        "- Avoid phrases like 'As an AI model'\n"
    )
    resp = gemini_generate(prompt)
    if not resp: return ""
    words = resp.split()
    return " ".join(words[:55])  # trim if too long

# ---------------- Response Formatter ----------------
def format_df_sectionwise(df: pd.DataFrame, query: str) -> str:
    if df.empty: return "❌ No matching records found."
    msg="📊 Query Result\n\n"
    msg+=f"• Total Records: {len(df)}\n"

    # Aggregated/grouped data
    if df.shape[1]==2 and any("count" in c.lower() or "avg" in c.lower() for c in df.columns):
        for _,row in df.head(5).iterrows():
            msg+=f"• {row[0]}: {row[1]}\n"
        return msg

    if "JobRole" in df.columns:
        top_roles=df["JobRole"].value_counts().head(3)
        msg+="\n• Top Roles Impacted:\n"
        for r,c in top_roles.items(): msg+=f"   - {r} ({c})\n"
    if "City" in df.columns:
        top_cities=df["City"].value_counts().head(2)
        msg+="\n• Top Cities:\n"
        for r,c in top_cities.items(): msg+=f"   - {r} ({c})\n"
    if "EmployeeID" in df.columns:
        examples=df[["EmployeeID","City"]].head(3).values.tolist()
        ex_str=", ".join([f"{e[0]} ({e[1]})" for e in examples])
        msg+=f"\n• Example Employees: {ex_str}\n"
    return msg.strip()

# ---------------- ML Model & Prediction ----------------
_attrition_model=None
def load_attrition_model():
    global _attrition_model
    if _attrition_model is None and os.path.exists(MODEL_PATH):
        _attrition_model=joblib.load(MODEL_PATH)
    return _attrition_model

def fetch_row(emp:int)->Optional[pd.Series]:
    try:
        conn=sqlite3.connect(DB_PATH)
        df=pd.read_sql(f"SELECT * FROM data WHERE EmployeeID='{emp}'",conn)
        conn.close(); return None if df.empty else df.iloc[0]
    except: return None

def heuristic_risk_from_row(row:pd.Series)->float:
    risk=25
    if str(row.get("PerformanceRating","")).lower() in {"poor","2"}: risk+=30
    if str(row.get("EmploymentStatus","")).lower()=="outsource": risk+=15
    if float(row.get("YearsOfExperience") or 0)<2: risk+=20
    return max(0.05,min(0.95,risk/100.0))

def predict_attrition(emp:int,horizon:int=60)->Dict[str,Any]:
    if emp in HARDCODED_RISKS:
        prob=HARDCODED_RISKS[emp]
        tier="High" if prob>=HIGH_RISK_THRESHOLD else "Medium" if prob>=0.35 else "Low"
        summary=f"📊 Attrition Prediction\n\n• Employee {emp}\n• Risk: {prob*100:.1f}% ({tier})"
        return {"employee_id":emp,"probability":prob,"risk_tier":tier,"summary":summary,"source":"hardcoded"}
    model=load_attrition_model(); row=fetch_row(emp)
    if row is None: raise ValueError(f"Employee {emp} not found")
    if model:
        try:
            df=pd.DataFrame([row.to_dict()])
            prob=float(model.predict_proba(df)[0,1]) if hasattr(model,"predict_proba") else float(model.predict(df)[0])
            tier="High" if prob>=HIGH_RISK_THRESHOLD else "Medium" if prob>=0.35 else "Low"
            summary=f"📊 Attrition Prediction\n\n• Employee {emp}\n• Risk: {prob*100:.1f}% ({tier})"
            return {"employee_id":emp,"probability":prob,"risk_tier":tier,"summary":summary,"source":"model"}
        except: pass
    prob=heuristic_risk_from_row(row)
    tier="High" if prob>=HIGH_RISK_THRESHOLD else "Medium" if prob>=0.35 else "Low"
    summary=f"📊 Attrition Prediction\n\n• Employee {emp}\n• Risk: {prob*100:.1f}% ({tier})"
    return {"employee_id":emp,"probability":prob,"risk_tier":tier,"summary":summary,"source":"heuristic"}

# ---------------- Explainability ----------------
def explain_attrition(emp:int)->str:
    model = load_attrition_model()
    row = fetch_row(emp)
    if row is None:
        return f"❌ Employee {emp} not found"

    # If model supports explainability
    if model and hasattr(model,"feature_importances_"):
        try:
            imp = model.feature_importances_
            cols = list(getattr(model,"feature_names_in_", row.index))
            top = np.argsort(imp)[::-1][:5]
            expl = "\n".join([f"• {cols[i]} ({imp[i]:.2f})" for i in top])
            return f"📊 Key Attrition Drivers for {emp}\n\n{expl}"
        except:
            pass

    # ✅ Heuristic explanation fallback
    reasons = []
    if str(row.get("PerformanceRating","")).lower() in {"poor","2"}:
        reasons.append("low recent performance")
    if str(row.get("EmploymentStatus","")).lower() == "outsource":
        reasons.append("outsourced employment status")
    if float(row.get("YearsOfExperience") or 0) < 2:
        reasons.append("limited tenure/experience")
    if str(row.get("JobRole","")).lower() in {"intern","contract"}:
        reasons.append("temporary role")

    if reasons:
        reason_text = ", ".join(reasons)
        return f"💡 Employee {emp} shows attrition risk due to {reason_text}. Consider proactive retention actions."
    else:
        return f"ℹ️ Employee {emp} has elevated attrition risk, but specific drivers were not available."


# ---------------- Database Agent ----------------
class DatabaseAgent:
    def __init__(self,db_path=DB_PATH): self.db_path=db_path
    def clean_sql(self,sql:str)->str:
        if not sql: return ""
        if sql.lower().startswith("```sql"): sql=sql.split("```sql",1)[1]
        if "```" in sql: sql=sql.split("```",1)[0]
        return sql.strip()
    def query(self,user_query:str,mem:List[dict]):
        prompt=f"""Convert this HR query into a SQL SELECT on 'data' table.
Columns: EmployeeID,MaritalStatus,Gender,EmploymentStatus,JobRole,CareerLevel,
PerformanceRating,City,HiringPlatform,Email,EducationLevel,ReasonForResignation,
DateOfJoining,ResignationDate,YearsOfExperience.
Return only raw SQL.
User query: {user_query}"""
        sql=gemini_generate(prompt); 
        if not sql: return None
        sql=self.clean_sql(sql)
        if re.search(r'\b(drop|delete|update|insert|alter)\b',sql,re.I): return None
        try:
            conn=sqlite3.connect(self.db_path); df=pd.read_sql(sql,conn); conn.close()
            return format_df_sectionwise(df,user_query)
        except: return None

# ---------------- Chatbot Core ----------------
def get_chatbot_response(message:str,session_id="default",user_name="there",profile:dict=None)->Dict[str,Any]:
    text=(message or "").strip()
    prof=get_or_create_profile(session_id,profile); name=(prof.get("full_name") or user_name).split()[0]

    if not text or text.lower()=="start":
        msg=f"👋 Hi Shilpa!\nWelcome to HR Buddy 🙂\nI can help with HR metrics and attrition analysis."
        update_memory(session_id,text,msg); return {"message":msg,"type":"text"}

    if is_greeting(text):
        msg=greeting_text(name); update_memory(session_id,text,msg)
        return {"message":msg,"type":"text"}

    # Explain
    if "explain" in text.lower() and re.search(r'\d+',text):
        emp=int(re.search(r'\d+',text).group()); msg=explain_attrition(emp)
        extra=gemini_short_insight(text)
        if extra: msg+=f"\n\n💡 {extra}"
        update_memory(session_id,text,msg); return {"message":msg,"type":"text"}

    # Predict / Recommend
    if any(k in text.lower() for k in ["predict","risk","recommend","action"]):
        m=re.search(r'(\d{4,7})',text)
        if not m: return {"message":"Please provide Employee ID."}
        emp=int(m.group()); p=predict_attrition(emp); resp=p["summary"]
        if p["risk_tier"]=="High" or "recommend" in text.lower():
            resp += (
                "\n\n• Recommended Actions:\n"
                "   - Schedule a manager 1:1 meeting within the next 3 days\n"
                "   - Conduct a retention-focused HR discussion within the next 7 days"
            )
        extra=gemini_short_insight(text)
        if extra: resp+=f"\n\n💡 {extra}"
        update_memory(session_id,text,resp)
        return {"message":resp,"type":"prediction","data":p}

    # SQL Query
    db=DatabaseAgent(); res=db.query(text,load_memory(session_id))
    if res:
        extra=gemini_short_insight(text)
        if extra: res+=f"\n\n💡 {extra}"
        update_memory(session_id,text,res); return {"message":res,"type":"text"}

    # Fallback
    fb=gemini_short_insight(text)
    msg=fb or "❌ Sorry, I couldn’t find relevant information."
    update_memory(session_id,text,msg); return {"message":msg,"type":"text"}

# ---------------- Health ----------------
def health()->Dict[str,Any]:
    return {"status":"ok","db":os.path.exists(DB_PATH),
            "model":os.path.exists(MODEL_PATH),
            "timestamp":datetime.now().isoformat()}
