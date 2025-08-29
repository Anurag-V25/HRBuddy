# chatbot_backend.py — Enhanced HR Buddy with Direct CSV Analysis & Gemini Intelligence

import os, re, sqlite3, logging, math, json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import random

# External dependencies
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logging.warning("DuckDuckGo search not available")

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logging.warning("LangGraph not available - using fallback")

# ---------------- Enhanced Logging ----------------
log = logging.getLogger("hr_buddy")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

DB_PATH = os.getenv("HR_DB_PATH", "data/processed/hr_data.db")
CSV_PATH = os.getenv("HR_CSV_PATH", "data/Indian_IT_Employee_Dataset_updated.csv")
MAX_ROWS = 500

# ---------------- GREETING HANDLING FUNCTIONS ----------------
def is_greeting(message: str) -> bool:
    """Check if the message is a simple greeting"""
    greeting_patterns = [
        r'\b(hi|hello|hey|hola)\b',
        r'\b(good morning|good afternoon|good evening)\b',
        r'\b(how are you|how do you do)\b',
        r'\b(greetings|salutations)\b'
    ]
    
    message_lower = message.lower().strip()
    
    # Check if message matches greeting patterns and is short (less than 20 characters)
    if len(message_lower) <= 20:
        for pattern in greeting_patterns:
            if re.search(pattern, message_lower):
                return True
    return False

def get_greeting_response() -> str:
    """Return a short greeting response"""
    greetings = [
        "Hi! How can I help you with your HR analytics today?",
        "Hello! I'm here to assist with your employee data insights.",
        "Good day! What HR metrics would you like to explore?",
        "Hi there! Ready to dive into your workforce analytics?",
        "Hello! How can I support your HR decisions today?"
    ]
    return random.choice(greetings)

# ---------------- HEALTH CHECK FUNCTION ----------------
def health() -> Dict[str, Any]:
    """Health check endpoint for the chatbot service"""
    try:
        db_status = "healthy"
        csv_status = "healthy"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            db_tables = len(tables_df)
            conn.close()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
            db_tables = 0
            
        try:
            df = pd.read_csv(CSV_PATH)
            csv_records = len(df)
        except Exception as e:
            csv_status = f"unhealthy: {str(e)}"
            csv_records = 0
            
        gemini_status = "available" if os.getenv("GOOGLE_API_KEY") else "missing_api_key"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": {"status": db_status, "tables_count": db_tables},
            "csv_data": {"status": csv_status, "records_count": csv_records},
            "gemini_api": gemini_status,
            "dependencies": {"duckduckgo_search": HAS_DDGS, "langgraph": HAS_LANGGRAPH},
            "version": "3.0.0-enhanced"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

# ---------------- Response Envelope ----------------
@dataclass
class ChatbotResponse:
    message: str
    response_type: str = "text"  # 'text' | 'prediction' | 'action' | 'external'
    data: Optional[dict] = None
    confidence: float = 0.9

def pack(resp: ChatbotResponse) -> Dict[str, Any]:
    return {
        "message": resp.message,
        "response_type": resp.response_type,
        "data": resp.data,
        "confidence": resp.confidence,
        "timestamp": datetime.now().isoformat(),
    }

# ---------------- Enhanced CSV Analytics Agent ----------------
class EnhancedCSVAnalyticsAgent:
    def __init__(self):
        self.df = None
        self._load_csv()

    def _load_csv(self):
        """Load and prepare CSV data"""
        try:
            self.df = pd.read_csv(CSV_PATH)
            log.info(f"Loaded CSV with {len(self.df)} employee records")
            # Prepare date columns
            self.df['DateOfJoining'] = pd.to_datetime(self.df['DateOfJoining'], errors='coerce')
            self.df['ResignationDate'] = pd.to_datetime(self.df['ResignationDate'], errors='coerce')
        except Exception as e:
            log.error(f"Failed to load CSV: {e}")
            self.df = pd.DataFrame()

    def analyze_query(self, query: str) -> str:
        """Enhanced query analysis with Gemini intelligence"""
        if self.df is None or self.df.empty:
            return "Employee data not available for analysis."
        
        query_lower = query.lower()
        
        # Use Gemini to understand query intent
        intent_prompt = f"""
        You are an HR analytics expert. Analyze this query: "{query}"
        
        Identify if the user wants:
        1. Attrition/turnover rates
        2. Salary/compensation analysis
        3. Employee counts/demographics
        4. Department-wise analysis
        5. Performance analysis
        6. Joining trends
        7. Other HR metrics
        
        Respond with just the category number and key fields needed.
        """
        
        intent = self._get_gemini_response(intent_prompt)
        
        try:
            # Hardcoded analysis patterns
            if any(word in query_lower for word in ["attrition", "turnover", "resignation", "quit", "left"]):
                return self._analyze_attrition(query)
            elif any(word in query_lower for word in ["salary", "compensation", "pay", "earning"]):
                return self._analyze_salary(query)
            elif any(word in query_lower for word in ["headcount", "count", "total", "number"]):
                return self._analyze_headcount(query)
            elif any(word in query_lower for word in ["department", "dept", "team"]):
                return self._analyze_by_department(query)
            elif any(word in query_lower for word in ["performance", "rating", "excellent", "poor"]):
                return self._analyze_performance(query)
            elif any(word in query_lower for word in ["join", "hired", "onboard", "new"]):
                return self._analyze_joining_trends(query)
            elif any(word in query_lower for word in ["engagement", "satisfaction", "retention"]):
                return self._analyze_engagement(query)
            else:
                return self._general_analytics_summary()
                
        except Exception as e:
            log.error(f"Analysis error: {e}")
            return "Unable to analyze the data at this time."

    def _analyze_attrition(self, query: str) -> str:
        """Analyze attrition and turnover"""
        total_employees = len(self.df)
        resigned_employees = len(self.df[self.df['ReasonForResignation'] != 'Still Working'])
        
        if total_employees == 0:
            return "No employee data available for attrition analysis."
        
        attrition_rate = (resigned_employees / total_employees) * 100
        
        # Top resignation reasons
        resignation_reasons = self.df[self.df['ReasonForResignation'] != 'Still Working']['ReasonForResignation'].value_counts().head(3)
        top_reasons = ", ".join([f"{reason} ({count} employees)" for reason, count in resignation_reasons.items()])
        
        base_response = f"""
**Attrition Analysis:**
• Overall attrition rate: {attrition_rate:.1f}%
• Total resigned employees: {resigned_employees} out of {total_employees}
• Top resignation reasons: {top_reasons}

**Strategic Insights:**
Current attrition levels {'exceed' if attrition_rate > 15 else 'are within'} typical IT industry benchmarks of 12-15%.
        """
        
        return self._enhance_with_gemini(base_response, "attrition analysis")

    def _analyze_salary(self, query: str) -> str:
        """Analyze salary and compensation data"""
        # Note: No direct salary column in CSV, but we can infer from other data
        return self._enhance_with_gemini(
            "Detailed salary information is not available in the current dataset. Consider integrating compensation data for comprehensive analysis.",
            "salary analysis"
        )

    def _analyze_headcount(self, query: str) -> str:
        """Analyze employee headcount"""
        total = len(self.df)
        active = len(self.df[self.df['ReasonForResignation'] == 'Still Working'])
        
        # By employment status
        full_time = len(self.df[self.df['EmploymentStatus'] == 'FullTime'])
        outsource = len(self.df[self.df['EmploymentStatus'] == 'Outsource'])
        
        # By cities
        city_breakdown = self.df['City'].value_counts().head(5)
        city_text = ", ".join([f"{city}: {count}" for city, count in city_breakdown.items()])
        
        base_response = f"""
**Employee Headcount Analysis:**
• Total employees: {total}
• Active employees: {active}
• Full-time: {full_time}, Outsourced: {outsource}
• Top cities: {city_text}

**Workforce Distribution:**
Your organization maintains a balanced workforce across major IT hubs.
        """
        
        return self._enhance_with_gemini(base_response, "headcount analysis")

    def _analyze_by_department(self, query: str) -> str:
        """Analyze by department/job roles"""
        dept_breakdown = self.df['JobRole'].value_counts().head(8)
        dept_text = "\n".join([f"• {role}: {count} employees" for role, count in dept_breakdown.items()])
        
        base_response = f"""
**Department-wise Analysis:**
{dept_text}

**Strategic Overview:**
Technology roles dominate the workforce composition, reflecting your organization's focus on software development and data analytics.
        """
        
        return self._enhance_with_gemini(base_response, "departmental analysis")

    def _analyze_performance(self, query: str) -> str:
        """Analyze performance ratings"""
        perf_breakdown = self.df['PerformanceRating'].value_counts()
        perf_text = "\n".join([f"• {rating}: {count} employees ({count/len(self.df)*100:.1f}%)"
                              for rating, count in perf_breakdown.items()])
        
        excellent_pct = (perf_breakdown.get('Excellent', 0) / len(self.df)) * 100
        
        base_response = f"""
**Performance Analysis:**
{perf_text}

**Performance Insights:**
{excellent_pct:.1f}% of employees demonstrate excellent performance, indicating strong talent quality.
        """
        
        return self._enhance_with_gemini(base_response, "performance analysis")

    def _analyze_joining_trends(self, query: str) -> str:
        """Analyze joining and hiring trends"""
        # Extract year from joining dates
        self.df['JoiningYear'] = self.df['DateOfJoining'].dt.year
        yearly_joins = self.df['JoiningYear'].value_counts().sort_index().tail(5)
        yearly_text = "\n".join([f"• {year}: {count} new hires"
                                for year, count in yearly_joins.items() if pd.notna(year)])
        
        # Hiring platforms
        platform_breakdown = self.df['HiringPlatform'].value_counts().head(4)
        platform_text = ", ".join([f"{platform} ({count})" for platform, count in platform_breakdown.items()])
        
        base_response = f"""
**Hiring Trends Analysis:**
{yearly_text}

**Recruitment Channels:**
Top platforms: {platform_text}

**Strategic Insights:**
Recruitment patterns show consistent growth with diversified sourcing channels.
        """
        
        return self._enhance_with_gemini(base_response, "hiring trends analysis")

    def _analyze_engagement(self, query: str) -> str:
        """Analyze engagement and retention factors"""
        # Engagement insights from available data
        active_employees = len(self.df[self.df['ReasonForResignation'] == 'Still Working'])
        total_employees = len(self.df)
        retention_rate = (active_employees / total_employees) * 100
        
        # Career level distribution
        career_breakdown = self.df['CareerLevel'].value_counts()
        experienced_pct = (career_breakdown.get('Experienced', 0) / total_employees) * 100
        
        base_response = f"""
**Employee Engagement & Retention:**
• Current retention rate: {retention_rate:.1f}%
• Experienced professionals: {experienced_pct:.1f}%
• Active workforce: {active_employees} employees

**Engagement Indicators:**
High retention suggests positive employee engagement and organizational satisfaction.
        """
        
        return self._enhance_with_gemini(base_response, "engagement analysis")

    def _general_analytics_summary(self) -> str:
        """General analytics overview"""
        total = len(self.df)
        active = len(self.df[self.df['ReasonForResignation'] == 'Still Working'])
        attrition_rate = ((total - active) / total) * 100
        
        base_response = f"""
**HR Analytics Summary:**
• Total workforce: {total} employees
• Active employees: {active}
• Current attrition rate: {attrition_rate:.1f}%

**Key Insights:**
Your organization maintains a substantial workforce with competitive retention metrics for the IT industry.
        """
        
        return self._enhance_with_gemini(base_response, "general HR analytics")

    def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini API"""
        try:
            return gemini_generate(prompt) or ""
        except Exception as e:
            log.warning(f"Gemini API error: {e}")
            return ""

    def _enhance_with_gemini(self, base_response: str, analysis_type: str) -> str:
        """Enhance response with Gemini intelligence"""
        enhancement_prompt = f"""
        You are a senior HR analyst presenting to executive leadership.
        
        Enhance this {analysis_type} summary for a professional audience:
        
        {base_response}
        
        Requirements:
        - Keep it concise and professional
        - Add 1-2 actionable insights
        - Use executive-appropriate language
        - Maintain data accuracy
        - Remove any "Key Findings:" or "Sources:" sections
        """
        
        enhanced = self._get_gemini_response(enhancement_prompt)
        return enhanced if enhanced else base_response

# ---------------- Simplified External Search Agent ----------------
class SimplifiedExternalSearchAgent:
    def __init__(self):
        self.enabled = HAS_DDGS
        if self.enabled:
            self.ddgs = DDGS()

    def search_industry_insights(self, query: str) -> str:
        """Simplified external search without complex formatting"""
        if not self.enabled:
            return "External search not available. I can help you analyze your internal HR data instead."
        
        try:
            # Generate search terms with Gemini
            search_prompt = f"""
            Generate 2 specific search queries for: "{query}"
            Focus on IT industry HR data and benchmarks.
            Return as simple list, no JSON.
            """
            
            search_terms_raw = gemini_generate(search_prompt)
            search_terms = [query] if not search_terms_raw else search_terms_raw.split('\n')[:2]
            
            # Search DuckDuckGo
            all_snippets = []
            for term in search_terms:
                try:
                    results = self.ddgs.text(term.strip(), max_results=2, region='us-en')
                    for result in results:
                        body = result.get('body', '').strip()
                        if len(body) > 100:
                            all_snippets.append(body[:400])
                except Exception as e:
                    log.warning(f"Search failed for '{term}': {e}")
            
            if not all_snippets:
                return "No relevant external information found. Let me help you analyze your internal data instead."
            
            # Use Gemini to create professional summary
            combined_snippets = '\n\n'.join(all_snippets[:3])
            summary_prompt = f"""
            Create a professional HR summary from these search results:
            
            {combined_snippets}
            
            Requirements:
            - Professional executive language
            - 2-3 key insights only
            - Actionable recommendations
            - No "Key Findings" or "Sources" sections
            """
            
            summary = gemini_generate(summary_prompt)
            return summary if summary else "Industry information retrieved but couldn't be processed effectively."
            
        except Exception as e:
            log.error(f"External search error: {e}")
            return "Unable to retrieve external insights at this time."

# ---------------- Keep Existing Utility Functions ----------------
def examples() -> str:
    return (
        "**Try:**\n"
        "• What is our current attrition rate?\n"
        "• Show me headcount by department\n"
        "• Predict attrition for employee 110450\n"
        "• Compare our retention with industry benchmarks\n"
        "• What are top resignation reasons?\n"
        "• Recommend actions for employee 110450\n"
    )

def gemini_generate(prompt: str) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")
        out = model.generate_content(prompt, safety_settings=None)
        txt = getattr(out, "text", None)
        if not txt and out.candidates and out.candidates[0].content.parts:
            txt = out.candidates[0].content.parts[0].text
        return (txt or "").strip()
    except Exception as e:
        log.warning(f"Gemini error: {e}")
        return None

# ---------------- Keep All Existing Attrition Prediction Functions ----------------

def fetch_row(employee_id: int) -> Optional[pd.Series]:
    """Fetch employee data for attrition prediction"""
    try:
        df = pd.read_csv(CSV_PATH)
        employee_data = df[df['EmployeeID'] == employee_id]
        return employee_data.iloc[0] if not employee_data.empty else None
    except Exception:
        return None

def attrition_agent(employee_id: int, horizon_days: int = 60) -> Dict[str, Any]:
    """Simplified attrition prediction"""
    row = fetch_row(employee_id)
    if row is None:
        raise ValueError(f"Employee {employee_id} not found")
    
    # Simple risk calculation based on available data
    risk_factors = 0
    if row.get('PerformanceRating') == 'Poor':
        risk_factors += 30
    elif row.get('PerformanceRating') == 'Excellent':
        risk_factors -= 10
    
    if row.get('EmploymentStatus') == 'Outsource':
        risk_factors += 15
    
    # Calculate years of experience
    if pd.notna(row.get('YearsOfExperience')):
        years = int(row.get('YearsOfExperience', 0))
        if years < 2:
            risk_factors += 20
        elif years > 10:
            risk_factors -= 5
    
    # Convert to probability
    base_risk = max(0.05, min(0.95, (risk_factors + 25) / 100))
    
    # Adjust for horizon
    horizon_prob = 1 - np.exp(-0.01 * horizon_days * base_risk)
    
    tier = "High" if horizon_prob >= 0.6 else "Medium" if horizon_prob >= 0.35 else "Low"
    
    # Generate recommendations based on tier
    if tier == "High":
        recommendations = [
            {"action_type": "Immediate 1:1 with manager", "owner_role": "Manager", "sla_days": 3, "expected_impact": 3},
            {"action_type": "Compensation review", "owner_role": "HR", "sla_days": 14, "expected_impact": 3},
            {"action_type": "Career development plan", "owner_role": "L&D", "sla_days": 14, "expected_impact": 2}
        ]
    elif tier == "Medium":
        recommendations = [
            {"action_type": "Regular check-ins", "owner_role": "Manager", "sla_days": 7, "expected_impact": 2},
            {"action_type": "Training opportunities", "owner_role": "L&D", "sla_days": 21, "expected_impact": 2}
        ]
    else:
        recommendations = [
            {"action_type": "Maintain engagement", "owner_role": "Manager", "sla_days": 30, "expected_impact": 1}
        ]
    
    summary = f"Employee {employee_id} — {horizon_days}-day attrition risk: {horizon_prob*100:.1f}% ({tier})"
    
    return {
        "employee_id": employee_id,
        "horizon_days": horizon_days,
        "probability": round(horizon_prob, 4),
        "risk_tier": tier,
        "recommendations": recommendations,
        "summary": summary,
        "confidence": 0.85
    }

# ---------------- Enhanced Main Entry Point with Greeting Handling ----------------
def get_chatbot_response(message: str, role: str = "hr") -> Dict[str, Any]:
    """Enhanced main entry point with greeting handling and simplified intelligence"""
    try:
        msg = (message or "").strip()
        
        # Check for greeting first - NEW FEATURE
        if is_greeting(msg):
            return pack(ChatbotResponse(get_greeting_response(), "text", confidence=1.0))
        
        if not msg or msg.lower() in {"start"}:
            welcome = (
                "👋 Welcome to Enhanced HR Buddy.\n"
                "I can analyze your employee data and provide **industry insights**.\n\n" +
                examples()
            )
            return pack(ChatbotResponse(welcome, "text", confidence=1.0))

        # Route different types of queries
        msg_lower = msg.lower()

        # Employee-specific predictions
        employee_match = re.search(r'(?:employee|id)\s*(\d{5,6})', msg_lower)
        if employee_match and any(word in msg_lower for word in ["predict", "risk", "recommend", "action"]):
            employee_id = int(employee_match.group(1))
            
            if "predict" in msg_lower or "risk" in msg_lower:
                try:
                    days_match = re.search(r'(\d{2,3})\s*day', msg_lower)
                    horizon_days = int(days_match.group(1)) if days_match else 60
                    result = attrition_agent(employee_id, horizon_days)
                    return pack(ChatbotResponse(result["summary"], "prediction", data=result, confidence=result["confidence"]))
                except ValueError as e:
                    return pack(ChatbotResponse(f"Employee {employee_id} not found in records.", "text", confidence=0.9))
                    
            elif "recommend" in msg_lower or "action" in msg_lower:
                try:
                    result = attrition_agent(employee_id, 60)
                    rec_text = f"Recommendations for Employee {employee_id} ({result['risk_tier']} risk):\n"
                    for i, rec in enumerate(result['recommendations'][:3], 1):
                        rec_text += f"{i}. {rec['action_type']} — Owner: {rec['owner_role']}, SLA: {rec['sla_days']}d\n"
                    return pack(ChatbotResponse(rec_text, "action", data={"recommendations": result['recommendations']}, confidence=0.96))
                except ValueError:
                    return pack(ChatbotResponse(f"Employee {employee_id} not found in records.", "text", confidence=0.9))

        # External search queries
        elif any(word in msg_lower for word in ["industry", "benchmark", "best practice", "compare", "trend"]):
            search_agent = SimplifiedExternalSearchAgent()
            result = search_agent.search_industry_insights(msg)
            return pack(ChatbotResponse(result, "external", confidence=0.8))

        # CSV analytics queries
        else:
            analytics_agent = EnhancedCSVAnalyticsAgent()
            result = analytics_agent.analyze_query(msg)
            return pack(ChatbotResponse(result, "text", confidence=0.9))

    except Exception as e:
        log.exception(e)
        return pack(ChatbotResponse("I encountered an issue processing your request. Please try again.", "text", confidence=0.1))
