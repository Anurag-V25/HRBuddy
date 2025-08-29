import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Import the existing data loading functions from your original file
from advanced_dashboard import (
    load_comprehensive_data,
    process_data,
    create_chart_card,
    create_chart_1_financial_impact,
    create_chart_2_risk_heatmap,
    create_chart_3_workforce_roi,
    create_chart_4_forecast,
    create_chart_5_attrition_analysis,
    create_chart_6_recruitment,
    create_chart_7_engagement,
    create_chart_8_performance,
    create_chart_9_demographics,
    create_chart_10_compensation,
    create_chart_11_learning_roi,
    create_chart_12_manager_performance,
    create_chart_13_daily_pulse,
    create_chart_14_risk_monitoring,
    create_chart_15_talent_pipeline,
    create_chart_16_journey_mapping,
    create_chart_17_compensation_analytics,
    create_chart_18_workforce_planning
)

# --- Configuration ---
load_dotenv()
DB_PATH = os.getenv('DATABASE_PATH', '../data/processed/hr_data.db')

# Import chatbot backend
from chatbot_backend import get_chatbot_response

def create_enhanced_dashboard(flask_app):
    """Create the production-ready HR dashboard with chatbot and enhanced interactivity"""
    app = dash.Dash(
        server=flask_app,
        name="EnhancedHRDashboard",
        url_base_pathname="/dashboard/",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    # Load data once
    df = load_comprehensive_data()

    # Override the default HTML template
    app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>🏢 Enterprise HR Analytics Suite</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="/assets/enhanced-style.css" rel="stylesheet">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {%favicon%}
    {%css%}
  </head>
  <body>
    <!-- Loading Overlay -->
    <div id="loading-overlay" class="loading-overlay">
      <div>
        <div class="spinner"></div>
        <h4>Loading Enterprise Analytics...</h4>
        <p>Connecting to data sources and preparing insights</p>
      </div>
    </div>

    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-gradient-primary fixed-top">
      <div class="container-fluid">
        <span class="navbar-brand">
          <i class="fas fa-chart-line me-2"></i>
          HR Analytics Suite
        </span>
        <div class="navbar-nav ms-auto">
          <span class="nav-text text-light me-3">
            <i class="far fa-clock me-1"></i>
          </span>
            <ul class="dropdown-menu">
              <li><a class="dropdown-item" href="#" onclick="exportDashboard('PDF')">
                <i class="fas fa-file-pdf me-2"></i> Export as PDF</a></li>
              <li><a class="dropdown-item" href="#" onclick="exportDashboard('Excel')">
                <i class="fas fa-file-excel me-2"></i> Export as Excel</a></li>
              <li><a class="dropdown-item" href="#" onclick="exportDashboard('PowerPoint')">
                <i class="fas fa-file-powerpoint me-2"></i> Export as PPT</a></li>
            </ul>
          </div>
        </div>
      </div>
    </nav>

    <!-- Sidebar Toggle Button -->
    <button class="sidebar-toggle" onclick="toggleSidebar()">
      <i class="fas fa-bars"></i>
    </button>

    <!-- Sidebar -->
    <div id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <h5 class="mb-1">📊 Dashboard Sections</h5>
        <small class="text-muted">Navigate through analytics</small>
      </div>
      <nav class="sidebar-nav">
        <a href="#executive-summary" class="sidebar-link active" onclick="scrollToSection('executive-summary')">
          <i class="fas fa-chart-pie"></i> Executive Summary
        </a>
        <a href="#hr-operations" class="sidebar-link" onclick="scrollToSection('hr-operations')">
          <i class="fas fa-users-cog"></i> HR Operations
        </a>
        <a href="#strategic-planning" class="sidebar-link" onclick="scrollToSection('strategic-planning')">
          <i class="fas fa-chess"></i> Strategic Planning
        </a>
        <a href="#real-time" class="sidebar-link" onclick="scrollToSection('real-time')">
          <i class="fas fa-tachometer-alt"></i> Real-Time Metrics
        </a>
        <a href="#advanced-analytics" class="sidebar-link" onclick="scrollToSection('advanced-analytics')">
          <i class="fas fa-brain"></i> Advanced Analytics
        </a>
        <hr class="my-3">
        <a href="#" class="sidebar-link" onclick="alertHighRisk()">
          <i class="fas fa-exclamation-triangle text-warning"></i> Alert High Risk
        </a>
        <a href="#" class="sidebar-link" onclick="openPlanningModal()">
          <i class="fas fa-calendar-alt"></i> Workforce Planning
        </a>
      </nav>
    </div>

    <!-- Main Content -->
    <div id="main-content" class="main-content">
      {%app_entry%}
    </div>

    <!-- Footer -->
    <footer class="dashboard-footer">
      <div class="container-fluid">
        <div class="row align-items-center">
          <div class="col-md-6">
            <p class="mb-0">
              <i class="fas fa-shield-alt me-2"></i>
              Enterprise HR Analytics Suite v2.0
            </p>
          </div>
          <div class="col-md-6 text-end">
            <p class="mb-0">
              <i class="fas fa-sync-alt me-1"></i>
              Auto-refresh: <span id="refresh-timer">Off</span>
            </p>
          </div>
        </div>
      </div>
    </footer>

    {%config%}
    {%scripts%}
    {%renderer%}

    <!-- Custom JavaScript -->
  <!-- <script src="/assets/dashboard-core.js"></script> -->
  <!-- <script src="/assets/dashboard-animations.js"></script> -->
  <!-- <script src="/assets/dashboard-interactions.js"></script> -->

    <!-- === Chatbot: Styles + HTML + JS (self-contained) === -->
    <style>
      /* Reliance-style widget, Poppins font */
      .chat-widget, .chat-launcher {
        font-family: "Poppins", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      }
      .chat-widget {
        position: fixed; bottom: 20px; right: 20px;
        width: 360px; max-height: 92vh;
        height: 600px;
        display: none; flex-direction: column;
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 6px 25px rgba(0,0,0,.3);
        background: #fff; z-index: 10002;
      }
      .chat-header {
        background: #0056D2; color: #fff;
        padding: 12px 14px; font-weight: 600;
        display:flex; align-items:center; justify-content: space-between;
      }
      .chat-body {
        flex: 1; overflow-y: auto;
        background: #f7f8fa; padding: 12px;
        display:flex; flex-direction:column; gap:10px;
      }
      .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px;
             font-size: 14px; line-height: 1.45; }
  .msg.bot  { background:#e5e7eb; align-self:flex-start; color:#0f172a; }
  .msg.user { background:#0056D2; color:#fff; align-self:flex-end; padding: 12px 18px; }
      .timestamp { font-size: 11px; color:#6b7280; margin-top:2px; }
      .button-group { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
      .quick-btn {
        background:#0056D2; color:#fff; border:none;
        border-radius: 18px; padding:8px 14px;
        font-size:13px; cursor:pointer; transition: background .2s, transform .05s;
      }
      .quick-btn:hover { background:#0043a6; transform: translateY(-1px); }
      .chat-footer {
        display:flex; align-items:center; gap:6px;
        padding: 8px; border-top:1px solid #ddd; background:#fff;
      }
      .chat-footer input {
        flex:1; padding:8px 12px; border:1px solid #ccc;
        border-radius:20px; outline:none; font-size:14px;
      }
      .chat-footer button {
        background:#0056D2; border:none; border-radius:50%;
        width:36px; height:36px; color:#fff; font-size:16px; cursor:pointer;
      }
      /* Launcher */
      .chat-launcher {
        position: fixed; bottom: 22px; right: 22px;
        width: 56px; height: 56px; border-radius: 50%;
        display: grid; place-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
        cursor: pointer; z-index: 10001;
      }
      .chat-launcher:hover {
        background: #fff;
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
        border: 1.5px solid #0056D2;
      }
      @media (max-width:560px){ .chat-widget { width: 92vw; right: 4vw; } }
    </style>

    <!-- Launcher -->
    <div id="chatLauncher" class="chat-launcher" title="Chat with AI">
      <img src="https://marketplace.canva.com/_o3Sg/MAGaMB_o3Sg/1/tl/canva-3d-robot-character-reading-a-book-illustration-MAGaMB_o3Sg.png" alt="Chatbot" style="width:54px;height:70px;object-fit:contain;display:block;margin:auto;" />
    </div>

    <!-- Widget -->
    <div id="chatWidget" class="chat-widget" aria-label="HR Attrition Assistant">
      <div class="chat-header">
        HR Attrition Assistant
        <span id="chatClose" style="font-size:16px; cursor:pointer;" aria-label="Close">✕</span>
      </div>
      <div id="chatBody" class="chat-body"></div>
      <div class="chat-footer">
        <input id="msgInput" type="text" placeholder="Type here..."/>
        <button id="sendBtn" aria-label="Send">➤</button>
      </div>
    </div>

    <!-- Overlay: make sure the loader hides quickly -->
<script>
  (function(){
    const overlay = document.getElementById('loading-overlay');
    function hide(){ if(overlay) overlay.style.display='none'; }
    window.addEventListener('dash:rendered', hide);
    window.addEventListener('load', ()=> setTimeout(hide, 300));
    setTimeout(hide, 2500); // final fallback
  })();
</script>

<!-- Chatbot controller -->
<script>
  (function(){
    // Elements
    const widget   = document.getElementById("chatWidget");
    const launcher = document.getElementById("chatLauncher");
    const chatBody = document.getElementById("chatBody");
    const input    = document.getElementById("msgInput");
    const send     = document.getElementById("sendBtn");
    const closeBtn = document.getElementById("chatClose");

    // Endpoint
    const API_URL  = "/api/chatbot";

    // --- Helpers ---
    function mdToHtml(t=""){
      t = t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      t = t.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>");
      t = t.replace(/\*(.+?)\*/g,"<em>$1</em>");
      t = t.replace(/^(?:-|\\u2022)\\s+(.*)$/gmi,"• $1");
      return t.replace(/\\n/g,"<br>");
    }

    function addMsg(text, who="bot", buttons=[], showTime=true){
      const wrap = document.createElement("div");
      const msg  = document.createElement("div");
      msg.className = "msg " + who;
      msg.innerHTML = mdToHtml(text || "");
      wrap.appendChild(msg);

      if (buttons && buttons.length){
        const group = document.createElement("div");
        group.className = "button-group";
        buttons.forEach(b=>{
          const btn = document.createElement("button");
          btn.className = "quick-btn";
          btn.textContent = b;
          btn.onclick = () => sendMsg(b, false);
          group.appendChild(btn);
        });
        wrap.appendChild(group);
      }

      if (showTime){
        const ts = document.createElement("div");
        ts.className = "timestamp";
        ts.textContent = new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
        wrap.appendChild(ts);
      }
      chatBody.appendChild(wrap);
      chatBody.scrollTop = chatBody.scrollHeight;
    }

    async function sendMsg(text, showUser=true){
      if (!text) return;
      if (showUser) addMsg(text, "user", [], true);
      try{
        const r = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, role: "hr" })
        });
        const data = await r.json();
        addMsg(data.message || "", "bot", data.quick_replies || [], true);
      }catch(e){
        addMsg("⚠️ Error contacting server. Please try again.", "bot");
      }
    }

    // --- Open/Close ---
    launcher.addEventListener("click", () => {
      widget.style.display = "flex";
      launcher.style.display = "none";

      // Seed FIRST message + buttons immediately (no network wait)
      if (!chatBody.dataset.seeded) {
        // Try to get user name from localStorage/sessionStorage or backend
        let userName = "";
        try {
          userName = localStorage.getItem("user_name") || sessionStorage.getItem("user_name") || "";
        } catch(e) {}
        const welcomeText = userName
          ? `Hello ${userName}! I'm your HR Analytics Assistant.`
          : "Hello! I'm your HR Analytics Assistant.";
        const quick = [];
        addMsg(welcomeText, "bot", quick, true);

        // Then ask the backend for dynamic menu / follow-ups
        sendMsg("start", false);

        chatBody.dataset.seeded = "1";
      }
    });

    closeBtn.addEventListener("click", () => {
      widget.style.display = "none";
      launcher.style.display = "grid";
    });

    // --- Send handlers ---
    send.onclick = () => {
      const t = input.value.trim();
      if(!t) return;
      input.value="";
      sendMsg(t, true);
    };
    input.addEventListener("keypress", (e)=>{
      if(e.key==="Enter"){ e.preventDefault(); send.click(); }
    });
  })();
</script>
    <!-- === /Chatbot section === -->

  </body>
</html>
'''


    # Layout with comprehensive dashboard and chatbot
    app.layout = dbc.Container([
        # --- Section 1: Executive Summary ---
        html.Div(id="executive-summary", children=[
            dbc.Card([
                dbc.CardHeader([
            html.H2([
                html.I(className="fas fa-chart-pie section-icon"),
                "Executive Summary"
                    ], className="section-title mb-0"),
                    dbc.Button(
                        [html.I(className="fas fa-chevron-down"), " Collapse"],
                        id="executive-summary-collapse-btn",
                        color="link",
                        className="float-end"
                    )
                ]),
                dbc.Collapse(
                    dbc.CardBody([
            create_chart_1_financial_impact(df),
            html.Br(),
            dbc.Row([
                            dbc.Col(create_chart_card("🔥 Business Risk Heatmap", create_chart_2_risk_heatmap(df), 
                                "Heatmap showing attrition risk by department and job role. Darker colors indicate higher risk areas."), md=6),
                            dbc.Col(create_chart_card("💼 Workforce ROI Metrics", create_chart_3_workforce_roi(df),
                                "Scatter plot of salary vs revenue per employee by department. Larger circles indicate more employees."), md=6)
            ], className="mb-4"),
                        create_chart_card("📈 Predictive Attrition Forecast", create_chart_4_forecast(df),
                            "12-month attrition rate forecast using historical data and trend analysis."),
                    ]),
                    id="executive-summary-collapse",
                    is_open=True
                )
            ], className="section-card")
        ], className="dashboard-section"),

        # --- Section 2: HR Operations ---
        html.Div(id="hr-operations", children=[
            dbc.Card([
                dbc.CardHeader([
            html.H2([
                html.I(className="fas fa-users-cog section-icon"),
                "HR Operations"
                    ], className="section-title mb-0"),
                    dbc.Button(
                        [html.I(className="fas fa-chevron-down"), " Collapse"],
                        id="hr-operations-collapse-btn",
                        color="link",
                        className="float-end"
                    )
                ]),
                dbc.Collapse(
                    dbc.CardBody([
            dbc.Row([
                            dbc.Col(create_chart_card("📊 Attrition Analysis", create_chart_5_attrition_analysis(df),
                                "Bar chart showing attrition rates by department with color intensity indicating severity."), md=6),
                            dbc.Col(create_chart_card("🎯 Recruitment Performance", create_chart_6_recruitment(df),
                                "Scatter plot of time-to-fill vs conversion rate for different job roles."), md=6)
            ], className="mb-4"),
            dbc.Row([
                            dbc.Col(create_chart_card("❤️ Employee Engagement", create_chart_7_engagement(df),
                                "Grouped bar chart comparing job satisfaction and work-life balance scores by department."), md=6),
                            dbc.Col(create_chart_card("⭐ Performance Analytics", create_chart_8_performance(df),
                                "Pie chart distribution of performance ratings across the organization."), md=6)
                        ]),
                    ]),
                    id="hr-operations-collapse",
                    is_open=True
                )
            ], className="section-card")
        ], className="dashboard-section"),

        # --- Section 3: Strategic Planning ---
        html.Div(id="strategic-planning", children=[
            dbc.Card([
                dbc.CardHeader([
            html.H2([
                html.I(className="fas fa-chess section-icon"),
                "Strategic Planning"
                    ], className="section-title mb-0"),
                    dbc.Button(
                        [html.I(className="fas fa-chevron-down"), " Collapse"],
                        id="strategic-planning-collapse-btn",
                        color="link",
                        className="float-end"
                    )
                ]),
                dbc.Collapse(
                    dbc.CardBody([
            dbc.Row([
                            dbc.Col(create_chart_card("👥 Workforce Demographics", create_chart_9_demographics(df),
                                "Sunburst chart showing employee distribution by department and gender."), md=6),
                            dbc.Col(create_chart_card("💰 Compensation Intelligence", create_chart_10_compensation(df),
                                "Scatter plot of salary vs risk score by job role, with circle size indicating risk level."), md=6)
            ], className="mb-4"),
            dbc.Row([
                            dbc.Col(create_chart_card("🎓 Learning & Development ROI", create_chart_11_learning_roi(df),
                                "Line chart showing the relationship between training hours and employee satisfaction/risk scores."), md=6),
                            dbc.Col(create_chart_card("👨‍💼 Manager Performance", create_chart_12_manager_performance(df),
                                "Scatter plot of manager satisfaction vs team risk scores, with circle size indicating team size."), md=6)
                        ]),
                    ]),
                    id="strategic-planning-collapse",
                    is_open=True
                )
            ], className="section-card")
        ], className="dashboard-section"),

        # --- Section 4: Real-Time Dashboard ---
        html.Div(id="real-time", children=[
            dbc.Card([
                dbc.CardHeader([
            html.H2([
                html.I(className="fas fa-tachometer-alt section-icon"),
                "Real-Time Dashboard"
                    ], className="section-title mb-0"),
                    dbc.Button(
                        [html.I(className="fas fa-chevron-down"), " Collapse"],
                        id="real-time-collapse-btn",
                        color="link",
                        className="float-end"
                    )
                ]),
                dbc.Collapse(
                    dbc.CardBody([
            create_chart_13_daily_pulse(df),
            html.Br(),
            dbc.Row([
                            dbc.Col(create_chart_card("⚠️ Risk Monitoring", create_chart_14_risk_monitoring(df),
                                "Table showing top 10 high-risk employees with their risk scores and details."), md=8),
                            dbc.Col(create_chart_card("🎯 Talent Pipeline", create_chart_15_talent_pipeline(df),
                                "Funnel chart showing recruitment pipeline stages and conversion rates."), md=4)
            ], className="my-4"),
                    ]),
                    id="real-time-collapse",
                    is_open=True
                )
            ], className="section-card")
        ], className="dashboard-section"),

        # --- Section 5: Advanced Analytics ---
        html.Div(id="advanced-analytics", children=[
            dbc.Card([
                dbc.CardHeader([
            html.H2([
                html.I(className="fas fa-brain section-icon"),
                "Advanced Analytics"
                    ], className="section-title mb-0"),
                    dbc.Button(
                        [html.I(className="fas fa-chevron-down"), " Collapse"],
                        id="advanced-analytics-collapse-btn",
                        color="link",
                        className="float-end"
                    )
                ]),
                dbc.Collapse(
                    dbc.CardBody([
            dbc.Row([
                            dbc.Col(create_chart_card("🗺️ Employee Journey Mapping", create_chart_16_journey_mapping(df),
                                "Timeline visualization of employee lifecycle stages and transition points."), md=6),
                            dbc.Col(create_chart_card("💎 Compensation Analytics", create_chart_17_compensation_analytics(df),
                                "Detailed compensation analysis with market comparisons and internal equity metrics."), md=6)
            ], className="mb-4"),
                        create_chart_card("🔮 Workforce Planning", create_chart_18_workforce_planning(df),
                            "Strategic workforce planning dashboard with headcount projections and skill gap analysis.")
                    ]),
                    id="advanced-analytics-collapse",
                    is_open=True
                )
            ], className="section-card")
        ], className="dashboard-section"),

        # --- Chatbot Callback Store ---
        dcc.Store(id='chatbot-messages-store', data=[]),
        dcc.Store(id='section-collapse-store', data={
            'executive-summary': True,
            'hr-operations': True,
            'strategic-planning': True,
            'real-time': True,
            'advanced-analytics': True
        }),

    ], fluid=True, className="p-0")

    # Callbacks for section collapse functionality
    @app.callback(
        [Output(f"{section}-collapse", "is_open") for section in 
         ["executive-summary", "hr-operations", "strategic-planning", "real-time", "advanced-analytics"]],
        [Input(f"{section}-collapse-btn", "n_clicks") for section in 
         ["executive-summary", "hr-operations", "strategic-planning", "real-time", "advanced-analytics"]],
        [State("section-collapse-store", "data")]
    )
    def toggle_section_collapse(*args):
        ctx = callback_context
        if not ctx.triggered:
            return [True] * 5
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        section_name = button_id.replace('-collapse-btn', '')
        
        # Get current state
        current_state = args[-1] if args[-1] else {}
        current_state[section_name] = not current_state.get(section_name, True)
        
        # Return new states
        return [
            current_state.get("executive-summary", True),
            current_state.get("hr-operations", True),
            current_state.get("strategic-planning", True),
            current_state.get("real-time", True),
            current_state.get("advanced-analytics", True)
        ]

    return app

# --- Export Function ---
def create_advanced_dashboard(flask_app):
    return create_enhanced_dashboard(flask_app)