# enhanced_advanced_dashboard.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from advanced_dashboard import (
    load_comprehensive_data,
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

def create_enhanced_dashboard(flask_app):
    app = dash.Dash(
        server=flask_app,
        name="EnhancedHRDashboard",
        url_base_pathname="/dashboard/",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    df = load_comprehensive_data()

    # ✅ Load chatbot widget snippet
    with open("templates/attration.html") as f:
        chatbot_html = f.read()

    # ✅ Build a valid Dash index_string with required placeholders
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
      <head>
        {{%metas%}}
        <title>🏢 Enterprise HR Analytics Suite</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
        <link href="/assets/enhanced-style.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {{%favicon%}}
        {{%css%}}
      </head>
      <body>
        <div id="main-content">
          {{%app_entry%}}
        </div>
        <footer>
          {{%config%}}
          {{%scripts%}}
          {{%renderer%}}
        </footer>

        <!-- 🔥 Inject Chatbot Widget -->
        {chatbot_html}
      </body>
    </html>
    """

    # ---------------- Dashboard Layout ----------------
    app.layout = dbc.Container([
        # === Section 1: Executive Summary ===
        html.Div(id="executive-summary", children=[
            dbc.Card([
                dbc.CardHeader(html.H2("Executive Summary")),
                dbc.CardBody([
                    create_chart_1_financial_impact(df),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(create_chart_card("🔥 Business Risk Heatmap",
                                create_chart_2_risk_heatmap(df)), md=6),
                        dbc.Col(create_chart_card("💼 Workforce ROI Metrics",
                                create_chart_3_workforce_roi(df)), md=6),
                    ], className="mb-4"),
                    create_chart_card("📈 Predictive Attrition Forecast",
                                      create_chart_4_forecast(df))
                ])
            ])
        ], className="dashboard-section"),

        # === Section 2: HR Operations ===
        html.Div(id="hr-operations", children=[
            dbc.Card([
                dbc.CardHeader(html.H2("HR Operations")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(create_chart_card("📊 Attrition Analysis",
                                create_chart_5_attrition_analysis(df)), md=6),
                        dbc.Col(create_chart_card("🎯 Recruitment Performance",
                                create_chart_6_recruitment(df)), md=6)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(create_chart_card("❤️ Employee Engagement",
                                create_chart_7_engagement(df)), md=6),
                        dbc.Col(create_chart_card("⭐ Performance Analytics",
                                create_chart_8_performance(df)), md=6)
                    ])
                ])
            ])
        ], className="dashboard-section"),

        # === Section 3: Strategic Planning ===
        html.Div(id="strategic-planning", children=[
            dbc.Card([
                dbc.CardHeader(html.H2("Strategic Planning")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(create_chart_card("👥 Workforce Demographics",
                                create_chart_9_demographics(df)), md=6),
                        dbc.Col(create_chart_card("💰 Compensation Intelligence",
                                create_chart_10_compensation(df)), md=6)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(create_chart_card("📚 Learning ROI",
                                create_chart_11_learning_roi(df)), md=6),
                        dbc.Col(create_chart_card("👩‍💼 Manager Performance",
                                create_chart_12_manager_performance(df)), md=6)
                    ])
                ])
            ])
        ], className="dashboard-section"),

        # === Section 4: Real-Time Metrics ===
        html.Div(id="real-time", children=[
            dbc.Card([
                dbc.CardHeader(html.H2("Real-Time Metrics")),
                dbc.CardBody([
                    create_chart_13_daily_pulse(df),
                    html.Br(),
                    create_chart_card("🚨 High Risk Monitor",
                                      create_chart_14_risk_monitoring(df))
                ])
            ])
        ], className="dashboard-section"),

        # === Section 5: Advanced Analytics ===
        html.Div(id="advanced-analytics", children=[
            dbc.Card([
                dbc.CardHeader(html.H2("Advanced Analytics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(create_chart_card("📌 Talent Pipeline",
                                create_chart_15_talent_pipeline(df)), md=6),
                        dbc.Col(create_chart_card("🗺 Employee Journey Mapping",
                                create_chart_16_journey_mapping(df)), md=6)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(create_chart_card("📊 Compensation Analytics",
                                create_chart_17_compensation_analytics(df)), md=6),
                        dbc.Col(create_chart_card("📅 Workforce Planning",
                                create_chart_18_workforce_planning(df)), md=6)
                    ])
                ])
            ])
        ], className="dashboard-section"),
    ], fluid=True)

    return app
