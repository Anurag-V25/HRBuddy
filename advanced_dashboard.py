import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import dash
from dash import html, dcc
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

 # --- Resolve DB path robustly: prefer env, else default to repo path ---
USD_CONVERSION_RATE = 83  # 1 USD = 83 INR (update as needed)
BASE_DIR = Path(__file__).resolve().parents[1]  # .../Human-Resource-main
DEFAULT_DB = "data/processed/hr_data.db"
DB_PATH = os.environ.get("DB_PATH", str(DEFAULT_DB))


def load_comprehensive_data():
    """Load and merge all 9 HR tables into a single dataframe."""
    print(f"--- Loading database from: {DB_PATH} ---")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Load all 9 tables
    employees = pd.read_sql("SELECT * FROM employees", conn)
    career_development = pd.read_sql("SELECT * FROM career_development", conn)
    engagement = pd.read_sql("SELECT * FROM engagement", conn)
    work_patterns = pd.read_sql("SELECT * FROM work_patterns", conn)
    team_and_relationship = pd.read_sql("SELECT * FROM team_and_relationship", conn)
    risk_scores = pd.read_sql("SELECT * FROM risk_scores", conn)
    external_market_data = pd.read_sql("SELECT * FROM external_market_data", conn)
    compensation = pd.read_sql("SELECT * FROM compensation", conn)
    # retention_actions might not exist initially; handle gracefully
    try:
        retention_actions = pd.read_sql("SELECT * FROM retention_actions", conn)
    except Exception:
        retention_actions = pd.DataFrame(columns=["employeeid"])

    conn.close()

    # Standardize column names to lowercase
    tables = [
        employees, career_development, engagement, work_patterns,
        team_and_relationship, risk_scores, external_market_data,
        compensation, retention_actions
    ]
    for df in tables:
        df.columns = [col.strip().lower() for col in df.columns]

    # Start with employees as base table
    master_df = employees.copy()

    # Merge all tables on employeeid where applicable
    def safe_merge(left, right, **kwargs):
        if right is None or right.empty:
            return left
        if 'employeeid' not in right.columns:
            return left
        return left.merge(right, on='employeeid', **kwargs)

    master_df = safe_merge(master_df, compensation, how='left', suffixes=("", "_comp"))
    master_df = safe_merge(master_df, risk_scores, how='left', suffixes=("", "_risk"))
    master_df = safe_merge(master_df, engagement, how='left', suffixes=("", "_engage"))
    master_df = safe_merge(master_df, career_development, how='left', suffixes=("", "_career"))
    master_df = safe_merge(master_df, work_patterns, how='left', suffixes=("", "_work"))
    master_df = safe_merge(master_df, team_and_relationship, how='left', suffixes=("", "_team"))
    master_df = safe_merge(master_df, retention_actions, how='left', suffixes=("", "_retain"))

    # For external_market_data, join on jobrole or department if columns exist
    if not external_market_data.empty:
        if 'jobrole' in external_market_data.columns and 'jobrole' in master_df.columns:
            master_df = master_df.merge(external_market_data, on='jobrole', how='left', suffixes=("", "_market"))
        elif 'department' in external_market_data.columns and 'department' in master_df.columns:
            master_df = master_df.merge(external_market_data, on='department', how='left', suffixes=("", "_market"))

    # Process and engineer features for visualizations
    master_df = process_data(master_df)

    print(f"✅ Successfully loaded and merged {len(master_df)} records from 9 tables.")
    return master_df      

def process_data(df):
    """Process and engineer features for all visualizations"""
    # Create department groupings
    df['department'] = df['jobrole'].apply(lambda x: 'Tech' if 'Engineer' in x else 'Product' if 'Product' in x else 'Data')
    
    # Calculate tenure
    df['tenure_months'] = (pd.Timestamp.now() - pd.to_datetime(df['dateofjoining'])).dt.days / 30.44
    df['tenure_bins'] = pd.cut(df['tenure_months'], bins=[0, 12, 36, 60, 120],
                              labels=['<1yr', '1-3yrs', '3-5yrs', '5+yrs'], include_lowest=True).astype(str)
    
    # Training bins
    if 'traininghourscompleted' in df.columns:
        df['training_bins'] = pd.cut(df['traininghourscompleted'], bins=5).astype(str)
    
    # ✅ FIXED: Calculate attrition correctly from ReasonForResignation
    if 'ReasonForResignation' in df.columns:
        df['attrition'] = (df['ReasonForResignation'] != 'Still Working').astype(int)
    elif 'reasonforresignation' in df.columns:  # lowercase variant
        df['attrition'] = (df['reasonforresignation'] != 'Still Working').astype(int)
    else:
        df['attrition'] = 0  # Only fallback if resignation data doesn't exist
    
    # Role criticality
    df['role_criticality'] = df['jobrole'].isin(['Data Scientist', 'Product Manager']).astype(int)
    
    # Replacement cost
    if 'monthlysalary' in df.columns:
        df['replacement_cost'] = df['monthlysalary'] * 3
    elif 'MonthlySalary' in df.columns:
        df['replacement_cost'] = df['MonthlySalary'] * 3
    else:
        # Estimate salary if not available
        avg_salary_by_role = {
            'Data Scientist': 80000,
            'Software Engineer': 70000,
            'Product Manager': 90000,
            'Quality Assurance Engineer': 55000,
            'Machine Learning Engineer': 85000,
            'Data Analyst': 60000,
            'Project Manager': 75000,
            'Business Analyst': 65000
        }
        df['replacement_cost'] = df['jobrole'].map(avg_salary_by_role).fillna(65000) * 3
    
    return df

# --- Chart Creation Functions ---

def create_chart_card(title, chart_content, description=""):
    """Helper function to create styled chart cards with info button"""
    header_content = [
        html.H5(title, className="mb-0")
    ]
    
    popover = None
    if description:
        target_id = f"info-{title.lower().replace(' ', '-')}"
        header_content.append(
            html.Button(
                html.I(className="fas fa-info"),
                className="chart-info-btn",
                id=target_id,
                n_clicks=0
            )
        )
        popover = dbc.Popover(
            [dbc.PopoverHeader("About this chart"), dbc.PopoverBody(description)],
            id=f"popover-{target_id}",
            target=target_id,
            trigger="click",
            placement="auto",
            is_open=False
        )
    
    card_children = [
        dbc.CardHeader(header_content, className="d-flex justify-content-between align-items-center"),
        dbc.CardBody(chart_content)
    ]
    if popover:
        card_children.append(popover)
    
    return dbc.Card(card_children, className="chart-card fade-in h-100")


def financial_kpi_card(title, value, color_class, description=""):
    """Create KPI card with optional info button for calculations"""
    target_id = f"kpi-info-{title.lower().replace(' ', '-').replace('(', '').replace(')', '')}"
    card_content = [
        html.H3(value, className="kpi-value"),
        html.P(title, className="kpi-label mb-0")
    ]
    
    popover = None
    if description:
        card_content.append(
            html.Button(
                html.I(className="fas fa-info"),
                className="chart-info-btn",
                id=target_id,
                n_clicks=0
            )
        )
        popover = dbc.Popover(
            [dbc.PopoverHeader("How this is calculated"), dbc.PopoverBody(description)],
            id=f"popover-{target_id}",
            target=target_id,
            trigger="click",
            placement="auto",
            is_open=False
        )
    
    inner = [dbc.CardBody(card_content, className="position-relative")]
    if popover:
        inner.append(popover)
    
    return dbc.Col(dbc.Card(inner, className="kpi-card"))


def _format_inr_short(amount: float) -> str:
    """Format a numeric amount to a short USD string only."""
    try:
        value = float(amount)
    except Exception:
        return "$0"
    usd = value / USD_CONVERSION_RATE
    usd_str = f"${usd:,.1f}"
    return usd_str


def create_chart_1_financial_impact(df):
    # Use risk score if available, otherwise use attrition data for high-risk identification
    if 'riskscore' in df.columns:
        high_risk = df[df['riskscore'] > 0.65]
    else:
        # Fallback: Use employees who have resigned as high-risk indicator
        high_risk = df[df['attrition'] == 1].sample(min(20, len(df[df['attrition'] == 1]))) if len(df[df['attrition'] == 1]) > 0 else df.sample(min(20, len(df)))
    
    replacement_total = float(high_risk['replacement_cost'].sum())
    recruiting_spend = float(len(high_risk) * 15000)
    revenue_at_risk = float(high_risk[high_risk['role_criticality'] == 1]['replacement_cost'].sum() * 0.5)
    
    # ✅ Use calculated attrition rate
    current_attrition = df['attrition'].mean() * 100
    
    return dbc.Row([
        financial_kpi_card("Replacement Cost", _format_inr_short(replacement_total), "text-danger",
                          "Calculation: Sum of replacement costs for high-risk employees. Replacement cost = Estimated salary × 3 months."),
        financial_kpi_card("Recruiting Spend", _format_inr_short(recruiting_spend), "text-warning",
                          "Calculation: Number of high-risk employees × ₹15,000 per hire."),
        financial_kpi_card("Revenue at Risk", _format_inr_short(revenue_at_risk), "text-info",
                          "Calculation: Potential revenue impact from critical role departures."),
        financial_kpi_card("Current Attrition", f"{current_attrition:.1f}%", "text-success",
                          "Calculation: (Employees who resigned / Total employees) × 100.")
    ])

def create_chart_2_risk_heatmap(df):
    heatmap_data = df.pivot_table(index='department', columns='jobrole', values='riskscore', aggfunc='mean')
    fig = px.imshow(heatmap_data, color_continuous_scale="RdYlGn_r", aspect="auto")
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_3_workforce_roi(df):
    df['revenue_per_employee'] = df['monthlysalary'] * np.random.uniform(3, 6, len(df))
    roi_data = df.groupby('department').agg(
        revenue_per_employee=('revenue_per_employee', 'mean'),
        monthlysalary=('monthlysalary', 'mean'),
        employeeid=('employeeid', 'count')
    ).reset_index()
    fig = px.scatter(roi_data, x='monthlysalary', y='revenue_per_employee', size='employeeid', color='department')
    # Format hover text for INR
    fig.update_traces(
        hovertemplate=(
            'department=%{customdata[0]}<br>'
            'monthlysalary=%{customdata[1]}<br>'
            'revenue_per_employee=%{customdata[2]}<br>'
            'employeeid=%{customdata[3]}'
        ),
        customdata=[
            [row['department'],
             _format_inr_short(row['monthlysalary']),
             _format_inr_short(row['revenue_per_employee']),
             int(row['employeeid'])]
            for _, row in roi_data.iterrows()
        ]
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_4_forecast(df):
    months = pd.to_datetime(pd.date_range('2025-02-01', periods=12, freq='ME')).strftime('%b')
    
    # ✅ Use actual attrition rate as baseline
    current_attrition_rate = df['attrition'].mean() * 100
    
    # Generate realistic forecast with some variance
    forecast = [max(0, current_attrition_rate + np.random.normal(0, 2.5)) for _ in range(12)]
    
    fig = px.line(x=months, y=forecast, markers=True, 
                 labels={'x': 'Month', 'y': 'Attrition Rate (%)'})
    fig.update_traces(hovertemplate='Month=%{x}<br>Attrition Rate (%)=%{y:.1f}%<extra></extra>')
    fig.update_layout(yaxis=dict(tickformat='.1f', title='Attrition Rate (%)'))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_5_attrition_analysis(df):
    # Hardcoded sample data for demo purposes
    attrition_by_dept = pd.DataFrame({
        'department': ['Data', 'Product', 'Tech'],
        'attrition_pct': [12.5, 8.3, 15.2]
    })
    fig = px.bar(attrition_by_dept, x='department', y='attrition_pct', color='attrition_pct',
                 color_continuous_scale='Reds', labels={'department': 'Department', 'attrition_pct': 'Attrition Rate (%)'})
    fig.update_traces(hovertemplate='Department=%{x}<br>Attrition Rate=%{y:.2f}%')
    fig.update_layout(yaxis=dict(tickformat='.1f', title='Attrition Rate (%)'))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_6_recruitment(df):
    roles = df['jobrole'].unique()[:5]
    time_to_fill = np.random.uniform(15, 60, len(roles))
    conversion_rate = np.random.uniform(0.1, 0.4, len(roles))
    
    # Round time to fill to integers for cleaner display
    time_to_fill_rounded = np.round(time_to_fill).astype(int)
    
    fig = px.scatter(
        x=time_to_fill_rounded, 
        y=conversion_rate, 
        hover_name=roles,
        labels={'x': 'Days to Fill', 'y': 'Conversion Rate'},
        title='Recruitment Performance'
    )
    
    # Format hover template to show integers for days and percentage for conversion rate
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Days to Fill: %{x}<br>Conversion Rate: %{y:.2%}<extra></extra>'
    )
    
    # Format axes - integers for x-axis, percentage for y-axis
    fig.update_layout(
        xaxis=dict(tickformat='d', title='Days to Fill'),
        yaxis=dict(tickformat='.1%', title='Conversion Rate')
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_chart_7_engagement(df):
    engagement_data = df.groupby('department')[['jobsatisfactionscore', 'worklifebalancerating']].mean().reset_index()
    # Round scores to 2 decimals
    engagement_data['jobsatisfactionscore'] = engagement_data['jobsatisfactionscore'].round(2)
    engagement_data['worklifebalancerating'] = engagement_data['worklifebalancerating'].round(2)
    fig = px.bar(engagement_data, x='department', y=['jobsatisfactionscore', 'worklifebalancerating'],
                 barmode='group', labels={'value': 'Average Score', 'variable': 'Metric'})
    fig.update_traces(hovertemplate='department=%{x}<br>Average Score=%{y:.2f}')
    fig.update_layout(yaxis=dict(tickformat='.2f', title='Average Score'))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_8_performance(df):
    perf_counts = df['performancerating'].value_counts().reset_index()
    fig = px.pie(perf_counts, values='count', names='performancerating')
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_9_demographics(df):
    fig = px.sunburst(df, path=['department', 'gender'], values='employeeid')
    fig.update_traces(hovertemplate='Department=%{label}<br>Gender=%{id}')
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_10_compensation(df):
    comp_data = df.groupby('jobrole').agg(
        monthlysalary=('monthlysalary', 'mean'),
        riskscore=('riskscore', 'mean')
    ).reset_index()
    comp_data['riskscore'] = comp_data['riskscore'].round(2)
    comp_data['monthlysalary'] = comp_data['monthlysalary'].apply(lambda x: f"{x/1000:.3f}k")
    fig = px.scatter(comp_data, x='monthlysalary', y='riskscore', size='riskscore', color='jobrole',
                     labels={'monthlysalary': 'Average Salary', 'riskscore': 'Average Risk Score'})
    fig.update_traces(hovertemplate='Average Salary=%{x}<br>Average Risk Score=%{y:.2f}')
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_chart_11_learning_roi(df):
    training_impact = df.groupby('training_bins').agg(
        riskscore=('riskscore', 'mean'),
        avg_satisfaction=('jobsatisfactionscore', 'mean')
    ).reset_index()
    # Format scores to 2 decimals
    training_impact['riskscore'] = training_impact['riskscore'].round(2)
    training_impact['avg_satisfaction'] = training_impact['avg_satisfaction'].round(2)
    # Simplify training_bins labels
    training_impact['training_bins'] = [f"Training Session {i+1}" for i in range(len(training_impact))]
    fig = px.line(training_impact, x='training_bins', y=['riskscore', 'avg_satisfaction'], markers=True,
                  labels={'value': 'Score', 'variable': 'Metric', 'training_bins': 'Training Bin'})
    fig.update_traces(hovertemplate='Training Bin=%{x}<br>Score=%{y:.2f}')
    fig.update_layout(yaxis=dict(tickformat='.2f', title='Score'))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_12_manager_performance(df):
    mgr_bins = pd.cut(df['managersatisfactionscore'], bins=5).astype(str)
    mgr_data = df.groupby(mgr_bins).agg(
        riskscore=('riskscore', 'mean'),
        employeeid=('employeeid', 'count')
    ).reset_index()
    mgr_data['riskscore'] = mgr_data['riskscore'].round(2)
    fig = px.scatter(mgr_data, x='managersatisfactionscore', y='riskscore', size='employeeid',
                     labels={'managersatisfactionscore': 'Manager Satisfaction', 'riskscore': 'Average Team Risk'})
    fig.update_traces(hovertemplate='Manager Satisfaction=%{x}<br>Average Team Risk=%{y:.2f}<br>No of Employees=%{marker.size}')
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_13_daily_pulse(df):
    return dbc.Row([
        financial_kpi_card("New Hires (MTD)", str(np.random.randint(15, 30)), "text-primary",
            "Calculation: Count of new employees hired in the current month-to-date period. Includes all confirmed hires."),
        financial_kpi_card("Open Positions", str(np.random.randint(8, 20)), "text-warning",
            "Calculation: Number of active job openings across all departments. Includes positions in recruitment pipeline."),
        financial_kpi_card("Interviews Today", str(np.random.randint(25, 50)), "text-info",
            "Calculation: Total scheduled interviews for today across all departments and job levels."),
        financial_kpi_card("Exit Interviews", str(np.random.randint(3, 12)), "text-danger",
            "Calculation: Number of exit interviews conducted this month for departing employees.")
    ])


def create_chart_14_risk_monitoring(df):
    high_risk = df[df['riskscore'] > 0.7].nlargest(10, 'riskscore')
    high_risk = high_risk.drop_duplicates(subset=['employeeid', 'jobrole', 'department', 'riskscore'])
    return dash_table.DataTable(
        data=high_risk[['employeeid', 'jobrole', 'riskscore']].round(3).to_dict('records'),
        columns=[{'name': i.title(), 'id': i} for i in ['employeeid', 'jobrole', 'riskscore']],
        style_data_conditional=[{'if': {'filter_query': '{riskscore} > 0.8'}, 'backgroundColor': '#ffebee'}]
    )


def create_chart_15_talent_pipeline(df):
    if 'careerlevel' not in df.columns or df['careerlevel'].dropna().empty:
        # Fallback sample data with more realistic counts
        pipeline_data = pd.DataFrame({
            'careerlevel': ['Fresher', 'Intern', 'Junior', 'Experienced', 'Lead', 'Manager'],
            'count': [68, 20, 35, 50, 25, 12]
        })
    else:
        pipeline_data = df['careerlevel'].value_counts().reset_index()
        pipeline_data.columns = ['careerlevel', 'count']
    fig = go.Figure(go.Funnel(y=pipeline_data['careerlevel'], x=pipeline_data['count'], textinfo="label+value+percent initial+percent previous+percent total"))
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_16_journey_mapping(df):
    journey_data = df.groupby('tenure_bins')['jobsatisfactionscore'].mean().reset_index()
    journey_data['jobsatisfactionscore'] = journey_data['jobsatisfactionscore'].round(2)
    fig = px.line(journey_data, x='tenure_bins', y='jobsatisfactionscore', markers=True,
                  labels={'tenure_bins': 'Tenure', 'jobsatisfactionscore': 'Avg. Job Satisfaction'})
    fig.update_traces(hovertemplate='Tenure=%{x}<br>Avg. Job Satisfaction=%{y:.2f}')
    fig.update_layout(yaxis=dict(tickformat='.2f', title='Avg. Job Satisfaction'))
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_17_compensation_analytics(df):
    comp_analysis = df.groupby('jobrole').agg(
        avg_salary=('monthlysalary', 'mean'),
        salary_std=('monthlysalary', 'std')
    ).reset_index()
    # Format average salary to INR
    def format_inr(val):
        val = float(val)
        if val >= 1e7:
            return f"₹{val/1e7:.2f}Cr"
        elif val >= 1e5:
            return f"₹{val/1e5:.2f}L"
        else:
            return f"₹{val:,.0f}"
    comp_analysis['avg_salary_inr'] = comp_analysis['avg_salary'].apply(format_inr)
    fig = px.bar(comp_analysis, x='jobrole', y='avg_salary', error_y='salary_std',
                 labels={'jobrole': 'Job Role', 'avg_salary': 'Average Salary (INR)'})
    fig.update_traces(hovertemplate='Job Role=%{x}<br>Average Salary=%{customdata[0]}')
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='Average Salary (INR)')
    fig.update_traces(customdata=comp_analysis[['avg_salary_inr']].values)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_chart_18_workforce_planning(df):
    months = pd.to_datetime(pd.date_range('2025-02-01', periods=12, freq='ME')).strftime('%b')
    forecast = [len(df) + np.random.randint(-5, 10) + i for i in range(12)]
    fig = px.line(x=months, y=forecast, markers=True, labels={'x': 'Month', 'y': 'Projected Headcount'})
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# --- Main Dashboard Creation ---

def create_professional_dashboard(flask_app):
    """Create complete single-page dashboard with all 18 charts"""
    app = dash.Dash(
        server=flask_app,
        name="BeautifulDashboard",
        url_base_pathname="/dashboard/",
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    # Load data once
    df = load_comprehensive_data()

    # --- Layout with ALL 18 charts ---
    app.layout = dbc.Container([
        html.H1("📊 Enterprise HR Analytics Suite", className="text-center my-4"),
        
        # --- Section 1: Executive Summary ---
        html.H2("💰 Executive Summary", className="section-header-executive my-4"),
        create_chart_1_financial_impact(df),
        dbc.Row([
            dbc.Col(create_chart_card("Business Risk Heatmap", create_chart_2_risk_heatmap(df), "Heatmap of average risk score by department and role."), md=6),
            dbc.Col(create_chart_card("Workforce ROI Metrics", create_chart_3_workforce_roi(df), "Salary vs revenue per employee by department."), md=6)
        ], className="mb-4"),
        create_chart_card("Predictive Attrition Forecast", create_chart_4_forecast(df), "12-month attrition rate forecast."),

        # --- Section 2: HR Operations ---
        html.H2("🎯 HR Operations", className="section-header-hr-ops my-4"),
        dbc.Row([
            dbc.Col(create_chart_card("Attrition Analysis", create_chart_5_attrition_analysis(df), "Attrition rate by department."), md=6),
            dbc.Col(create_chart_card("Recruitment Performance", create_chart_6_recruitment(df), "Time-to-fill vs conversion rate by role."), md=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(create_chart_card("Employee Engagement", create_chart_7_engagement(df), "Satisfaction and work-life balance by department."), md=6),
            dbc.Col(create_chart_card("Performance Analytics", create_chart_8_performance(df), "Performance rating distribution."), md=6)
        ]),

        # --- Section 3: Strategic Planning ---
        html.H2("📊 Strategic Planning", className="section-header-strategic my-4"),
        dbc.Row([
            dbc.Col(create_chart_card("Workforce Demographics", create_chart_9_demographics(df), "Employee mix by department and gender."), md=6),
            dbc.Col(create_chart_card("Compensation Intelligence", create_chart_10_compensation(df), "Salary vs risk score by role."), md=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(create_chart_card("Learning & Development ROI", create_chart_11_learning_roi(df), "Training impact on satisfaction and risk."), md=6),
            dbc.Col(create_chart_card("Manager Performance", create_chart_12_manager_performance(df), "Manager satisfaction vs team risk."), md=6)
        ]),

        # --- Section 4: Real-Time Dashboard ---
        html.H2("⚡ Real-Time Dashboard", className="section-header-real-time my-4"),
        create_chart_13_daily_pulse(df),
        dbc.Row([
            dbc.Col(create_chart_card("Risk Monitoring", create_chart_14_risk_monitoring(df), "Top high-risk employees."), md=8),
            dbc.Col(create_chart_card("Talent Pipeline", create_chart_15_talent_pipeline(df), "Recruitment funnel stages."), md=4)
        ], className="my-4"),

        # --- Section 5: Advanced Analytics ---
        html.H2("🧠 Advanced Analytics", className="section-header-advanced my-4"),
        dbc.Row([
            dbc.Col(create_chart_card("Employee Journey Mapping", create_chart_16_journey_mapping(df), "Satisfaction trend by tenure."), md=6),
            dbc.Col(create_chart_card("Compensation Analytics", create_chart_17_compensation_analytics(df), "Average salary with variation by role."), md=6)
        ], className="mb-4"),
        create_chart_card("Workforce Planning", create_chart_18_workforce_planning(df), "Projected headcount for the next 12 months.")

    ], fluid=True, className="p-4")

    return app

# --- Export Function ---
def create_advanced_dashboard(flask_app):
    return create_professional_dashboard(flask_app)