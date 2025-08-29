# app/main.py
import os
from pathlib import Path
from flask import Flask, request, jsonify, redirect, send_from_directory
import pandas as pd
from flask import render_template

# ----------------- Resolve project paths (no .env needed) -----------------
APP_DIR = Path(__file__).resolve().parent                   # .../Human-Resource-main/app
BASE_DIR = APP_DIR.parent                                   # .../Human-Resource-main

DB_PATH = "data/processed/hr_data.db"
MODEL_PATH = "models/attrition_pipeline_real.joblib"
ASSETS_PATH = str(APP_DIR / "assets")

# Ensure dashboard modules see the right DB path (they read from env)
os.environ["DB_PATH"] = DB_PATH

# ----------------- Imports that depend on paths above -----------------
from enhanced_advanced_dashboard import create_enhanced_dashboard
from chatbot_backend import (
    get_chatbot_response,
    api_forecast,
    api_bulk_forecast,
    api_create_action_plan,
)

# ----------------- Flask -----------------
server = Flask(__name__)

# --------- Static assets (only if you have files under app/assets) -----
@server.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(ASSETS_PATH, filename)

# ----------------- Dashboard (unchanged UI) -----------------
app = create_enhanced_dashboard(server)

@server.route("/")
def index():
    return redirect("/dashboard/")

# ================== New: Attrition APIs ===================

@server.route("/api/attrition/predict", methods=["POST"])
def api_predict_single():
    """
    Body: {"employee_id": 111065, "horizon_days": 60}
    Allowed horizons: 30,40,50,60,90,180
    """
    try:
        data = request.get_json(force=True) or {}
        employee_id = int(data.get("employee_id"))
        horizon_days = int(data.get("horizon_days", 60))
        if horizon_days not in {30, 40, 50, 60, 90, 180}:
            return jsonify({"error": "Invalid horizon_days; use 30,40,50,60,90,180"}), 400
        result = api_forecast(employee_id, horizon_days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to predict: {e}"}), 500


@server.route("/api/attrition/predict/bulk", methods=["POST"])
def api_predict_bulk():
    """
    Body: {"filters":{"department":"Engineering"}, "horizon_days":60, "limit":50}
    """
    try:
        data = request.get_json(force=True) or {}
        filters = data.get("filters", {}) or {}
        horizon_days = int(data.get("horizon_days", 60))
        limit = int(data.get("limit", 100))
        if horizon_days not in {30, 40, 50, 60, 90, 180}:
            return jsonify({"error": "Invalid horizon_days; use 30,40,50,60,90,180"}), 400
        results = api_bulk_forecast(filters, horizon_days, limit)
        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": f"Failed to bulk predict: {e}"}), 500


@server.route("/api/attrition/action-plan", methods=["POST"])
def api_action_plan():
    """
    Body:
    {
      "employee_id": 111065,
      "actions": [
        {"action_type":"Manager 1:1","owner_role":"Manager","sla_days":3,"expected_impact":3,"cost_band":"₹","notes":"High workload"}
      ]
    }
    """
    try:
        data = request.get_json(force=True) or {}
        employee_id = int(data.get("employee_id"))
        actions = data.get("actions", [])
        if not isinstance(actions, list) or not actions:
            return jsonify({"error": "actions must be a non-empty list"}), 400
        result = api_create_action_plan(employee_id, actions)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to create plan: {e}"}), 500


# ================== Chatbot endpoint (button-driven) ===================

@server.route("/api/chatbot", methods=["POST"])
def chatbot_endpoint():
    """
    Body: {"message":"Predict risk (60 days)","role":"hr"}
    role: employee | manager | hr | admin
    """
    try:
        data = request.get_json(force=True) or {}
        msg = (data.get("message") or "").strip()
        role = (data.get("role") or "hr").lower()
        if not msg:
            return jsonify({"error": "Message cannot be empty"}), 400
        response = get_chatbot_response(msg, role=role)
        return jsonify(response)
    except Exception:
        return jsonify({"message": "Sorry, I'm having trouble right now.", "response_type": "text", "confidence": 0.0}), 500


# ================== Health ===================


@server.route("/chat/attrition")
def chat_attrition():
    return render_template("attration.html")

