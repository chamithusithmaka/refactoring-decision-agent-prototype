"""
Flask Web UI for the Refactoring Decision & Planning Agent.
============================================================

A simple web interface for uploading JSON quality reports and generating
refactoring plans.

Usage:
    python app.py
    # Open http://localhost:5000 in your browser.
"""

import json
import os

from flask import Flask, render_template, request, jsonify

from rdp_agent import generate_plan_from_dict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    """Render the upload form."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Accept a JSON quality report and return the refactoring plan."""
    # --- Handle file upload ---
    if "file" in request.files:
        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not uploaded.filename.lower().endswith(".json"):
            return jsonify({"error": "Please upload a .json file."}), 400
        try:
            content = uploaded.read().decode("utf-8")
            data = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Invalid JSON file: {exc}"}), 400

    # --- Handle raw JSON body ---
    elif request.is_json:
        data = request.get_json()
    else:
        return jsonify({"error": "No file or JSON body provided."}), 400

    # --- Generate plan ---
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if not os.path.isfile(config_path):
            config_path = None
        plan = generate_plan_from_dict(data, config_path=config_path)
        return jsonify({"success": True, "plan": plan})
    except Exception as exc:
        return jsonify({"error": f"Plan generation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
