"""StudyCast – Exam Score Estimator (CSCI104 Option B).

Flask web app: serves the page and a /api/predict endpoint that runs the
scikit-learn model trained in notebook/Exam_Score_Model.ipynb (model.joblib).
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__, template_folder=str(BASE / "templates"))

# name -> (accepted min, accepted max, whole number?)
NUMERIC = {
    "Hours_Studied": (0, 60, False),
    "Attendance": (0, 100, False),
    "Previous_Scores": (0, 100, False),
    "Tutoring_Sessions": (0, 20, True),
}
RESOURCES = ("Low", "Medium", "High")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(BASE / "model.joblib")
    return _model


def validate(data):
    clean, errors = {}, {}
    for name, (lo, hi, whole) in NUMERIC.items():
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = f"This field is required. Enter a number from {lo} to {hi}."
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"Enter a number from {lo} to {hi}."
            continue
        if not lo <= v <= hi:
            errors[name] = f"{v:g} is out of range. Enter a number from {lo} to {hi}."
        elif whole and v != int(v):
            errors[name] = "Enter a whole number of sessions (e.g. 0, 1, 2)."
        else:
            clean[name] = v
    if data.get("Access_to_Resources") not in RESOURCES:
        errors["Access_to_Resources"] = "Choose Low, Medium or High."
    else:
        clean["Access_to_Resources"] = data["Access_to_Resources"]
    return clean, errors


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    clean, errors = validate(data)
    if errors:
        return jsonify(ok=False, errors=errors), 400
    try:
        if data.get("simulate_failure"):  # used only for test 4 in the report
            raise RuntimeError("Simulated model failure")
        row = pd.DataFrame([clean])[list(NUMERIC) + ["Access_to_Resources"]]
        score = float(get_model().predict(row)[0])
    except Exception:
        return jsonify(ok=False, message="We couldn't calculate an estimate. Please try again."), 500
    return jsonify(ok=True, score=round(max(0.0, min(100.0, score)), 1))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
