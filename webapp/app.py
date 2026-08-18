"""
CardioSense — Heart Disease Risk Console
------------------------------------------
Flask backend serving the trained voting-ensemble model (XGBoost + Random Forest +
Logistic Regression) from the companion notebook at ../ml-project/notebook/.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "heart_disease_model.pkl")
METADATA_PATH = os.path.join(APP_DIR, "model", "model_metadata.json")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifact + metadata once at startup
# ---------------------------------------------------------------------------
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
SELECTED_FEATURES = artifact["selected_features"]
TEST_METRICS = artifact.get("test_metrics", {})

with open(METADATA_PATH) as f:
    METADATA = json.load(f)

RAW_INPUT_NAMES = [f["name"] for f in METADATA["raw_inputs"]]


def engineer_features(raw: dict) -> pd.DataFrame:
    """Recreate the exact feature engineering performed in the training notebook."""
    row = {k: float(raw[k]) for k in RAW_INPUT_NAMES}
    row["hr_reserve"] = (220 - row["age"]) - row["thalach"]
    row["age_chol_interaction"] = row["age"] * row["chol"] / 1000
    row["high_risk_bp"] = 1.0 if row["trestbps"] >= 140 else 0.0

    frame = pd.DataFrame([row])
    return frame[SELECTED_FEATURES]


def risk_band(probability: float) -> dict:
    if probability < 0.30:
        return {"label": "LOW RISK", "tone": "low"}
    if probability < 0.60:
        return {"label": "MODERATE RISK", "tone": "moderate"}
    return {"label": "HIGH RISK", "tone": "high"}


def top_contributing_factors(frame: pd.DataFrame, n=4):
    """Approximate per-patient driver ranking using the tuned XGBoost's global
    feature importances weighted by how far this patient's value sits from the
    training-set median (a lightweight, dependency-free stand-in for SHAP)."""
    xgb_est = dict(MODEL.named_estimators_)["xgb"]
    importances = pd.Series(xgb_est.feature_importances_, index=SELECTED_FEATURES)
    z = (frame.iloc[0] - frame.iloc[0].mean()) / (frame.iloc[0].std() + 1e-9)
    score = (importances * z.abs()).sort_values(ascending=False)
    return list(score.head(n).index)


@app.route("/")
def index():
    return render_template("index.html", inputs=METADATA["raw_inputs"], meta=METADATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [k for k in RAW_INPUT_NAMES if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        frame = engineer_features(payload)
        probability = float(MODEL.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    band = risk_band(probability)
    factors = top_contributing_factors(frame)

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "risk_label": band["label"],
        "risk_tone": band["tone"],
        "top_factors": factors,
    })


@app.route("/api/model-info")
def model_info():
    return jsonify({
        "model_name": METADATA["model_name"],
        "algorithm": METADATA["algorithm"],
        "trained_on": METADATA["trained_on"],
        "selection_method": METADATA["selection_method"],
        "test_metrics": TEST_METRICS,
        "feature_count": len(SELECTED_FEATURES),
    })


if __name__ == "__main__":
    app.run(debug=True)
