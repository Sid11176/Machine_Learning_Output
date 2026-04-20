"""
main.py
-------
FastAPI inference service for the Solar Inverter Anomaly Detection model.

Endpoints
---------
GET  /              → health check
POST /predict       → single reading prediction
POST /predict/batch → batch of readings prediction

Start the server:
    uvicorn api.main:app --reload
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import FEATURES_V2

# ── Load model artifacts ─────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try:
    model      = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest_v2.pkl'))
    scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler_v2.pkl'))
    p_rated_map = joblib.load(os.path.join(MODELS_DIR, 'p_rated_map.pkl'))
except FileNotFoundError:
    raise RuntimeError(
        "Model artifacts not found. Run 'python api/train_and_save.py' first."
    )

# ── Constants ────────────────────────────────────────────────────────────────
TEMP_COEFFICIENT = 0.005

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Solar Inverter Anomaly Detection API",
    description=(
        "Predicts anomalies in solar inverter readings using "
        "Isolation Forest V2. Trained on Plant 1 data from the "
        "Kaggle Solar Power Generation Dataset."
    ),
    version="1.0.0",
)


# ── Schema ────────────────────────────────────────────────────────────────────
class InverterReading(BaseModel):
    """A single 15-minute inverter reading."""
    date_time:          str   = Field(..., example="2020-05-15 10:15:00",
                                      description="Datetime string (YYYY-MM-DD HH:MM:SS)")
    source_key:         str   = Field(..., example="sjndEbLyjtCKgGv",
                                      description="Inverter identifier")
    dc_power:           float = Field(..., example=8500.0,  description="DC power output (kW)")
    ac_power:           float = Field(..., example=8330.0,  description="AC power output (kW)")
    irradiation:        float = Field(..., example=0.75,    description="Solar irradiation (W/m²)")
    module_temperature: float = Field(..., example=42.0,    description="Module temperature (°C)")


class PredictionResult(BaseModel):
    source_key:        str
    date_time:         str
    is_anomaly:        bool
    anomaly_label:     int   = Field(..., description="-1 = anomaly, 1 = normal (raw IF output)")
    anomaly_score:     float = Field(..., description="Isolation Forest anomaly score (lower = more anomalous)")
    features_used:     dict


class BatchPredictionResult(BaseModel):
    total_readings:  int
    total_anomalies: int
    flag_rate_pct:   float
    results:         List[PredictionResult]


# ── Feature engineering (inference) ──────────────────────────────────────────
def build_features(reading: InverterReading) -> dict:
    """
    Compute all V2 features for a single reading.
    NOTE: ROLLING_DEVIATION requires historical context.
    For single-reading inference, we set it to 0.0 (neutral).
    For batch inference over a time series, pass readings in chronological
    order and this function computes rolling deviation across the batch.
    """
    dt = pd.to_datetime(reading.date_time)
    hour = dt.hour

    efficiency_ratio = reading.ac_power / reading.dc_power
    irr_norm_dc      = reading.dc_power / reading.irradiation

    p_rated = p_rated_map.get(reading.source_key)
    if p_rated is None:
        # Unknown inverter: use fleet mean as fallback
        p_rated = float(np.mean(list(p_rated_map.values())))

    expected_dc  = p_rated * (1 - TEMP_COEFFICIENT * (reading.module_temperature - 25))
    temp_derating = ((expected_dc - reading.dc_power) / p_rated) * 100

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return {
        'EFFICIENCY_RATIO':  efficiency_ratio,
        'IRR_NORM_DC':       irr_norm_dc,
        'TEMP_DERATING':     temp_derating,
        'ROLLING_DEVIATION': 0.0,   # neutral for single-point inference
        'HOUR_SIN':          hour_sin,
        'HOUR_COS':          hour_cos,
    }


def predict_single(reading: InverterReading) -> PredictionResult:
    features = build_features(reading)
    X = np.array([[features[f] for f in FEATURES_V2]])
    X_scaled = scaler.transform(X)

    label = int(model.predict(X_scaled)[0])        # 1 = normal, -1 = anomaly
    score = float(model.score_samples(X_scaled)[0]) # lower = more anomalous

    return PredictionResult(
        source_key    = reading.source_key,
        date_time     = reading.date_time,
        is_anomaly    = label == -1,
        anomaly_label = label,
        anomaly_score = round(score, 6),
        features_used = {k: round(v, 6) for k, v in features.items()},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "online",
        "model":   "Isolation Forest V2",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict(reading: InverterReading):
    """
    Predict whether a single inverter reading is anomalous.

    - **is_anomaly**: True if the reading is flagged as an anomaly
    - **anomaly_score**: Lower values indicate more anomalous behavior
    - **features_used**: Engineered feature values sent to the model
    """
    try:
        return predict_single(reading)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Inference"])
def predict_batch(readings: List[InverterReading]):
    """
    Predict anomalies for a batch of readings.

    For time-series batches, send readings in **chronological order**
    per inverter so that rolling deviation is computed correctly.
    The batch endpoint computes rolling deviation from the batch itself.
    """
    if not readings:
        raise HTTPException(status_code=400, detail="Readings list is empty.")
    if len(readings) > 5000:
        raise HTTPException(status_code=400, detail="Maximum batch size is 5000 readings.")

    try:
        # Build DataFrame for rolling deviation computation
        rows = []
        for r in readings:
            feat = build_features(r)
            rows.append({
                'date_time':  pd.to_datetime(r.date_time),
                'source_key': r.source_key,
                'dc_power':   r.dc_power,
                **feat,
            })

        df = pd.DataFrame(rows).sort_values(['source_key', 'date_time'])

        # Recompute rolling deviation across the batch
        df = df.set_index('date_time')
        df['ROLLING_DEVIATION'] = (
            df.groupby('source_key')['dc_power']
              .transform(lambda x: x - x.rolling('1h', min_periods=1).mean())
        )
        df = df.reset_index()

        X = df[FEATURES_V2].values
        X_scaled = scaler.transform(X)
        labels = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)

        results = []
        for i, reading in enumerate(readings):
            label = int(labels[i])
            score = float(scores[i])
            features = {f: round(float(df[FEATURES_V2].iloc[i][f]), 6) for f in FEATURES_V2}
            results.append(PredictionResult(
                source_key    = reading.source_key,
                date_time     = reading.date_time,
                is_anomaly    = label == -1,
                anomaly_label = label,
                anomaly_score = round(score, 6),
                features_used = features,
            ))

        n_anomalies = sum(1 for r in results if r.is_anomaly)

        return BatchPredictionResult(
            total_readings  = len(results),
            total_anomalies = n_anomalies,
            flag_rate_pct   = round(n_anomalies / len(results) * 100, 2),
            results         = results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
