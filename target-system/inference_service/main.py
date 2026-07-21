"""
Inference service for the binary churn classifier.

Start: uvicorn inference_service.main:app --host 0.0.0.0 --port 8000
       (run from target-system/ directory)

Endpoints:
  POST /predict          — single prediction
  POST /predict/batch    — batch predictions (up to 1000)
  GET  /health           — service health
  GET  /model/info       — model metadata and feature names
"""

from __future__ import annotations
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", str(_ROOT / "model" / "artifacts"))


def _load():
    model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_info.json")) as f:
        info = json.load(f)
    return model, scaler, info


_model, _scaler, _feature_info = _load()
_feature_names: list[str] = _feature_info["feature_names"]
_n_features = len(_feature_names)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    features: dict[str, float] = Field(
        ...,
        description="Feature values keyed by feature name. All 12 features required.",
        example={
            "account_age_days": 720.0,
            "monthly_spend": 245.0,
            "num_transactions_30d": 40.0,
            "avg_transaction_value": 55.0,
            "days_since_last_login": 2.0,
            "support_tickets_90d": 1.0,
            "product_category": 1.0,
            "region": 2.0,
            "device_type": 1.0,
            "login_failure_rate": 0.05,
            "session_duration_min": 18.0,
            "referral_source": 1.0,
        },
    )


class PredictResponse(BaseModel):
    request_id: str
    prediction: int          # 0 = not churn, 1 = churn
    probability: float       # P(churn)
    latency_ms: float


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., max_length=1000)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    training_stats: dict
    feature_distributions: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_vector(features: dict[str, float]) -> np.ndarray:
    missing = [f for f in _feature_names if f not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}. Expected: {_feature_names}",
        )
    return np.array([[features[f] for f in _feature_names]], dtype=np.float32)


def _predict_single(request_id: str, features: dict[str, float]) -> PredictResponse:
    t0 = time.perf_counter()
    X = _build_feature_vector(features)
    X_scaled = _scaler.transform(X)
    proba = float(_model.predict_proba(X_scaled)[0, 1])
    prediction = int(proba >= 0.5)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        probability=proba,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Churn Classifier — Inference Service",
    description="Binary classifier predicting customer churn from tabular features.",
    version="2.1.1",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version="2.1.1",
        n_features=_n_features,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        feature_names=_feature_names,
        training_stats=_feature_info["training_stats"],
        feature_distributions=_feature_info["feature_distributions"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict_single(req.request_id or str(uuid.uuid4()), req.features)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    t0 = time.perf_counter()
    results = [_predict_single(r.request_id or str(uuid.uuid4()), r.features) for r in req.requests]
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(results=results, total_latency_ms=total_ms)
