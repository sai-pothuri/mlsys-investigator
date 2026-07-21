"""Simple inference server wrapping the XGBoost model."""
import joblib
import numpy as np
from feature_engineering import build_feature_vector, FEATURE_NAMES


_model = joblib.load("artifacts/model.pkl")
_scaler = joblib.load("artifacts/scaler.pkl")

REQUEST_TIMEOUT_MS = 250   # bumped from 200ms (see ops ticket OPS-1142)
WORKER_POOL_SIZE = 6       # bumped from 4 (p99 latency SLA breach under load)


def predict(record: dict) -> dict:
    features = build_feature_vector(record)
    X = _scaler.transform(features.reshape(1, -1))
    proba = float(_model.predict_proba(X)[0, 1])
    return {"prediction": int(proba >= 0.5), "probability": proba}


def get_features(request_id: str, timeout: float = 0.25):
    features = feature_store.lookup(request_id, timeout=timeout)
    return features
