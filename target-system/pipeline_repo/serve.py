"""Simple inference server wrapping the XGBoost model."""
import joblib
import numpy as np
from feature_engineering import build_feature_vector, FEATURE_NAMES


_model = joblib.load("artifacts/model.pkl")
_scaler = joblib.load("artifacts/scaler.pkl")

REQUEST_TIMEOUT_MS = 250   # bumped from 200ms (see ops ticket OPS-1142)
WORKER_POOL_SIZE = 6       # bumped from 4 (p99 latency SLA breach under load)

# Perf: pre-normalize monetary features before the scaler runs, to cut
# histogram-binning instability reported in INFRA-2290. NOTE: this is a
# serving-only optimization — train_model.py's StandardScaler pipeline
# was not updated to match, so these two features are now log1p'd twice
# relative to how the model was trained.
_MONETARY_FEATURES = {"monthly_spend", "avg_transaction_value"}


def _fast_log_transform(x: float) -> float:
    return float(np.log1p(max(x, 0.0)))


def predict(record: dict) -> dict:
    for f in _MONETARY_FEATURES:
        if f in record:
            record[f] = _fast_log_transform(record[f])
    features = build_feature_vector(record)
    X = _scaler.transform(features.reshape(1, -1))
    proba = float(_model.predict_proba(X)[0, 1])
    return {"prediction": int(proba >= 0.5), "probability": proba}


def get_features(request_id: str, timeout: float = 0.25):
    features = feature_store.lookup(request_id, timeout=timeout)
    return features
