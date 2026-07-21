"""
Train a binary XGBoost classifier on synthetic tabular data and save artifacts.

Run:  python -m model.train [--output-dir model/artifacts]

Artifacts written:
  model.pkl          — trained XGBoost model (joblib)
  scaler.pkl         — StandardScaler fitted on training features (joblib)
  feature_info.json  — feature names, ground-truth weights, training stats
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------------------------------------------------------------------------
# Ground-truth label function
#
# Labels are a deterministic function of features + noise. Both training and
# simulation use this same function so accuracy is well-defined and shifts
# when features drift.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "account_age_days",      # normal(730, 200)
    "monthly_spend",         # lognormal(mean=5.5, std=0.8) → ~$245 avg
    "num_transactions_30d",  # normal(42, 15)
    "avg_transaction_value", # lognormal(mean=4.0, std=0.6)
    "days_since_last_login", # normal(3, 2) clipped to [0, ∞)
    "support_tickets_90d",   # normal(1.2, 1.5) clipped to [0, ∞)
    "product_category",      # categorical: 0=Basic, 1=Pro, 2=Enterprise
    "region",                # categorical: 0-4
    "device_type",           # categorical: 0=mobile, 1=desktop, 2=tablet
    "login_failure_rate",    # uniform(0, 0.3)
    "session_duration_min",  # normal(18, 8) clipped to [0, ∞)
    "referral_source",       # categorical: 0-3
]

# Weights for linear score → label (ground truth, not learned by model)
GROUND_TRUTH_WEIGHTS = np.array([
    -0.0010,   # account_age_days: older → less churn
    -0.0005,   # monthly_spend: higher spend → less churn
    -0.0080,   # num_transactions_30d: more engaged → less churn
    -0.0030,   # avg_transaction_value: higher value → less churn
     0.1500,   # days_since_last_login: longer away → churn
     0.2000,   # support_tickets_90d: more tickets → churn
    -0.3000,   # product_category: Enterprise(2) strongly anti-churn
     0.0500,   # region: slight effect
     0.1000,   # device_type: slight effect
     1.5000,   # login_failure_rate: high failure rate → strongest churn signal
    -0.0150,   # session_duration_min: longer sessions → less churn
     0.0500,   # referral_source: slight effect
])
GROUND_TRUTH_BIAS = 0.35   # calibrates overall churn rate to ~25%


def compute_label_score(X: np.ndarray) -> np.ndarray:
    """Deterministic score used for both training labels and simulation labels."""
    return X @ GROUND_TRUTH_WEIGHTS + GROUND_TRUTH_BIAS


def assign_labels(X: np.ndarray, noise_std: float = 0.3, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    score = compute_label_score(X) + rng.normal(0, noise_std, len(X))
    return (score > 0).astype(int)


# ---------------------------------------------------------------------------
# Feature generation for training data
# ---------------------------------------------------------------------------

def generate_training_features(n: int, rng: np.random.Generator) -> np.ndarray:
    cols = [
        np.clip(rng.normal(730, 200, n), 0, None),           # account_age_days
        np.exp(rng.normal(5.5, 0.8, n)),                     # monthly_spend
        np.clip(rng.normal(42, 15, n), 0, None),             # num_transactions_30d
        np.exp(rng.normal(4.0, 0.6, n)),                     # avg_transaction_value
        np.clip(rng.normal(3, 2, n), 0, None),               # days_since_last_login
        np.clip(rng.normal(1.2, 1.5, n), 0, None),          # support_tickets_90d
        rng.choice([0, 1, 2], size=n, p=[0.50, 0.35, 0.15]).astype(float),  # product_category
        rng.choice([0, 1, 2, 3, 4], size=n, p=[0.20, 0.30, 0.25, 0.15, 0.10]).astype(float),  # region
        rng.choice([0, 1, 2], size=n, p=[0.55, 0.35, 0.10]).astype(float),  # device_type
        rng.uniform(0, 0.3, n),                              # login_failure_rate
        np.clip(rng.normal(18, 8, n), 0, None),             # session_duration_min
        rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.35, 0.25, 0.15]).astype(float),  # referral_source
    ]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(output_dir: str, n_train: int = 40_000, n_val: int = 10_000, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Generating {n_train + n_val} synthetic samples...")
    X_all = generate_training_features(n_train + n_val, rng)
    y_all = assign_labels(X_all, rng=rng)

    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    print(f"Training churn rate: {y_train.mean():.3f}")
    print(f"Validation churn rate: {y_val.mean():.3f}")

    # Scale features (XGBoost doesn't strictly need it, but scaler is stored
    # so the inference service can apply it consistently)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=seed,
        verbosity=1,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val_s)
    val_probas = model.predict_proba(X_val_s)[:, 1]
    val_acc = (val_preds == y_val).mean()
    print(f"Validation accuracy: {val_acc:.4f}")

    # Save artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    info_path = os.path.join(output_dir, "feature_info.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    feature_info = {
        "feature_names": FEATURE_NAMES,
        "ground_truth_weights": GROUND_TRUTH_WEIGHTS.tolist(),
        "ground_truth_bias": GROUND_TRUTH_BIAS,
        "training_stats": {
            "n_train": n_train,
            "n_val": n_val,
            "train_churn_rate": float(y_train.mean()),
            "val_accuracy": float(val_acc),
            "val_mean_proba": float(val_probas.mean()),
        },
        "feature_distributions": {
            "account_age_days":      {"distribution": "normal",      "mean": 730,  "std": 200},
            "monthly_spend":         {"distribution": "lognormal",   "mean": 5.5,  "std": 0.8},
            "num_transactions_30d":  {"distribution": "normal",      "mean": 42,   "std": 15},
            "avg_transaction_value": {"distribution": "lognormal",   "mean": 4.0,  "std": 0.6},
            "days_since_last_login": {"distribution": "normal",      "mean": 3,    "std": 2,   "clip_low": 0},
            "support_tickets_90d":   {"distribution": "normal",      "mean": 1.2,  "std": 1.5, "clip_low": 0},
            "product_category":      {"distribution": "categorical", "categories": [0, 1, 2],     "probs": [0.50, 0.35, 0.15]},
            "region":                {"distribution": "categorical", "categories": [0, 1, 2, 3, 4], "probs": [0.20, 0.30, 0.25, 0.15, 0.10]},
            "device_type":           {"distribution": "categorical", "categories": [0, 1, 2],     "probs": [0.55, 0.35, 0.10]},
            "login_failure_rate":    {"distribution": "uniform",     "low": 0,     "high": 0.3},
            "session_duration_min":  {"distribution": "normal",      "mean": 18,   "std": 8,   "clip_low": 0},
            "referral_source":       {"distribution": "categorical", "categories": [0, 1, 2, 3],  "probs": [0.25, 0.35, 0.25, 0.15]},
        },
    }
    with open(info_path, "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"Saved: {model_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {info_path}")
    return model, scaler, feature_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="model/artifacts")
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.output_dir, args.n_train, args.n_val, args.seed)
