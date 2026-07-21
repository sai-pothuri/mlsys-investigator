"""Train XGBoost binary churn classifier."""
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from feature_engineering import FEATURE_NAMES, build_feature_vector


MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

# Tuned hyperparams (rolling window 3mo, includes referral_source)
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "early_stopping_rounds": 20,
}

TRAINING_WINDOW_DAYS = 90   # was 180


def train(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = xgb.XGBClassifier(**HYPERPARAMS, random_state=42, eval_metric="logloss")
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )
    return model, scaler
