"""
Creates pipeline_repo/ with realistic git commit history.

Each commit is timestamped to a point in the simulated time window and
reflects a real code change. Commit SHAs are printed at the end and can
be patched into generator/defaults.py or a chaos injection config.

Run: python setup_pipeline_repo.py [--repo-dir pipeline_repo]
     (from target-system/ directory)
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path


# Simulated start: 2024-01-01 00:00:00 UTC
SIM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _ts(day: float, hour: float = 0) -> str:
    """ISO 8601 string for git commit date."""
    from datetime import timedelta
    dt = SIM_START + timedelta(days=day, hours=hour)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def run(cmd: list[str], cwd: str, env: dict = None):
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=cwd, env=full_env, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write(repo: str, path: str, content: str):
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(textwrap.dedent(content))


def commit(repo: str, message: str, date_iso: str) -> str:
    env = {"GIT_AUTHOR_DATE": date_iso, "GIT_COMMITTER_DATE": date_iso}
    run(["git", "add", "-A"], cwd=repo)
    run(["git", "commit", "-m", message], cwd=repo, env=env)
    sha = run(["git", "rev-parse", "HEAD"], cwd=repo)
    print(f"  [{sha[:8]}] {message}")
    return sha


def setup(repo_dir: str) -> dict[str, str]:
    """Create the repo. Returns mapping of label → commit SHA."""
    if os.path.exists(os.path.join(repo_dir, ".git")):
        print(f"Repo already exists at {repo_dir!r}. Remove it to re-create.")
        # Still read existing SHAs
        shas = {}
        log = run(["git", "log", "--oneline"], cwd=repo_dir)
        for line in log.splitlines():
            sha, *msg_parts = line.split(" ", 1)
            msg = msg_parts[0] if msg_parts else ""
            if "Initial commit" in msg:           shas["sha_1"] = run(["git", "rev-parse", sha], cwd=repo_dir)
            elif "Add feature normalization" in msg: shas["sha_2"] = run(["git", "rev-parse", sha], cwd=repo_dir)
            elif "Add referral_source" in msg:    shas["sha_3"] = run(["git", "rev-parse", sha], cwd=repo_dir)
            elif "Tune XGBoost" in msg:           shas["sha_4"] = run(["git", "rev-parse", sha], cwd=repo_dir)
        return shas

    os.makedirs(repo_dir, exist_ok=True)
    run(["git", "init", "-b", "main"], cwd=repo_dir)
    run(["git", "config", "user.email", "mlops@example.com"], cwd=repo_dir)
    run(["git", "config", "user.name", "MLOps Bot"], cwd=repo_dir)
    print(f"Initialized git repo at {repo_dir!r}")

    shas: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Commit 1 — initial implementation (simulated: 2023-12-01, pre-window)
    # -----------------------------------------------------------------------
    write(repo_dir, "feature_engineering.py", """\
        \"\"\"Feature engineering pipeline for churn classifier.\"\"\"
        import numpy as np


        FEATURE_NAMES = [
            "account_age_days",
            "monthly_spend",
            "num_transactions_30d",
            "avg_transaction_value",
            "days_since_last_login",
            "support_tickets_90d",
            "product_category",
            "region",
            "device_type",
            "login_failure_rate",
            "session_duration_min",
        ]
        EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)


        def validate_features(record: dict) -> dict:
            for field in FEATURE_NAMES:
                if field not in record:
                    raise ValidationError(f"Missing required field: {field}")
            return record


        def build_feature_vector(record: dict) -> np.ndarray:
            validate_features(record)
            features = [record[f] for f in FEATURE_NAMES]
            assert len(features) == EXPECTED_FEATURE_COUNT
            return np.array(features, dtype=np.float32)
        """)

    write(repo_dir, "train_model.py", """\
        \"\"\"Train XGBoost binary churn classifier.\"\"\"
        import joblib
        import numpy as np
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        from feature_engineering import FEATURE_NAMES, build_feature_vector


        MODEL_PATH = "artifacts/model.pkl"
        SCALER_PATH = "artifacts/scaler.pkl"

        HYPERPARAMS = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        }


        def train(X_train, y_train, X_val, y_val):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = xgb.XGBClassifier(**HYPERPARAMS, random_state=42)
            model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            return model, scaler
        """)

    write(repo_dir, "serve.py", """\
        \"\"\"Simple inference server wrapping the XGBoost model.\"\"\"
        import joblib
        import numpy as np
        from feature_engineering import build_feature_vector, FEATURE_NAMES


        _model = joblib.load("artifacts/model.pkl")
        _scaler = joblib.load("artifacts/scaler.pkl")

        REQUEST_TIMEOUT_MS = 200
        WORKER_POOL_SIZE = 4


        def predict(record: dict) -> dict:
            features = build_feature_vector(record)
            X = _scaler.transform(features.reshape(1, -1))
            proba = float(_model.predict_proba(X)[0, 1])
            return {"prediction": int(proba >= 0.5), "probability": proba}


        def get_features(request_id: str, timeout: float = 0.2):
            features = feature_store.lookup(request_id, timeout=timeout)
            return features
        """)

    write(repo_dir, "README.md", """\
        # Churn Classifier Pipeline

        Binary churn prediction model. XGBoost on tabular features.

        ## Features
        See `feature_engineering.py` for the full feature list.

        ## Training
        Run `python train_model.py`
        """)

    shas["sha_initial"] = commit(repo_dir, "Initial commit: churn classifier v1.0", _ts(-31))

    # -----------------------------------------------------------------------
    # Commit 2 — Add feature normalization (2023-12-10)
    # -----------------------------------------------------------------------
    write(repo_dir, "feature_engineering.py", """\
        \"\"\"Feature engineering pipeline for churn classifier.\"\"\"
        import numpy as np


        FEATURE_NAMES = [
            "account_age_days",
            "monthly_spend",
            "num_transactions_30d",
            "avg_transaction_value",
            "days_since_last_login",
            "support_tickets_90d",
            "product_category",
            "region",
            "device_type",
            "login_failure_rate",
            "session_duration_min",
        ]
        EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)

        # Clip bounds for normalizing continuous features
        CLIP_BOUNDS = {
            "account_age_days":     (0, 3650),
            "monthly_spend":        (0, 10000),
            "num_transactions_30d": (0, 500),
            "days_since_last_login": (0, 365),
            "support_tickets_90d":  (0, 50),
            "session_duration_min": (0, 480),
        }


        def validate_features(record: dict) -> dict:
            for field in FEATURE_NAMES:
                if field not in record:
                    raise ValidationError(f"Missing required field: {field}")
            return record


        def normalize_feature(name: str, value: float) -> float:
            if name in CLIP_BOUNDS:
                lo, hi = CLIP_BOUNDS[name]
                return float(np.clip(value, lo, hi))
            return value


        def build_feature_vector(record: dict) -> np.ndarray:
            validate_features(record)
            features = [normalize_feature(f, record[f]) for f in FEATURE_NAMES]
            assert len(features) == EXPECTED_FEATURE_COUNT
            return np.array(features, dtype=np.float32)
        """)

    shas["sha_1"] = commit(repo_dir, "Add feature normalization with clip bounds", _ts(-21))

    # -----------------------------------------------------------------------
    # Commit 3 — Fix off-by-one in session_duration (2023-12-20)
    # -----------------------------------------------------------------------
    write(repo_dir, "feature_engineering.py", """\
        \"\"\"Feature engineering pipeline for churn classifier.\"\"\"
        import numpy as np


        FEATURE_NAMES = [
            "account_age_days",
            "monthly_spend",
            "num_transactions_30d",
            "avg_transaction_value",
            "days_since_last_login",
            "support_tickets_90d",
            "product_category",
            "region",
            "device_type",
            "login_failure_rate",
            "session_duration_min",
        ]
        EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)

        CLIP_BOUNDS = {
            "account_age_days":     (0, 3650),
            "monthly_spend":        (0, 10000),
            "num_transactions_30d": (0, 500),
            "days_since_last_login": (0, 365),
            "support_tickets_90d":  (0, 50),
            "session_duration_min": (0, 120),  # fix: was 480, users cap at 2h sessions
        }


        def validate_features(record: dict) -> dict:
            for field in FEATURE_NAMES:
                if field not in record:
                    raise ValidationError(f"Missing required field: {field}")
            return record


        def normalize_feature(name: str, value: float) -> float:
            if name in CLIP_BOUNDS:
                lo, hi = CLIP_BOUNDS[name]
                return float(np.clip(value, lo, hi))
            return value


        def build_feature_vector(record: dict) -> np.ndarray:
            validate_features(record)
            features = [normalize_feature(f, record[f]) for f in FEATURE_NAMES]
            assert len(features) == EXPECTED_FEATURE_COUNT
            return np.array(features, dtype=np.float32)
        """)

    shas["sha_fix_session"] = commit(repo_dir, "Fix session_duration_min clip bound (480 → 120)", _ts(-11))

    # -----------------------------------------------------------------------
    # Commit 4 — Config change: bump timeout + workers (Day 0 of sim window)
    # -----------------------------------------------------------------------
    write(repo_dir, "serve.py", """\
        \"\"\"Simple inference server wrapping the XGBoost model.\"\"\"
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
        """)

    shas["sha_2"] = commit(repo_dir, "Bump request timeout to 250ms; increase worker pool to 6", _ts(0, 8))

    # -----------------------------------------------------------------------
    # Commit 5 — Add referral_source feature (Day 2)
    # -----------------------------------------------------------------------
    write(repo_dir, "feature_engineering.py", """\
        \"\"\"Feature engineering pipeline for churn classifier.\"\"\"
        import numpy as np


        FEATURE_NAMES = [
            "account_age_days",
            "monthly_spend",
            "num_transactions_30d",
            "avg_transaction_value",
            "days_since_last_login",
            "support_tickets_90d",
            "product_category",
            "region",
            "device_type",
            "login_failure_rate",
            "session_duration_min",
            "referral_source",   # NEW: 0=organic, 1=paid, 2=partner, 3=unknown
        ]
        EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)

        CLIP_BOUNDS = {
            "account_age_days":     (0, 3650),
            "monthly_spend":        (0, 10000),
            "num_transactions_30d": (0, 500),
            "days_since_last_login": (0, 365),
            "support_tickets_90d":  (0, 50),
            "session_duration_min": (0, 120),
        }

        VALID_CATEGORICALS = {
            "product_category": {0, 1, 2},
            "region":           {0, 1, 2, 3, 4},
            "device_type":      {0, 1, 2},
            "referral_source":  {0, 1, 2, 3},
        }


        def validate_features(record: dict) -> dict:
            for field in FEATURE_NAMES:
                if field not in record:
                    # referral_source defaults to 0 if absent (backwards compat)
                    if field == "referral_source":
                        record[field] = 0
                    else:
                        raise ValidationError(f"Missing required field: {field}")
            for cat, valid in VALID_CATEGORICALS.items():
                if record.get(cat) not in valid:
                    import warnings
                    warnings.warn(f"Unexpected value for {cat}: {record[cat]}")
            return record


        def normalize_feature(name: str, value: float) -> float:
            if name in CLIP_BOUNDS:
                lo, hi = CLIP_BOUNDS[name]
                return float(np.clip(value, lo, hi))
            return value


        def build_feature_vector(record: dict) -> np.ndarray:
            validate_features(record)
            features = [normalize_feature(f, record[f]) for f in FEATURE_NAMES]
            assert len(features) == EXPECTED_FEATURE_COUNT
            return np.array(features, dtype=np.float32)
        """)

    shas["sha_3"] = commit(repo_dir, "Add referral_source feature; add categorical validation", _ts(2, 13))

    # -----------------------------------------------------------------------
    # Commit 6 — Retrain with new feature (Day 4) — tied to model_retrain deploy
    # -----------------------------------------------------------------------
    write(repo_dir, "train_model.py", """\
        \"\"\"Train XGBoost binary churn classifier.\"\"\"
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
        """)

    shas["sha_retrain"] = commit(repo_dir, "Tune XGBoost hyperparams; retrain on 3-month window", _ts(4, 10))

    # -----------------------------------------------------------------------
    # Commit 7 — Dependency bump (Day 6)
    # -----------------------------------------------------------------------
    write(repo_dir, "requirements.txt", """\
        xgboost==2.0.3      # was 1.7.6
        scikit-learn>=1.3.0
        numpy>=1.24.0
        joblib>=1.3.0
        fastapi>=0.100.0
        uvicorn>=0.23.0
        """)

    shas["sha_4"] = commit(repo_dir, "Upgrade xgboost 1.7.6 → 2.0.3", _ts(6, 7))

    # -----------------------------------------------------------------------
    # Print SHA summary for defaults.py
    # -----------------------------------------------------------------------
    print("\nCommit SHAs (update generator/defaults.py with these):")
    for label, sha in shas.items():
        print(f"  {label}: {sha}")

    return shas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="pipeline_repo")
    args = parser.parse_args()
    shas = setup(args.repo_dir)

    # Patch defaults.py with actual SHAs
    defaults_path = os.path.join(os.path.dirname(__file__), "generator", "defaults.py")
    with open(defaults_path) as f:
        content = f.read()

    # Each deploy event maps to the commit that represents what changed
    # sha_2 = "Bump timeout" → config_change deploy (day 0)
    # sha_3 = "Add referral_source" → feature_pipeline_change deploy (day 2)
    # sha_retrain = "Tune XGBoost" → model_retrain deploy (day 4)
    # sha_4 = "Upgrade xgboost" → dependency_bump deploy (day 6)
    mapping = {
        shas.get("sha_2", ""):       "PLACEHOLDER_SHA_1",
        shas.get("sha_3", ""):       "PLACEHOLDER_SHA_2",
        shas.get("sha_retrain", ""): "PLACEHOLDER_SHA_3",
        shas.get("sha_4", ""):       "PLACEHOLDER_SHA_4",
    }
    for sha, placeholder in mapping.items():
        if sha:
            content = content.replace(placeholder, sha)

    with open(defaults_path, "w") as f:
        f.write(content)
    print(f"\nPatched {defaults_path} with real commit SHAs.")
