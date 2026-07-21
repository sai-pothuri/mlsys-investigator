"""
Shared pytest fixtures for the ML system test suite.

All fixtures that touch the filesystem use tmp_path so tests stay isolated.
The `generated_data` fixture is session-scoped and generates 1 day of data
once — reused across all tests that need a populated DB.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

# Ensure target-system/ is on the path regardless of where pytest is invoked
_TARGET = Path(__file__).parent.parent
if str(_TARGET) not in sys.path:
    sys.path.insert(0, str(_TARGET))

from generator.defaults import SIM_START, make_default_config
from generator.generate import generate_data, load_model_artifacts
from generator.schema import setup_databases


MODEL_DIR = str(_TARGET / "model" / "artifacts")
PIPELINE_REPO = str(_TARGET / "pipeline_repo")


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tmp_db_paths(tmp_path):
    """Empty databases with correct schema, in a temporary directory."""
    return setup_databases(str(tmp_path))


@pytest.fixture(scope="session")
def loaded_model():
    """Loaded model, scaler, and feature_info (session-scoped — expensive)."""
    return load_model_artifacts(MODEL_DIR)


@pytest.fixture(scope="session")
def feature_names(loaded_model):
    _, _, info = loaded_model
    return info["feature_names"]


@pytest.fixture(scope="session")
def gt_weights(loaded_model):
    _, _, info = loaded_model
    return np.array(info["ground_truth_weights"])


@pytest.fixture(scope="session")
def gt_bias(loaded_model):
    _, _, info = loaded_model
    return float(info["ground_truth_bias"])


@pytest.fixture(scope="session")
def generated_data(tmp_path_factory, loaded_model):
    """
    1-day generated dataset in a temporary directory.
    Session-scoped so it's generated once and reused.
    """
    out_dir = str(tmp_path_factory.mktemp("generated"))
    config = make_default_config(n_days=1)
    config.output_dir = out_dir
    config.seed = 42
    db_paths = generate_data(config, MODEL_DIR, verbose=False)
    return db_paths, config


@pytest.fixture
def default_config(tmp_path):
    """Fresh 1-day GenerationConfig pointing to tmp_path."""
    config = make_default_config(n_days=1)
    config.output_dir = str(tmp_path)
    return config


# ---------------------------------------------------------------------------
# Inference service client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def inference_client():
    from fastapi.testclient import TestClient
    from inference_service.main import app
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers used by multiple test modules
# ---------------------------------------------------------------------------

VALID_FEATURES = {
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
}
