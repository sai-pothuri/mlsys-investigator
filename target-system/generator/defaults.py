"""
Default normal-operation GenerationConfig for 7 simulated days.

SIM_START is the canonical simulated epoch — all timestamps in generated
data are offsets from this value. Investigation tools receive absolute
timestamps derived from this base.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone

from .params import (
    DeploymentEvent,
    FeatureConfig,
    FeatureParams,
    GenerationConfig,
    InferenceParams,
    LogParams,
)

# Canonical simulated start: 2024-01-01 00:00:00 UTC
SIM_START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
HOURS_PER_DAY = 24


def _ts(day: float, hour: float = 0) -> float:
    """Offset from SIM_START in days + hours."""
    return SIM_START + day * 86_400 + hour * 3600


def default_feature_params(requests_per_hour: float = 500.0) -> FeatureParams:
    return FeatureParams(
        requests_per_hour=requests_per_hour,
        feature_configs=[
            FeatureConfig("account_age_days",      "normal",      mean=730,  std=200),
            FeatureConfig("monthly_spend",         "lognormal",   mean=5.5,  std=0.8),
            FeatureConfig("num_transactions_30d",  "normal",      mean=42,   std=15),
            FeatureConfig("avg_transaction_value", "lognormal",   mean=4.0,  std=0.6),
            FeatureConfig("days_since_last_login", "normal",      mean=3,    std=2),
            FeatureConfig("support_tickets_90d",   "normal",      mean=1.2,  std=1.5),
            FeatureConfig("product_category",      "categorical", categories=[0, 1, 2],       probs=[0.50, 0.35, 0.15]),
            FeatureConfig("region",                "categorical", categories=[0, 1, 2, 3, 4], probs=[0.20, 0.30, 0.25, 0.15, 0.10]),
            FeatureConfig("device_type",           "categorical", categories=[0, 1, 2],       probs=[0.55, 0.35, 0.10]),
            FeatureConfig("login_failure_rate",    "uniform",     low=0.0,   high=0.3),
            FeatureConfig("session_duration_min",  "normal",      mean=18,   std=8),
            FeatureConfig("referral_source",       "categorical", categories=[0, 1, 2, 3],    probs=[0.25, 0.35, 0.25, 0.15]),
        ],
    )


def default_inference_params() -> InferenceParams:
    return InferenceParams(
        baseline_latency_ms=45.0,
        latency_std_ms=8.0,
        tail_prob=0.02,
        tail_latency_ms=300.0,
        error_rate=0.005,
        timeout_rate=0.001,
    )


def default_log_params() -> LogParams:
    return LogParams(
        info_log_rate=0.05,
        warning_rate=0.01,
        background_error_rate=0.002,
    )


def default_deployments(n_days: int = 7) -> list[DeploymentEvent]:
    """
    Realistic deploy history for a 7-day window.
    commit_sha values match commits in pipeline_repo (created by setup_pipeline_repo.py).
    """
    end_day = n_days
    return [
        DeploymentEvent(
            timestamp=_ts(0, 9),
            version_before="v1.3.1",
            version_after="v1.3.2",
            service="inference_service",
            change_type="config_change",
            changelog="Bump request timeout from 200ms to 250ms; increase worker pool size from 4 to 6.",
            commit_sha="44c6ed470a59111000897e0cc9596f6d3e6ac35f",
            deployed_by="mlops-bot",
        ),
        DeploymentEvent(
            timestamp=_ts(2, 14),
            version_before="v1.3.2",
            version_after="v1.4.0",
            service="feature_pipeline",
            change_type="feature_pipeline_change",
            changelog=(
                "Add referral_source feature; normalize session_duration_min "
                "by clipping at 120 min (previously uncapped, caused outlier issues)."
            ),
            commit_sha="bee909c319a3e4fb7b2d7419d69d3d2ee153e6e6",
            deployed_by="alice",
        ),
        DeploymentEvent(
            timestamp=_ts(4, 11),
            version_before="v2.0.0",
            version_after="v2.1.0",
            service="inference_service",
            change_type="model_retrain",
            changelog=(
                "Retrain XGBoost on 3-month rolling window (previously 6-month). "
                "Adds referral_source feature. Validation AUC 0.847 → 0.861."
            ),
            commit_sha="23c603f791915fd1ca2b900236fcbe5c40be5c5c",
            deployed_by="alice",
        ),
        DeploymentEvent(
            timestamp=_ts(6, 8),
            version_before="v2.1.0",
            version_after="v2.1.1",
            service="inference_service",
            change_type="dependency_bump",
            changelog="Upgrade xgboost 1.7.6 → 2.0.3. No model changes.",
            commit_sha="cf1246e2b5b6714b6308573a1d222b17841b9a7b",
            deployed_by="mlops-bot",
        ),
    ]


def make_default_config(n_days: int = 7) -> GenerationConfig:
    return GenerationConfig(
        start_ts=SIM_START,
        end_ts=SIM_START + n_days * 86_400,
        default_feature_params=default_feature_params(),
        default_inference_params=default_inference_params(),
        default_log_params=default_log_params(),
        overrides=[],
        deployment_events=default_deployments(n_days),
        label_delay_hours=24.0,
        seed=42,
    )
