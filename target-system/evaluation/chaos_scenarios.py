"""
Five easy-tier chaos injection scenarios for the evaluation harness.

Each InjectionScenario specifies:
  - id / ground_truth_category: what the agent should diagnose
  - sub_range: SubRangeConfig covering the failure window (days 5-7)
  - deployment_events: deployments to write into deployments.db
  - diagnostic_logs: log rows injected directly into logs.db after generation
  - alert: text handed to the agent at investigation time
  - investigation_start_ts: the agent's simulated "now" (day 6 hour 6)

Timeline:
  SIM_START = 2024-01-01 00:00 UTC  (Unix 1704067200)
  Failure window : days 5–7  (2024-01-06 to 2024-01-08)
  Investigation  : 2024-01-07 06:00 UTC  (30 h into failure)
  Baseline       : days 1–5  (clean; label_delay_hours=0 so accuracy is immediate)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.params import (
    DeploymentEvent,
    FeatureConfig,
    FeatureParams,
    InferenceParams,
    LogParams,
    SubRangeConfig,
)
from generator.defaults import SIM_START, default_inference_params, default_log_params


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ts(day: float, hour: float = 0.0) -> float:
    return SIM_START + day * 86_400 + hour * 3_600


_log_counters: dict[str, int] = {}

def _log_id(scenario_id: str) -> str:
    """Return a stable, unique log ID with the 'injected-' prefix so the harness
    can clear prior-scenario rows before inserting new ones."""
    n = _log_counters.get(scenario_id, 0)
    _log_counters[scenario_id] = n + 1
    return f"injected-{scenario_id}-{n}"


FAILURE_START_TS  = _ts(5)                # 2024-01-06 00:00 UTC
FAILURE_END_TS    = _ts(7)                # 2024-01-08 00:00 UTC
INVESTIGATION_TS  = _ts(6, 6)            # 2024-01-07 06:00 UTC  (agent's "now")

_INV_DT = datetime.fromtimestamp(INVESTIGATION_TS, tz=timezone.utc)
INVESTIGATION_START = _INV_DT


# ---------------------------------------------------------------------------
# Base deployment history shared by all scenarios
# (only days 0 and 2 — nothing near the day-5 failure window)
# ---------------------------------------------------------------------------

_BASE_DEPLOYMENTS = [
    DeploymentEvent(
        timestamp=_ts(0, 9),
        version_before="v1.3.1",
        version_after="v1.3.2",
        service="inference_service",
        change_type="config_change",
        changelog="Bump request timeout 200→250 ms; increase worker pool 4→6.",
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
            "by clipping at 120 min."
        ),
        commit_sha="bee909c319a3e4fb7b2d7419d69d3d2ee153e6e6",
        deployed_by="alice",
    ),
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class InjectionScenario:
    id: str                        # matches FailureCategory.value
    ground_truth_category: str     # same
    label: str
    alert: str
    failure_window_start_ts: float
    failure_window_end_ts: float
    investigation_start_ts: float
    sub_range: SubRangeConfig      # overrides for the failure window
    deployment_events: list        # full list incl. base deployments
    diagnostic_logs: list[dict]    # extra log rows injected into logs.db


# ---------------------------------------------------------------------------
# Shared feature configs (normal-operation defaults, from defaults.py)
# ---------------------------------------------------------------------------

def _default_feature_configs() -> list[FeatureConfig]:
    return [
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
    ]


def _replace_feature(configs: list[FeatureConfig], name: str, new_cfg: FeatureConfig) -> list[FeatureConfig]:
    return [new_cfg if c.name == name else c for c in configs]


# ---------------------------------------------------------------------------
# Scenario 1 — feature_drift
# ---------------------------------------------------------------------------
# Ground truth: login_failure_rate drifts DOWN to near-zero (uniform[0.01, 0.06])
# while support_tickets_90d spikes UP (N(8.0, 3.0)).  The model is heavily
# dominated by login_failure_rate (weight +1.5) — it sees low LFR and predicts
# LOW churn risk.  But the ground-truth function (weight +0.2 for tickets) now
# assigns most users as churn=1 via high support_tickets_90d.  The model
# systematically under-predicts churn → accuracy and prediction_confidence both drop.
# Discriminating signals:
#   • accuracy ↓  prediction_confidence ↓  (model gets OOD inputs → wrong direction)
#   • latency flat, error_rate flat  (no serving-layer issues)
#   • feature_pipeline logs: PSI drift alerts for login_failure_rate + tickets

_drift_features = _default_feature_configs()
_drift_features = _replace_feature(_drift_features, "login_failure_rate",
    FeatureConfig("login_failure_rate", "uniform", low=0.01, high=0.06))
_drift_features = _replace_feature(_drift_features, "support_tickets_90d",
    FeatureConfig("support_tickets_90d", "normal", mean=8.0, std=3.0))

FEATURE_DRIFT = InjectionScenario(
    id="feature_drift",
    ground_truth_category="feature_drift",
    label="Feature Drift — OOD Input Distributions",
    alert=(
        "ALERT: Model accuracy has been declining since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Model prediction behavior has changed significantly. "
        "No system errors visible on the infrastructure dashboard."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_range=SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_drift_features,
        ),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    ),
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("feature_drift"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "Distribution monitor: login_failure_rate PSI=0.53 over past 1h "
                "(threshold 0.25); mean shifted 0.15→0.035 — significant input drift detected"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_drift"),
            "timestamp": _ts(5, 1) + 120,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "Distribution monitor: support_tickets_90d PSI=0.62 over past 1h "
                "(threshold 0.25); mean shifted 1.2→8.1 — significant input drift detected"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_drift"),
            "timestamp": _ts(5, 2),
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   (
                "Drift alert: 2 features exceed PSI threshold (0.25). "
                "login_failure_rate PSI=0.53, support_tickets_90d PSI=0.62. "
                "Model may be receiving out-of-distribution inputs."
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_drift"),
            "timestamp": _ts(5, 6),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "Hourly feature summary: login_failure_rate range [0.01, 0.06] "
                "(trained range [0.00, 0.30]); values now compressed to low tail of training distribution"
            ),
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 2 — bad_deployment
# ---------------------------------------------------------------------------
# Ground truth: inference_service v2.1.0 → v2.2.0 deployed at day 5 hour 2.
# The new model's feature normalizer was trained on 12 features; the serving
# pipeline added 2 new features it doesn't know about.  Scaler raises ValueError
# on ~4% of requests; mis-normalizes the rest, degrading accuracy.
# Discriminating signals:
#   • accuracy ↓  error_rate SPIKES (0.005→0.04)  latency_p99 elevated
#   • Deployment event at exact start of degradation
#   • inference_service logs: ValueError in feature normalization

_bad_deploy_event = DeploymentEvent(
    timestamp=_ts(5, 2),
    version_before="v2.1.0",
    version_after="v2.2.0",
    service="inference_service",
    change_type="model_retrain",
    changelog="Retrain XGBoost on Q4 data; updated feature normalization pipeline.",
    commit_sha="f3a2c9b1d847e6c05a3d1f2b4e8c7a9d0b6e3f1a",
    deployed_by="alice",
)

BAD_DEPLOYMENT = InjectionScenario(
    id="bad_deployment",
    ground_truth_category="bad_deployment",
    label="Bad Deployment — Feature Normalizer Shape Mismatch",
    alert=(
        "ALERT: Model accuracy dropped sharply starting ~2024-01-06 02:00 UTC "
        "(approximately 28 hours ago). Error rate has spiked significantly. "
        "A model deployment occurred around the same time."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_range=SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_default_feature_configs(),
        ),
        inference_params=InferenceParams(
            baseline_latency_ms=55.0,
            latency_std_ms=20.0,
            tail_prob=0.04,
            tail_latency_ms=400.0,
            error_rate=0.04,
            timeout_rate=0.004,
        ),
        log_params=default_log_params(),
    ),
    deployment_events=_BASE_DEPLOYMENTS + [_bad_deploy_event],
    diagnostic_logs=[
        {
            "log_id":    _log_id("bad_deployment"),
            "timestamp": _ts(5, 2, ),
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Deployment started: pulling artifact model-v2.2.0 from registry",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("bad_deployment"),
            "timestamp": _ts(5, 2) + 120,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Deployment complete: now serving model-v2.2.0",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("bad_deployment"),
            "timestamp": _ts(5, 2) + 300,
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Error rate elevated post-deploy: 0.041 (baseline: 0.005) — monitoring"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("bad_deployment"),
            "timestamp": _ts(5, 2) + 420,
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Prediction failed request_id=a8f2c3: "
                "ValueError: scaler expected input shape (1, 12), got (1, 14) "
                "— feature normalization mismatch post v2.2.0 deploy"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("bad_deployment"),
            "timestamp": _ts(5, 2) + 480,
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Prediction failed request_id=b3c1d9: "
                "ValueError: scaler expected input shape (1, 12), got (1, 14) "
                "— feature normalization mismatch post v2.2.0 deploy"
            ),
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 3 — upstream_schema_change
# ---------------------------------------------------------------------------
# Ground truth: upstream 'customer_profile' table dropped column 'created_at'.
# Feature pipeline falls back to row insertion_timestamp for account_age_days,
# producing wildly out-of-range values (mean ≈ 5000 days instead of 730).
#
# Effect of account_age_days corruption (weight = -0.001 in GT):
#   GT score shifts by -0.001*(5000-730)≈-4.3 → nearly everyone labeled non-churn.
#   Model also sees extreme scaled values (21+ σ) → reduces churn predictions.
#   Both truth and model converge to "non-churn" → accuracy paradoxically IMPROVES,
#   BUT prediction_confidence drops significantly (0.27 → ~0.14) and account_age
#   PSI is enormous.
# Discriminating signals:
#   • prediction_confidence ↓ significantly  accuracy roughly stable (or slightly up)
#   • account_age_days PSI > 1.0 (enormously out of range)
#   • latency flat, error_rate flat  (no serving-layer issues)
#   • feature_pipeline logs: schema validation FAIL + missing column message

_schema_features = _default_feature_configs()
_schema_features = _replace_feature(_schema_features, "account_age_days",
    FeatureConfig("account_age_days", "normal", mean=5000.0, std=3000.0))

UPSTREAM_SCHEMA_CHANGE = InjectionScenario(
    id="upstream_schema_change",
    ground_truth_category="upstream_schema_change",
    label="Upstream Schema Change — account_age_days Corruption",
    alert=(
        "ALERT: Model prediction confidence has dropped significantly since "
        "2024-01-06 00:00 UTC (approximately 30 hours ago). Feature monitoring "
        "has flagged anomalous account_age_days values (observed range up to 9000+ days). "
        "No deployment events or infrastructure anomalies detected."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_range=SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_schema_features,
        ),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    ),
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("upstream_schema_change"),
            "timestamp": _ts(5, 0) + 720,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "account_age_days: unexpected spike to 4847.3 "
                "(expected range 0–1825); 234 rows affected"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("upstream_schema_change"),
            "timestamp": _ts(5, 0) + 900,
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   (
                "Schema validation FAILED: column 'account_age_days' has 312 values "
                "outside expected range [0, 1825]; job continued with suppressed errors"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("upstream_schema_change"),
            "timestamp": _ts(5, 0) + 1320,
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   (
                "Upstream source 'customer_profile' missing column 'created_at'; "
                "falling back to row insertion_timestamp — this affects account_age_days"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("upstream_schema_change"),
            "timestamp": _ts(5, 1),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   (
                "Batch job completed: 12,847 records processed, "
                "312 validation failures suppressed and filled with column mean"
            ),
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 4 — infrastructure_latency_spike
# ---------------------------------------------------------------------------
# Ground truth: feature store I/O latency spiked due to disk saturation on
# the feature store host.  Inference latency p99 jumped from ~235ms to >1500ms.
# Accuracy is UNCHANGED (latency doesn't affect model quality).
# Discriminating signals:
#   • latency_p50 SPIKES  latency_p99 SPIKES  error_rate elevated (timeouts)
#   • accuracy flat  prediction_confidence flat  (model is fine)
#   • inference_service logs: feature store timeout errors, latency SLO breaches

INFRASTRUCTURE_LATENCY_SPIKE = InjectionScenario(
    id="infrastructure_latency_spike",
    ground_truth_category="infrastructure_latency_spike",
    label="Infrastructure Latency Spike — Feature Store I/O Saturation",
    alert=(
        "ALERT: Inference latency p99 has been elevated since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Error rate is also elevated. "
        "Model accuracy appears unchanged."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_range=SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_default_feature_configs(),
        ),
        inference_params=InferenceParams(
            baseline_latency_ms=250.0,
            latency_std_ms=60.0,
            tail_prob=0.15,
            tail_latency_ms=2000.0,
            error_rate=0.025,
            timeout_rate=0.020,
        ),
        log_params=default_log_params(),
    ),
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("infrastructure_latency_spike"),
            "timestamp": _ts(5, 0) + 600,
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   "Latency SLO breach: p99=1247ms (SLO: 200ms) — monitoring",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("infrastructure_latency_spike"),
            "timestamp": _ts(5, 0) + 900,
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Feature store lookup timed out after 200ms for request_id=c7e4a1; "
                "returning error (timeout budget exceeded)"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("infrastructure_latency_spike"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Feature store host disk I/O saturation detected: "
                "await=842ms (normal: <5ms); connection pool queue depth=38"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("infrastructure_latency_spike"),
            "timestamp": _ts(5, 2),
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Cascading latency: feature-store p99=1842ms → inference p99=2103ms; "
                "10.4% of requests timing out (budget: 200ms)"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("infrastructure_latency_spike"),
            "timestamp": _ts(5, 3),
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Model prediction accuracy metrics nominal — latency issue is infrastructure-only",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 5 — model_version_rollback_regression
# ---------------------------------------------------------------------------
# Ground truth: v2.1.0 was rolled back to v2.0.5 at day 5 hour 1 after a brief
# test window showed elevated errors.  v2.0.5 is 4 months old and was trained
# before the referral_source feature was added to the pipeline; it errors on
# ~2.5% of requests that include referral_source.
# Discriminating signals:
#   • error_rate ELEVATED (0.005→0.025) right after rollback
#   • Rollback deployment event at exact degradation onset
#   • inference_service logs: rollback message + errors about unknown feature

_rollback_deploy = DeploymentEvent(
    timestamp=_ts(5, 1),
    version_before="v2.1.0",
    version_after="v2.0.5",
    service="inference_service",
    change_type="model_retrain",
    changelog="Emergency rollback: v2.1.0 showed elevated p99 latency in canary; reverting to v2.0.5.",
    commit_sha="23c603f791915fd1ca2b900236fcbe5c40be5c5c",
    deployed_by="oncall-bot",
    is_rollback=True,
)

MODEL_VERSION_ROLLBACK_REGRESSION = InjectionScenario(
    id="model_version_rollback_regression",
    ground_truth_category="model_version_rollback_regression",
    label="Model Version Rollback Regression — Feature Schema Mismatch",
    alert=(
        "ALERT: Error rate has been elevated since 2024-01-06 01:00 UTC "
        "(approximately 29 hours ago). An emergency rollback was triggered around that time. "
        "The rollback appears to have introduced a regression."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_range=SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_default_feature_configs(),
        ),
        inference_params=InferenceParams(
            baseline_latency_ms=50.0,
            latency_std_ms=10.0,
            tail_prob=0.03,
            tail_latency_ms=350.0,
            error_rate=0.025,
            timeout_rate=0.002,
        ),
        log_params=default_log_params(),
    ),
    deployment_events=_BASE_DEPLOYMENTS + [_rollback_deploy],
    diagnostic_logs=[
        {
            "log_id":    _log_id("model_version_rollback_regression"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   "Initiating emergency rollback: v2.1.0 → v2.0.5 (reason: elevated p99 latency in canary)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_version_rollback_regression"),
            "timestamp": _ts(5, 1) + 90,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Rollback complete: now serving model-v2.0.5 (deployed originally 2023-09-12)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_version_rollback_regression"),
            "timestamp": _ts(5, 1) + 300,
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Post-rollback error rate still elevated: 0.024 (expected: 0.005) "
                "— rollback has not resolved the issue"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_version_rollback_regression"),
            "timestamp": _ts(5, 1) + 480,
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Prediction failed request_id=d4f1e8: "
                "KeyError: 'referral_source' not found in feature schema for model-v2.0.5 "
                "(feature was added after this model was trained)"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_version_rollback_regression"),
            "timestamp": _ts(5, 2),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "model-v2.0.5 (4 months old) was trained before referral_source was added "
                "to the feature pipeline on 2024-01-03; "
                "requests including referral_source will continue to error"
            ),
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, InjectionScenario] = {
    FEATURE_DRIFT.id:                    FEATURE_DRIFT,
    BAD_DEPLOYMENT.id:                   BAD_DEPLOYMENT,
    UPSTREAM_SCHEMA_CHANGE.id:           UPSTREAM_SCHEMA_CHANGE,
    INFRASTRUCTURE_LATENCY_SPIKE.id:     INFRASTRUCTURE_LATENCY_SPIKE,
    MODEL_VERSION_ROLLBACK_REGRESSION.id: MODEL_VERSION_ROLLBACK_REGRESSION,
}
