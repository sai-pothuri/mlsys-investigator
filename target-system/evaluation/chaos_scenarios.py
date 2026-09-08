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
    sub_ranges: list               # list[SubRangeConfig] — overrides covering the failure window
                                    # (usually one entry; >1 entry produces a ramp instead of a step)
    deployment_events: list        # full list incl. base deployments
    diagnostic_logs: list[dict]    # extra log rows injected into logs.db
    label_delay_hours: float = 0.0 # 0.0 = accuracy reacts immediately (default for all easy/most
                                    # medium scenarios); only delayed_label_feedback_shift needs >0
    metric_overrides: list = None  # list[dict] of {timestamp, metric_name, metric_value, service, tags}
                                    # rows that replace the organically-generated metric row at that
                                    # exact (timestamp, metric_name). Used only for signals the
                                    # feature-driven simulation cannot produce on its own (corrupted
                                    # labels, miscalibrated confidence, a leaking shadow model) — see
                                    # _metric_override() below.

    def __post_init__(self):
        if self.metric_overrides is None:
            self.metric_overrides = []


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


def _metric_override(ts: float, metric_name: str, value: float, service: str = "inference_service") -> dict:
    """Build a metrics.db row that replaces whatever the feature-driven simulation
    would have produced for this exact hour bucket. `ts` must be hour-aligned
    (use _ts(day, hour) with whole-hour values) to land on the same bucket
    aggregate_metrics() writes to. Tagged so run_eval.py can find-and-replace
    on rerun without touching organically-generated rows."""
    return {
        "timestamp": ts,
        "metric_name": metric_name,
        "metric_value": value,
        "service": service,
        "tags": '{"injected": true}',
    }


def _hourly_overrides(start_ts: float, end_ts: float, metric_name: str, value: float,
                       service: str = "inference_service") -> list[dict]:
    """One override per hour bucket covering [start_ts, end_ts)."""
    overrides = []
    ts = start_ts
    while ts < end_ts:
        overrides.append(_metric_override(ts, metric_name, value, service))
        ts += 3600
    return overrides


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
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_drift_features,
        ),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
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
    sub_ranges=[SubRangeConfig(
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
    )],
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
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(
            requests_per_hour=500.0,
            feature_configs=_schema_features,
        ),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
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
    sub_ranges=[SubRangeConfig(
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
    )],
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
    sub_ranges=[SubRangeConfig(
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
    )],
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
# Scenario 6 — label_pipeline_corruption  (medium)
# ---------------------------------------------------------------------------
# Ground truth: label pipeline join key changed session_id -> request_id,
# misaligning 38% of labels. Features and model are both fine; the labels
# used to SCORE the model are wrong. Not representable via feature drift
# (labels are assigned by a fixed ground-truth function, not a simulated
# pipeline) so accuracy is overridden directly for the failure window.
# Discriminating signals:
#   • accuracy ↓  prediction_confidence UNCHANGED (the key discriminator)
#   • feature distributions clean, no deployment event
#   • label_pipeline logs: join alignment rate drop, label match errors

LABEL_PIPELINE_CORRUPTION = InjectionScenario(
    id="label_pipeline_corruption",
    ground_truth_category="label_pipeline_corruption",
    label="Label Pipeline Corruption — Join Key Misalignment",
    alert=(
        "ALERT: Model accuracy has dropped sharply since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Prediction confidence and input feature "
        "distributions both look unchanged. No deployment events or infrastructure "
        "anomalies in the window."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("label_pipeline_corruption"),
            "timestamp": _ts(5, 0) + 600,
            "severity":  "WARNING",
            "service":   "label_pipeline",
            "message":   (
                "Join alignment rate dropped: 62% of labels matched to a request "
                "(baseline: 99.2%) — investigating join key"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("label_pipeline_corruption"),
            "timestamp": _ts(5, 0) + 900,
            "severity":  "ERROR",
            "service":   "label_pipeline",
            "message":   (
                "Label match error: join on 'request_id' returned 0 rows for 38% of "
                "batch 2024-01-06; falling back to nearest-timestamp heuristic match"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("label_pipeline_corruption"),
            "timestamp": _ts(5, 1),
            "severity":  "ERROR",
            "service":   "label_pipeline",
            "message":   (
                "Schema drift detected: upstream label export switched primary key "
                "from 'session_id' to 'request_id' as of 2024-01-05 23:40 UTC; "
                "downstream join was not updated"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("label_pipeline_corruption"),
            "timestamp": _ts(5, 2),
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Prediction confidence distribution nominal — mean 0.271 (baseline 0.273)",
            "context":   "{}",
        },
    ],
    metric_overrides=_hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.52),
)


# ---------------------------------------------------------------------------
# Scenario 7 — training_serving_skew  (medium)
# ---------------------------------------------------------------------------
# Ground truth: a "perf optimization" pre-normalizes monetary features in
# serve.py that train_model.py's StandardScaler pipeline doesn't know about
# (see pipeline_repo commit 4612fcb). Each feature's own distribution still
# looks plausible in isolation (low PSI); only the code diff reveals the
# divergence. Discriminating signals:
#   • accuracy ↓  prediction_confidence ↓ moderately
#   • feature PSI low per-feature — query_feature_distributions looks clean
#   • query_code_diffs on the config_change deploy shows the divergence

_SKEW_DEPLOY = DeploymentEvent(
    timestamp=FAILURE_START_TS,
    version_before="v2.1.1",
    version_after="v2.1.2",
    service="inference_service",
    change_type="config_change",
    changelog="Perf: pre-normalize monetary features before scaler (INFRA-2290).",
    commit_sha="4612fcbb617ce881c1ae2dce2a5f412b1830423e",
    deployed_by="mlops-bot",
)

TRAINING_SERVING_SKEW = InjectionScenario(
    id="training_serving_skew",
    ground_truth_category="training_serving_skew",
    label="Training/Serving Skew — Duplicate Monetary-Feature Normalization",
    alert=(
        "ALERT: Model accuracy and prediction confidence have both declined "
        "since 2024-01-06 00:00 UTC (approximately 30 hours ago). Individual "
        "feature distributions look within normal ranges. A minor config deployment "
        "occurred around the same time."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS + [_SKEW_DEPLOY],
    diagnostic_logs=[
        {
            "log_id":    _log_id("training_serving_skew"),
            "timestamp": FAILURE_START_TS + 300,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Deployment complete: now serving config v2.1.2 (perf optimization, INFRA-2290)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("training_serving_skew"),
            "timestamp": FAILURE_START_TS + 3600,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "monthly_spend, avg_transaction_value: per-feature distributions within "
                "expected range (PSI < 0.05) — no drift detected"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("training_serving_skew"),
            "timestamp": FAILURE_START_TS + 7200,
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Prediction confidence trending down (mean 0.19, baseline 0.27) despite "
                "clean feature distributions — model behavior does not match input data"
            ),
            "context":   "{}",
        },
    ],
    metric_overrides=(
        _hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.66)
        + _hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "prediction_confidence", 0.14)
    ),
)


# ---------------------------------------------------------------------------
# Scenario 8 — data_freshness_degradation  (medium)
# ---------------------------------------------------------------------------
# Ground truth: the feature pipeline batch job stalled for 12h; feature
# values served during the window reflect user state from ~12h earlier
# (days_since_last_login reads artificially low, as if everyone just logged
# in). Organic accuracy effect via GT weight (+0.15) on days_since_last_login.
# Discriminating signals:
#   • query_logs on feature_pipeline: "batch job delayed" / stale run messages
#   • feature values are stale, not invalid — PSI moderate, not extreme
#   • accuracy drifts moderately; no schema errors, no deploy event

_freshness_features = _default_feature_configs()
_freshness_features = _replace_feature(_freshness_features, "days_since_last_login",
    FeatureConfig("days_since_last_login", "normal", mean=1.0, std=1.0))

DATA_FRESHNESS_DEGRADATION = InjectionScenario(
    id="data_freshness_degradation",
    ground_truth_category="data_freshness_degradation",
    label="Data Freshness Degradation — Stalled Feature Batch Job",
    alert=(
        "ALERT: Model accuracy has drifted moderately since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Feature values are present but the "
        "feature_pipeline scheduler is flagging delays. No schema errors, no "
        "deployment events."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_freshness_features),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("data_freshness_degradation"),
            "timestamp": _ts(5, 0) + 300,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "Batch job 'user_activity_rollup' delayed: last successful run 11.6h ago (SLA: 1h)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("data_freshness_degradation"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "days_since_last_login: serving values computed from snapshot "
                "2024-01-05T12:03:00Z (12h stale); no upstream error, job is just backed up"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("data_freshness_degradation"),
            "timestamp": _ts(5, 3),
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   "user_activity_rollup queue depth: 14,200 (normal: <500) — worker pool saturated",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("data_freshness_degradation"),
            "timestamp": _ts(5, 6),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Feature values are structurally valid (no nulls, no schema errors) — freshness issue only",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 9 — feature_encoding_bug  (medium)
# ---------------------------------------------------------------------------
# Ground truth: product_category emits a 4th category value (3) not present
# in the training encoder; the pipeline should reject/remap it but instead
# passes it through, and it defaults to index 0 downstream. Affects only the
# ~10% of requests carrying the new category (organic, via GT weight -0.3).
# Discriminating signals:
#   • product_category distribution shows a new mode; other features clean
#   • feature_pipeline logs: encoding warning for unseen category
#   • overall accuracy drops moderately, not sharply; error_rate unchanged

_encoding_features = _default_feature_configs()
_encoding_features = _replace_feature(_encoding_features, "product_category",
    FeatureConfig("product_category", "categorical",
                  categories=[0, 1, 2, 3], probs=[0.45, 0.30, 0.15, 0.10]))

FEATURE_ENCODING_BUG = InjectionScenario(
    id="feature_encoding_bug",
    ground_truth_category="feature_encoding_bug",
    label="Feature Encoding Bug — Unseen product_category Value",
    alert=(
        "ALERT: Model accuracy has degraded moderately since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Most feature distributions look normal. "
        "Error rate is unchanged. No deployment events."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_encoding_features),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("feature_encoding_bug"),
            "timestamp": _ts(5, 0) + 600,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   (
                "Unexpected value for product_category: 3 (valid set: {0, 1, 2}); "
                "encoder defaulting to index 0 — 9.8% of requests in the last hour affected"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_encoding_bug"),
            "timestamp": _ts(5, 2),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "product_category value=3 recurring (612 occurrences in past 2h) — likely a new catalog category not yet registered in the encoder",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_encoding_bug"),
            "timestamp": _ts(5, 4),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "All other feature distributions (11 of 12) within expected ranges",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 10 — gradual_concept_drift  (medium)
# ---------------------------------------------------------------------------
# Ground truth: user behavior has been slowly shifting across days 3-7 —
# three features ramp gradually rather than stepping. No single feature hits
# an extreme PSI; the trend is only visible in a wide query_metrics window.
# Discriminating signals:
#   • accuracy decays slowly across days, not a sharp step
#   • moderate PSI (0.1-0.2) across several features, not one outlier
#   • no deployment event, no schema/log errors

_GCD_START = _ts(3)   # day 3
_GCD_END   = _ts(7)   # day 7

def _gcd_features(step: int) -> list[FeatureConfig]:
    """step 0..3 — increasingly shifted, still within plausible ranges."""
    cfgs = _default_feature_configs()
    cfgs = _replace_feature(cfgs, "login_failure_rate",
        FeatureConfig("login_failure_rate", "uniform", low=0.0, high=0.3 + step * 0.10))
    cfgs = _replace_feature(cfgs, "session_duration_min",
        FeatureConfig("session_duration_min", "normal", mean=18 + step * 4, std=8))
    cfgs = _replace_feature(cfgs, "support_tickets_90d",
        FeatureConfig("support_tickets_90d", "normal", mean=1.2 + step * 0.8, std=1.5))
    return cfgs

GRADUAL_CONCEPT_DRIFT = InjectionScenario(
    id="gradual_concept_drift",
    ground_truth_category="gradual_concept_drift",
    label="Gradual Concept Drift — Multi-Feature Ramp Over 4 Days",
    alert=(
        "ALERT: Model accuracy has been slowly trending down over the past several "
        "days (not a sharp step change). No single feature or deployment event stands "
        "out. Latency and error rate are nominal."
    ),
    failure_window_start_ts=_GCD_START,
    failure_window_end_ts=_GCD_END,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[
        SubRangeConfig(
            start_ts=_ts(3 + i), end_ts=_ts(3 + i + 1),
            feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_gcd_features(i)),
            inference_params=default_inference_params(),
            log_params=default_log_params(),
        )
        for i in range(4)
    ],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("gradual_concept_drift"),
            "timestamp": _ts(3, 12),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Weekly distribution summary: login_failure_rate PSI=0.09 vs. 7-day-prior baseline",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("gradual_concept_drift"),
            "timestamp": _ts(4, 12),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Weekly distribution summary: login_failure_rate PSI=0.13, session_duration_min PSI=0.11 vs. 7-day-prior baseline",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("gradual_concept_drift"),
            "timestamp": _ts(5, 12),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "Weekly distribution summary: 3 features now at PSI 0.14-0.18 vs. 7-day-prior baseline — gradual, no single outlier",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("gradual_concept_drift"),
            "timestamp": _ts(6, 12),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "Weekly distribution summary: 3 features now at PSI 0.17-0.21 vs. 7-day-prior baseline — trend continuing, no step change detected",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 11 — model_calibration_drift  (medium)
# ---------------------------------------------------------------------------
# Ground truth: predicted probabilities are systematically overconfident
# relative to outcomes. Not representable via feature drift (it's a property
# of the score->outcome relationship, not the inputs), so both metrics are
# overridden directly. Inverted signature vs. label_pipeline_corruption:
# there, confidence is flat and accuracy drops; here, confidence goes UP
# while accuracy drops — the model is *more* sure of *worse* predictions.
# Discriminating signals:
#   • prediction_confidence HIGH, accuracy LOW for that confidence level
#   • features clean, no deployment event, no errors

MODEL_CALIBRATION_DRIFT = InjectionScenario(
    id="model_calibration_drift",
    ground_truth_category="model_calibration_drift",
    label="Model Calibration Drift — Systematic Overconfidence",
    alert=(
        "ALERT: Since 2024-01-06 00:00 UTC (approximately 30 hours ago), prediction "
        "confidence has risen while accuracy has fallen — the model is more confident "
        "in worse predictions. Feature distributions and infrastructure are nominal."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("model_calibration_drift"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Prediction confidence distribution shifted: mean 0.85 (baseline 0.27) "
                "with accuracy at 0.55 (baseline 0.87) — confidence/accuracy divergence, "
                "consider recalibration"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_calibration_drift"),
            "timestamp": _ts(5, 3),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "All 12 features within expected distribution ranges (max PSI 0.04)",
            "context":   "{}",
        },
    ],
    metric_overrides=(
        _hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "prediction_confidence", 0.85)
        + _hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.55)
    ),
)


# ---------------------------------------------------------------------------
# Scenario 12 — shadow_mode_leak  (medium)
# ---------------------------------------------------------------------------
# Ground truth: an experimental model (v4.0.0-shadow) is accidentally
# handling ~20% of live traffic alongside the primary (v3.2.1), with no
# official promotion in deployment history. Not representable as a feature
# or single-model metric shift, so accuracy is overridden to reflect the
# blended primary+shadow output quality.
# Discriminating signals:
#   • inference_service logs show two model versions serving requests
#   • deployment history shows no official promotion event
#   • accuracy degraded but not catastrophically (blended, not total failure)

SHADOW_MODE_LEAK = InjectionScenario(
    id="shadow_mode_leak",
    ground_truth_category="shadow_mode_leak",
    label="Shadow Mode Leak — Unpromoted Model Serving Live Traffic",
    alert=(
        "ALERT: Model accuracy has degraded since 2024-01-06 00:00 UTC (approximately "
        "30 hours ago). No deployment event coincides with the onset. Feature "
        "distributions and error rate are nominal."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("shadow_mode_leak"),
            "timestamp": _ts(5, 0) + 400,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Prediction served by model_version=v3.2.1 for request_id=e91a3c",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("shadow_mode_leak"),
            "timestamp": _ts(5, 0) + 460,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Prediction served by model_version=v4.0.0-shadow for request_id=f02b4d",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("shadow_mode_leak"),
            "timestamp": _ts(5, 2),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   (
                "Traffic split observed across model_version labels: v3.2.1 (80.3%), "
                "v4.0.0-shadow (19.7%) — no promotion record for v4.0.0-shadow in deploy history"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("shadow_mode_leak"),
            "timestamp": _ts(5, 5),
            "severity":  "ERROR",
            "service":   "inference_service",
            "message":   (
                "Routing config anomaly: experiment flag 'shadow_v4_eval' has traffic_pct=20 "
                "in production routing table (expected: shadow-only, 0% live traffic)"
            ),
            "context":   "{}",
        },
    ],
    metric_overrides=_hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.68),
)


# ---------------------------------------------------------------------------
# Scenario 13 — feature_pipeline_partial_failure  (medium)
# ---------------------------------------------------------------------------
# Ground truth: the job computing support_tickets_90d failed silently for
# the whole window; the feature defaults to a constant fill value (0). Other
# 11 features are unaffected. Organic accuracy effect via GT weight (+0.2).
# Discriminating signals:
#   • support_tickets_90d shows near-zero variance (degenerate distribution)
#   • other features normal; error_rate unchanged
#   • feature_pipeline logs: partial job failure naming the specific feature

_partial_failure_features = _default_feature_configs()
_partial_failure_features = _replace_feature(_partial_failure_features, "support_tickets_90d",
    FeatureConfig("support_tickets_90d", "normal", mean=0.0, std=0.0))

FEATURE_PIPELINE_PARTIAL_FAILURE = InjectionScenario(
    id="feature_pipeline_partial_failure",
    ground_truth_category="feature_pipeline_partial_failure",
    label="Feature Pipeline Partial Failure — support_tickets_90d Stuck at Fill Value",
    alert=(
        "ALERT: Model accuracy has degraded moderately since 2024-01-06 00:00 UTC "
        "(approximately 30 hours ago). Error rate is unchanged. Most feature "
        "distributions look normal."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_partial_failure_features),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("feature_pipeline_partial_failure"),
            "timestamp": _ts(5, 0) + 500,
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   (
                "Job 'support_tickets_rollup' failed silently at 2024-01-06T00:08:00Z "
                "(uncaught exception in aggregation step); downstream consumers received "
                "the default fill value (0) for support_tickets_90d"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_pipeline_partial_failure"),
            "timestamp": _ts(5, 1),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "support_tickets_90d: 100% of values in window = 0.0 (expected mean ~1.2) — likely fill-value default, not real signal",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_pipeline_partial_failure"),
            "timestamp": _ts(5, 3),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "11 of 12 features unaffected; only support_tickets_rollup job is failing",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 14 — delayed_label_feedback_shift  (hard)
# ---------------------------------------------------------------------------
# Ground truth: a real fix was deployed at day 6 hour 4, but the 24h label
# delay means accuracy metrics for the "now" window still reflect requests
# from the still-degraded day-5 period. Investigation happens only 6h after
# the fix — nowhere near enough for the fix to show up in accuracy. Requires
# reasoning about label delay to conclude the fix hasn't had time to
# propagate, not that it failed.
# Discriminating signals:
#   • the degradation onset is >24h before "now"; the fix is <24h before "now"
#   • accuracy in the most recent window is still bad despite the fix
#   • deployment log explicitly documents label-delay-driven metric lag

_DLFS_DEGRADE_START = _ts(5, 0)
_DLFS_FIX_TS         = _ts(6, 4)
_DLFS_END            = _ts(7, 0)
_DLFS_INVESTIGATION  = _ts(6, 10)   # 6h after the fix — well under the 24h label delay

_dlfs_fix_deploy = DeploymentEvent(
    timestamp=_DLFS_FIX_TS,
    version_before="v2.1.1",
    version_after="v2.1.1-hotfix1",
    service="inference_service",
    change_type="config_change",
    changelog="Hotfix: revert feature normalization regression introduced 2024-01-06 00:00 UTC.",
    commit_sha="ed7471329c5a25b0a67624a2fb94e47aae36a337",
    deployed_by="oncall-bot",
)

DELAYED_LABEL_FEEDBACK_SHIFT = InjectionScenario(
    id="delayed_label_feedback_shift",
    ground_truth_category="delayed_label_feedback_shift",
    label="Delayed Label Feedback Shift — Fix Deployed, Metrics Lag 24h",
    alert=(
        "ALERT: Model accuracy has been degraded since 2024-01-06 00:00 UTC. A hotfix "
        "was deployed at 2024-01-06 04:00 UTC intended to resolve it, but accuracy "
        "metrics still show degradation as of now (2024-01-06 10:00 UTC). Determine "
        "whether the hotfix failed or whether something else explains why accuracy "
        "hasn't recovered yet."
    ),
    failure_window_start_ts=_DLFS_DEGRADE_START,
    failure_window_end_ts=_DLFS_FIX_TS,
    investigation_start_ts=_DLFS_INVESTIGATION,
    label_delay_hours=24.0,
    sub_ranges=[
        SubRangeConfig(
            start_ts=_DLFS_DEGRADE_START, end_ts=_DLFS_FIX_TS,
            feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_drift_features),
            inference_params=default_inference_params(),
            log_params=default_log_params(),
        ),
        SubRangeConfig(
            start_ts=_DLFS_FIX_TS, end_ts=_DLFS_END,
            feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
            inference_params=default_inference_params(),
            log_params=default_log_params(),
        ),
    ],
    deployment_events=_BASE_DEPLOYMENTS + [_dlfs_fix_deploy],
    diagnostic_logs=[
        {
            "log_id":    _log_id("delayed_label_feedback_shift"),
            "timestamp": _DLFS_DEGRADE_START + 3600,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "Distribution monitor: login_failure_rate PSI=0.51 — significant input drift detected",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("delayed_label_feedback_shift"),
            "timestamp": _DLFS_FIX_TS,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Hotfix deployed: v2.1.1 -> v2.1.1-hotfix1, reverting the 2024-01-06 00:00 UTC normalization regression",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("delayed_label_feedback_shift"),
            "timestamp": _DLFS_FIX_TS + 120,
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Feature distributions returned to baseline immediately post-hotfix (login_failure_rate PSI=0.02)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("delayed_label_feedback_shift"),
            "timestamp": _DLFS_FIX_TS + 300,
            "severity":  "WARNING",
            "service":   "label_pipeline",
            "message":   (
                "Reminder: accuracy metrics are computed from labels with a 24h delay "
                "(label_ts = request_ts + 24h). Requests made before this hotfix will "
                "continue to show as degraded in the accuracy metric until their labels "
                "clear, ~24h after each request — this is expected and does not indicate "
                "the hotfix failed."
            ),
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 15 — cascading_upstream_failure  (hard)
# ---------------------------------------------------------------------------
# Ground truth: two unrelated upstream failures hit simultaneously — the
# same account_age_days schema loss as upstream_schema_change, PLUS a label
# pipeline join corruption on top. Neither alone would explain the magnitude
# of the drop; reconstructing it requires synthesizing feature_pipeline logs,
# label_pipeline logs, feature_distributions, and metrics together.
# Discriminating signals:
#   • no single cause explains the drop's magnitude
#   • feature_pipeline AND label_pipeline both show errors in the same window
#   • accuracy collapses further than either failure alone would predict

CASCADING_UPSTREAM_FAILURE = InjectionScenario(
    id="cascading_upstream_failure",
    ground_truth_category="cascading_upstream_failure",
    label="Cascading Upstream Failure — Schema Loss + Label Join Corruption",
    alert=(
        "ALERT: Model accuracy has collapsed since 2024-01-06 00:00 UTC (approximately "
        "30 hours ago) — a much larger drop than typical single-cause incidents. "
        "Multiple subsystems are reporting anomalies simultaneously. No deployment events."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_schema_features),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS,
    diagnostic_logs=[
        {
            "log_id":    _log_id("cascading_upstream_failure"),
            "timestamp": _ts(5, 0) + 600,
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   (
                "Upstream source 'customer_profile' missing column 'created_at'; "
                "falling back to row insertion_timestamp — affects account_age_days"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("cascading_upstream_failure"),
            "timestamp": _ts(5, 0) + 900,
            "severity":  "ERROR",
            "service":   "label_pipeline",
            "message":   (
                "Join alignment rate dropped: 58% of labels matched to a request "
                "(baseline: 99.2%) — upstream label export changed primary key independently "
                "of the feature_pipeline incident"
            ),
            "context":   "{}",
        },
        {
            "log_id":    _log_id("cascading_upstream_failure"),
            "timestamp": _ts(5, 1),
            "severity":  "ERROR",
            "service":   "feature_pipeline",
            "message":   "account_age_days: unexpected spike to 4900+ (expected range 0-1825); 289 rows affected",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("cascading_upstream_failure"),
            "timestamp": _ts(5, 2),
            "severity":  "ERROR",
            "service":   "label_pipeline",
            "message":   "Label match error: join on 'request_id' returned 0 rows for 41% of batch 2024-01-06",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("cascading_upstream_failure"),
            "timestamp": _ts(5, 4),
            "severity":  "CRITICAL",
            "service":   "inference_service",
            "message":   (
                "Accuracy drop (0.87 -> 0.31) exceeds what either the feature_pipeline "
                "schema incident or the label_pipeline join incident would explain in "
                "isolation — two independent root causes suspected"
            ),
            "context":   "{}",
        },
    ],
    metric_overrides=_hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.31),
)


# ---------------------------------------------------------------------------
# Scenario 16 — model_staleness  (hard)
# ---------------------------------------------------------------------------
# Ground truth: the current model was last retrained 65 days before the
# simulated window even begins; user behavior has been slowly drifting the
# whole time. No pipeline failure, no in-window deployment event — the only
# clue to "why now" is a 65-day-old retrain event well outside the default
# investigation window, plus a slow multi-feature trend across all 7 days.
# Discriminating signals:
#   • no single feature shows extreme PSI; broad, moderate drift across many
#   • no errors, no in-window deploy events
#   • query_deployment_history over a WIDE range surfaces the stale retrain

_MS_START = SIM_START           # day 0
_MS_END   = _ts(7)              # day 7
_MS_INVESTIGATION = _ts(6, 20)  # near the end of the window — most drift accumulated

def _staleness_features(day_idx: int) -> list[FeatureConfig]:
    """day_idx 0..6 — small linear ramp across the whole week, capped moderate."""
    frac = day_idx / 6.0
    cfgs = _default_feature_configs()
    cfgs = _replace_feature(cfgs, "days_since_last_login",
        FeatureConfig("days_since_last_login", "normal", mean=3 + frac * 4, std=2))
    cfgs = _replace_feature(cfgs, "session_duration_min",
        FeatureConfig("session_duration_min", "normal", mean=18 - frac * 6, std=8))
    cfgs = _replace_feature(cfgs, "login_failure_rate",
        FeatureConfig("login_failure_rate", "uniform", low=0.0, high=0.3 + frac * 0.15))
    return cfgs

_stale_retrain_deploy = DeploymentEvent(
    timestamp=_ts(-65),
    version_before="v1.9.0",
    version_after="v2.0.0",
    service="inference_service",
    change_type="model_retrain",
    changelog="Routine quarterly retrain on 6-month rolling window.",
    commit_sha="11b78fd0f6702860504524d4d319ab2601926739",
    deployed_by="alice",
)

MODEL_STALENESS = InjectionScenario(
    id="model_staleness",
    ground_truth_category="model_staleness",
    label="Model Staleness — 65-Day-Old Model, Slow Behavior Drift",
    alert=(
        "ALERT: Model accuracy has been slowly eroding over the past week and just "
        "crossed the alerting threshold. No errors, no deployment events in the past "
        "week, no single feature stands out. Unclear when the model was last retrained — "
        "may require checking further back than the usual 24-48h window."
    ),
    failure_window_start_ts=_MS_START,
    failure_window_end_ts=_MS_END,
    investigation_start_ts=_MS_INVESTIGATION,
    sub_ranges=[
        SubRangeConfig(
            start_ts=_ts(d), end_ts=_ts(d + 1),
            feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_staleness_features(d)),
            inference_params=default_inference_params(),
            log_params=default_log_params(),
        )
        for d in range(7)
    ],
    deployment_events=[_stale_retrain_deploy],
    diagnostic_logs=[
        {
            "log_id":    _log_id("model_staleness"),
            "timestamp": _ts(3, 12),
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Weekly distribution summary: all features within normal ranges (max PSI 0.06)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("model_staleness"),
            "timestamp": _ts(6, 12),
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   "Accuracy alert threshold crossed (0.87 -> 0.74 over trailing 7 days); no single incident identified",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Scenario 17 — feature_importance_inversion  (hard)
# ---------------------------------------------------------------------------
# Ground truth: login_failure_rate's polarity was flipped by a product-side
# redefinition (see pipeline_repo commit 5bc3343 — "PROD-482"), but the field
# name and raw value RANGE are unchanged, so PSI on the raw values looks
# normal. Only the code diff reveals the semantic inversion.
# Discriminating signals:
#   • login_failure_rate PSI looks normal (values in expected range)
#   • accuracy still drops — the numbers alone don't explain why
#   • query_code_diffs on the feature_pipeline_change deploy reveals the flip

_INVERSION_DEPLOY = DeploymentEvent(
    timestamp=FAILURE_START_TS,
    version_before="v1.4.0",
    version_after="v1.4.1",
    service="feature_pipeline",
    change_type="feature_pipeline_change",
    changelog="Redefine login_failure_rate polarity per PROD-482 (field name unchanged).",
    commit_sha="5bc334393b9388a5cfe0284cbae6585c166bef94",
    deployed_by="bob",
)

FEATURE_IMPORTANCE_INVERSION = InjectionScenario(
    id="feature_importance_inversion",
    ground_truth_category="feature_importance_inversion",
    label="Feature Importance Inversion — login_failure_rate Polarity Flip",
    alert=(
        "ALERT: Model accuracy has dropped since 2024-01-06 00:00 UTC (approximately "
        "30 hours ago). All feature distributions, including login_failure_rate, look "
        "within normal ranges. A feature_pipeline deployment occurred around the same time."
    ),
    failure_window_start_ts=FAILURE_START_TS,
    failure_window_end_ts=FAILURE_END_TS,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=FAILURE_START_TS,
        end_ts=FAILURE_END_TS,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_default_feature_configs()),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS + [_INVERSION_DEPLOY],
    diagnostic_logs=[
        {
            "log_id":    _log_id("feature_importance_inversion"),
            "timestamp": FAILURE_START_TS + 300,
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "Deployment complete: now serving feature_pipeline v1.4.1 (PROD-482)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_importance_inversion"),
            "timestamp": FAILURE_START_TS + 3600,
            "severity":  "INFO",
            "service":   "feature_pipeline",
            "message":   "login_failure_rate: distribution within expected range [0.00, 0.30] (PSI=0.03) — no drift detected",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("feature_importance_inversion"),
            "timestamp": FAILURE_START_TS + 7200,
            "severity":  "WARNING",
            "service":   "inference_service",
            "message":   "Accuracy declining (0.87 -> 0.60) with no corresponding feature drift signal — recommend reviewing recent feature_pipeline code changes",
            "context":   "{}",
        },
    ],
    metric_overrides=_hourly_overrides(FAILURE_START_TS, FAILURE_END_TS, "accuracy", 0.60),
)


# ---------------------------------------------------------------------------
# Scenario 18 — compound_drift_plus_deployment  (hard)
# ---------------------------------------------------------------------------
# Ground truth: feature drift begins at day 4 hour 12; a benign, unrelated
# config redeploy happens 12h later at day 5 hour 0. The deploy is a red
# herring — query_code_diffs on it shows only the (already-shipped,
# genuinely harmless) session_duration_min clip-bound fix. The accuracy
# drop is fully explained by drift that predates the deploy.
# Discriminating signals:
#   • feature drift onset is BEFORE the deploy timestamp
#   • the deploy's diff is small and unrelated to the drifting feature
#   • no hard errors anywhere — timeline reconstruction is the only way to tell

_CDD_START = _ts(4, 12)
_CDD_END   = _ts(7)

_cdd_features = _default_feature_configs()
_cdd_features = _replace_feature(_cdd_features, "login_failure_rate",
    FeatureConfig("login_failure_rate", "uniform", low=0.05, high=0.15))

_cdd_benign_deploy = DeploymentEvent(
    timestamp=_ts(5, 0),
    version_before="v2.1.1",
    version_after="v2.1.1a",
    service="inference_service",
    change_type="config_change",
    changelog="Routine redeploy of session_duration_min clip-bound fix to secondary region.",
    commit_sha="ed7471329c5a25b0a67624a2fb94e47aae36a337",
    deployed_by="mlops-bot",
)

COMPOUND_DRIFT_PLUS_DEPLOYMENT = InjectionScenario(
    id="compound_drift_plus_deployment",
    ground_truth_category="compound_drift_plus_deployment",
    label="Compound Drift + Deployment — Deploy Is a Red Herring",
    alert=(
        "ALERT: Model accuracy has declined since approximately 2024-01-04 12:00 UTC. "
        "A configuration deployment occurred at 2024-01-05 00:00 UTC, roughly 12 hours "
        "after the decline started. Determine whether the deployment caused the "
        "degradation or is unrelated."
    ),
    failure_window_start_ts=_CDD_START,
    failure_window_end_ts=_CDD_END,
    investigation_start_ts=INVESTIGATION_TS,
    sub_ranges=[SubRangeConfig(
        start_ts=_CDD_START,
        end_ts=_CDD_END,
        feature_params=FeatureParams(requests_per_hour=500.0, feature_configs=_cdd_features),
        inference_params=default_inference_params(),
        log_params=default_log_params(),
    )],
    deployment_events=_BASE_DEPLOYMENTS + [_cdd_benign_deploy],
    diagnostic_logs=[
        {
            "log_id":    _log_id("compound_drift_plus_deployment"),
            "timestamp": _CDD_START + 1800,
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "Distribution monitor: login_failure_rate PSI=0.31 over past 1h — drift detected, onset ~2024-01-04T12:00Z",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("compound_drift_plus_deployment"),
            "timestamp": _ts(5, 0) + 60,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "Deployment complete: now serving config v2.1.1a (session_duration_min clip-bound fix, secondary region)",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("compound_drift_plus_deployment"),
            "timestamp": _ts(5, 0) + 600,
            "severity":  "INFO",
            "service":   "inference_service",
            "message":   "No errors observed post-deploy; error_rate and latency both nominal since v2.1.1a rollout",
            "context":   "{}",
        },
        {
            "log_id":    _log_id("compound_drift_plus_deployment"),
            "timestamp": _ts(5, 6),
            "severity":  "WARNING",
            "service":   "feature_pipeline",
            "message":   "login_failure_rate PSI still elevated (0.29) — drift predates the 2024-01-05T00:00Z deploy by ~12h and continues unabated",
            "context":   "{}",
        },
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, InjectionScenario] = {
    FEATURE_DRIFT.id:                         FEATURE_DRIFT,
    BAD_DEPLOYMENT.id:                        BAD_DEPLOYMENT,
    UPSTREAM_SCHEMA_CHANGE.id:                UPSTREAM_SCHEMA_CHANGE,
    INFRASTRUCTURE_LATENCY_SPIKE.id:          INFRASTRUCTURE_LATENCY_SPIKE,
    MODEL_VERSION_ROLLBACK_REGRESSION.id:     MODEL_VERSION_ROLLBACK_REGRESSION,
    LABEL_PIPELINE_CORRUPTION.id:             LABEL_PIPELINE_CORRUPTION,
    TRAINING_SERVING_SKEW.id:                 TRAINING_SERVING_SKEW,
    DATA_FRESHNESS_DEGRADATION.id:            DATA_FRESHNESS_DEGRADATION,
    FEATURE_ENCODING_BUG.id:                  FEATURE_ENCODING_BUG,
    GRADUAL_CONCEPT_DRIFT.id:                 GRADUAL_CONCEPT_DRIFT,
    MODEL_CALIBRATION_DRIFT.id:               MODEL_CALIBRATION_DRIFT,
    SHADOW_MODE_LEAK.id:                      SHADOW_MODE_LEAK,
    FEATURE_PIPELINE_PARTIAL_FAILURE.id:      FEATURE_PIPELINE_PARTIAL_FAILURE,
    DELAYED_LABEL_FEEDBACK_SHIFT.id:          DELAYED_LABEL_FEEDBACK_SHIFT,
    CASCADING_UPSTREAM_FAILURE.id:            CASCADING_UPSTREAM_FAILURE,
    MODEL_STALENESS.id:                       MODEL_STALENESS,
    FEATURE_IMPORTANCE_INVERSION.id:          FEATURE_IMPORTANCE_INVERSION,
    COMPOUND_DRIFT_PLUS_DEPLOYMENT.id:        COMPOUND_DRIFT_PLUS_DEPLOYMENT,
}

assert len(SCENARIOS) == 18, f"Expected 18 chaos scenarios (full taxonomy), got {len(SCENARIOS)}"
