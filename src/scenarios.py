"""All hardcoded investigation scenarios for the mock runner.

Three scenarios, each representing a distinct failure category:
  feature_drift    — upstream schema change corrupts a feature silently
  bad_deployment   — new model artifact has a feature normalization shape mismatch
  label_corruption — label pipeline join key change misaligns evaluation labels

Each scenario is self-contained: its own alert text, ground truth, tool
responses, and GraphUpdate fixture sequence.

Usage:
  PYTHONPATH=src python3 src/run_mock.py --scenario bad_deployment
  PYTHONPATH=src python3 src/run_mock.py --list
"""

from dataclasses import dataclass
from typing import Callable

from fixtures import Fixture
from hypothesis_graph import (
    EvidenceInput,
    FailureCategory,
    GraphUpdate,
    Hypothesis,
    HypothesisStatus,
    Severity,
)


# ── Scenario dataclass ────────────────────────────────────────────────────────

@dataclass
class Scenario:
    id: str
    label: str
    alert: str
    ground_truth: str
    fixtures: list[Fixture]
    dispatch_tool: Callable[[str, dict], dict]


# ── Shared malformed fixture (reused across all scenarios) ────────────────────
# Tests that the Output Validator catches hallucinated hypothesis IDs.
# H99 and H88 never exist in any scenario's initial graph.

_MALFORMED_FIXTURE = Fixture(
    label="MALFORMED — HALLUCINATED HYPOTHESIS ID",
    thought=(
        "I want to update the likelihood of H99 which I believe I created "
        "earlier, and record deployment evidence."
    ),
    tool_called="query_deployment_history",
    tool_inputs={
        "service": "inference_service",
        "time_range": {"start": "2026-06-24T08:00:00Z", "end": "2026-06-25T08:00:00Z"},
    },
    observation="(this turn should be rejected before the update is applied)",
    current_focus="H1",
    update=GraphUpdate(
        new_evidence=EvidenceInput(
            tool_called="query_deployment_history",  # unregistered in MVP
            observation="No deployments found.",
            supports=False,
            confidence_delta=-0.10,
        ),
        likelihood_changes={"H1": -0.05, "H99": +0.40},  # H99 hallucinated
        hypotheses_to_rule_out=["H88"],                    # H88 hallucinated
        new_hypotheses=[],
        new_established_facts=[],
        next_experiment_rationale="Updating H99 based on deployment check.",
    ),
    expect_valid=False,
)


# ── Shared metrics response builder ───────────────────────────────────────────

def _metrics_response(
    inputs: dict,
    values: dict[str, tuple[float, float]],  # metric → (primary, comparison)
) -> dict:
    """Build a standard query_metrics response from per-metric (primary, comparison) pairs."""
    requested = set(inputs.get("metric_names", []))
    has_comparison = bool(inputs.get("comparison_window"))
    series = []
    delta = {}
    for metric, (primary, comparison) in values.items():
        if metric not in requested:
            continue
        series.append({"metric_name": metric, "window": "primary",
                        "aggregated_value": primary, "sample_count": 1440})
        if has_comparison:
            series.append({"metric_name": metric, "window": "comparison",
                            "aggregated_value": comparison, "sample_count": 1440})
            delta[metric] = round(primary - comparison, 4)
    data: dict = {"series": series}
    if has_comparison and delta:
        data["delta"] = delta
    return {
        "tool_name": "query_metrics", "status": "ok", "data": data,
        "query_metadata": {"latency_ms": 22, "result_count": len(series)},
    }


def _logs_response(tool_name: str, service: str, entries: list[dict],
                   severity_filter: set[str]) -> dict:
    filtered = [e for e in entries if e["severity"] in severity_filter]
    return {
        "tool_name": "query_logs", "status": "ok",
        "data": {"entries": filtered, "truncated": False},
        "query_metadata": {"latency_ms": 18, "result_count": len(filtered)},
    }


def _unknown_tool(tool_name: str) -> dict:
    return {
        "tool_name": tool_name, "status": "error",
        "error": {"error_type": "invalid_parameter",
                  "message": f"Unknown tool: {tool_name!r}", "retryable": False},
        "query_metadata": {"latency_ms": 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Feature Drift (Upstream Schema Change)
# ══════════════════════════════════════════════════════════════════════════════
# Ground truth: the 'customer_events' table dropped its 'event_date' column.
# The feature pipeline silently fell back to row insertion_timestamp, producing
# feature_age_days values up to 847 days (expected 0–365).
# Discriminating signals:
#   • accuracy ↓  prediction_confidence ↓  (model is getting bad inputs)
#   • latency flat, error_rate flat         (not an infra or deployment issue)
#   • feature_pipeline logs: schema errors + upstream column removal message

def _dispatch_feature_drift(tool_name: str, inputs: dict) -> dict:
    if tool_name == "query_metrics":
        return _metrics_response(inputs, {
            "accuracy":              (0.76, 0.91),
            "prediction_confidence": (0.61, 0.84),
            "latency_p99":           (143.0, 139.0),
            "error_rate":            (0.002, 0.002),
            "throughput":            (481.0, 478.0),
        })
    if tool_name == "query_logs":
        service = inputs.get("service", "")
        sev = set(inputs.get("severity") or ["error", "warning", "info"])
        if service == "feature_pipeline":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T08:12:03Z", "service": "feature_pipeline",
                 "severity": "warning",
                 "message": "feature_age_days: unexpected spike to 847.3 (expected range 0–365); 156 rows affected"},
                {"timestamp": "2026-06-25T08:14:17Z", "service": "feature_pipeline",
                 "severity": "warning",
                 "message": "feature_transaction_count_30d: null values in 23.1% of rows (threshold: 1%); filling with column mean"},
                {"timestamp": "2026-06-25T08:15:01Z", "service": "feature_pipeline",
                 "severity": "error",
                 "message": "Schema validation FAILED: column 'feature_age_days' has 12 values outside expected range [0, 365]; job continued with suppressed errors"},
                {"timestamp": "2026-06-25T08:22:44Z", "service": "feature_pipeline",
                 "severity": "error",
                 "message": "Upstream source 'customer_events' missing column 'event_date'; falling back to row insertion_timestamp — this affects feature_age_days"},
                {"timestamp": "2026-06-25T08:30:00Z", "service": "feature_pipeline",
                 "severity": "info",
                 "message": "Batch job completed: 14,203 records processed, 168 validation failures suppressed and filled with defaults"},
            ], sev)
        if service == "inference_service":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T08:00:01Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Serving model v3.2.1 — no deployment events in past 48h"},
                {"timestamp": "2026-06-25T09:05:12Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Prediction throughput nominal: 481 req/s (SLO: 300 req/s)"},
            ], sev)
        return _logs_response(tool_name, service, [
            {"timestamp": "2026-06-25T07:00:00Z", "service": service,
             "severity": "info", "message": "All systems normal."},
        ], sev)
    return _unknown_tool(tool_name)


_FEATURE_DRIFT_FIXTURES = [
    Fixture(
        label="NORMAL EVIDENCE UPDATE",
        thought=(
            "I'll start with metrics to quantify the degradation and check "
            "whether the pattern is consistent with a model quality issue "
            "or an infrastructure failure."
        ),
        tool_called="query_metrics",
        tool_inputs={
            "metric_names": ["accuracy", "prediction_confidence", "latency_p99", "error_rate"],
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "comparison_window": {"start": "2026-06-24T20:00:00Z", "end": "2026-06-25T02:00:00Z"},
        },
        observation=(
            "accuracy: 0.91→0.76 (−0.15), prediction_confidence: 0.84→0.61 (−0.23). "
            "latency_p99 and error_rate are flat. Both quality metrics degraded "
            "with no infra signal — consistent with bad input data."
        ),
        current_focus="H1",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_metrics",
                observation=(
                    "accuracy dropped 0.91→0.76 (−15pp) and prediction_confidence "
                    "0.84→0.61 (−23pp); latency_p99 and error_rate unchanged — no infra signal."
                ),
                supports=True,
                confidence_delta=0.20,
            ),
            likelihood_changes={"H1": +0.20, "H2": -0.08, "H3": -0.06, "H4": -0.04},
            new_established_facts=[
                "Accuracy degraded 15pp and prediction_confidence 23pp over 6 hours.",
                "Latency and error_rate flat — no infrastructure degradation.",
            ],
            next_experiment_rationale=(
                "Flat error_rate weakens bad_deployment (H2); I'll query "
                "inference_service logs to rule it out."
            ),
        ),
    ),
    Fixture(
        label="HYPOTHESIS RULED OUT",
        thought=(
            "The flat error_rate already weakened bad_deployment. I'll check "
            "inference_service logs for recent deployments or serving errors."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning"],
        },
        observation=(
            "inference_service: serving model v3.2.1 unchanged, no deployments "
            "in past 48h, throughput nominal at 481 req/s. Zero errors."
        ),
        current_focus="H2",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "inference_service: model v3.2.1 unchanged, no deployments "
                    "in past 48h, throughput 481 req/s, zero errors."
                ),
                supports=False,
                confidence_delta=-0.30,
            ),
            likelihood_changes={"H2": -0.25},
            hypotheses_to_rule_out=["H2"],
            new_established_facts=[
                "No model or code deployments in the past 48 hours.",
                "inference_service is healthy — not the failure origin.",
            ],
            next_experiment_rationale=(
                "H2 ruled out. H1 (feature_drift) is highest-likelihood. "
                "Querying feature_pipeline logs next."
            ),
        ),
    ),
    Fixture(
        label="SURPRISING RESULT — NEW HYPOTHESIS",
        thought=(
            "H1 (feature_drift) is highest-likelihood. I'll query feature_pipeline "
            "logs for validation errors that explain why features shifted."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "feature_pipeline",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "feature_pipeline logs: 'feature_age_days' spiked to 847.3 (expected 0–365). "
            "CRITICAL: upstream 'customer_events' table missing column 'event_date'; "
            "pipeline fell back to row insertion_timestamp. 168 validation failures suppressed silently."
        ),
        current_focus="H1",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "feature_pipeline: schema validation errors — feature_age_days up to 847.3 "
                    "(expected ≤365); upstream customer_events missing event_date column; "
                    "pipeline silently using insertion_timestamp; 168 failures suppressed."
                ),
                supports=True,
                confidence_delta=0.25,
            ),
            likelihood_changes={"H1": +0.25, "H3": -0.15},
            new_hypotheses=[
                Hypothesis(
                    id="H5",
                    root_cause_category=FailureCategory.FEATURE_DRIFT,
                    description=(
                        "Upstream 'customer_events' dropped column 'event_date'; "
                        "feature_pipeline fell back to insertion_timestamp, producing "
                        "out-of-range feature_age_days values (up to 847 days)"
                    ),
                    likelihood=0.40,
                    severity=Severity.CRITICAL,
                    status=HypothesisStatus.ACTIVE,
                    evidence=[],
                ),
            ],
            new_established_facts=[
                "feature_pipeline has active schema validation errors on feature_age_days.",
                "Upstream customer_events table is missing event_date column.",
                "Pipeline silently using insertion_timestamp as fallback.",
            ],
            next_experiment_rationale=(
                "H5 is now the most specific and best-supported hypothesis. "
                "Evidence sufficient to call stop_investigation."
            ),
        ),
    ),
    _MALFORMED_FIXTURE,
]

FEATURE_DRIFT = Scenario(
    id="feature_drift",
    label="Feature Drift — Upstream Schema Change",
    alert=(
        "ALERT: Model accuracy dropped from 0.91 to 0.76 over the past 6 hours. "
        "Prediction confidence also degraded. No system errors visible on dashboard."
    ),
    ground_truth=(
        "Upstream 'customer_events' table dropped the 'event_date' column. "
        "Feature pipeline silently fell back to row insertion_timestamp, "
        "producing out-of-range feature_age_days values (up to 847 days). "
        "168 validation failures were suppressed without alerting."
    ),
    fixtures=_FEATURE_DRIFT_FIXTURES,
    dispatch_tool=_dispatch_feature_drift,
)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Bad Deployment (Feature Normalization Shape Mismatch)
# ══════════════════════════════════════════════════════════════════════════════
# Ground truth: model v3.3.0 deployed at 14:58 UTC. Its feature normalizer was
# trained on 14 features; the serving pipeline now provides 28 (after a schema
# addition). The scaler raises ValueError on ~2% of requests, and mis-normalizes
# the rest, degrading accuracy 18pp.
# Discriminating signals:
#   • accuracy ↓  prediction_confidence ↓  error_rate SPIKES (0.002→0.019)
#   • latency_p99 elevated (error handling overhead)
#   • inference_service logs: deployment event at exact start of degradation,
#     ValueError in feature normalization scaler

def _dispatch_bad_deployment(tool_name: str, inputs: dict) -> dict:
    if tool_name == "query_metrics":
        return _metrics_response(inputs, {
            "accuracy":              (0.73, 0.91),
            "prediction_confidence": (0.69, 0.84),
            "latency_p99":           (158.0, 139.0),   # elevated — error handling overhead
            "error_rate":            (0.019, 0.002),   # SPIKE — key signal
            "throughput":            (476.0, 481.0),
        })
    if tool_name == "query_logs":
        service = inputs.get("service", "")
        sev = set(inputs.get("severity") or ["error", "warning", "info"])
        if service == "inference_service":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T14:57:01Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Serving model v3.2.1 (stable for 14 days, zero issues)"},
                {"timestamp": "2026-06-25T14:58:22Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Deployment started: pulling artifact model-v3.3.0 from registry"},
                {"timestamp": "2026-06-25T15:01:17Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Deployment complete: now serving model-v3.3.0"},
                {"timestamp": "2026-06-25T15:02:03Z", "service": "inference_service",
                 "severity": "warning",
                 "message": "Error rate elevated post-deploy: 0.019 (baseline: 0.002) — monitoring"},
                {"timestamp": "2026-06-25T15:05:33Z", "service": "inference_service",
                 "severity": "error",
                 "message": "Prediction failed request_id=a8f2c3: ValueError: scaler expected input shape (1, 14), got (1, 28) — feature normalization mismatch"},
                {"timestamp": "2026-06-25T15:05:34Z", "service": "inference_service",
                 "severity": "error",
                 "message": "Prediction failed request_id=b3c1d9: ValueError: scaler expected input shape (1, 14), got (1, 28) — feature normalization mismatch"},
                {"timestamp": "2026-06-25T15:10:00Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Serving continues on v3.3.0 — auto-rollback not triggered (error rate below 5% threshold)"},
            ], sev)
        if service == "feature_pipeline":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T08:00:00Z", "service": "feature_pipeline",
                 "severity": "info",
                 "message": "Batch job completed normally: 14,203 records processed, 0 validation failures"},
                {"timestamp": "2026-06-25T14:00:00Z", "service": "feature_pipeline",
                 "severity": "info",
                 "message": "Batch job completed normally: 14,311 records processed, 0 validation failures"},
            ], sev)
        return _logs_response(tool_name, service, [
            {"timestamp": "2026-06-25T07:00:00Z", "service": service,
             "severity": "info", "message": "All systems normal."},
        ], sev)
    return _unknown_tool(tool_name)


_BAD_DEPLOYMENT_FIXTURES = [
    Fixture(
        label="ERROR RATE SPIKE POINTS TO CODE REGRESSION",
        thought=(
            "I'll start with metrics to quantify the degradation and look "
            "for signals that distinguish a data issue from a code or model issue. "
            "An error rate spike would strongly suggest a deployment regression."
        ),
        tool_called="query_metrics",
        tool_inputs={
            "metric_names": ["accuracy", "prediction_confidence", "latency_p99", "error_rate"],
            "time_range": {"start": "2026-06-25T09:00:00Z", "end": "2026-06-25T22:00:00Z"},
            "comparison_window": {"start": "2026-06-25T00:00:00Z", "end": "2026-06-25T09:00:00Z"},
        },
        observation=(
            "accuracy dropped 0.91→0.73 (−18pp), error_rate spiked 0.002→0.019 (+850%), "
            "latency_p99 elevated 139→158ms. Prediction_confidence also degraded. "
            "The error_rate spike is the key signal — feature drift or label issues "
            "rarely cause serving errors, but a bad deployment does."
        ),
        current_focus="H2",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_metrics",
                observation=(
                    "accuracy −18pp, error_rate spiked 850% (0.002→0.019), "
                    "latency_p99 elevated 19ms — error spike alongside accuracy drop "
                    "is consistent with a code or model regression."
                ),
                supports=True,
                confidence_delta=0.18,
            ),
            likelihood_changes={"H2": +0.18, "H1": -0.10, "H3": -0.05, "H4": -0.03},
            new_established_facts=[
                "Accuracy dropped 18pp with error_rate spiking 850% — suggests code or model regression.",
                "Latency_p99 elevated 19ms — consistent with error handling overhead.",
            ],
            next_experiment_rationale=(
                "Error rate spike strongly suggests a deployment regression. "
                "Querying inference_service logs for recent deployment events."
            ),
        ),
    ),
    Fixture(
        label="DEPLOYMENT EVENT CONFIRMS ROOT CAUSE",
        thought=(
            "The error rate spike points to a code-level failure. I'll check "
            "inference_service logs for deployment events — a deployment that "
            "coincides with the degradation window confirms bad_deployment."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-25T09:00:00Z", "end": "2026-06-25T22:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "Model v3.3.0 deployed at 14:58 UTC (7 hours ago). Error rate elevated "
            "within 4 minutes of deployment. Error logs: 'ValueError: scaler expected "
            "input shape (1, 14), got (1, 28)' — the new model's normalizer was trained "
            "on 14 features; the serving pipeline now provides 28."
        ),
        current_focus="H2",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "Model v3.3.0 deployed 14:58 UTC; errors began 15:02 UTC "
                    "(4 min post-deploy); ValueError: feature normalization scaler "
                    "shape mismatch (1,14) vs (1,28)."
                ),
                supports=True,
                confidence_delta=0.25,
            ),
            likelihood_changes={"H2": +0.25, "H1": -0.15, "H3": -0.05, "H4": -0.05},
            hypotheses_to_rule_out=["H1", "H3"],
            new_established_facts=[
                "Model v3.3.0 deployed at 14:58 UTC — 7 hours ago.",
                "ValueError in feature normalization: scaler shape mismatch (1,14) vs (1,28).",
                "Errors began 4 minutes post-deployment — confirms deployment as root cause.",
            ],
            next_experiment_rationale=(
                "Deployment event at exact degradation onset with matching error signature. "
                "Confidence sufficient to call stop_investigation."
            ),
        ),
    ),
    _MALFORMED_FIXTURE,
]

BAD_DEPLOYMENT = Scenario(
    id="bad_deployment",
    label="Bad Deployment — Feature Normalizer Shape Mismatch",
    alert=(
        "ALERT: Model accuracy dropped from 0.91 to 0.73 over the past 7 hours. "
        "Error rate has elevated significantly. A model deployment occurred earlier today."
    ),
    ground_truth=(
        "Model v3.3.0 deployed at 14:58 UTC. Its feature normalization scaler was "
        "trained on 14 features; the serving pipeline provides 28 (schema addition "
        "not reflected in the training pipeline). Scaler raises ValueError on ~2% of "
        "requests; mis-normalizes the rest, degrading accuracy 18pp."
    ),
    fixtures=_BAD_DEPLOYMENT_FIXTURES,
    dispatch_tool=_dispatch_bad_deployment,
)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Label Pipeline Corruption (Join Key Misconfiguration)
# ══════════════════════════════════════════════════════════════════════════════
# Ground truth: label_pipeline config changed from 'session_id' to 'request_id'
# as the join key for matching outcome labels to predictions. The two IDs don't
# correspond, so 38.8% of labels are misaligned. The model itself is fine —
# the accuracy drop is a measurement artifact.
# Discriminating signals:
#   • accuracy ↓  BUT prediction_confidence STABLE (0.84→0.83) — KEY SIGNAL
#     (model is confident; it's being evaluated against wrong labels)
#   • error_rate flat, latency flat — no serving-layer issues
#   • label_pipeline logs: join key change, label alignment rate 61.2%

def _dispatch_label_corruption(tool_name: str, inputs: dict) -> dict:
    if tool_name == "query_metrics":
        return _metrics_response(inputs, {
            "accuracy":              (0.74, 0.91),
            "prediction_confidence": (0.83, 0.84),   # essentially flat — KEY SIGNAL
            "latency_p99":           (140.0, 139.0),
            "error_rate":            (0.002, 0.002),
            "throughput":            (480.0, 481.0),
        })
    if tool_name == "query_logs":
        service = inputs.get("service", "")
        sev = set(inputs.get("severity") or ["error", "warning", "info"])
        if service == "label_pipeline":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T07:14:02Z", "service": "label_pipeline",
                 "severity": "info",
                 "message": "Starting label ingestion job — config updated to use 'request_id' as join key (previously 'session_id')"},
                {"timestamp": "2026-06-25T07:22:41Z", "service": "label_pipeline",
                 "severity": "warning",
                 "message": "847 outcome records could not be matched to prediction logs via 'request_id' — label not found; marking as incorrect predictions"},
                {"timestamp": "2026-06-25T07:30:00Z", "service": "label_pipeline",
                 "severity": "error",
                 "message": "Label alignment rate: 61.2% (threshold: 95%) — 38.8% of labels are misaligned or missing; evaluation metrics will be unreliable"},
                {"timestamp": "2026-06-25T07:30:01Z", "service": "label_pipeline",
                 "severity": "warning",
                 "message": "Proceeding with partial label set — downstream accuracy metrics will under-report true model performance"},
                {"timestamp": "2026-06-25T08:00:00Z", "service": "label_pipeline",
                 "severity": "info",
                 "message": "Label ingestion completed: 8,710 of 14,203 records labelled (61.3% coverage)"},
            ], sev)
        if service == "inference_service":
            return _logs_response(tool_name, service, [
                {"timestamp": "2026-06-25T08:00:01Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Serving model v3.2.1 — no deployment events in past 48h"},
                {"timestamp": "2026-06-25T09:05:12Z", "service": "inference_service",
                 "severity": "info",
                 "message": "Prediction throughput nominal: 480 req/s; zero prediction errors"},
            ], sev)
        return _logs_response(tool_name, service, [
            {"timestamp": "2026-06-25T07:00:00Z", "service": service,
             "severity": "info", "message": "All systems normal."},
        ], sev)
    return _unknown_tool(tool_name)


_LABEL_CORRUPTION_FIXTURES = [
    Fixture(
        label="STABLE CONFIDENCE FINGERPRINTS LABEL CORRUPTION",
        thought=(
            "I'll start with metrics, paying close attention to prediction_confidence. "
            "If accuracy dropped but confidence is stable, the model isn't degrading — "
            "it's being evaluated against wrong labels. If both dropped, the model "
            "is receiving bad inputs (feature_drift or bad_deployment)."
        ),
        tool_called="query_metrics",
        tool_inputs={
            "metric_names": ["accuracy", "prediction_confidence", "latency_p99", "error_rate"],
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "comparison_window": {"start": "2026-06-24T20:00:00Z", "end": "2026-06-25T02:00:00Z"},
        },
        observation=(
            "accuracy dropped 0.91→0.74 (−17pp) BUT prediction_confidence is essentially "
            "unchanged: 0.84→0.83 (−0.01). Error rate flat, latency flat. "
            "The model is just as confident as before — it's not receiving bad inputs. "
            "It's being evaluated against wrong labels."
        ),
        current_focus="H3",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_metrics",
                observation=(
                    "accuracy −17pp but prediction_confidence stable (0.84→0.83, −0.01); "
                    "error_rate and latency flat. Stable model confidence with accuracy drop "
                    "is the fingerprint of label corruption, not model or data degradation."
                ),
                supports=True,
                confidence_delta=0.22,
            ),
            likelihood_changes={"H3": +0.22, "H1": -0.08, "H2": -0.08, "H4": -0.06},
            new_established_facts=[
                "Prediction confidence stable (0.84→0.83) despite 17pp accuracy drop.",
                "Stable confidence + accuracy drop = model is healthy, evaluation labels are wrong.",
            ],
            next_experiment_rationale=(
                "Stable confidence rules out feature_drift and bad_deployment (both degrade "
                "model confidence). Querying label_pipeline logs directly."
            ),
        ),
    ),
    Fixture(
        label="LABEL PIPELINE JOIN KEY CHANGE CONFIRMED",
        thought=(
            "The stable confidence strongly points to H3 (label_pipeline_corruption). "
            "I'll check label_pipeline logs for any configuration change or error "
            "that would explain why labels are misaligned."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "label_pipeline",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T09:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "Label pipeline config updated 8 hours ago to join on 'request_id' instead "
            "of 'session_id'. Since the two IDs don't correspond, 847 predictions could "
            "not be matched to outcome labels. Label alignment rate dropped to 61.2% "
            "(threshold 95%). 38.8% of labels are misaligned — accuracy drop is a "
            "measurement artifact, not model degradation."
        ),
        current_focus="H3",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "label_pipeline join key changed session_id→request_id at 07:14 UTC; "
                    "label alignment rate dropped from ~99% to 61.2%; "
                    "847 predictions marked incorrect due to missing label match."
                ),
                supports=True,
                confidence_delta=0.25,
            ),
            likelihood_changes={"H3": +0.25, "H1": -0.10, "H2": -0.05, "H4": -0.10},
            hypotheses_to_rule_out=["H1", "H2", "H4"],
            new_established_facts=[
                "Label pipeline join key changed session_id→request_id at 07:14 UTC.",
                "Label alignment rate: 61.2% (down from ~99%) — 38.8% of labels are wrong.",
                "Accuracy drop is a measurement artifact. Model performance is unchanged.",
            ],
            next_experiment_rationale=(
                "Root cause confirmed: label_pipeline config change caused label misalignment. "
                "Calling stop_investigation."
            ),
        ),
    ),
    _MALFORMED_FIXTURE,
]

LABEL_CORRUPTION = Scenario(
    id="label_corruption",
    label="Label Pipeline Corruption — Join Key Misconfiguration",
    alert=(
        "ALERT: Model accuracy dropped from 0.91 to 0.74 over the past 8 hours. "
        "No infrastructure errors reported. Engineers unsure if the model degraded "
        "or if something changed in the evaluation pipeline."
    ),
    ground_truth=(
        "Label pipeline config changed to use 'request_id' as the join key instead "
        "of 'session_id'. The two IDs don't correspond, so 38.8% of outcome labels "
        "are assigned to wrong predictions or left unmatched (marked as incorrect). "
        "The model itself is performing normally — the accuracy drop is a measurement artifact."
    ),
    fixtures=_LABEL_CORRUPTION_FIXTURES,
    dispatch_tool=_dispatch_label_corruption,
)


# ── Registry ──────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, Scenario] = {
    FEATURE_DRIFT.id:    FEATURE_DRIFT,
    BAD_DEPLOYMENT.id:   BAD_DEPLOYMENT,
    LABEL_CORRUPTION.id: LABEL_CORRUPTION,
}
