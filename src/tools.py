"""Tool definitions for the Anthropic API and hardcoded implementations.

Two data tools with static responses for MVP:
  - query_metrics: static Prometheus-style accuracy/confidence degradation data
  - query_logs:    static log snippet showing feature_pipeline schema errors

Plus stop_investigation as the termination control tool.
"""

import json

# ── Anthropic API tool definitions ────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "query_metrics",
        "description": (
            "Query time-series metrics for the ML system. Returns aggregated "
            "values over the requested window, with optional comparison window "
            "for before/after deltas. Use to check whether accuracy, "
            "prediction_confidence, latency_p50, latency_p99, throughput, "
            "or error_rate have shifted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_names": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "accuracy", "prediction_confidence",
                            "latency_p50", "latency_p99",
                            "throughput", "error_rate",
                        ],
                    },
                    "description": "Which metrics to query",
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "required": ["start", "end"],
                    "description": "Primary window (absolute timestamps, anchored to investigation_start)",
                },
                "comparison_window": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "description": "Optional baseline window for delta computation",
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "p50", "p95", "p99", "min", "max"],
                    "default": "mean",
                },
            },
            "required": ["metric_names", "time_range"],
        },
    },
    {
        "name": "query_logs",
        "description": (
            "Search logs for a service over a time window. Returns structured "
            "log entries filtered by severity and/or keyword. Use to detect "
            "errors, anomalies, or warnings in inference_service, "
            "feature_pipeline, label_pipeline, or training_pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": [
                        "inference_service", "feature_pipeline",
                        "label_pipeline", "training_pipeline",
                    ],
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "required": ["start", "end"],
                },
                "filter": {
                    "type": "string",
                    "description": "Substring or regex to filter log messages",
                },
                "severity": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["error", "warning", "info"],
                    },
                    "description": "Restrict to these severity levels (default: all)",
                },
                "max_results": {
                    "type": "integer",
                    "default": 50,
                    "maximum": 200,
                },
            },
            "required": ["service", "time_range"],
        },
    },
    {
        "name": "stop_investigation",
        "description": (
            "Terminate the investigation and submit the final diagnosis. "
            "Call when one hypothesis is clearly dominant (likelihood > 0.60) "
            "or when the tool budget is nearly exhausted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause_category": {
                    "type": "string",
                    "enum": [
                        "feature_drift",
                        "bad_deployment",
                        "label_pipeline_corruption",
                        "training_serving_skew",
                    ],
                },
                "diagnosis": {
                    "type": "string",
                    "description": "Explanation of the root cause and supporting evidence",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in this diagnosis (0.0–1.0)",
                },
                "recommended_action": {
                    "type": "string",
                    "description": "Immediate remediation step",
                },
            },
            "required": [
                "root_cause_category", "diagnosis",
                "confidence", "recommended_action",
            ],
        },
    },
]


# ── Hardcoded implementations ──────────────────────────────────────────────────
#
# Scenario: ML model accuracy dropped 15 pp over 6 hours.
# Ground truth: feature_pipeline started silently corrupting feature_age_days
# due to an upstream schema change, causing significant input distribution shift.

def _query_metrics(inputs: dict) -> dict:
    requested = set(inputs.get("metric_names", []))
    has_comparison = bool(inputs.get("comparison_window"))

    series = []
    delta: dict = {}

    # Accuracy: degraded in primary window vs comparison
    if "accuracy" in requested:
        series.append({
            "metric_name": "accuracy",
            "window": "primary",
            "aggregated_value": 0.76,
            "sample_count": 1440,
        })
        if has_comparison:
            series.append({
                "metric_name": "accuracy",
                "window": "comparison",
                "aggregated_value": 0.91,
                "sample_count": 1440,
            })
            delta["accuracy"] = -0.15

    # Prediction confidence: also degraded (model uncertain about its own outputs)
    if "prediction_confidence" in requested:
        series.append({
            "metric_name": "prediction_confidence",
            "window": "primary",
            "aggregated_value": 0.61,
            "sample_count": 1440,
        })
        if has_comparison:
            series.append({
                "metric_name": "prediction_confidence",
                "window": "comparison",
                "aggregated_value": 0.84,
                "sample_count": 1440,
            })
            delta["prediction_confidence"] = -0.23

    # Latency: essentially unchanged (rules out infra degradation)
    if "latency_p99" in requested:
        series.append({
            "metric_name": "latency_p99",
            "window": "primary",
            "aggregated_value": 143.0,
            "sample_count": 1440,
        })
        if has_comparison:
            series.append({
                "metric_name": "latency_p99",
                "window": "comparison",
                "aggregated_value": 139.0,
                "sample_count": 1440,
            })
            delta["latency_p99"] = 4.0

    # Error rate: flat (rules out hard failures / bad deployment causing crashes)
    if "error_rate" in requested:
        series.append({
            "metric_name": "error_rate",
            "window": "primary",
            "aggregated_value": 0.002,
            "sample_count": 1440,
        })
        if has_comparison:
            series.append({
                "metric_name": "error_rate",
                "window": "comparison",
                "aggregated_value": 0.002,
                "sample_count": 1440,
            })
            delta["error_rate"] = 0.0

    if "throughput" in requested:
        series.append({
            "metric_name": "throughput",
            "window": "primary",
            "aggregated_value": 481.0,
            "sample_count": 1440,
        })
        if has_comparison:
            series.append({
                "metric_name": "throughput",
                "window": "comparison",
                "aggregated_value": 478.0,
                "sample_count": 1440,
            })
            delta["throughput"] = 3.0

    data: dict = {"series": series}
    if has_comparison and delta:
        data["delta"] = delta

    return {
        "tool_name": "query_metrics",
        "status": "ok",
        "data": data,
        "query_metadata": {"latency_ms": 23, "result_count": len(series)},
    }


def _query_logs(inputs: dict) -> dict:
    service = inputs.get("service", "")
    severity_filter = set(inputs.get("severity") or ["error", "warning", "info"])

    if service == "feature_pipeline":
        all_entries = [
            {
                "timestamp": "2026-06-25T08:12:03Z",
                "service": "feature_pipeline",
                "severity": "warning",
                "message": (
                    "feature_age_days: unexpected spike to 847.3 "
                    "(expected range 0–365); 156 rows affected"
                ),
            },
            {
                "timestamp": "2026-06-25T08:14:17Z",
                "service": "feature_pipeline",
                "severity": "warning",
                "message": (
                    "feature_transaction_count_30d: null values in 23.1% of rows "
                    "(threshold: 1%); filling with column mean"
                ),
            },
            {
                "timestamp": "2026-06-25T08:15:01Z",
                "service": "feature_pipeline",
                "severity": "error",
                "message": (
                    "Schema validation FAILED: column 'feature_age_days' has 12 values "
                    "outside expected range [0, 365]; job continued with suppressed errors"
                ),
            },
            {
                "timestamp": "2026-06-25T08:22:44Z",
                "service": "feature_pipeline",
                "severity": "error",
                "message": (
                    "Upstream source 'customer_events' missing column 'event_date'; "
                    "falling back to row insertion_timestamp — this affects feature_age_days"
                ),
            },
            {
                "timestamp": "2026-06-25T08:30:00Z",
                "service": "feature_pipeline",
                "severity": "info",
                "message": (
                    "Batch job completed: 14,203 records processed, "
                    "168 validation failures suppressed and filled with defaults"
                ),
            },
        ]

    elif service == "inference_service":
        all_entries = [
            {
                "timestamp": "2026-06-25T08:00:01Z",
                "service": "inference_service",
                "severity": "info",
                "message": "Serving model v3.2.1 — no deployment events in past 48h",
            },
            {
                "timestamp": "2026-06-25T09:05:12Z",
                "service": "inference_service",
                "severity": "info",
                "message": "Prediction throughput nominal: 481 req/s (SLO: 300 req/s)",
            },
        ]

    elif service == "label_pipeline":
        all_entries = [
            {
                "timestamp": "2026-06-25T07:00:00Z",
                "service": "label_pipeline",
                "severity": "info",
                "message": "Daily label batch job completed successfully: 14,203 labels written",
            },
        ]

    else:
        all_entries = []

    filtered = [e for e in all_entries if e["severity"] in severity_filter]

    return {
        "tool_name": "query_logs",
        "status": "ok",
        "data": {"entries": filtered, "truncated": False},
        "query_metadata": {"latency_ms": 18, "result_count": len(filtered)},
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool call to its implementation. stop_investigation is handled by the loop."""
    if tool_name == "query_metrics":
        return _query_metrics(tool_input)
    if tool_name == "query_logs":
        return _query_logs(tool_input)
    return {
        "tool_name": tool_name,
        "status": "error",
        "error": {
            "error_type": "invalid_parameter",
            "message": f"Unknown tool: {tool_name!r}",
            "retryable": False,
        },
        "query_metadata": {"latency_ms": 0},
    }
