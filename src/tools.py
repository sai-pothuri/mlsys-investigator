"""Tool definitions for the Anthropic API and dispatcher implementations.

Data tools:
  - query_metrics:              static responses (real SQLite wired once chaos injection exists)
  - query_logs:                 static responses (same caveat)
  - query_deployment_history:   live query against target-system/data/deployments.db
  - query_feature_distributions: live PSI computation against target-system/data/feature_store.db
  - query_code_diffs:           git diff inside target-system/pipeline_repo/

Plus stop_investigation as the termination control tool.
"""

import json
import math
import os
import sqlite3
import subprocess
import time
from pathlib import Path

from prompts import (
    ALTERNATIVE_CATEGORIES_DESCRIPTION,
    ROOT_CAUSE_CATEGORY_DESCRIPTION,
    STOP_INVESTIGATION_DESCRIPTION,
)

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
        "name": "update_hypothesis_graph",
        "description": (
            "Record what you learned from the last tool result into the structured "
            "hypothesis graph. Call this after EVERY query tool result, before calling "
            "another tool or stop_investigation. Does NOT count against your tool budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["update", "create", "merge"],
                    "description": (
                        "'update' (most common): evidence tests an existing hypothesis — "
                        "provide current_focus. "
                        "'create': evidence suggests a root cause not yet in the graph — "
                        "provide new_hypothesis_* fields. "
                        "'merge': your proposed new hypothesis is semantically equivalent "
                        "to an existing one — provide merge_into_id."
                    ),
                },
                "current_focus": {
                    "type": "string",
                    "description": "Required for action='update'. Hypothesis ID this evidence most directly tests (e.g. 'H1')",
                },
                "merge_into_id": {
                    "type": "string",
                    "description": "Required for action='merge'. The existing hypothesis ID to merge evidence into.",
                },
                "new_hypothesis_description": {
                    "type": "string",
                    "description": "Required for action='create'. What the new hypothesis claims as root cause.",
                },
                "new_hypothesis_severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "For action='create'. Severity of the new hypothesis (default: medium).",
                },
                "new_hypothesis_initial_likelihood": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "For action='create'. Starting likelihood weight, e.g. 0.10–0.20 (default: 0.15).",
                },
                "new_evidence": {
                    "type": "object",
                    "properties": {
                        "tool_called": {
                            "type": "string",
                            "enum": [
                                "query_metrics",
                                "query_logs",
                                "query_deployment_history",
                                "query_feature_distributions",
                                "query_code_diffs",
                            ],
                            "description": "The tool you just called",
                        },
                        "observation": {
                            "type": "string",
                            "description": (
                                "Precise, concise statement of what the tool returned. "
                                "Quote specific numbers."
                            ),
                        },
                        "supports": {
                            "type": "boolean",
                            "description": "True if this observation supports the focused hypothesis being the root cause",
                        },
                        "confidence_delta": {
                            "type": "number",
                            "minimum": -1.0,
                            "maximum": 1.0,
                            "description": "Confidence shift for the focused hypothesis (-1.0 to +1.0; typical ±0.05–0.25)",
                        },
                    },
                    "required": ["tool_called", "observation", "supports", "confidence_delta"],
                },
                "likelihood_changes": {
                    "type": "object",
                    "description": "Map of hypothesis ID → likelihood delta. Will be normalized.",
                    "additionalProperties": {"type": "number"},
                },
                "hypotheses_to_rule_out": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hypothesis IDs you are now confident are NOT the root cause",
                },
                "new_established_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Definitive facts established by this observation",
                },
                "next_experiment_rationale": {
                    "type": "string",
                    "description": "Why you're calling the next tool, or why you're now confident enough to stop",
                },
            },
            "required": [
                "action", "new_evidence", "likelihood_changes",
                "next_experiment_rationale",
            ],
        },
    },
    {
        "name": "query_deployment_history",
        "description": (
            "Retrieve deployment events for a service within a time window. "
            "Returns timestamps, version transitions, commit SHAs, and whether each "
            "event is a rollback. Use to check whether a deployment coincides with "
            "the onset of degradation. Pass a commit_sha from a result here to "
            "query_code_diffs to see what actually changed."
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
                    "description": "Which service's deployments to retrieve",
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "required": ["start", "end"],
                },
            },
            "required": ["service", "time_range"],
        },
    },
    {
        "name": "query_feature_distributions",
        "description": (
            "Compare feature value distributions between a baseline window and a "
            "comparison window using PSI (Population Stability Index). "
            "PSI < 0.1: no significant change. 0.1–0.25: moderate shift. >0.25: significant drift. "
            "Use to detect input feature drift that could explain model degradation. "
            "Supply feature names as strings; invalid names return valid_values for self-correction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature names to analyse (free text — validated at dispatch)",
                },
                "baseline_window": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "required": ["start", "end"],
                    "description": "Earlier window representing normal behaviour",
                },
                "comparison_window": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end":   {"type": "string", "format": "date-time"},
                    },
                    "required": ["start", "end"],
                    "description": "Window to compare against the baseline",
                },
            },
            "required": ["features", "baseline_window", "comparison_window"],
        },
    },
    {
        "name": "query_code_diffs",
        "description": (
            "Retrieve the unified diff between two commits in the ML pipeline repo. "
            "Typical flow: get a commit_sha from query_deployment_history, then call "
            "this tool with commit_before=<sha_before_deploy> and commit_after=<sha_after_deploy> "
            "to see exactly what changed. Results are capped at ~200 lines per file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commit_before": {
                    "type": "string",
                    "description": "Git commit SHA for the earlier state (or 'HEAD~N')",
                },
                "commit_after": {
                    "type": "string",
                    "description": "Git commit SHA for the later state (or 'HEAD')",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional path filters (e.g. ['feature_engineering.py']). All files if omitted.",
                },
            },
            "required": ["commit_before", "commit_after"],
        },
    },
    {
        "name": "stop_investigation",
        "description": STOP_INVESTIGATION_DESCRIPTION.content,
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause_category": {
                    "type": "string",
                    "description": ROOT_CAUSE_CATEGORY_DESCRIPTION.content,
                    "enum": [
                        "feature_drift",
                        "bad_deployment",
                        "upstream_schema_change",
                        "infrastructure_latency_spike",
                        "model_version_rollback_regression",
                        "label_pipeline_corruption",
                        "training_serving_skew",
                        "data_freshness_degradation",
                        "feature_encoding_bug",
                        "gradual_concept_drift",
                        "model_calibration_drift",
                        "shadow_mode_leak",
                        "feature_pipeline_partial_failure",
                        "delayed_label_feedback_shift",
                        "cascading_upstream_failure",
                        "model_staleness",
                        "feature_importance_inversion",
                        "compound_drift_plus_deployment",
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
                "alternative_categories": {
                    "type": "array",
                    "description": ALTERNATIVE_CATEGORIES_DESCRIPTION.content,
                    "items": {
                        "type": "string",
                        "enum": [
                            "feature_drift",
                            "bad_deployment",
                            "upstream_schema_change",
                            "infrastructure_latency_spike",
                            "model_version_rollback_regression",
                            "label_pipeline_corruption",
                            "training_serving_skew",
                            "data_freshness_degradation",
                            "feature_encoding_bug",
                            "gradual_concept_drift",
                            "model_calibration_drift",
                            "shadow_mode_leak",
                            "feature_pipeline_partial_failure",
                            "delayed_label_feedback_shift",
                            "cascading_upstream_failure",
                            "model_staleness",
                            "feature_importance_inversion",
                            "compound_drift_plus_deployment",
                        ],
                    },
                    "maxItems": 2,
                },
            },
            "required": [
                "root_cause_category", "diagnosis",
                "confidence", "recommended_action",
            ],
        },
    },
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _parse_ts(s: str) -> float:
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _aggregate_vals(vals: list, aggregation: str) -> float:
    if not vals:
        return 0.0
    if aggregation == "mean":
        return sum(vals) / len(vals)
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    pct_map = {"p50": 0.50, "p95": 0.95, "p99": 0.99}
    if aggregation in pct_map:
        idx = min(int(n * pct_map[aggregation]), n - 1)
        return sorted_vals[idx]
    if aggregation == "min":
        return sorted_vals[0]
    if aggregation == "max":
        return sorted_vals[-1]
    return sum(vals) / len(vals)


# ── Real SQLite-backed tool implementations ────────────────────────────────────

def _query_metrics(inputs: dict) -> dict:
    requested = list(inputs.get("metric_names", []))
    tr = inputs.get("time_range", {})
    cw_input = inputs.get("comparison_window")
    aggregation = inputs.get("aggregation", "mean")

    try:
        t_start = _parse_ts(tr["start"])
        t_end   = _parse_ts(tr["end"])
    except Exception as exc:
        return {
            "tool_name": "query_metrics",
            "status": "error",
            "error": {"error_type": "invalid_parameter", "message": str(exc), "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    db_path = _data_dir() / "metrics.db"
    if not db_path.exists():
        return {
            "tool_name": "query_metrics",
            "status": "error",
            "error": {"error_type": "service_unavailable",
                      "message": "metrics.db not found — run the data generator first",
                      "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    t0 = time.monotonic()

    def _fetch_window(start: float, end: float, window_label: str) -> list:
        placeholders = ",".join("?" * len(requested))
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            f"SELECT metric_name, metric_value FROM metrics "
            f"WHERE timestamp >= ? AND timestamp < ? AND metric_name IN ({placeholders}) "
            f"ORDER BY timestamp",
            [start, end, *requested],
        ).fetchall()
        con.close()
        by_metric: dict = {}
        for metric_name, metric_value in rows:
            by_metric.setdefault(metric_name, []).append(metric_value)
        result = []
        for metric in requested:
            vals = by_metric.get(metric, [])
            result.append({
                "metric_name":      metric,
                "window":           window_label,
                "aggregated_value": round(_aggregate_vals(vals, aggregation), 6),
                "sample_count":     len(vals),
            })
        return result

    series = _fetch_window(t_start, t_end, "primary")
    delta: dict = {}

    if cw_input:
        try:
            c_start = _parse_ts(cw_input["start"])
            c_end   = _parse_ts(cw_input["end"])
        except Exception as exc:
            return {
                "tool_name": "query_metrics",
                "status": "error",
                "error": {"error_type": "invalid_parameter", "message": str(exc), "retryable": False},
                "query_metadata": {"latency_ms": 0},
            }
        comp_series = _fetch_window(c_start, c_end, "comparison")
        primary_vals = {s["metric_name"]: s["aggregated_value"] for s in series}
        comp_vals    = {s["metric_name"]: s["aggregated_value"] for s in comp_series}
        for metric in primary_vals:
            if metric in comp_vals:
                delta[metric] = round(primary_vals[metric] - comp_vals[metric], 4)
        series = series + comp_series

    latency_ms = int((time.monotonic() - t0) * 1000)
    data: dict = {"series": series}
    if delta:
        data["delta"] = delta

    return {
        "tool_name": "query_metrics",
        "status": "ok",
        "data": data,
        "query_metadata": {"latency_ms": latency_ms, "result_count": len(series)},
    }


def _query_logs(inputs: dict) -> dict:
    service    = inputs.get("service", "")
    tr         = inputs.get("time_range", {})
    # DB stores severity as uppercase; tool input uses lowercase
    severity_filter = [s.upper() for s in (inputs.get("severity") or ["error", "warning", "info"])]
    filter_text = inputs.get("filter")
    max_results = min(int(inputs.get("max_results") or 50), 200)

    try:
        t_start = _parse_ts(tr["start"])
        t_end   = _parse_ts(tr["end"])
    except Exception as exc:
        return {
            "tool_name": "query_logs",
            "status": "error",
            "error": {"error_type": "invalid_parameter", "message": str(exc), "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    db_path = _data_dir() / "logs.db"
    if not db_path.exists():
        return {
            "tool_name": "query_logs",
            "status": "error",
            "error": {"error_type": "service_unavailable",
                      "message": "logs.db not found — run the data generator first",
                      "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    t0 = time.monotonic()
    params: list = [t_start, t_end, service]
    sql = (
        "SELECT timestamp, service, severity, message "
        "FROM logs WHERE timestamp >= ? AND timestamp < ? AND service = ?"
    )
    if severity_filter:
        placeholders = ",".join("?" * len(severity_filter))
        sql += f" AND severity IN ({placeholders})"
        params.extend(severity_filter)
    if filter_text:
        escaped = filter_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " AND message LIKE ? ESCAPE '\\'"
        params.append(f"%{escaped}%")
    sql += " ORDER BY timestamp LIMIT ?"
    params.append(max_results + 1)

    con = sqlite3.connect(str(db_path))
    rows = con.execute(sql, params).fetchall()
    con.close()
    latency_ms = int((time.monotonic() - t0) * 1000)

    truncated = len(rows) > max_results
    entries = []
    from datetime import datetime, timezone
    for ts, svc, sev, msg in rows[:max_results]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        entries.append({
            "timestamp": dt.isoformat().replace("+00:00", "Z"),
            "service":   svc,
            "severity":  sev.lower(),
            "message":   msg,
        })

    return {
        "tool_name": "query_logs",
        "status": "ok",
        "data": {"entries": entries, "truncated": truncated},
        "query_metadata": {"latency_ms": latency_ms, "result_count": len(entries)},
    }


# ── Real backends ─────────────────────────────────────────────────────────────

_DATA_DIR_DEFAULT = Path(__file__).parent.parent / "target-system" / "data"
_PIPELINE_REPO = Path(__file__).parent.parent / "target-system" / "pipeline_repo"


def _data_dir() -> Path:
    override = os.environ.get("MLSYS_DATA_DIR", "")
    return Path(override) if override else _DATA_DIR_DEFAULT

_KNOWN_FEATURES = [
    "account_age_days", "monthly_spend", "num_transactions_30d",
    "avg_transaction_value", "days_since_last_login", "support_tickets_90d",
    "product_category", "region", "device_type",
    "login_failure_rate", "session_duration_min", "referral_source",
]


def _query_deployment_history(inputs: dict) -> dict:
    service = inputs.get("service", "")
    tr = inputs.get("time_range", {})
    try:
        t_start = _parse_ts(tr["start"])
        t_end   = _parse_ts(tr["end"])
    except Exception as exc:
        return {
            "tool_name": "query_deployment_history",
            "status": "error",
            "error": {"error_type": "invalid_parameter", "message": str(exc), "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    db_path = _data_dir() / "deployments.db"
    if not db_path.exists():
        return {
            "tool_name": "query_deployment_history",
            "status": "error",
            "error": {"error_type": "service_unavailable",
                      "message": "deployments.db not found — run the data generator first",
                      "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }
    t0 = time.monotonic()
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        """
        SELECT deploy_id, timestamp, service, version_before, version_after,
               commit_sha, change_type, changelog, deployed_by, is_rollback
        FROM deployments
        WHERE service = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """,
        (service, t_start, t_end),
    ).fetchall()
    con.close()
    latency_ms = int((time.monotonic() - t0) * 1000)

    from datetime import datetime, timezone as _tz
    deployments = []
    for row in rows:
        deployments.append({
            "deploy_id":      row[0],
            "timestamp":      datetime.fromtimestamp(row[1], tz=_tz.utc).isoformat(),
            "service":        row[2],
            "version_before": row[3],
            "version_after":  row[4],
            "commit_sha":     row[5],
            "change_type":    row[6],
            "changelog":      row[7],
            "deployed_by":    row[8],
            "is_rollback":    bool(row[9]),
        })

    return {
        "tool_name": "query_deployment_history",
        "status": "ok",
        "data": {"deployments": deployments},
        "query_metadata": {"latency_ms": latency_ms, "result_count": len(deployments)},
    }


def _psi(baseline: list, comparison: list, bins: int = 10) -> float:
    """Population Stability Index between two numerical samples."""
    if not baseline or not comparison:
        return 0.0
    lo = min(min(baseline), min(comparison))
    hi = max(max(baseline), max(comparison))
    if hi == lo:
        return 0.0

    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]

    def _bin(vals: list) -> list:
        counts = [0] * bins
        for v in vals:
            idx = int((v - lo) / (hi - lo) * bins)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        total = len(vals)
        return [max(c / total, 1e-8) for c in counts]

    b = _bin(baseline)
    c = _bin(comparison)
    return sum((c[i] - b[i]) * math.log(c[i] / b[i]) for i in range(bins))


def _query_feature_distributions(inputs: dict) -> dict:
    features = inputs.get("features", [])
    bw = inputs.get("baseline_window", {})
    cw = inputs.get("comparison_window", {})

    # Validate feature names and return valid_values on miss.
    unknown = [f for f in features if f not in _KNOWN_FEATURES]
    if unknown:
        return {
            "tool_name": "query_feature_distributions",
            "status": "error",
            "error": {
                "error_type": "invalid_parameter",
                "message": f"Unknown feature(s): {unknown}. Use valid_values to self-correct.",
                "retryable": False,
                "valid_values": {"features": _KNOWN_FEATURES},
            },
            "query_metadata": {"latency_ms": 0},
        }

    try:
        b_start, b_end = _parse_ts(bw["start"]), _parse_ts(bw["end"])
        c_start, c_end = _parse_ts(cw["start"]), _parse_ts(cw["end"])
    except Exception as exc:
        return {
            "tool_name": "query_feature_distributions",
            "status": "error",
            "error": {"error_type": "invalid_parameter", "message": str(exc), "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }

    db_path = _data_dir() / "feature_store.db"
    if not db_path.exists():
        return {
            "tool_name": "query_feature_distributions",
            "status": "error",
            "error": {"error_type": "service_unavailable",
                      "message": "feature_store.db not found — run the data generator first",
                      "retryable": False},
            "query_metadata": {"latency_ms": 0},
        }
    t0 = time.monotonic()
    con = sqlite3.connect(str(db_path))
    try:
        def _fetch_window(t_start: float, t_end: float) -> list:
            return con.execute(
                "SELECT features FROM requests WHERE timestamp >= ? AND timestamp <= ?",
                (t_start, t_end),
            ).fetchall()

        baseline_rows  = _fetch_window(b_start, b_end)
        comparison_rows = _fetch_window(c_start, c_end)
    finally:
        con.close()
    latency_ms = int((time.monotonic() - t0) * 1000)

    def _extract(rows: list, feat: str) -> list:
        vals = []
        for (feat_json,) in rows:
            try:
                d = json.loads(feat_json)
                if feat in d and d[feat] is not None:
                    vals.append(float(d[feat]))
            except Exception:
                pass
        return vals

    results = []
    for feat in features:
        b_vals = _extract(baseline_rows, feat)
        c_vals = _extract(comparison_rows, feat)
        score  = _psi(b_vals, c_vals)
        b_mean = sum(b_vals) / len(b_vals) if b_vals else None
        c_mean = sum(c_vals) / len(c_vals) if c_vals else None
        entry: dict = {
            "feature_name":      feat,
            "drift_score":       round(score, 4),
            "drift_metric_used": "psi",
            "exceeds_threshold": score > 0.25,
            "baseline_mean":     round(b_mean, 4) if b_mean is not None else None,
            "comparison_mean":   round(c_mean, 4) if c_mean is not None else None,
            "baseline_n":        len(b_vals),
            "comparison_n":      len(c_vals),
        }
        if not b_vals:
            entry["warning"] = "baseline window contains no data — PSI score is 0.0 and unreliable"
        results.append(entry)

    return {
        "tool_name": "query_feature_distributions",
        "status": "ok",
        "data": {"results": results},
        "query_metadata": {"latency_ms": latency_ms, "result_count": len(results)},
    }


_MAX_DIFF_LINES = 200


def _query_code_diffs(inputs: dict) -> dict:
    commit_before = inputs.get("commit_before", "").strip()
    commit_after  = inputs.get("commit_after",  "").strip()
    paths         = inputs.get("paths") or []

    import re as _re
    _SHA_RE = _re.compile(r"^[0-9a-f]{7,40}$")

    if not commit_before or not commit_after:
        return {
            "tool_name": "query_code_diffs",
            "status": "error",
            "error": {
                "error_type": "invalid_parameter",
                "message": "commit_before and commit_after are required",
                "retryable": False,
            },
            "query_metadata": {"latency_ms": 0},
        }

    for ref_name, ref_val in (("commit_before", commit_before), ("commit_after", commit_after)):
        if not _SHA_RE.match(ref_val):
            return {
                "tool_name": "query_code_diffs",
                "status": "error",
                "error": {
                    "error_type": "invalid_parameter",
                    "message": f"{ref_name} {ref_val!r} is not a valid commit SHA (7–40 hex chars)",
                    "retryable": False,
                },
                "query_metadata": {"latency_ms": 0},
            }

    repo = str(_PIPELINE_REPO)
    cmd  = ["git", "diff", commit_before, commit_after, "--", *paths] if paths else \
           ["git", "diff", commit_before, commit_after]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {
            "tool_name": "query_code_diffs",
            "status": "error",
            "error": {
                "error_type": "service_unavailable",
                "message": "git not found or pipeline_repo does not exist",
                "retryable": False,
            },
            "query_metadata": {"latency_ms": 0},
        }
    except subprocess.TimeoutExpired:
        return {
            "tool_name": "query_code_diffs",
            "status": "error",
            "error": {"error_type": "timeout", "message": "git diff timed out", "retryable": True},
            "query_metadata": {"latency_ms": 10000},
        }

    if result.returncode != 0:
        return {
            "tool_name": "query_code_diffs",
            "status": "error",
            "error": {
                "error_type": "invalid_parameter",
                "message": result.stderr.strip(),
                "retryable": False,
            },
            "query_metadata": {"latency_ms": 0},
        }

    raw_diff = result.stdout or ""
    lines    = raw_diff.splitlines()
    truncated = len(lines) > _MAX_DIFF_LINES
    if truncated:
        lines = lines[:_MAX_DIFF_LINES]

    files_changed: list = []
    current_file: str | None = None
    current_patch: list = []
    additions = deletions = 0

    for line in lines:
        if line.startswith("diff --git"):
            if current_file is not None:
                files_changed.append({
                    "path": current_file,
                    "additions": additions,
                    "deletions": deletions,
                    "patch": "\n".join(current_patch),
                })
            current_file  = line.split(" b/")[-1] if " b/" in line else line
            current_patch = [line]
            additions = deletions = 0
        else:
            if current_file is not None:
                current_patch.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1

    if current_file is not None:
        files_changed.append({
            "path": current_file,
            "additions": additions,
            "deletions": deletions,
            "patch": "\n".join(current_patch),
        })

    return {
        "tool_name": "query_code_diffs",
        "status": "ok",
        "data": {"files_changed": files_changed, "truncated": truncated},
        "query_metadata": {"latency_ms": 0, "result_count": len(files_changed)},
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool call to its implementation. stop_investigation is handled by the loop."""
    if tool_name == "query_metrics":
        return _query_metrics(tool_input)
    if tool_name == "query_logs":
        return _query_logs(tool_input)
    if tool_name == "query_deployment_history":
        return _query_deployment_history(tool_input)
    if tool_name == "query_feature_distributions":
        return _query_feature_distributions(tool_input)
    if tool_name == "query_code_diffs":
        return _query_code_diffs(tool_input)
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
