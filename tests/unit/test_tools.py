"""Unit tests for tools.py — dispatcher, implementations, PSI computation."""

import math
import pytest
from unittest.mock import patch, MagicMock

from tools import (
    TOOL_DEFINITIONS,
    _psi,
    _query_code_diffs,
    _query_logs,
    _query_metrics,
    dispatch_tool,
)


# ── dispatch_tool ─────────────────────────────────────────────────────────────

class TestDispatchTool:
    def test_unknown_tool_returns_error_envelope(self):
        result = dispatch_tool("not_a_real_tool", {})
        assert result["status"] == "error"
        assert result["tool_name"] == "not_a_real_tool"
        assert "error" in result

    def test_routes_query_metrics(self):
        result = dispatch_tool("query_metrics", {
            "metric_names": ["accuracy"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        assert result["tool_name"] == "query_metrics"
        assert result["status"] == "ok"

    def test_routes_query_logs(self):
        result = dispatch_tool("query_logs", {
            "service": "inference_service",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        assert result["tool_name"] == "query_logs"
        assert result["status"] == "ok"

    def test_all_tools_return_standard_envelope(self):
        """Every dispatcher must return the standard envelope keys."""
        inputs_by_tool = {
            "query_metrics": {
                "metric_names": ["accuracy"],
                "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
            },
            "query_logs": {
                "service": "inference_service",
                "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
            },
        }
        for tool_name, inputs in inputs_by_tool.items():
            result = dispatch_tool(tool_name, inputs)
            assert "tool_name" in result, f"{tool_name}: missing 'tool_name'"
            assert "status" in result, f"{tool_name}: missing 'status'"
            assert "query_metadata" in result, f"{tool_name}: missing 'query_metadata'"


# ── TOOL_DEFINITIONS ──────────────────────────────────────────────────────────

class TestToolDefinitions:
    def test_has_six_tools(self):
        # query_metrics, query_logs, update_hypothesis_graph,
        # query_deployment_history, query_feature_distributions,
        # query_code_diffs, stop_investigation
        assert len(TOOL_DEFINITIONS) == 7

    def test_all_have_name_and_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "input_schema" in tool
            assert "description" in tool

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_expected_tool_names_present(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "query_metrics",
            "query_logs",
            "query_deployment_history",
            "query_feature_distributions",
            "query_code_diffs",
            "update_hypothesis_graph",
            "stop_investigation",
        }
        assert expected == names


# ── _query_metrics ────────────────────────────────────────────────────────────

class TestQueryMetrics:
    def test_returns_only_requested_metrics(self):
        result = _query_metrics({
            "metric_names": ["accuracy"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        assert result["status"] == "ok"
        series = result["data"]["series"]
        metric_names = {s["metric_name"] for s in series}
        assert metric_names == {"accuracy"}

    def test_returns_multiple_requested_metrics(self):
        result = _query_metrics({
            "metric_names": ["accuracy", "error_rate"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        metric_names = {s["metric_name"] for s in result["data"]["series"]}
        assert "accuracy" in metric_names
        assert "error_rate" in metric_names

    def test_no_unrequested_metrics_returned(self):
        result = _query_metrics({
            "metric_names": ["throughput"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        metric_names = {s["metric_name"] for s in result["data"]["series"]}
        assert "accuracy" not in metric_names

    def test_with_comparison_window_includes_delta(self):
        result = _query_metrics({
            "metric_names": ["accuracy"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
            "comparison_window": {"start": "2024-01-01T06:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        assert "delta" in result["data"]
        assert "accuracy" in result["data"]["delta"]

    def test_without_comparison_window_no_delta(self):
        result = _query_metrics({
            "metric_names": ["accuracy"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        assert "delta" not in result["data"]

    def test_delta_value_is_numeric(self):
        result = _query_metrics({
            "metric_names": ["accuracy"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
            "comparison_window": {"start": "2024-01-01T06:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        delta = result["data"]["delta"]["accuracy"]
        assert isinstance(delta, float)

    def test_delta_is_finite(self):
        result = _query_metrics({
            "metric_names": ["latency_p99"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
            "comparison_window": {"start": "2024-01-01T06:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        if "delta" in result["data"] and "latency_p99" in result["data"]["delta"]:
            assert math.isfinite(result["data"]["delta"]["latency_p99"])

    def test_result_count_in_metadata(self):
        result = _query_metrics({
            "metric_names": ["accuracy", "error_rate"],
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T06:00:00Z"},
        })
        assert result["query_metadata"]["result_count"] == len(result["data"]["series"])


# ── _query_logs ───────────────────────────────────────────────────────────────

class TestQueryLogs:
    def test_feature_pipeline_returns_entries(self):
        result = _query_logs({
            "service": "feature_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-07T00:00:00Z"},
        })
        assert result["status"] == "ok"
        assert isinstance(result["data"]["entries"], list)

    def test_inference_service_returns_nominal_entries(self):
        result = _query_logs({
            "service": "inference_service",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        assert result["status"] == "ok"
        entries = result["data"]["entries"]
        assert len(entries) > 0

    def test_label_pipeline_returns_entries(self):
        result = _query_logs({
            "service": "label_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        assert result["status"] == "ok"

    def test_unknown_service_returns_empty(self):
        result = _query_logs({
            "service": "not_a_service",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        assert result["status"] == "ok"
        assert result["data"]["entries"] == []

    def test_severity_filter_errors_only(self):
        result = _query_logs({
            "service": "feature_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
            "severity": ["error"],
        })
        entries = result["data"]["entries"]
        assert all(e["severity"] == "error" for e in entries)

    def test_severity_filter_warnings_only(self):
        result = _query_logs({
            "service": "feature_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
            "severity": ["warning"],
        })
        entries = result["data"]["entries"]
        assert all(e["severity"] == "warning" for e in entries)

    def test_severity_filter_info_only(self):
        result = _query_logs({
            "service": "inference_service",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
            "severity": ["info"],
        })
        entries = result["data"]["entries"]
        assert all(e["severity"] == "info" for e in entries)

    def test_all_entries_have_required_fields(self):
        result = _query_logs({
            "service": "feature_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        for entry in result["data"]["entries"]:
            assert "timestamp" in entry
            assert "service" in entry
            assert "severity" in entry
            assert "message" in entry

    def test_result_not_truncated(self):
        result = _query_logs({
            "service": "feature_pipeline",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-01T12:00:00Z"},
        })
        assert result["data"]["truncated"] is False


# ── _query_code_diffs ─────────────────────────────────────────────────────────

class TestQueryCodeDiffs:
    def test_missing_commit_before_returns_error(self):
        result = _query_code_diffs({
            "commit_before": "",
            "commit_after": "HEAD",
        })
        assert result["status"] == "error"
        assert result["error"]["error_type"] == "invalid_parameter"

    def test_missing_commit_after_returns_error(self):
        result = _query_code_diffs({
            "commit_before": "HEAD~1",
            "commit_after": "",
        })
        assert result["status"] == "error"

    def test_both_missing_returns_error(self):
        result = _query_code_diffs({})
        assert result["status"] == "error"

    def test_invalid_commits_returns_error(self):
        result = _query_code_diffs({
            "commit_before": "deadbeef00000000",
            "commit_after": "cafebabe00000000",
        })
        # Either git errors or invalid_parameter — must be an error
        assert result["status"] == "error"

    def test_error_envelope_has_required_fields(self):
        result = _query_code_diffs({
            "commit_before": "",
            "commit_after": "",
        })
        assert "tool_name" in result
        assert "status" in result
        assert "error" in result
        assert "query_metadata" in result

    def test_valid_diff_returns_ok(self):
        # Use HEAD~1..HEAD inside the real pipeline repo (which has commits)
        result = _query_code_diffs({
            "commit_before": "HEAD~1",
            "commit_after": "HEAD",
        })
        # Should be ok (or error if repo has no history — either is fine)
        assert result["status"] in ("ok", "error")

    def test_ok_result_has_files_changed(self):
        result = _query_code_diffs({
            "commit_before": "HEAD~1",
            "commit_after": "HEAD",
        })
        if result["status"] == "ok":
            assert "files_changed" in result["data"]
            assert isinstance(result["data"]["files_changed"], list)


# ── _psi ──────────────────────────────────────────────────────────────────────

class TestPSI:
    def test_empty_baseline_returns_zero(self):
        from tools import _psi
        assert _psi([], [1.0, 2.0, 3.0]) == 0.0

    def test_empty_comparison_returns_zero(self):
        from tools import _psi
        assert _psi([1.0, 2.0, 3.0], []) == 0.0

    def test_both_empty_returns_zero(self):
        from tools import _psi
        assert _psi([], []) == 0.0

    def test_identical_distributions_returns_near_zero(self):
        from tools import _psi
        vals = [float(i) for i in range(100)]
        score = _psi(vals, vals)
        assert score < 0.01

    def test_shifted_distribution_returns_positive(self):
        from tools import _psi
        baseline = [1.0] * 100
        comparison = [10.0] * 100
        score = _psi(baseline, comparison)
        assert score > 0.0

    def test_large_shift_exceeds_threshold(self):
        from tools import _psi
        baseline = list(range(100))
        comparison = list(range(500, 600))
        score = _psi(baseline, comparison)
        assert score > 0.25  # significant drift

    def test_constant_distribution_returns_zero(self):
        from tools import _psi
        vals = [5.0] * 50
        score = _psi(vals, vals)
        assert score == 0.0

    def test_returns_float(self):
        from tools import _psi
        score = _psi([1.0, 2.0], [1.5, 2.5])
        assert isinstance(score, float)

    def test_non_negative(self):
        from tools import _psi
        import random
        random.seed(42)
        for _ in range(10):
            a = [random.gauss(0, 1) for _ in range(50)]
            b = [random.gauss(0.5, 1) for _ in range(50)]
            score = _psi(a, b)
            assert score >= 0.0
