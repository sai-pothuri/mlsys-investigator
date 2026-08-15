"""Integration tests: tool dispatch against real SQLite backends.

These tests call the actual implementations that hit the filesystem
(deployments.db, feature_store.db, pipeline_repo git).
"""

import pytest
from pathlib import Path

from tools import (
    _query_deployment_history,
    _query_feature_distributions,
    _query_code_diffs,
    dispatch_tool,
)

_DATA_DIR = Path(__file__).parent.parent.parent / "target-system" / "data"
_PIPELINE_REPO = Path(__file__).parent.parent.parent / "target-system" / "pipeline_repo"


# ── query_deployment_history (SQLite) ─────────────────────────────────────────

class TestQueryDeploymentHistoryIntegration:
    def test_returns_ok_status(self):
        result = _query_deployment_history({
            "service": "inference_service",
            "time_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-12-31T23:59:59Z",
            },
        })
        assert result["status"] == "ok"

    def test_result_has_deployments_key(self):
        result = _query_deployment_history({
            "service": "inference_service",
            "time_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-12-31T23:59:59Z",
            },
        })
        assert "deployments" in result["data"]

    def test_deployments_is_list(self):
        result = _query_deployment_history({
            "service": "feature_pipeline",
            "time_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-12-31T23:59:59Z",
            },
        })
        assert isinstance(result["data"]["deployments"], list)

    def test_deployment_records_have_required_fields(self):
        result = _query_deployment_history({
            "service": "inference_service",
            "time_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-12-31T23:59:59Z",
            },
        })
        required = {"deploy_id", "timestamp", "service", "version_before",
                    "version_after", "commit_sha", "is_rollback"}
        for deployment in result["data"]["deployments"]:
            assert required.issubset(deployment.keys()), (
                f"Deployment record missing fields: {required - deployment.keys()}"
            )

    def test_narrow_time_range_returns_fewer_records(self):
        wide = _query_deployment_history({
            "service": "inference_service",
            "time_range": {"start": "2020-01-01T00:00:00Z", "end": "2030-01-01T00:00:00Z"},
        })
        narrow = _query_deployment_history({
            "service": "inference_service",
            "time_range": {"start": "2050-01-01T00:00:00Z", "end": "2050-01-02T00:00:00Z"},
        })
        assert len(wide["data"]["deployments"]) >= len(narrow["data"]["deployments"])

    def test_invalid_timestamp_returns_error(self):
        result = _query_deployment_history({
            "service": "inference_service",
            "time_range": {"start": "not-a-date", "end": "also-not-a-date"},
        })
        assert result["status"] == "error"
        assert result["error"]["error_type"] == "invalid_parameter"

    def test_metadata_has_latency_and_result_count(self):
        result = _query_deployment_history({
            "service": "inference_service",
            "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
        })
        assert "latency_ms" in result["query_metadata"]
        assert "result_count" in result["query_metadata"]
        assert result["query_metadata"]["result_count"] == len(result["data"]["deployments"])

    def test_all_services_queryable(self):
        for service in ("inference_service", "feature_pipeline", "label_pipeline", "training_pipeline"):
            result = _query_deployment_history({
                "service": service,
                "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
            })
            assert result["status"] == "ok", f"service={service!r} returned error"


# ── query_feature_distributions (SQLite) ─────────────────────────────────────

class TestQueryFeatureDistributionsIntegration:
    def test_unknown_feature_returns_error(self):
        result = _query_feature_distributions({
            "features": ["not_a_real_feature"],
            "baseline_window": {"start": "2024-01-01T00:00:00Z", "end": "2024-06-01T00:00:00Z"},
            "comparison_window": {"start": "2024-06-01T00:00:00Z", "end": "2024-12-31T00:00:00Z"},
        })
        assert result["status"] == "error"
        assert "valid_values" in result["error"]
        assert "features" in result["error"]["valid_values"]

    def test_valid_features_return_ok(self):
        result = _query_feature_distributions({
            "features": ["account_age_days"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        assert result["status"] == "ok"

    def test_result_has_psi_scores(self):
        result = _query_feature_distributions({
            "features": ["account_age_days", "monthly_spend"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        assert result["status"] == "ok"
        results = result["data"]["results"]
        assert len(results) == 2
        for r in results:
            assert "drift_score" in r
            assert "drift_metric_used" in r
            assert "exceeds_threshold" in r

    def test_psi_values_non_negative(self):
        result = _query_feature_distributions({
            "features": ["account_age_days"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        if result["status"] == "ok":
            for r in result["data"]["results"]:
                assert r["drift_score"] >= 0.0

    def test_exceeds_threshold_consistent_with_drift_score(self):
        result = _query_feature_distributions({
            "features": ["account_age_days"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        if result["status"] == "ok":
            for r in result["data"]["results"]:
                expected_exceeds = r["drift_score"] > 0.25
                assert r["exceeds_threshold"] == expected_exceeds

    def test_invalid_timestamp_returns_error(self):
        result = _query_feature_distributions({
            "features": ["account_age_days"],
            "baseline_window": {"start": "not-a-date", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        assert result["status"] == "error"
        assert result["error"]["error_type"] == "invalid_parameter"

    def test_mixed_valid_invalid_features_returns_error(self):
        result = _query_feature_distributions({
            "features": ["account_age_days", "fake_feature"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        assert result["status"] == "error"
        assert "fake_feature" in str(result["error"]["message"])

    def test_valid_features_list_provided_in_error(self):
        result = _query_feature_distributions({
            "features": ["bad_feature"],
            "baseline_window": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
            "comparison_window": {"start": "2023-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        })
        assert result["error"]["valid_values"]["features"]
        assert len(result["error"]["valid_values"]["features"]) > 0

    def test_empty_window_returns_warning_not_crash(self):
        # Window far in the future — no data, should return warning not exception
        result = _query_feature_distributions({
            "features": ["account_age_days"],
            "baseline_window": {"start": "2050-01-01T00:00:00Z", "end": "2050-06-01T00:00:00Z"},
            "comparison_window": {"start": "2050-06-01T00:00:00Z", "end": "2050-12-31T00:00:00Z"},
        })
        assert result["status"] == "ok"
        for r in result["data"]["results"]:
            # PSI should be 0 when one or both windows are empty
            assert r["drift_score"] == 0.0 or "warning" in r


# ── query_code_diffs (git) ────────────────────────────────────────────────────

class TestQueryCodeDiffsIntegration:
    def test_pipeline_repo_has_commits(self):
        """Sanity check: the pipeline repo must have at least one commit."""
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(_PIPELINE_REPO),
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, "pipeline_repo has no git history"

    def test_diff_against_itself_returns_empty(self):
        result = _query_code_diffs({
            "commit_before": "HEAD",
            "commit_after": "HEAD",
        })
        if result["status"] == "ok":
            assert result["data"]["files_changed"] == []

    def test_result_count_in_metadata(self):
        result = _query_code_diffs({
            "commit_before": "HEAD",
            "commit_after": "HEAD",
        })
        if result["status"] == "ok":
            assert result["query_metadata"]["result_count"] == len(result["data"]["files_changed"])

    def test_files_changed_records_have_patch(self):
        # Use HEAD~1..HEAD — there may be changes
        result = _query_code_diffs({
            "commit_before": "HEAD~1",
            "commit_after": "HEAD",
        })
        if result["status"] == "ok":
            for fc in result["data"]["files_changed"]:
                assert "path" in fc
                assert "additions" in fc
                assert "deletions" in fc
                assert "patch" in fc

    def test_truncated_flag_present(self):
        result = _query_code_diffs({
            "commit_before": "HEAD",
            "commit_after": "HEAD",
        })
        if result["status"] == "ok":
            assert "truncated" in result["data"]
            assert isinstance(result["data"]["truncated"], bool)

    def test_path_filter_narrows_diff(self):
        import subprocess
        # Get all changed files from HEAD~1..HEAD
        proc = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            cwd=str(_PIPELINE_REPO),
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            pytest.skip("No changes between HEAD~1 and HEAD in pipeline_repo")

        all_result = _query_code_diffs({"commit_before": "HEAD~1", "commit_after": "HEAD"})
        fake_result = _query_code_diffs({
            "commit_before": "HEAD~1",
            "commit_after": "HEAD",
            "paths": ["__nonexistent_path__.py"],
        })
        if all_result["status"] == "ok" and fake_result["status"] == "ok":
            assert len(all_result["data"]["files_changed"]) >= len(fake_result["data"]["files_changed"])
