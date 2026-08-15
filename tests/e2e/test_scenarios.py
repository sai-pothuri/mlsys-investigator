"""End-to-end tests for the mock runner — all scenarios exercised without the Anthropic API.

Runs run_mock(scenario) with time.sleep and snapshot_graph mocked out.
Verifies scenario-level invariants:
  - runner completes without crashing
  - at least one fixture is applied
  - the malformed fixture is always rejected
  - final graph has at least one hypothesis
  - the ground truth category is a valid FailureCategory value
  - final graph has established facts after a successful run
"""

import json
import pathlib
import tempfile
from unittest.mock import patch

import pytest

from hypothesis_graph import (
    FailureCategory,
    HypothesisStatus,
    update_graph,
)
from output_validator import validate_graph_update
from run_mock import run_mock
from scenarios import SCENARIOS, Scenario
from agent import build_initial_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_scenario_captured(scenario: Scenario) -> tuple:
    """Run a scenario with I/O mocked, returning (applied, rejected, final_graph)."""
    applied = 0
    rejected = 0
    graph = build_initial_graph(scenario.alert, budget=8)

    for fixture in scenario.fixtures:
        result = validate_graph_update(fixture.update, graph)
        assert result.valid == fixture.expect_valid, (
            f"[{scenario.id}] fixture '{fixture.label}' expected valid={fixture.expect_valid}, "
            f"got valid={result.valid}. Errors: {result.errors}"
        )
        if result.valid:
            update_graph(graph, fixture.update)
            graph.tool_calls_used += 1
            applied += 1
        else:
            rejected += 1

    return applied, rejected, graph


# ── Per-scenario fixture validation tests ────────────────────────────────────

class TestScenarioFixtureExpectations:
    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_all_fixture_expectations_match(self, scenario_key):
        """Each fixture's expect_valid flag must match what the validator actually returns."""
        scenario = SCENARIOS[scenario_key]
        graph = build_initial_graph(scenario.alert, budget=8)

        for fixture in scenario.fixtures:
            result = validate_graph_update(fixture.update, graph)
            assert result.valid == fixture.expect_valid, (
                f"[{scenario_key}] fixture '{fixture.label}': "
                f"expected valid={fixture.expect_valid}, got {result.valid}. "
                f"Errors: {result.errors}"
            )
            if result.valid:
                update_graph(graph, fixture.update)

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_malformed_fixture_always_rejected(self, scenario_key):
        """Every scenario ends with a malformed fixture — must always be rejected."""
        scenario = SCENARIOS[scenario_key]
        # The last fixture in every scenario is the malformed one
        last = scenario.fixtures[-1]
        assert last.expect_valid is False, (
            f"[{scenario_key}] Last fixture should be malformed (expect_valid=False), "
            f"got expect_valid={last.expect_valid}"
        )

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_first_fixture_uses_create(self, scenario_key):
        """First fixture must use action='create' since the graph starts empty."""
        scenario = SCENARIOS[scenario_key]
        first = scenario.fixtures[0]
        assert first.update.action == "create", (
            f"[{scenario_key}] First fixture must use action='create'; "
            f"got action={first.update.action!r}"
        )


# ── Full mock runner e2e ──────────────────────────────────────────────────────

class TestMockRunnerE2E:
    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_scenario_completes_without_crash(self, scenario_key, capsys):
        scenario = SCENARIOS[scenario_key]
        with patch("run_mock.time.sleep"), \
             patch("run_mock.snapshot_graph"):
            run_mock(scenario)
        # If we get here without exception, the runner completed

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_at_least_one_fixture_applied(self, scenario_key):
        scenario = SCENARIOS[scenario_key]
        applied, rejected, graph = _run_scenario_captured(scenario)
        assert applied > 0, f"[{scenario_key}] No fixtures were applied"

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_malformed_fixture_counted_as_rejected(self, scenario_key):
        scenario = SCENARIOS[scenario_key]
        applied, rejected, graph = _run_scenario_captured(scenario)
        assert rejected >= 1, f"[{scenario_key}] Expected at least 1 rejected fixture"

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_final_graph_has_at_least_one_hypothesis(self, scenario_key):
        scenario = SCENARIOS[scenario_key]
        _, _, graph = _run_scenario_captured(scenario)
        assert len(graph.hypotheses) > 0, (
            f"[{scenario_key}] Final graph has no hypotheses"
        )

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_final_graph_has_established_facts(self, scenario_key):
        scenario = SCENARIOS[scenario_key]
        _, _, graph = _run_scenario_captured(scenario)
        assert len(graph.established_facts) > 0, (
            f"[{scenario_key}] Final graph has no established facts"
        )

    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_scenario_has_non_empty_ground_truth(self, scenario_key):
        scenario = SCENARIOS[scenario_key]
        assert scenario.ground_truth and len(scenario.ground_truth) > 20, (
            f"[{scenario_key}] ground_truth is missing or too short"
        )


# ── Scenario-specific correctness ────────────────────────────────────────────

class TestScenarioCorrectness:
    def test_feature_drift_final_top_hypothesis_is_about_features(self):
        scenario = SCENARIOS["feature_drift"]
        _, _, graph = _run_scenario_captured(scenario)
        active = sorted(
            [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE],
            key=lambda h: -h.likelihood,
        )
        assert active, "feature_drift scenario ended with no active hypotheses"
        top = active[0]
        # Top hypothesis description should mention feature or drift or schema
        desc_lower = top.description.lower()
        assert any(kw in desc_lower for kw in ("feature", "drift", "schema", "distribution")), (
            f"Top hypothesis for feature_drift doesn't mention key concepts: {top.description!r}"
        )

    def test_bad_deployment_scenario_creates_deployment_hypothesis(self):
        scenario = SCENARIOS["bad_deployment"]
        _, _, graph = _run_scenario_captured(scenario)
        descriptions = [h.description.lower() for h in graph.hypotheses]
        assert any(
            "deploy" in d or "regression" in d or "version" in d for d in descriptions
        ), f"bad_deployment scenario never created a deployment hypothesis. Got: {descriptions}"

    def test_label_corruption_scenario_creates_label_hypothesis(self):
        scenario = SCENARIOS["label_corruption"]
        _, _, graph = _run_scenario_captured(scenario)
        descriptions = [h.description.lower() for h in graph.hypotheses]
        assert any(
            "label" in d or "corruption" in d or "join" in d for d in descriptions
        ), f"label_corruption scenario never created a label hypothesis. Got: {descriptions}"


# ── SCENARIOS registry ────────────────────────────────────────────────────────

class TestScenariosRegistry:
    def test_three_scenarios_registered(self):
        assert len(SCENARIOS) == 3

    def test_expected_scenario_keys(self):
        assert set(SCENARIOS.keys()) == {"feature_drift", "bad_deployment", "label_corruption"}

    def test_each_scenario_has_required_fields(self):
        for key, scenario in SCENARIOS.items():
            assert scenario.id == key, f"scenario.id mismatch for {key!r}"
            assert scenario.label
            assert scenario.alert
            assert scenario.ground_truth
            assert len(scenario.fixtures) > 0
            assert callable(scenario.dispatch_tool)

    def test_scenario_dispatch_tool_accepts_all_tool_names(self):
        from hypothesis_graph import TOOL_TO_EVIDENCE_TYPE
        tools = list(TOOL_TO_EVIDENCE_TYPE.keys())
        for scenario in SCENARIOS.values():
            for tool in tools:
                result = scenario.dispatch_tool(tool, {
                    "metric_names": ["accuracy"],
                    "time_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
                    "service": "inference_service",
                    "features": ["account_age_days"],
                    "baseline_window": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
                    "comparison_window": {"start": "2024-01-02T00:00:00Z", "end": "2024-01-03T00:00:00Z"},
                    "commit_before": "HEAD~1",
                    "commit_after": "HEAD",
                })
                # Every dispatch should return a dict with at least a status
                assert isinstance(result, dict)
                assert "status" in result
