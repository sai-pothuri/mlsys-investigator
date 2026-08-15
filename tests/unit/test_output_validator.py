"""Unit tests for output_validator.py — ValidationResult and validate_graph_update."""

import pytest

from hypothesis_graph import (
    EvidenceInput,
    GraphUpdate,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
)
from output_validator import ValidationResult, validate_graph_update
from datetime import datetime, timezone


# ── Helpers ───────────────────────────────────────────────────────────────────

def _graph(*hypothesis_ids) -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary="test alert",
        investigation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        tool_call_budget=8,
        hypotheses=[
            Hypothesis(
                id=hid,
                description="test",
                likelihood=0.5,
                severity=Severity.MEDIUM,
                status=HypothesisStatus.ACTIVE,
            )
            for hid in hypothesis_ids
        ],
    )


def _evidence(tool="query_metrics") -> EvidenceInput:
    return EvidenceInput(
        tool_called=tool,
        observation="test observation",
        supports=True,
        confidence_delta=0.10,
    )


def _update(
    action="update",
    current_focus="H1",
    merge_into_id=None,
    description=None,
    severity=None,
    initial_likelihood=None,
    tool="query_metrics",
    likelihood_changes=None,
    rule_out=None,
) -> GraphUpdate:
    return GraphUpdate(
        action=action,
        current_focus=current_focus,
        merge_into_id=merge_into_id,
        new_hypothesis_description=description,
        new_hypothesis_severity=severity,
        new_hypothesis_initial_likelihood=initial_likelihood,
        new_evidence=_evidence(tool=tool),
        likelihood_changes=likelihood_changes or {},
        hypotheses_to_rule_out=rule_out or [],
        next_experiment_rationale="testing",
    )


# ── ValidationResult ──────────────────────────────────────────────────────────

class TestValidationResult:
    def test_str_pass(self):
        r = ValidationResult(valid=True)
        assert str(r) == "PASS"

    def test_str_fail_single_error(self):
        r = ValidationResult(valid=False, errors=["bad id"])
        s = str(r)
        assert "FAIL" in s
        assert "bad id" in s

    def test_str_fail_multiple_errors(self):
        r = ValidationResult(valid=False, errors=["err1", "err2"])
        s = str(r)
        assert "err1" in s
        assert "err2" in s

    def test_valid_true_means_no_errors(self):
        r = ValidationResult(valid=True, errors=[])
        assert r.valid is True

    def test_default_errors_empty_list(self):
        r = ValidationResult(valid=True)
        assert r.errors == []


# ── action="update" ───────────────────────────────────────────────────────────

class TestValidateUpdateAction:
    def test_valid_update(self):
        g = _graph("H1", "H2")
        u = _update(action="update", current_focus="H1")
        result = validate_graph_update(u, g)
        assert result.valid

    def test_missing_current_focus(self):
        g = _graph("H1")
        u = _update(action="update", current_focus=None)
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("current_focus" in e for e in result.errors)

    def test_current_focus_not_in_graph(self):
        g = _graph("H1")
        u = _update(action="update", current_focus="H99")
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("H99" in e for e in result.errors)

    def test_current_focus_pointing_to_existing_id(self):
        g = _graph("H1", "H2", "H3")
        u = _update(action="update", current_focus="H3")
        result = validate_graph_update(u, g)
        assert result.valid


# ── action="merge" ────────────────────────────────────────────────────────────

class TestValidateMergeAction:
    def test_valid_merge(self):
        g = _graph("H1", "H2")
        u = _update(action="merge", current_focus=None, merge_into_id="H1")
        result = validate_graph_update(u, g)
        assert result.valid

    def test_missing_merge_into_id(self):
        g = _graph("H1")
        u = _update(action="merge", current_focus=None, merge_into_id=None)
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("merge_into_id" in e for e in result.errors)

    def test_merge_into_id_not_in_graph(self):
        g = _graph("H1")
        u = _update(action="merge", current_focus=None, merge_into_id="H99")
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("H99" in e for e in result.errors)


# ── action="create" ───────────────────────────────────────────────────────────

class TestValidateCreateAction:
    def test_valid_create(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="new hypothesis")
        result = validate_graph_update(u, g)
        assert result.valid

    def test_empty_description_fails(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="")
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("description" in e.lower() for e in result.errors)

    def test_none_description_fails(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description=None)
        result = validate_graph_update(u, g)
        assert not result.valid

    def test_invalid_severity_fails(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h", severity="catastrophic")
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("severity" in e.lower() for e in result.errors)

    def test_valid_severities_pass(self):
        for sev in ("critical", "high", "medium", "low"):
            g = _graph()
            u = _update(action="create", current_focus=None, description="h", severity=sev)
            result = validate_graph_update(u, g)
            assert result.valid, f"severity={sev!r} should be valid"

    def test_likelihood_above_one_fails(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h", initial_likelihood=1.5)
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("likelihood" in e.lower() for e in result.errors)

    def test_likelihood_below_zero_fails(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h", initial_likelihood=-0.1)
        result = validate_graph_update(u, g)
        assert not result.valid

    def test_likelihood_at_bounds_passes(self):
        for val in (0.0, 0.5, 1.0):
            g = _graph()
            u = _update(action="create", current_focus=None, description="h", initial_likelihood=val)
            result = validate_graph_update(u, g)
            assert result.valid, f"likelihood={val} should be valid"


# ── Common checks (all actions) ───────────────────────────────────────────────

class TestCommonValidation:
    def test_invalid_tool_called_fails(self):
        g = _graph("H1")
        u = _update(action="update", current_focus="H1", tool="not_a_tool")
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("not_a_tool" in e for e in result.errors)

    def test_hallucinated_id_in_likelihood_changes_fails(self):
        g = _graph("H1")
        u = _update(action="update", current_focus="H1", likelihood_changes={"H99": 0.10})
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("H99" in e for e in result.errors)

    def test_valid_id_in_likelihood_changes_passes(self):
        g = _graph("H1", "H2")
        u = _update(action="update", current_focus="H1", likelihood_changes={"H2": -0.10})
        result = validate_graph_update(u, g)
        assert result.valid

    def test_hallucinated_id_in_rule_out_fails(self):
        g = _graph("H1")
        u = _update(action="update", current_focus="H1", rule_out=["H88"])
        result = validate_graph_update(u, g)
        assert not result.valid
        assert any("H88" in e for e in result.errors)

    def test_valid_id_in_rule_out_passes(self):
        g = _graph("H1", "H2")
        u = _update(action="update", current_focus="H1", rule_out=["H2"])
        result = validate_graph_update(u, g)
        assert result.valid

    def test_multiple_errors_collected(self):
        g = _graph("H1")
        # Two bad IDs — both should be reported
        u = _update(
            action="update",
            current_focus="H1",
            likelihood_changes={"H99": 0.10},
            rule_out=["H88"],
        )
        result = validate_graph_update(u, g)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_all_registered_tools_pass_validation(self):
        tools = [
            "query_metrics",
            "query_logs",
            "query_deployment_history",
            "query_feature_distributions",
            "query_code_diffs",
        ]
        for tool in tools:
            g = _graph("H1")
            u = _update(action="update", current_focus="H1", tool=tool)
            result = validate_graph_update(u, g)
            assert result.valid, f"tool={tool!r} should be valid"

    def test_validate_never_raises(self):
        # validate_graph_update must never raise; it always returns ValidationResult
        g = _graph()
        u = _update(action="update", current_focus="H99", tool="not_a_tool", likelihood_changes={"H88": 1.0})
        result = validate_graph_update(u, g)
        assert isinstance(result, ValidationResult)
        assert not result.valid
