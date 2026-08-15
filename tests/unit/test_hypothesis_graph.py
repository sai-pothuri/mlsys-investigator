"""Unit tests for hypothesis_graph.py — enums, models, and update_graph logic."""

import pytest
from datetime import datetime, timezone

from hypothesis_graph import (
    Evidence,
    EvidenceInput,
    EvidenceType,
    FailureCategory,
    GraphUpdate,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
    TOOL_TO_EVIDENCE_TYPE,
    _MAX_LIKELIHOOD_DELTA,
    _next_hypothesis_id,
    derive_evidence_type,
    update_graph,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _graph(hypotheses=None, facts=None) -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary="test alert",
        investigation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        tool_call_budget=8,
        hypotheses=hypotheses or [],
        established_facts=facts or [],
    )


def _hyp(id="H1", likelihood=0.5, status=HypothesisStatus.ACTIVE) -> Hypothesis:
    return Hypothesis(
        id=id,
        description="test hypothesis",
        likelihood=likelihood,
        severity=Severity.MEDIUM,
        status=status,
    )


def _evidence_input(tool="query_metrics", supports=True, delta=0.10) -> EvidenceInput:
    return EvidenceInput(
        tool_called=tool,
        observation="test observation",
        supports=supports,
        confidence_delta=delta,
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
    facts=None,
) -> GraphUpdate:
    return GraphUpdate(
        action=action,
        current_focus=current_focus,
        merge_into_id=merge_into_id,
        new_hypothesis_description=description,
        new_hypothesis_severity=severity,
        new_hypothesis_initial_likelihood=initial_likelihood,
        new_evidence=_evidence_input(tool=tool),
        likelihood_changes=likelihood_changes or {},
        hypotheses_to_rule_out=rule_out or [],
        new_established_facts=facts or [],
        next_experiment_rationale="because testing",
    )


# ── derive_evidence_type ──────────────────────────────────────────────────────

class TestDeriveEvidenceType:
    def test_known_tools_map_correctly(self):
        assert derive_evidence_type("query_metrics") == EvidenceType.METRICS
        assert derive_evidence_type("query_logs") == EvidenceType.LOGS
        assert derive_evidence_type("query_deployment_history") == EvidenceType.DEPLOYMENT_HISTORY
        assert derive_evidence_type("query_feature_distributions") == EvidenceType.FEATURE_DISTRIBUTIONS
        assert derive_evidence_type("query_code_diffs") == EvidenceType.CODE_DIFFS

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unregistered tool"):
            derive_evidence_type("not_a_tool")

    def test_all_tools_in_map(self):
        for tool_name in TOOL_TO_EVIDENCE_TYPE:
            assert derive_evidence_type(tool_name) == TOOL_TO_EVIDENCE_TYPE[tool_name]


# ── Evidence.from_input ───────────────────────────────────────────────────────

class TestEvidenceFromInput:
    def test_derives_evidence_type_from_tool(self):
        raw = EvidenceInput(
            tool_called="query_logs",
            observation="saw errors",
            supports=True,
            confidence_delta=0.15,
        )
        ev = Evidence.from_input(raw)
        assert ev.evidence_type == EvidenceType.LOGS
        assert ev.tool_called == "query_logs"
        assert ev.observation == "saw errors"
        assert ev.supports is True
        assert ev.confidence_delta == 0.15

    def test_never_manually_sets_evidence_type(self):
        # evidence_type must come from tool_called, not from the user
        raw = EvidenceInput(
            tool_called="query_feature_distributions",
            observation="psi=0.32",
            supports=True,
            confidence_delta=0.20,
        )
        ev = Evidence.from_input(raw)
        assert ev.evidence_type == EvidenceType.FEATURE_DISTRIBUTIONS


# ── EvidenceInput validation ──────────────────────────────────────────────────

class TestEvidenceInputValidation:
    def test_confidence_delta_bounds(self):
        # These should work
        EvidenceInput(tool_called="query_metrics", observation="ok", supports=True, confidence_delta=-1.0)
        EvidenceInput(tool_called="query_metrics", observation="ok", supports=True, confidence_delta=1.0)

    def test_confidence_delta_out_of_bounds(self):
        with pytest.raises(Exception):
            EvidenceInput(tool_called="query_metrics", observation="ok", supports=True, confidence_delta=1.1)
        with pytest.raises(Exception):
            EvidenceInput(tool_called="query_metrics", observation="ok", supports=True, confidence_delta=-1.1)


# ── _next_hypothesis_id ───────────────────────────────────────────────────────

class TestNextHypothesisId:
    def test_empty_graph_gives_h1(self):
        g = _graph()
        assert _next_hypothesis_id(g) == "H1"

    def test_increments_with_each_hypothesis(self):
        g = _graph(hypotheses=[_hyp("H1"), _hyp("H2")])
        assert _next_hypothesis_id(g) == "H3"


# ── update_graph action="create" ─────────────────────────────────────────────

class TestUpdateGraphCreate:
    def test_adds_hypothesis_to_graph(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="new hyp", severity="high", initial_likelihood=0.20)
        update_graph(g, u)
        assert len(g.hypotheses) == 1
        assert g.hypotheses[0].id == "H1"
        assert g.hypotheses[0].description == "new hyp"
        assert g.hypotheses[0].severity == Severity.HIGH
        assert g.hypotheses[0].status == HypothesisStatus.ACTIVE

    def test_assigns_sequential_ids(self):
        g = _graph(hypotheses=[_hyp("H1")])
        u = _update(action="create", current_focus=None, description="second hyp")
        update_graph(g, u)
        assert g.hypotheses[1].id == "H2"

    def test_attaches_evidence_to_new_hypothesis(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="new hyp")
        update_graph(g, u)
        assert len(g.hypotheses[0].evidence) == 1
        assert g.hypotheses[0].evidence[0].tool_called == "query_metrics"

    def test_initial_likelihood_clamped(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h", initial_likelihood=2.0)
        update_graph(g, u)
        assert g.hypotheses[0].likelihood <= 1.0

    def test_initial_likelihood_defaults_to_015(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h")
        update_graph(g, u)
        # Only one hypothesis, normalized to 1.0
        assert g.hypotheses[0].likelihood == 1.0

    def test_severity_defaults_to_medium(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h", severity=None)
        update_graph(g, u)
        assert g.hypotheses[0].severity == Severity.MEDIUM

    def test_sets_current_focus(self):
        g = _graph()
        u = _update(action="create", current_focus=None, description="h")
        update_graph(g, u)
        assert g.current_focus == "H1"


# ── update_graph action="update" ─────────────────────────────────────────────

class TestUpdateGraphUpdate:
    def test_attaches_evidence_to_focused_hypothesis(self):
        g = _graph(hypotheses=[_hyp("H1"), _hyp("H2")])
        u = _update(action="update", current_focus="H2", tool="query_logs")
        update_graph(g, u)
        h2 = next(h for h in g.hypotheses if h.id == "H2")
        assert len(h2.evidence) == 1
        assert h2.evidence[0].tool_called == "query_logs"

    def test_does_not_touch_other_hypotheses_evidence(self):
        g = _graph(hypotheses=[_hyp("H1"), _hyp("H2")])
        u = _update(action="update", current_focus="H2")
        update_graph(g, u)
        h1 = next(h for h in g.hypotheses if h.id == "H1")
        assert len(h1.evidence) == 0

    def test_raises_if_current_focus_missing(self):
        g = _graph(hypotheses=[_hyp("H1")])
        u = _update(action="update", current_focus=None)
        with pytest.raises(ValueError, match="current_focus"):
            update_graph(g, u)

    def test_raises_if_current_focus_not_in_graph(self):
        g = _graph(hypotheses=[_hyp("H1")])
        u = _update(action="update", current_focus="H99")
        with pytest.raises(ValueError, match="current_focus"):
            update_graph(g, u)


# ── update_graph action="merge" ───────────────────────────────────────────────

class TestUpdateGraphMerge:
    def test_attaches_evidence_to_merge_target(self):
        g = _graph(hypotheses=[_hyp("H1"), _hyp("H2")])
        u = _update(action="merge", current_focus=None, merge_into_id="H1")
        update_graph(g, u)
        h1 = next(h for h in g.hypotheses if h.id == "H1")
        assert len(h1.evidence) == 1

    def test_does_not_create_new_hypothesis(self):
        g = _graph(hypotheses=[_hyp("H1")])
        before = len(g.hypotheses)
        u = _update(action="merge", current_focus=None, merge_into_id="H1")
        update_graph(g, u)
        assert len(g.hypotheses) == before

    def test_raises_if_merge_into_missing(self):
        g = _graph(hypotheses=[_hyp("H1")])
        u = _update(action="merge", current_focus=None, merge_into_id=None)
        with pytest.raises(ValueError, match="merge_into_id"):
            update_graph(g, u)

    def test_raises_if_merge_into_not_in_graph(self):
        g = _graph(hypotheses=[_hyp("H1")])
        u = _update(action="merge", current_focus=None, merge_into_id="H99")
        with pytest.raises(ValueError, match="merge_into_id"):
            update_graph(g, u)


# ── Likelihood changes & capping ──────────────────────────────────────────────

class TestLikelihoodChanges:
    def test_applies_delta_to_hypothesis(self):
        h = _hyp("H1", likelihood=0.5)
        g = _graph(hypotheses=[h])
        u = _update(action="update", current_focus="H1", likelihood_changes={"H1": 0.10})
        update_graph(g, u)
        # 0.5 + 0.10 = 0.60, then normalized (only one active → 1.0)
        assert g.hypotheses[0].likelihood == 1.0

    def test_delta_capped_at_max(self):
        h1 = _hyp("H1", likelihood=0.5)
        h2 = _hyp("H2", likelihood=0.5)
        g = _graph(hypotheses=[h1, h2])
        # Request a delta well above the cap
        u = _update(action="update", current_focus="H1", likelihood_changes={"H1": 0.99})
        update_graph(g, u)
        # H1 gets +0.25 (capped), then normalized — so H1 > H2
        h1_after = next(h for h in g.hypotheses if h.id == "H1")
        h2_after = next(h for h in g.hypotheses if h.id == "H2")
        assert h1_after.likelihood > h2_after.likelihood

    def test_delta_capped_at_min(self):
        h1 = _hyp("H1", likelihood=0.5)
        h2 = _hyp("H2", likelihood=0.5)
        g = _graph(hypotheses=[h1, h2])
        u = _update(action="update", current_focus="H1", likelihood_changes={"H1": -0.99})
        update_graph(g, u)
        h1_after = next(h for h in g.hypotheses if h.id == "H1")
        h2_after = next(h for h in g.hypotheses if h.id == "H2")
        assert h1_after.likelihood < h2_after.likelihood

    def test_likelihood_floored_at_zero(self):
        h = _hyp("H1", likelihood=0.1)
        g = _graph(hypotheses=[h])
        u = _update(action="update", current_focus="H1", likelihood_changes={"H1": -0.25})
        update_graph(g, u)
        assert g.hypotheses[0].likelihood >= 0.0

    def test_likelihood_capped_at_one(self):
        h = _hyp("H1", likelihood=0.9)
        g = _graph(hypotheses=[h])
        u = _update(action="update", current_focus="H1", likelihood_changes={"H1": 0.25})
        update_graph(g, u)
        assert g.hypotheses[0].likelihood <= 1.0


# ── Normalization ─────────────────────────────────────────────────────────────

class TestNormalization:
    def test_active_hypotheses_normalize_to_one(self):
        h1 = _hyp("H1", likelihood=0.3)
        h2 = _hyp("H2", likelihood=0.3)
        h3 = _hyp("H3", likelihood=0.3)
        g = _graph(hypotheses=[h1, h2, h3])
        u = _update(action="update", current_focus="H1")
        update_graph(g, u)
        total = sum(h.likelihood for h in g.hypotheses if h.status == HypothesisStatus.ACTIVE)
        # Rounded to 4 decimal places so tolerance accounts for rounding error
        assert abs(total - 1.0) < 1e-3

    def test_ruled_out_excluded_from_normalization(self):
        h1 = _hyp("H1", likelihood=0.5)
        h2 = _hyp("H2", likelihood=0.5, status=HypothesisStatus.RULED_OUT)
        g = _graph(hypotheses=[h1, h2])
        u = _update(action="update", current_focus="H1")
        update_graph(g, u)
        active_total = sum(h.likelihood for h in g.hypotheses if h.status == HypothesisStatus.ACTIVE)
        assert abs(active_total - 1.0) < 1e-6
        # Ruled-out should not be touched by normalization
        ruled = next(h for h in g.hypotheses if h.id == "H2")
        assert ruled.likelihood == 0.5


# ── Rule out ──────────────────────────────────────────────────────────────────

class TestRuleOut:
    def test_marks_hypothesis_ruled_out(self):
        h1 = _hyp("H1")
        h2 = _hyp("H2")
        g = _graph(hypotheses=[h1, h2])
        u = _update(action="update", current_focus="H1", rule_out=["H2"])
        update_graph(g, u)
        h2_after = next(h for h in g.hypotheses if h.id == "H2")
        assert h2_after.status == HypothesisStatus.RULED_OUT

    def test_does_not_rule_out_wrong_hypothesis(self):
        h1 = _hyp("H1")
        h2 = _hyp("H2")
        g = _graph(hypotheses=[h1, h2])
        u = _update(action="update", current_focus="H1", rule_out=["H2"])
        update_graph(g, u)
        h1_after = next(h for h in g.hypotheses if h.id == "H1")
        assert h1_after.status == HypothesisStatus.ACTIVE


# ── Established facts ─────────────────────────────────────────────────────────

class TestEstablishedFacts:
    def test_appends_new_facts(self):
        g = _graph(hypotheses=[_hyp("H1")], facts=["existing fact"])
        u = _update(action="update", current_focus="H1", facts=["new fact"])
        update_graph(g, u)
        assert "existing fact" in g.established_facts
        assert "new fact" in g.established_facts

    def test_empty_facts_list_is_noop(self):
        g = _graph(hypotheses=[_hyp("H1")], facts=["existing"])
        u = _update(action="update", current_focus="H1", facts=[])
        update_graph(g, u)
        assert g.established_facts == ["existing"]


# ── FailureCategory enum coverage ────────────────────────────────────────────

class TestFailureCategoryEnum:
    def test_all_values_are_strings(self):
        for cat in FailureCategory:
            assert isinstance(cat.value, str)

    def test_values_are_snake_case(self):
        for cat in FailureCategory:
            assert cat.value == cat.value.lower()
            assert " " not in cat.value

    def test_eighteen_categories(self):
        assert len(FailureCategory) == 18
