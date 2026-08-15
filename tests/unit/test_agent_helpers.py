"""Unit tests for agent.py helpers — no Anthropic API calls."""

import io
from datetime import datetime, timezone
from unittest.mock import patch

from agent import DiagnosisResult, _graph_context, build_initial_graph, print_top_hypothesis
from hypothesis_graph import (
    Evidence,
    EvidenceType,
    FailureCategory,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc)


def _graph(hypotheses=None) -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary="accuracy dropped",
        investigation_start=_NOW,
        tool_call_budget=8,
        hypotheses=hypotheses or [],
    )


def _hyp(
    id="H1",
    likelihood=0.5,
    status=HypothesisStatus.ACTIVE,
    evidence=None,
    category=None,
) -> Hypothesis:
    return Hypothesis(
        id=id,
        description=f"hypothesis {id}",
        likelihood=likelihood,
        severity=Severity.HIGH,
        status=status,
        evidence=evidence or [],
        root_cause_category=category,
    )


def _evidence(supports=True) -> Evidence:
    return Evidence(
        tool_called="query_metrics",
        evidence_type=EvidenceType.METRICS,
        observation="accuracy dropped 15pp",
        supports=supports,
        confidence_delta=0.20,
    )


# ── build_initial_graph ───────────────────────────────────────────────────────

class TestBuildInitialGraph:
    def test_creates_graph_with_alert(self):
        g = build_initial_graph("my alert")
        assert g.alert_summary == "my alert"

    def test_default_budget_is_eight(self):
        g = build_initial_graph("alert")
        assert g.tool_call_budget == 8

    def test_custom_budget(self):
        g = build_initial_graph("alert", budget=5)
        assert g.tool_call_budget == 5

    def test_starts_with_zero_tool_calls(self):
        g = build_initial_graph("alert")
        assert g.tool_calls_used == 0

    def test_starts_with_empty_hypotheses(self):
        g = build_initial_graph("alert")
        assert g.hypotheses == []

    def test_sets_investigation_start(self):
        g = build_initial_graph("alert", investigation_start=_NOW)
        assert g.investigation_start == _NOW

    def test_defaults_investigation_start_to_now(self):
        before = datetime.now(timezone.utc)
        g = build_initial_graph("alert")
        after = datetime.now(timezone.utc)
        assert before <= g.investigation_start <= after


# ── _graph_context ────────────────────────────────────────────────────────────

class TestGraphContext:
    def test_empty_graph_has_no_hypotheses_message(self):
        g = _graph()
        ctx = _graph_context(g)
        assert "No hypotheses yet" in ctx

    def test_empty_graph_shows_alert(self):
        g = _graph()
        ctx = _graph_context(g)
        assert "accuracy dropped" in ctx

    def test_empty_graph_shows_budget(self):
        g = _graph()
        ctx = _graph_context(g)
        assert "0 / 8" in ctx

    def test_shows_active_hypotheses(self):
        g = _graph(hypotheses=[_hyp("H1")])
        ctx = _graph_context(g)
        assert "H1" in ctx
        assert "hypothesis H1" in ctx

    def test_active_hypotheses_sorted_by_likelihood_desc(self):
        h_low = _hyp("H1", likelihood=0.2)
        h_high = _hyp("H2", likelihood=0.8)
        g = _graph(hypotheses=[h_low, h_high])
        ctx = _graph_context(g)
        # H2 (higher likelihood) should appear before H1
        assert ctx.index("H2") < ctx.index("H1")

    def test_shows_ruled_out_section(self):
        g = _graph(hypotheses=[_hyp("H1", status=HypothesisStatus.RULED_OUT)])
        ctx = _graph_context(g)
        assert "Ruled Out" in ctx
        assert "H1" in ctx

    def test_shows_evidence(self):
        ev = _evidence(supports=True)
        h = _hyp("H1", evidence=[ev])
        g = _graph(hypotheses=[h])
        ctx = _graph_context(g)
        assert "accuracy dropped 15pp" in ctx
        assert "SUPPORTS" in ctx

    def test_shows_contradicting_evidence(self):
        ev = _evidence(supports=False)
        h = _hyp("H1", evidence=[ev])
        g = _graph(hypotheses=[h])
        ctx = _graph_context(g)
        assert "CONTRADICTS" in ctx

    def test_shows_established_facts(self):
        g = _graph(hypotheses=[_hyp("H1")])
        g.established_facts = ["latency is flat", "error_rate unchanged"]
        ctx = _graph_context(g)
        assert "Established Facts" in ctx
        assert "latency is flat" in ctx
        assert "error_rate unchanged" in ctx

    def test_shows_open_questions(self):
        g = _graph(hypotheses=[_hyp("H1")])
        g.open_questions = ["Is it a deployment?"]
        ctx = _graph_context(g)
        assert "Open Questions" in ctx
        assert "Is it a deployment?" in ctx

    def test_shows_investigation_start(self):
        g = _graph()
        ctx = _graph_context(g)
        assert "2024-01-07T12:00:00Z" in ctx

    def test_shows_tool_calls_used(self):
        g = _graph()
        g.tool_calls_used = 3
        ctx = _graph_context(g)
        assert "3 / 8" in ctx


# ── print_top_hypothesis ──────────────────────────────────────────────────────

class TestPrintTopHypothesis:
    def test_no_active_hypotheses_prints_fallback(self, capsys):
        g = _graph()
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "no active hypotheses" in out

    def test_only_ruled_out_hypotheses_prints_fallback(self, capsys):
        g = _graph(hypotheses=[_hyp("H1", status=HypothesisStatus.RULED_OUT)])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "no active hypotheses" in out

    def test_picks_highest_likelihood_active(self, capsys):
        h_low = _hyp("H1", likelihood=0.2)
        h_high = _hyp("H2", likelihood=0.8)
        g = _graph(hypotheses=[h_low, h_high])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "hypothesis H2" in out

    def test_skips_ruled_out_when_picking_top(self, capsys):
        h_active = _hyp("H1", likelihood=0.3)
        h_ruled = _hyp("H2", likelihood=0.9, status=HypothesisStatus.RULED_OUT)
        g = _graph(hypotheses=[h_active, h_ruled])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "hypothesis H1" in out

    def test_shows_likelihood_percentage(self, capsys):
        h = _hyp("H1", likelihood=0.75)
        g = _graph(hypotheses=[h])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "75%" in out

    def test_shows_evidence_count(self, capsys):
        ev = _evidence(supports=True)
        h = _hyp("H1", evidence=[ev, ev])
        g = _graph(hypotheses=[h])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "2" in out

    def test_shows_category_when_set(self, capsys):
        h = _hyp("H1", category=FailureCategory.FEATURE_DRIFT)
        g = _graph(hypotheses=[h])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "FEATURE DRIFT" in out

    def test_shows_unknown_when_category_none(self, capsys):
        h = _hyp("H1", category=None)
        g = _graph(hypotheses=[h])
        print_top_hypothesis(g)
        out = capsys.readouterr().out
        assert "UNKNOWN" in out


# ── DiagnosisResult ───────────────────────────────────────────────────────────

class TestDiagnosisResult:
    def test_str_contains_all_fields(self):
        d = DiagnosisResult(
            root_cause="feature_drift",
            diagnosis="Feature drift detected",
            confidence=0.87,
            recommended_action="Roll back schema",
        )
        s = str(d)
        assert "feature_drift" in s
        assert "Feature drift detected" in s
        assert "87%" in s
        assert "Roll back schema" in s
