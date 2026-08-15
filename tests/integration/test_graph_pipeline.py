"""Integration tests: validate_graph_update → update_graph pipeline.

Tests multi-turn sequences where actions depend on prior state.
"""

import pytest
from datetime import datetime, timezone

from hypothesis_graph import (
    EvidenceInput,
    GraphUpdate,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
    update_graph,
)
from output_validator import validate_graph_update


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_graph() -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary="test alert",
        investigation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        tool_call_budget=8,
        hypotheses=[],
    )


def _update(action, tool="query_metrics", **kwargs) -> GraphUpdate:
    return GraphUpdate(
        action=action,
        new_evidence=EvidenceInput(
            tool_called=tool,
            observation="test",
            supports=True,
            confidence_delta=0.10,
        ),
        likelihood_changes=kwargs.pop("likelihood_changes", {}),
        hypotheses_to_rule_out=kwargs.pop("rule_out", []),
        new_established_facts=kwargs.pop("facts", []),
        next_experiment_rationale="testing",
        **kwargs,
    )


# ── Valid update is applied; invalid update leaves graph unchanged ─────────────

class TestValidateAndApply:
    def test_valid_update_applied_to_graph(self):
        g = _empty_graph()
        u = _update("create", new_hypothesis_description="feature drift", new_hypothesis_severity="high")
        result = validate_graph_update(u, g)
        assert result.valid
        update_graph(g, u)
        assert len(g.hypotheses) == 1

    def test_invalid_update_leaves_graph_unchanged(self):
        g = _empty_graph()
        # action="update" with no existing hypotheses — invalid
        u = _update("update", current_focus="H99")
        result = validate_graph_update(u, g)
        assert not result.valid
        # Do NOT call update_graph since invalid; graph must stay empty
        assert len(g.hypotheses) == 0

    def test_rejected_malformed_id_leaves_graph_unchanged(self):
        g = _empty_graph()
        # First create a hypothesis legitimately
        create = _update("create", new_hypothesis_description="H1")
        validate_graph_update(create, g)
        update_graph(g, create)
        assert len(g.hypotheses) == 1
        before_likelihood = g.hypotheses[0].likelihood

        # Now send a malformed update referencing H99 (doesn't exist)
        bad = _update(
            "update",
            current_focus="H1",
            likelihood_changes={"H1": 0.10, "H99": 0.50},
        )
        result = validate_graph_update(bad, g)
        assert not result.valid
        # Graph unchanged — do not apply
        assert g.hypotheses[0].likelihood == before_likelihood


# ── Multi-turn sequence ───────────────────────────────────────────────────────

class TestMultiTurnSequence:
    def test_create_then_update(self):
        g = _empty_graph()

        # Turn 1: create H1
        t1 = _update("create", new_hypothesis_description="feature drift")
        assert validate_graph_update(t1, g).valid
        update_graph(g, t1)
        assert g.hypotheses[0].id == "H1"

        # Turn 2: update H1
        t2 = _update("update", current_focus="H1", likelihood_changes={"H1": 0.15})
        assert validate_graph_update(t2, g).valid
        before = g.hypotheses[0].likelihood
        update_graph(g, t2)
        # Likelihood should be 1.0 (only active hyp, normalized)
        assert g.hypotheses[0].likelihood == 1.0

    def test_create_two_hypotheses_then_rule_one_out(self):
        g = _empty_graph()

        t1 = _update("create", new_hypothesis_description="feature drift")
        assert validate_graph_update(t1, g).valid
        update_graph(g, t1)

        t2 = _update("create", new_hypothesis_description="bad deployment",
                     tool="query_deployment_history")
        assert validate_graph_update(t2, g).valid
        update_graph(g, t2)
        assert len(g.hypotheses) == 2

        # Rule out H2
        t3 = _update(
            "update",
            current_focus="H1",
            tool="query_logs",
            rule_out=["H2"],
            likelihood_changes={"H1": 0.20},
        )
        assert validate_graph_update(t3, g).valid
        update_graph(g, t3)
        h2 = next(h for h in g.hypotheses if h.id == "H2")
        assert h2.status == HypothesisStatus.RULED_OUT

    def test_normalization_after_ruling_out(self):
        g = _empty_graph()

        for desc in ("H1 desc", "H2 desc"):
            u = _update("create", new_hypothesis_description=desc)
            update_graph(g, u)

        # Rule out H2
        u = _update("update", current_focus="H1", rule_out=["H2"])
        validate_graph_update(u, g)
        update_graph(g, u)

        active = [h for h in g.hypotheses if h.status == HypothesisStatus.ACTIVE]
        total = sum(h.likelihood for h in active)
        assert abs(total - 1.0) < 1e-5

    def test_merge_attaches_evidence_not_creates_new(self):
        g = _empty_graph()

        t1 = _update("create", new_hypothesis_description="feature drift")
        update_graph(g, t1)
        assert len(g.hypotheses) == 1

        t2 = _update("merge", merge_into_id="H1")
        assert validate_graph_update(t2, g).valid
        update_graph(g, t2)
        # merge should NOT create a new hypothesis
        assert len(g.hypotheses) == 1
        assert len(g.hypotheses[0].evidence) == 2  # two pieces of evidence on H1

    def test_five_turn_sequence_consistent_state(self):
        g = _empty_graph()

        # Turn 1: create H1
        update_graph(g, _update("create", new_hypothesis_description="feature drift"))
        # Turn 2: create H2
        update_graph(g, _update("create", new_hypothesis_description="bad deployment",
                                tool="query_deployment_history"))
        # Turn 3: create H3
        update_graph(g, _update("create", new_hypothesis_description="label corruption",
                                tool="query_logs"))
        # Turn 4: update H1
        update_graph(g, _update("update", current_focus="H1",
                                likelihood_changes={"H1": 0.20, "H2": -0.10}))
        # Turn 5: rule out H2, update H1 again
        update_graph(g, _update("update", current_focus="H1", rule_out=["H2"],
                                likelihood_changes={"H1": 0.20}))

        active = [h for h in g.hypotheses if h.status == HypothesisStatus.ACTIVE]
        ruled = [h for h in g.hypotheses if h.status == HypothesisStatus.RULED_OUT]

        assert len(g.hypotheses) == 3
        assert len(active) == 2
        assert len(ruled) == 1

        total = sum(h.likelihood for h in active)
        assert abs(total - 1.0) < 1e-5


# ── Evidence accumulation across turns ────────────────────────────────────────

class TestEvidenceAccumulation:
    def test_evidence_accumulates_on_hypothesis(self):
        g = _empty_graph()
        update_graph(g, _update("create", new_hypothesis_description="h1"))

        for _ in range(3):
            update_graph(g, _update("update", current_focus="H1",
                                    tool="query_metrics"))

        assert len(g.hypotheses[0].evidence) == 4  # 1 from create + 3 from updates

    def test_established_facts_accumulate(self):
        g = _empty_graph()
        update_graph(g, _update("create", new_hypothesis_description="h1",
                                facts=["fact A"]))
        update_graph(g, _update("update", current_focus="H1", facts=["fact B"]))
        update_graph(g, _update("update", current_focus="H1", facts=["fact C"]))

        assert "fact A" in g.established_facts
        assert "fact B" in g.established_facts
        assert "fact C" in g.established_facts
