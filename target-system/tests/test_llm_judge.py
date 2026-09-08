"""
Unit tests for evaluation.llm_judge — no live Anthropic API calls.

Mocks the Anthropic client the same way tests/unit/test_tools.py mocks
external calls, per this repo's established test conventions (see
docs/QA_Plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluation.llm_judge import (
    JudgeRubricScore,
    _pairwise_agreement,
    _single_judgment,
    build_judge_user_message,
    judge_diagnosis,
)


# ---------------------------------------------------------------------------
# Fakes — duck-typed stand-ins for HypothesisGraph / DiagnosisResult so this
# module doesn't need to import the full agent.py (which instantiates a
# Langfuse client and the Anthropic SDK at import time).
# ---------------------------------------------------------------------------

@dataclass
class _FakeDiagnosis:
    root_cause: str = "feature_drift"
    diagnosis: str = "login_failure_rate PSI=0.53, well above threshold; accuracy and confidence both dropped."
    confidence: float = 0.8
    recommended_action: str = "Roll back the login_failure_rate feature transform to the previous version."
    alternative_categories: list = field(default_factory=lambda: ["upstream_schema_change"])


class _FakeGraph:
    def model_dump(self, mode: str = "python") -> dict:
        return {
            "alert_summary": "accuracy dropped",
            "hypotheses": [
                {
                    "id": "H1",
                    "description": "feature drift",
                    "likelihood": 0.8,
                    "status": "active",
                    "evidence": [
                        {
                            "tool_called": "query_feature_distributions",
                            "observation": "login_failure_rate PSI=0.53",
                            "supports": True,
                            "confidence_delta": 0.2,
                        }
                    ],
                }
            ],
        }


def _tool_use_response(payload: dict):
    block = SimpleNamespace(type="tool_use", name="submit_judgment", input=payload)
    return SimpleNamespace(content=[block])


_VALID_PAYLOAD = {
    "evidence_grounding": 5,
    "reasoning_coherence": 4,
    "actionability": 4,
    "calibration_reasonableness": 3,
    "overall": 4,
    "hallucination_detected": False,
    "rationale": "The diagnosis cites the PSI figure that is actually in the evidence.",
}


# ---------------------------------------------------------------------------
# _pairwise_agreement — pure function, no mocking
# ---------------------------------------------------------------------------

class TestPairwiseAgreement:
    def test_single_value_is_perfect_agreement(self):
        assert _pairwise_agreement([4]) == 1.0

    def test_identical_values_is_perfect_agreement(self):
        assert _pairwise_agreement([3, 3, 3]) == 1.0

    def test_all_within_tolerance(self):
        # every pair differs by <= 1
        assert _pairwise_agreement([4, 5, 4]) == 1.0

    def test_partial_agreement(self):
        # pairs: (5,5)->agree, (5,1)->disagree, (5,1)->disagree = 1/3
        assert _pairwise_agreement([5, 5, 1]) == pytest.approx(1 / 3)

    def test_no_agreement(self):
        assert _pairwise_agreement([1, 5]) == 0.0

    def test_custom_tolerance(self):
        assert _pairwise_agreement([1, 3], tolerance=2) == 1.0
        assert _pairwise_agreement([1, 3], tolerance=1) == 0.0


# ---------------------------------------------------------------------------
# build_judge_user_message
# ---------------------------------------------------------------------------

class TestBuildJudgeUserMessage:
    def test_includes_alert_graph_and_diagnosis(self):
        msg = build_judge_user_message("ALERT: accuracy dropped", _FakeGraph(), _FakeDiagnosis())
        assert "ALERT: accuracy dropped" in msg
        assert "login_failure_rate PSI=0.53" in msg  # from the graph's evidence
        assert "feature_drift" in msg                # from the diagnosis
        assert "submit_judgment" in msg

    def test_does_not_leak_ground_truth_field(self):
        # the judge must never be handed a ground_truth field to grade against
        msg = build_judge_user_message("alert", _FakeGraph(), _FakeDiagnosis())
        assert "ground_truth" not in msg


# ---------------------------------------------------------------------------
# _single_judgment
# ---------------------------------------------------------------------------

class TestSingleJudgment:
    def test_parses_tool_use_response(self):
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        score = _single_judgment(client, "alert", _FakeGraph(), _FakeDiagnosis(), "claude-sonnet-4-6")

        assert isinstance(score, JudgeRubricScore)
        assert score.evidence_grounding == 5
        assert score.hallucination_detected is False

    def test_forces_tool_choice(self):
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        _single_judgment(client, "alert", _FakeGraph(), _FakeDiagnosis(), "claude-sonnet-4-6")

        _, kwargs = client.messages.create.call_args
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_judgment"}
        assert kwargs["tools"][0]["name"] == "submit_judgment"

    def test_raises_if_no_tool_use_block(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I refuse to call the tool.")]
        )
        with pytest.raises(RuntimeError, match="did not call submit_judgment"):
            _single_judgment(client, "alert", _FakeGraph(), _FakeDiagnosis(), "claude-sonnet-4-6")

    def test_rejects_out_of_range_score(self):
        client = MagicMock()
        bad_payload = {**_VALID_PAYLOAD, "overall": 7}  # out of [1,5]
        client.messages.create.return_value = _tool_use_response(bad_payload)
        with pytest.raises(Exception):  # pydantic.ValidationError
            _single_judgment(client, "alert", _FakeGraph(), _FakeDiagnosis(), "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# judge_diagnosis — orchestration, aggregation, error handling
# ---------------------------------------------------------------------------

class TestJudgeDiagnosis:
    def test_raises_on_none_diagnosis(self):
        with pytest.raises(ValueError, match="requires a completed diagnosis"):
            judge_diagnosis("alert", _FakeGraph(), None, client=MagicMock())

    def test_raises_on_zero_raters(self):
        with pytest.raises(ValueError, match="n_raters must be >= 1"):
            judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=0, client=MagicMock())

    def test_makes_exactly_n_rater_calls(self):
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=5, client=client)

        assert client.messages.create.call_count == 5

    def test_perfect_agreement_when_all_raters_identical(self):
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        result = judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=3, client=client)

        assert result.n_raters == 3
        assert result.overall_agreement == 1.0
        assert result.mean["overall"] == 4.0
        assert result.hallucination_rate == 0.0

    def test_disagreement_lowers_agreement_score(self):
        client = MagicMock()
        low = {**_VALID_PAYLOAD, "overall": 1}
        high = {**_VALID_PAYLOAD, "overall": 5}
        client.messages.create.side_effect = [
            _tool_use_response(low),
            _tool_use_response(high),
        ]

        result = judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=2, client=client)

        assert result.agreement["overall"] == 0.0
        assert result.mean["overall"] == 3.0

    def test_hallucination_rate_is_fraction_flagged(self):
        client = MagicMock()
        clean = {**_VALID_PAYLOAD, "hallucination_detected": False}
        flagged = {**_VALID_PAYLOAD, "hallucination_detected": True}
        client.messages.create.side_effect = [
            _tool_use_response(clean),
            _tool_use_response(flagged),
        ]

        result = judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=2, client=client)

        assert result.hallucination_rate == 0.5

    def test_to_dict_is_json_serializable(self):
        import json
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        result = judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=2, client=client)
        json.dumps(result.to_dict())  # must not raise

    def test_summary_mentions_every_dimension(self):
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_VALID_PAYLOAD)

        result = judge_diagnosis("alert", _FakeGraph(), _FakeDiagnosis(), n_raters=1, client=client)
        summary = result.summary()
        for dim in ["evidence_grounding", "reasoning_coherence", "actionability",
                    "calibration_reasonableness", "overall"]:
            assert dim in summary
        assert "overall_agreement" in summary
        assert "hallucination_rate" in summary
