"""
Unit tests for evaluation.run_sweep — aggregation logic and orchestration,
no live Anthropic API calls (generate_scenario_data / run_agent_and_score
are mocked, same convention as tests/test_llm_judge.py).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evaluation.chaos_scenarios import SCENARIOS
from evaluation.run_sweep import (
    TIER_MAP,
    _judge_summary,
    _mean,
    _rate,
    _tier_breakdown,
    run_sweep,
    summarize,
)


# ---------------------------------------------------------------------------
# TIER_MAP integrity
# ---------------------------------------------------------------------------

class TestTierMap:
    def test_covers_every_scenario_exactly(self):
        assert set(TIER_MAP.keys()) == set(SCENARIOS.keys())

    def test_every_value_is_a_valid_tier(self):
        assert set(TIER_MAP.values()) <= {"easy", "medium", "hard"}

    def test_tier_counts_match_taxonomy(self):
        # docs/chaos-taxonomy.md: 5 easy, 8 medium, 5 hard
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for tier in TIER_MAP.values():
            counts[tier] += 1
        assert counts == {"easy": 5, "medium": 8, "hard": 5}


# ---------------------------------------------------------------------------
# Aggregation helpers — pure functions
# ---------------------------------------------------------------------------

def _row(scenario_id="feature_drift", tier="easy", top1=True, top3=True,
         tool_calls=4, elapsed=12.0, error=None, judge=None):
    row = {
        "scenario_id": scenario_id,
        "tier": tier,
        "top1_correct": top1,
        "top3_correct": top3,
        "tool_calls_used": tool_calls,
        "elapsed_s": elapsed,
        "error": error,
    }
    if judge is not None:
        row["judge"] = judge
    return row


class TestRateAndMean:
    def test_rate_empty_list(self):
        assert _rate([], "top1_correct") is None

    def test_rate_all_true(self):
        rows = [_row(top1=True), _row(top1=True)]
        assert _rate(rows, "top1_correct") == 1.0

    def test_rate_mixed(self):
        rows = [_row(top1=True), _row(top1=False)]
        assert _rate(rows, "top1_correct") == 0.5

    def test_mean_empty_list(self):
        assert _mean([], "tool_calls_used") is None

    def test_mean_skips_none_values(self):
        rows = [_row(tool_calls=4), {"tool_calls_used": None}]
        assert _mean(rows, "tool_calls_used") == 4.0

    def test_mean_averages_correctly(self):
        rows = [_row(tool_calls=2), _row(tool_calls=6)]
        assert _mean(rows, "tool_calls_used") == 4.0


class TestTierBreakdown:
    def test_groups_by_tier(self):
        rows = [
            _row(scenario_id="feature_drift", tier="easy", top1=True),
            _row(scenario_id="model_staleness", tier="hard", top1=False),
        ]
        breakdown = _tier_breakdown(rows)
        assert set(breakdown.keys()) == {"easy", "hard"}
        assert breakdown["easy"]["n"] == 1
        assert breakdown["easy"]["top1_accuracy"] == 1.0
        assert breakdown["hard"]["top1_accuracy"] == 0.0

    def test_empty_rows(self):
        assert _tier_breakdown([]) == {}


class TestJudgeSummary:
    def _judge_dict(self, overall=4.0, agreement=1.0, hallucination_rate=0.0):
        return {
            "mean": {
                "evidence_grounding": overall, "reasoning_coherence": overall,
                "actionability": overall, "calibration_reasonableness": overall,
                "overall": overall,
            },
            "overall_agreement": agreement,
            "hallucination_rate": hallucination_rate,
        }

    def test_none_when_no_rows_have_judge(self):
        rows = [_row(), _row()]
        assert _judge_summary(rows) is None

    def test_averages_across_scored_rows(self):
        rows = [
            _row(judge=self._judge_dict(overall=3.0, agreement=1.0)),
            _row(judge=self._judge_dict(overall=5.0, agreement=0.5)),
        ]
        result = _judge_summary(rows)
        assert result["n_scored"] == 2
        assert result["mean"]["overall"] == 4.0
        assert result["mean_overall_agreement"] == 0.75

    def test_ignores_unjudged_rows_in_mixed_set(self):
        rows = [_row(judge=self._judge_dict(overall=4.0)), _row(judge=None)]
        result = _judge_summary(rows)
        assert result["n_scored"] == 1


class TestSummarize:
    def test_separates_completed_from_errored(self):
        rows = [
            _row(top1=True),
            {"scenario_id": "bad_deployment", "tier": "easy", "error": "boom"},
        ]
        summary = summarize(rows)
        assert summary["n_completed"] == 1
        assert summary["n_errored"] == 1
        assert summary["errored_scenarios"] == ["bad_deployment"]

    def test_top1_top3_only_over_completed_runs(self):
        rows = [
            _row(top1=True, top3=True),
            {"scenario_id": "bad_deployment", "tier": "easy", "error": "boom"},
        ]
        summary = summarize(rows)
        assert summary["top1_accuracy"] == 1.0
        assert summary["top3_accuracy"] == 1.0

    def test_no_judge_key_when_judge_not_used(self):
        rows = [_row()]
        summary = summarize(rows)
        assert "judge" not in summary

    def test_includes_judge_when_present(self):
        rows = [_row(judge={
            "mean": {d: 4.0 for d in ["evidence_grounding", "reasoning_coherence",
                                       "actionability", "calibration_reasonableness", "overall"]},
            "overall_agreement": 1.0,
            "hallucination_rate": 0.0,
        })]
        summary = summarize(rows)
        assert "judge" in summary
        assert summary["judge"]["n_scored"] == 1


# ---------------------------------------------------------------------------
# run_sweep — orchestration, with generate/run/score mocked out
# ---------------------------------------------------------------------------

class TestRunSweep:
    @patch("evaluation.run_sweep.run_agent_and_score")
    @patch("evaluation.run_sweep.generate_scenario_data")
    def test_calls_pipeline_once_per_scenario_by_default(self, mock_gen, mock_run):
        mock_run.return_value = {
            "scenario_id": "feature_drift", "ground_truth": "feature_drift",
            "predicted": "feature_drift", "top1_correct": True, "top3_correct": True,
            "confidence": 0.8, "tool_calls_used": 3, "tool_call_budget": 8,
            "termination_reason": "stop_investigation", "elapsed_s": 10.0,
        }
        scenario_ids = ["feature_drift", "bad_deployment"]

        report = run_sweep(scenario_ids, output_dir="/tmp/whatever")

        assert mock_gen.call_count == 2
        assert mock_run.call_count == 2
        assert report["n_runs"] == 2
        assert report["summary"]["n_completed"] == 2

    @patch("evaluation.run_sweep.run_agent_and_score")
    @patch("evaluation.run_sweep.generate_scenario_data")
    def test_repeats_multiply_run_count(self, mock_gen, mock_run):
        mock_run.return_value = {
            "scenario_id": "feature_drift", "top1_correct": True, "top3_correct": True,
            "tool_calls_used": 3, "elapsed_s": 10.0,
        }
        report = run_sweep(["feature_drift"], output_dir="/tmp/whatever", repeats=3)

        assert mock_run.call_count == 3
        assert report["n_runs"] == 3
        assert report["repeats_per_scenario"] == 3

    @patch("evaluation.run_sweep.run_agent_and_score")
    @patch("evaluation.run_sweep.generate_scenario_data")
    def test_one_scenario_failing_does_not_abort_the_sweep(self, mock_gen, mock_run):
        def _side_effect(scenario, output_dir, **kwargs):
            if scenario.id == "bad_deployment":
                raise RuntimeError("API timeout")
            return {"scenario_id": scenario.id, "top1_correct": True, "top3_correct": True,
                     "tool_calls_used": 2, "elapsed_s": 5.0}

        mock_run.side_effect = _side_effect
        scenario_ids = ["feature_drift", "bad_deployment", "upstream_schema_change"]

        report = run_sweep(scenario_ids, output_dir="/tmp/whatever")

        assert report["n_runs"] == 3
        assert report["summary"]["n_completed"] == 2
        assert report["summary"]["n_errored"] == 1
        assert report["summary"]["errored_scenarios"] == ["bad_deployment"]
        errored_run = next(r for r in report["runs"] if r["scenario_id"] == "bad_deployment")
        assert "RuntimeError" in errored_run["error"]

    @patch("evaluation.run_sweep.run_agent_and_score")
    @patch("evaluation.run_sweep.generate_scenario_data")
    def test_tags_each_run_with_tier_and_repeat_index(self, mock_gen, mock_run):
        mock_run.return_value = {"scenario_id": "model_staleness", "top1_correct": True,
                                  "top3_correct": True, "tool_calls_used": 3, "elapsed_s": 5.0}
        report = run_sweep(["model_staleness"], output_dir="/tmp/whatever")
        assert report["runs"][0]["tier"] == "hard"
        assert report["runs"][0]["repeat_index"] == 0

    @patch("evaluation.run_sweep.run_agent_and_score")
    @patch("evaluation.run_sweep.generate_scenario_data")
    def test_report_is_json_serializable(self, mock_gen, mock_run):
        import json
        mock_run.return_value = {"scenario_id": "feature_drift", "top1_correct": True,
                                  "top3_correct": True, "tool_calls_used": 3, "elapsed_s": 5.0}
        report = run_sweep(["feature_drift"], output_dir="/tmp/whatever")
        json.dumps(report)  # must not raise
