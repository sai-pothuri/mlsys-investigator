"""
LLM-as-judge scoring for completed investigations.

Rates the QUALITATIVE quality of a diagnosis narrative — evidence grounding,
reasoning coherence, actionability, and confidence calibration — independent
of whether the predicted root_cause_category matches ground truth. Top-1/
top-3 accuracy (computed in run_eval.py) already covers correctness; this
module answers a different question: even when the agent is right (or
wrong), is the diagnosis it wrote well-reasoned, non-hallucinated, and
actionable?

Architecture: raw Anthropic API only, per CLAUDE.md's non-negotiable
constraint — no separate eval framework, no third-party judge SDK. The
judge itself is a single forced tool-use call, the same pattern agent.py
uses for stop_investigation's final synthesis.

Inter-rater reliability: the project is scoped to one reasoning engine, so
"raters" here means independent repeated judge calls against the same
transcript (self-consistency), not multiple models or human annotators.
Agreement is reported per rubric dimension as the fraction of rater PAIRS
whose 1-5 scores land within `tolerance` of each other — this is the
standard way to report reliability for repeated ordinal ratings, and is a
substantive signal in its own right: an item every rater rates 3 on 3 tries
is a much steadier read than an item that swings 1-5-3, even before
touching top1/top3 accuracy.

Usage:
    from evaluation.llm_judge import judge_diagnosis
    result = judge_diagnosis(alert=scenario.alert, graph=graph, diagnosis=diagnosis)
    print(result.summary())
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

JUDGE_MODEL_DEFAULT = "claude-sonnet-4-6"

_NUMERIC_DIMS = [
    "evidence_grounding",
    "reasoning_coherence",
    "actionability",
    "calibration_reasonableness",
    "overall",
]


# ── Rubric ───────────────────────────────────────────────────────────────────

class JudgeRubricScore(BaseModel):
    """One rater pass over one diagnosis. Scores are always explicitly
    elicited from the model via a forced tool call — never inferred or
    rank-derived — for the same reason hypothesis confidence is (see
    hypothesis-graph-spec.md §5.4): a derived score can't be used to
    validate the elicitation itself."""

    evidence_grounding: int = Field(
        ge=1, le=5,
        description="Does every factual claim in the diagnosis trace back to evidence actually in the graph?",
    )
    reasoning_coherence: int = Field(
        ge=1, le=5,
        description="Does the cited evidence actually support the claimed root cause?",
    )
    actionability: int = Field(
        ge=1, le=5,
        description="Is recommended_action concrete enough to execute immediately?",
    )
    calibration_reasonableness: int = Field(
        ge=1, le=5,
        description="Is the stated confidence defensible given the evidence strength?",
    )
    overall: int = Field(ge=1, le=5, description="Holistic diagnosis quality.")
    hallucination_detected: bool = Field(
        description="True if the diagnosis asserts anything not traceable to the graph's evidence."
    )
    rationale: str = Field(description="2-4 sentences justifying the scores above.")


@dataclass
class JudgeResult:
    scores: list[JudgeRubricScore]      # one entry per rater pass
    mean: dict                          # per-dimension mean across raters
    agreement: dict                     # per-dimension pairwise agreement, 0.0-1.0
    overall_agreement: float            # mean of per-dimension agreement
    hallucination_rate: float           # fraction of raters that flagged a hallucination
    n_raters: int
    model: str

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_raters": self.n_raters,
            "mean": self.mean,
            "agreement": self.agreement,
            "overall_agreement": round(self.overall_agreement, 3),
            "hallucination_rate": round(self.hallucination_rate, 3),
            "scores": [s.model_dump() for s in self.scores],
        }

    def summary(self) -> str:
        lines = [
            f"LLM-as-judge ({self.n_raters} raters, {self.model}):",
        ]
        for dim in _NUMERIC_DIMS:
            lines.append(
                f"  {dim:<28} mean={self.mean[dim]:.2f}  agreement={self.agreement[dim]:.2f}"
            )
        lines.append(f"  {'overall_agreement':<28} {self.overall_agreement:.2f}")
        lines.append(f"  {'hallucination_rate':<28} {self.hallucination_rate:.2f}")
        return "\n".join(lines)


# ── Judge prompt ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring the QUALITY of an automated \
ML failure diagnosis — not whether its predicted category happens to match a ground-truth \
label. That is scored separately by exact-match accuracy; you are not shown a ground truth \
and should not try to guess or reward one.

You will be shown the original alert, the complete hypothesis graph the investigating agent \
built (every hypothesis it considered, all evidence it gathered, and whether each hypothesis \
was ruled out, confirmed, or left active), and the agent's final diagnosis.

Score the final diagnosis strictly against what is actually in the hypothesis graph:

- evidence_grounding: does every factual claim in the diagnosis text (numbers, event names, \
timestamps, service names) trace back to an observation actually recorded in the graph's \
evidence? Penalize any claim you cannot find grounded in the graph, even if it sounds plausible.
- reasoning_coherence: does the cited evidence actually support the claimed root cause, or is \
the connection weak, circular, or a non-sequitur?
- actionability: is recommended_action something an on-call engineer could execute right now \
(a specific rollback, a specific config change, a specific team to page), or is it vague \
("investigate further", "monitor closely")?
- calibration_reasonableness: given how much and how strong the supporting evidence is, is the \
stated confidence value defensible? Penalize high confidence backed by thin or one-sided \
evidence, and penalize low confidence when the evidence is actually strong and consistent.
- overall: your holistic 1-5 quality score, not a mechanical average of the above.
- hallucination_detected: true if the diagnosis text asserts anything — a number, an event, a \
log message, a timestamp — that does not appear anywhere in the hypothesis graph's evidence.

Call submit_judgment with your scores and a short rationale. Do not consider or guess at \
whether the root_cause_category is "correct" against any external label."""


def _judge_tool_schema() -> dict:
    return {
        "name": "submit_judgment",
        "description": "Submit rubric scores for this investigation's final diagnosis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evidence_grounding":          {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning_coherence":         {"type": "integer", "minimum": 1, "maximum": 5},
                "actionability":               {"type": "integer", "minimum": 1, "maximum": 5},
                "calibration_reasonableness":  {"type": "integer", "minimum": 1, "maximum": 5},
                "overall":                     {"type": "integer", "minimum": 1, "maximum": 5},
                "hallucination_detected":      {"type": "boolean"},
                "rationale":                   {"type": "string"},
            },
            "required": [
                "evidence_grounding", "reasoning_coherence", "actionability",
                "calibration_reasonableness", "overall",
                "hallucination_detected", "rationale",
            ],
        },
    }


def _diagnosis_to_dict(diagnosis) -> dict:
    return {
        "root_cause_category": diagnosis.root_cause,
        "diagnosis": diagnosis.diagnosis,
        "confidence": diagnosis.confidence,
        "recommended_action": diagnosis.recommended_action,
        "alternative_categories": diagnosis.alternative_categories,
    }


def build_judge_user_message(alert: str, graph, diagnosis) -> str:
    """Serialize the full graph (all hypotheses, all evidence, ruled-out included)
    and the final diagnosis into the judge's user turn. Uses the graph's own
    Pydantic serialization rather than agent.py's `_graph_context` — the judge
    needs the complete evidence trail to check grounding, not the condensed
    view the ReAct loop shows the investigating model mid-session."""
    graph_json = json.dumps(graph.model_dump(mode="json"), indent=2, default=str)
    diagnosis_json = json.dumps(_diagnosis_to_dict(diagnosis), indent=2)
    return (
        f"## Alert\n{alert}\n\n"
        f"## Complete Hypothesis Graph\n```json\n{graph_json}\n```\n\n"
        f"## Final Diagnosis\n```json\n{diagnosis_json}\n```\n\n"
        "Score this diagnosis by calling submit_judgment."
    )


def _single_judgment(client, alert: str, graph, diagnosis, model: str) -> JudgeRubricScore:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_JUDGE_SYSTEM_PROMPT,
        tools=[_judge_tool_schema()],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[{"role": "user", "content": build_judge_user_message(alert, graph, diagnosis)}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_judgment":
            return JudgeRubricScore(**block.input)
    raise RuntimeError("Judge model did not call submit_judgment")


# ── Inter-rater agreement ────────────────────────────────────────────────────

def _pairwise_agreement(values: list[int], tolerance: int = 1) -> float:
    """Fraction of rater PAIRS whose scores differ by <= tolerance.
    1.0 = every pair of raters agrees within tolerance; 0.0 = none do.
    A single rater (or all-identical values) trivially agrees with itself."""
    if len(values) < 2:
        return 1.0
    pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1:]]
    agree = sum(1 for a, b in pairs if abs(a - b) <= tolerance)
    return agree / len(pairs)


# ── Public entry point ───────────────────────────────────────────────────────

def judge_diagnosis(
    alert: str,
    graph,
    diagnosis,
    n_raters: int = 3,
    model: str = JUDGE_MODEL_DEFAULT,
    client: Optional["anthropic.Anthropic"] = None,
) -> JudgeResult:
    """Score a completed investigation's diagnosis narrative `n_raters`
    independent times and report per-dimension means alongside inter-rater
    agreement. `diagnosis` must not be None — call this only after
    stop_investigation has actually fired (or the budget-exhaustion
    fallback produced a diagnosis); there is nothing to judge otherwise.
    """
    if diagnosis is None:
        raise ValueError("judge_diagnosis requires a completed diagnosis, got None")
    if n_raters < 1:
        raise ValueError(f"n_raters must be >= 1, got {n_raters}")

    client = client or anthropic.Anthropic()
    scores = [_single_judgment(client, alert, graph, diagnosis, model) for _ in range(n_raters)]

    mean = {
        dim: statistics.mean(getattr(s, dim) for s in scores)
        for dim in _NUMERIC_DIMS
    }
    agreement = {
        dim: _pairwise_agreement([getattr(s, dim) for s in scores])
        for dim in _NUMERIC_DIMS
    }
    overall_agreement = statistics.mean(agreement.values())
    hallucination_rate = sum(1 for s in scores if s.hallucination_detected) / len(scores)

    return JudgeResult(
        scores=scores,
        mean=mean,
        agreement=agreement,
        overall_agreement=overall_agreement,
        hallucination_rate=hallucination_rate,
        n_raters=n_raters,
        model=model,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Judge a fresh investigation run against a chaos scenario.

    Usage:
      cd target-system/
      python -m evaluation.llm_judge --scenario feature_drift
      python -m evaluation.llm_judge --scenario feature_drift --raters 5
    """
    import argparse
    import os
    import sys
    from pathlib import Path as _Path

    _here = _Path(__file__).parent
    _root = _here.parent
    _src = _root.parent / "src"
    for p in [str(_root), str(_src)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser(description="Run an investigation and score it with the LLM judge.")
    parser.add_argument("--scenario", required=True, help="Chaos scenario ID (see run_eval.py --list)")
    parser.add_argument("--raters", type=int, default=3, help="Number of independent judge passes")
    parser.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from evaluation.chaos_scenarios import SCENARIOS
    from evaluation.run_eval import generate_scenario_data
    from datetime import datetime, timezone

    scenario = SCENARIOS[args.scenario]
    output_dir = args.output_dir or str(_root / "data")
    generate_scenario_data(scenario, output_dir)
    os.environ["MLSYS_DATA_DIR"] = output_dir

    from agent import run_investigation

    inv_start = datetime.fromtimestamp(scenario.investigation_start_ts, tz=timezone.utc)
    graph, diagnosis = run_investigation(
        alert=scenario.alert,
        budget=8,
        investigation_start=inv_start,
        verbose=False,
        ground_truth=scenario.ground_truth_category,
    )

    if diagnosis is None:
        print("Investigation ended without a diagnosis (budget exhausted, no synthesis) — nothing to judge.")
        return

    result = judge_diagnosis(scenario.alert, graph, diagnosis, n_raters=args.raters, model=args.judge_model)
    print(f"\nGround truth: {scenario.ground_truth_category}")
    print(f"Predicted:    {diagnosis.root_cause}")
    print()
    print(result.summary())
    print()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
