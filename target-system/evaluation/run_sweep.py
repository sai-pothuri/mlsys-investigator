"""
Batch evaluation sweep — runs the agent against all (or a subset of) chaos
injection scenarios and produces one structured report.

This is the piece run_eval.py doesn't have on its own: run_eval.py drives one
scenario per invocation (generate data, optionally run the agent, print a
result dict). run_sweep.py iterates that same generate+run+score pipeline
across the full scenario set (or a tier, or an explicit list), tolerates a
single scenario failing without aborting the rest, and aggregates everything
— including LLM-judge scores, when requested — into one JSON report with a
top-1/top-3/tool-calls/judge summary broken out overall and per difficulty
tier.

Usage:
  cd target-system/
  python -m evaluation.run_sweep                              # all 18 scenarios
  python -m evaluation.run_sweep --judge                       # + LLM-judge scoring
  python -m evaluation.run_sweep --tier easy
  python -m evaluation.run_sweep --scenarios feature_drift,bad_deployment
  python -m evaluation.run_sweep --repeats 3                   # N runs per scenario
  python -m evaluation.run_sweep --output-file report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_SRC = _ROOT.parent / "src"
for p in [str(_ROOT), str(_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.chaos_scenarios import SCENARIOS
from evaluation.run_eval import generate_scenario_data, run_agent_and_score


# Difficulty tier per docs/chaos-taxonomy.md. Not stored on InjectionScenario
# itself — this is purely a reporting concern, kept here rather than in
# chaos_scenarios.py so scenario definitions don't carry sweep-only metadata.
TIER_MAP: dict[str, str] = {
    "feature_drift":                     "easy",
    "bad_deployment":                    "easy",
    "upstream_schema_change":            "easy",
    "infrastructure_latency_spike":      "easy",
    "model_version_rollback_regression": "easy",
    "label_pipeline_corruption":         "medium",
    "training_serving_skew":             "medium",
    "data_freshness_degradation":        "medium",
    "feature_encoding_bug":              "medium",
    "gradual_concept_drift":             "medium",
    "model_calibration_drift":           "medium",
    "shadow_mode_leak":                  "medium",
    "feature_pipeline_partial_failure":  "medium",
    "delayed_label_feedback_shift":      "hard",
    "cascading_upstream_failure":        "hard",
    "model_staleness":                   "hard",
    "feature_importance_inversion":      "hard",
    "compound_drift_plus_deployment":    "hard",
}
assert set(TIER_MAP) == set(SCENARIOS), (
    f"TIER_MAP out of sync with SCENARIOS registry.\n"
    f"  In SCENARIOS but not TIER_MAP: {set(SCENARIOS) - set(TIER_MAP)}\n"
    f"  In TIER_MAP but not SCENARIOS: {set(TIER_MAP) - set(SCENARIOS)}"
)

_JUDGE_DIMS = [
    "evidence_grounding", "reasoning_coherence", "actionability",
    "calibration_reasonableness", "overall",
]


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def run_sweep(
    scenario_ids: list[str],
    output_dir: str,
    repeats: int = 1,
    judge: bool = False,
    judge_raters: int = 3,
    verbose: bool = False,
) -> dict:
    """Run generate+agent+score (and optionally judge) for every scenario ID,
    `repeats` times each. A single run's exception is caught and recorded as
    an errored run rather than aborting the sweep — an API hiccup on
    scenario 12 of 18 shouldn't cost the other 17 results."""
    runs: list[dict] = []
    total = len(scenario_ids) * repeats
    t0 = time.monotonic()

    for i, scenario_id in enumerate(scenario_ids):
        scenario = SCENARIOS[scenario_id]
        for rep in range(repeats):
            n = i * repeats + rep + 1
            print(f"\n{'=' * 70}\n[{n}/{total}] {scenario_id} (tier={TIER_MAP[scenario_id]}, run {rep + 1}/{repeats})\n{'=' * 70}")
            try:
                generate_scenario_data(scenario, output_dir)
                result = run_agent_and_score(
                    scenario, output_dir, verbose=verbose,
                    judge=judge, judge_raters=judge_raters,
                )
                result["tier"] = TIER_MAP[scenario_id]
                result["repeat_index"] = rep
                result["error"] = None
            except Exception as exc:
                print(f"\n[run_sweep] scenario '{scenario_id}' run {rep + 1} FAILED: {exc}")
                result = {
                    "scenario_id": scenario_id,
                    "ground_truth": scenario.ground_truth_category,
                    "tier": TIER_MAP[scenario_id],
                    "repeat_index": rep,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_traceback": traceback.format_exc(),
                }
            runs.append(result)

    elapsed = time.monotonic() - t0
    return {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "n_scenarios":            len(scenario_ids),
        "repeats_per_scenario":   repeats,
        "n_runs":                 len(runs),
        "judge_enabled":          judge,
        "judge_raters":           judge_raters if judge else None,
        "total_elapsed_s":        round(elapsed, 1),
        "runs":                   runs,
        "summary":                summarize(runs),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rate(rows: list[dict], key: str) -> float | None:
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _tier_breakdown(rows: list[dict]) -> dict:
    by_tier: dict[str, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    return {
        tier: {
            "n":               len(rs),
            "top1_accuracy":   _rate(rs, "top1_correct"),
            "top3_accuracy":   _rate(rs, "top3_correct"),
            "mean_tool_calls": _mean(rs, "tool_calls_used"),
        }
        for tier, rs in sorted(by_tier.items())
    }


def _judge_summary(rows: list[dict]) -> dict | None:
    judged = [r["judge"] for r in rows if r.get("judge")]
    if not judged:
        return None
    return {
        "n_scored":               len(judged),
        "mean":                   {d: round(statistics.mean(j["mean"][d] for j in judged), 3) for d in _JUDGE_DIMS},
        "mean_overall_agreement": round(statistics.mean(j["overall_agreement"] for j in judged), 3),
        "mean_hallucination_rate": round(statistics.mean(j["hallucination_rate"] for j in judged), 3),
    }


def summarize(runs: list[dict]) -> dict:
    completed = [r for r in runs if not r.get("error")]
    errored = [r for r in runs if r.get("error")]
    summary = {
        "n_completed":      len(completed),
        "n_errored":        len(errored),
        "errored_scenarios": [r["scenario_id"] for r in errored],
        "top1_accuracy":    _rate(completed, "top1_correct"),
        "top3_accuracy":    _rate(completed, "top3_correct"),
        "mean_tool_calls":  _mean(completed, "tool_calls_used"),
        "mean_elapsed_s":   _mean(completed, "elapsed_s"),
        "by_tier":          _tier_breakdown(completed),
    }
    judge_summary = _judge_summary(completed)
    if judge_summary:
        summary["judge"] = judge_summary
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agent across chaos injection scenarios and produce one structured report."
    )
    parser.add_argument("--scenarios", default=None,
                         help="Comma-separated scenario IDs (default: all 18)")
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], default=None,
                         help="Restrict the sweep to one difficulty tier")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Runs per scenario, for variance across repeated investigations (default: 1)")
    parser.add_argument("--judge", action="store_true",
                         help="Score each diagnosis with the LLM judge (see evaluation/llm_judge.py)")
    parser.add_argument("--judge-raters", type=int, default=3)
    parser.add_argument("--output-dir", default=None,
                         help="Scratch dir for generated SQLite data, reused across runs (default: target-system/data)")
    parser.add_argument("--output-file", default=None,
                         help="Where to write the JSON report (default: target-system/eval_report_<timestamp>.json)")
    parser.add_argument("--verbose", action="store_true",
                         help="Show the full agent reasoning trace for every run (noisy across 18 scenarios)")
    args = parser.parse_args()

    if args.scenarios and args.tier:
        parser.error("--scenarios and --tier are mutually exclusive")

    if args.scenarios:
        scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        unknown = set(scenario_ids) - set(SCENARIOS)
        if unknown:
            parser.error(f"Unknown scenario ID(s): {sorted(unknown)}. See --scenarios choices in SCENARIOS.")
    elif args.tier:
        scenario_ids = [sid for sid, tier in TIER_MAP.items() if tier == args.tier]
    else:
        scenario_ids = list(SCENARIOS.keys())

    output_dir = args.output_dir or str(_ROOT / "data")
    n_total = len(scenario_ids) * args.repeats
    print(f"Sweeping {len(scenario_ids)} scenario(s) x {args.repeats} repeat(s) = {n_total} run(s)"
          f"{' + LLM judge (' + str(args.judge_raters) + ' raters)' if args.judge else ''}")

    report = run_sweep(
        scenario_ids, output_dir,
        repeats=args.repeats, judge=args.judge, judge_raters=args.judge_raters,
        verbose=args.verbose,
    )

    output_file = args.output_file or str(
        _ROOT / f"eval_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}\nSWEEP COMPLETE\n{'=' * 70}")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report written to: {output_file}")


if __name__ == "__main__":
    main()
