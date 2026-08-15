"""
Evaluation harness for chaos injection scenarios.

Workflow:
  1. Select a scenario by ID.
  2. Generate 7 days of synthetic data, with the failure window overridden by
     the scenario's SubRangeConfig (via GenerationConfig.overrides).
  3. Inject scenario-specific diagnostic log rows into logs.db.
  4. Optionally run the agent and score top-1 accuracy against ground truth.

Usage:
  cd target-system/
  python -m evaluation.run_eval --scenario feature_drift
  python -m evaluation.run_eval --scenario feature_drift --run-agent
  python -m evaluation.run_eval --list
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_SRC  = _ROOT.parent / "src"

for p in [str(_ROOT), str(_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generator.generate import generate_data
from generator.defaults import make_default_config
from generator.params import GenerationConfig
from evaluation.chaos_scenarios import SCENARIOS, InjectionScenario


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_scenario_data(scenario: InjectionScenario, output_dir: str) -> None:
    config = make_default_config(n_days=7)
    config.output_dir = output_dir
    config.label_delay_hours = 0.0
    config.overrides = [scenario.sub_range]
    config.deployment_events = scenario.deployment_events
    config.seed = 42

    model_dir = str(_ROOT / "model" / "artifacts")
    print(f"\nGenerating data for scenario '{scenario.id}' → {output_dir!r}")
    generate_data(config, model_dir=model_dir, verbose=True)

    _inject_diagnostic_logs(output_dir, scenario.diagnostic_logs)
    print(f"Injected {len(scenario.diagnostic_logs)} diagnostic log entries.")


def _inject_diagnostic_logs(output_dir: str, logs: list[dict]) -> None:
    if not logs:
        return
    db_path = os.path.join(output_dir, "logs.db")
    con = sqlite3.connect(db_path)
    # Remove any diagnostic rows from a previous scenario run before inserting
    # the current scenario's rows. All injected rows use the "injected-" prefix.
    con.execute("DELETE FROM logs WHERE log_id LIKE 'injected-%'")
    con.executemany(
        "INSERT OR REPLACE INTO logs (log_id, timestamp, severity, service, message, context) "
        "VALUES (:log_id, :timestamp, :severity, :service, :message, :context)",
        logs,
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Quick sanity check — print key metric deltas for the generated scenario
# ---------------------------------------------------------------------------

def _print_metric_summary(output_dir: str, scenario: InjectionScenario) -> None:
    db_path = os.path.join(output_dir, "metrics.db")
    con = sqlite3.connect(db_path)
    failure_start = scenario.failure_window_start_ts
    failure_end   = scenario.failure_window_end_ts
    baseline_start = scenario.failure_window_start_ts - 2 * 86_400
    baseline_end   = scenario.failure_window_start_ts

    metrics_of_interest = ["accuracy", "error_rate", "latency_p99", "prediction_confidence"]
    print("\n── Metric sanity check ──────────────────────────────────────")
    print(f"  {'Metric':<28} {'Baseline (2d)':<18} {'Failure (2d)':<18} {'Delta':>8}")
    for m in metrics_of_interest:
        rows = con.execute(
            "SELECT metric_value, timestamp FROM metrics WHERE metric_name=? AND timestamp>=? AND timestamp<?",
            (m, baseline_start, baseline_end),
        ).fetchall()
        baseline_vals = [r[0] for r in rows]
        rows = con.execute(
            "SELECT metric_value FROM metrics WHERE metric_name=? AND timestamp>=? AND timestamp<?",
            (m, failure_start, failure_end),
        ).fetchall()
        failure_vals = [r[0] for r in rows]
        b_avg = sum(baseline_vals) / len(baseline_vals) if baseline_vals else float("nan")
        f_avg = sum(failure_vals) / len(failure_vals) if failure_vals else float("nan")
        delta = f_avg - b_avg if baseline_vals and failure_vals else float("nan")
        print(f"  {m:<28} {b_avg:<18.4f} {f_avg:<18.4f} {delta:>+8.4f}")
    con.close()
    print()


# ---------------------------------------------------------------------------
# Agent run + scoring
# ---------------------------------------------------------------------------

def run_agent_and_score(scenario: InjectionScenario, output_dir: str, verbose: bool = True) -> dict:
    os.environ["MLSYS_DATA_DIR"] = str(output_dir)

    # Import after setting env var so _DATA_DIR picks up the override.
    from datetime import datetime, timezone
    from agent import run_investigation

    print(f"\n── Running agent for scenario '{scenario.id}' ──────────────")
    print(f"  Alert: {scenario.alert[:80]}...")
    print(f"  Investigation start: {scenario.investigation_start_ts}")

    inv_start = datetime.fromtimestamp(scenario.investigation_start_ts, tz=timezone.utc)
    t0 = time.monotonic()
    graph, diagnosis = run_investigation(
        alert=scenario.alert,
        budget=8,
        investigation_start=inv_start,
        verbose=verbose,
        ground_truth=scenario.ground_truth_category,
    )
    elapsed = time.monotonic() - t0

    predicted = diagnosis.root_cause if diagnosis else None
    top1_correct = predicted == scenario.ground_truth_category

    top3_categories = (
        [diagnosis.root_cause] + diagnosis.alternative_categories if diagnosis else []
    )
    top3_correct = scenario.ground_truth_category in top3_categories

    result = {
        "scenario_id":          scenario.id,
        "ground_truth":         scenario.ground_truth_category,
        "predicted":            predicted,
        "top1_correct":         top1_correct,
        "top3_correct":         top3_correct,
        "confidence":           diagnosis.confidence if diagnosis else None,
        "tool_calls_used":      graph.tool_calls_used,
        "tool_call_budget":     graph.tool_call_budget,
        "termination_reason":   graph.termination_reason,
        "elapsed_s":            round(elapsed, 1),
    }

    print(f"\n── Scoring ─────────────────────────────────────────────────")
    print(f"  Ground truth : {scenario.ground_truth_category}")
    print(f"  Predicted    : {predicted}")
    print(f"  Top-1 correct: {top1_correct}")
    print(f"  Top-3 correct: {top3_correct}")
    print(f"  Tool calls   : {graph.tool_calls_used}/{graph.tool_call_budget}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chaos data and optionally run the agent for evaluation."
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        metavar="SCENARIO",
        help=f"Scenario ID. Choices: {', '.join(SCENARIOS)}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated SQLite databases (default: target-system/data)",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="After generating data, run the real agent and score the result.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress agent verbose output.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available chaos injection scenarios:\n")
        for key, s in SCENARIOS.items():
            print(f"  {key}")
            print(f"    {s.label}")
            print(f"    Ground truth: {s.ground_truth_category}")
            print()
        return

    if not args.scenario:
        parser.error("--scenario is required (or --list to see options)")

    scenario = SCENARIOS[args.scenario]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(_ROOT / "data")

    generate_scenario_data(scenario, output_dir)
    _print_metric_summary(output_dir, scenario)

    print(f"\n── Scenario ready ──────────────────────────────────────────")
    print(f"  Data dir         : {output_dir}")
    print(f"  Scenario         : {scenario.label}")
    print(f"  Ground truth     : {scenario.ground_truth_category}")
    print(f"  Alert for agent  : {scenario.alert}")
    print(f"  Investigation ts : {scenario.investigation_start_ts}")
    print(f"  Failure window   : {scenario.failure_window_start_ts} – {scenario.failure_window_end_ts}")
    print()
    print("  To run the agent manually:")
    print(f"    MLSYS_DATA_DIR={output_dir!r} \\")
    print(f"    PYTHONPATH=../src python3 ../src/agent.py")
    print(f"    (edit agent.py ALERT and investigation_start to match above)")

    if args.run_agent:
        result = run_agent_and_score(scenario, output_dir, verbose=not args.quiet)
        print("\n── Result JSON ─────────────────────────────────────────────")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
