"""Mock ReAct runner — drives the loop with canned fixtures instead of the Anthropic API.

Each fixture is one complete ReAct turn:
  Thought  → model's reasoning
  Action   → tool call (scenario-specific hardcoded dispatch)
  Observe  → tool result
  Update   → GraphUpdate validated then applied (or rejected)

Usage:
  PYTHONPATH=src python3 src/run_mock.py                          # feature_drift (default)
  PYTHONPATH=src python3 src/run_mock.py --scenario bad_deployment
  PYTHONPATH=src python3 src/run_mock.py --scenario label_corruption
  PYTHONPATH=src python3 src/run_mock.py --list
"""

import argparse
import json
import sys
import time

from hypothesis_graph import HypothesisGraph, HypothesisStatus, update_graph
from agent import build_initial_graph, _graph_context, print_top_hypothesis
from output_validator import validate_graph_update
from persistence import snapshot_graph
from scenarios import SCENARIOS, Scenario

# ── Display helpers ───────────────────────────────────────────────────────────

_W = 66


def _header(text: str) -> None:
    print(f"\n{'═' * _W}")
    print(f"  {text}")
    print(f"{'═' * _W}")


def _section(label: str, text: str) -> None:
    print(f"\n[{label}]")
    for line in text.strip().splitlines():
        print(f"  {line}")


def _graph_summary(graph: HypothesisGraph) -> str:
    lines = [f"tool calls used: {graph.tool_calls_used} / {graph.tool_call_budget}"]
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    ruled  = [h for h in graph.hypotheses if h.status == HypothesisStatus.RULED_OUT]
    for h in sorted(active, key=lambda x: -x.likelihood):
        cat = h.root_cause_category.value if h.root_cause_category else "–"
        lines.append(
            f"  {h.id}  {cat:<30}  "
            f"likelihood={h.likelihood:.3f}  [{h.status.value}]"
        )
    for h in ruled:
        cat = h.root_cause_category.value if h.root_cause_category else "–"
        lines.append(f"  {h.id}  {cat:<30}  [{h.status.value}]")
    if graph.established_facts:
        lines.append("  established facts:")
        for f in graph.established_facts:
            lines.append(f"    · {f}")
    return "\n".join(lines)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_mock(scenario: Scenario) -> None:
    _header(f"SCENARIO: {scenario.label}")
    print(f"  Alert      : {scenario.alert}")
    print(f"  Fixtures   : {len(scenario.fixtures)}")

    graph = build_initial_graph(scenario.alert, budget=8)
    snapshot_graph(graph)

    _header("INITIAL GRAPH STATE")
    print(_graph_summary(graph))

    rejected_count = 0
    applied_count = 0

    for i, fixture in enumerate(scenario.fixtures, start=1):
        _header(f"Turn {i} — {fixture.label}")

        _section("Thought", fixture.thought)

        _section("Action", f"Calling tool: {fixture.tool_called}")
        tool_result = scenario.dispatch_tool(fixture.tool_called, fixture.tool_inputs)
        tool_json = json.dumps(tool_result, indent=4)
        _section("Tool Result (raw)", tool_json)
        graph.tool_calls_used += 1

        _section("Observation", fixture.observation)

        result = validate_graph_update(fixture.update, graph)

        action = fixture.update.action
        action_detail = {
            "update": f"current_focus={fixture.update.current_focus}",
            "create": f"description='{fixture.update.new_hypothesis_description}'",
            "merge":  f"merge_into_id={fixture.update.merge_into_id}",
        }.get(action, "")
        _section("GraphUpdate", (
            f"action : {action}  ({action_detail})\n"
            f"new_evidence : tool={fixture.update.new_evidence.tool_called}"
            f"  supports={fixture.update.new_evidence.supports}"
            f"  delta={fixture.update.new_evidence.confidence_delta:+.2f}\n"
            f"likelihood_changes : {fixture.update.likelihood_changes}\n"
            f"rule_out : {fixture.update.hypotheses_to_rule_out}"
        ))

        _section("Validation", str(result))

        if result.valid != fixture.expect_valid:
            print(
                f"\n  !! FIXTURE EXPECTATION MISMATCH: "
                f"expected valid={fixture.expect_valid}, got valid={result.valid}",
                file=sys.stderr,
            )

        if result.valid:
            update_graph(graph, fixture.update)
            snapshot_graph(graph)
            applied_count += 1
            _section("Graph state after update", _graph_summary(graph))
            print_top_hypothesis(graph)
        else:
            rejected_count += 1
            _section(
                "Graph state UNCHANGED",
                "Update rejected — graph not mutated.\n" + _graph_summary(graph),
            )

        print(f"\n  [waiting 5s — check hypothesis_graph.json]")
        time.sleep(5)

    _header("RUN COMPLETE")
    print(f"  Fixtures applied  : {applied_count}")
    print(f"  Fixtures rejected : {rejected_count}")
    print(f"\n  Ground truth: {scenario.ground_truth}")
    print_top_hypothesis(graph)
    print()
    print(_graph_context(graph))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a mock ReAct investigation against a hardcoded scenario."
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="feature_drift",
        metavar="SCENARIO",
        help=f"Which scenario to run. Choices: {', '.join(SCENARIOS)}. Default: feature_drift",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:\n")
        for key, s in SCENARIOS.items():
            print(f"  {key}")
            print(f"    {s.label}")
            print(f"    Ground truth: {s.ground_truth}")
            print()
        return

    run_mock(SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
