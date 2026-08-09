"""Hand-rolled ReAct loop against the raw Anthropic API.

Architecture: no frameworks, no LangChain, no MCP abstractions.
Loop: send messages → model reasons + calls tool → dispatch → feed result back → repeat.
Terminates when the model calls stop_investigation or the tool budget is exhausted.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from persistence import snapshot_graph
from hypothesis_graph import (
    EvidenceInput,
    GraphUpdate,
    HypothesisGraph,
    HypothesisStatus,
    update_graph,
)
from output_validator import validate_graph_update
from tools import TOOL_DEFINITIONS, dispatch_tool

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert ML systems reliability engineer investigating a degraded ML system.

Your job: identify the root cause by querying available tools, maintaining a structured
hypothesis graph as you go, then call stop_investigation with your final diagnosis.

## Protocol (follow exactly)
1. Call a query tool (query_metrics, query_logs).
2. Immediately call update_hypothesis_graph to record what you learned.
   Never call two query tools back-to-back without an update in between.
   update_hypothesis_graph is FREE — it does not count against your tool budget.
3. Repeat until one hypothesis is clearly dominant (likelihood > 0.60) or budget is low.
4. Call stop_investigation to submit your diagnosis.

## update_hypothesis_graph — choose one action

action="update" (most common): evidence tests an EXISTING hypothesis.
  - current_focus: hypothesis ID this evidence most directly tests (e.g. "H1")

action="create": evidence suggests a root cause NOT yet in the graph.
  - new_hypothesis_description: what the new hypothesis claims
  - new_hypothesis_severity: critical | high | medium | low
  - new_hypothesis_initial_likelihood: starting weight, e.g. 0.10–0.20

action="merge": your proposed new hypothesis is semantically equivalent
to an existing one (e.g. "model staleness" ≡ "training-serving skew from
delayed retraining"). Name the duplicate target instead of creating a copy.
  - merge_into_id: the existing hypothesis ID to merge evidence into

## Common fields (all actions)
- new_evidence.tool_called: Exactly the tool you just called.
- new_evidence.observation: Precise statement quoting key numbers from the result.
- new_evidence.supports: true if the observation supports the focused hypothesis.
- new_evidence.confidence_delta: Confidence shift for the focused hypothesis (typical ±0.05–0.25).
- likelihood_changes: Adjust ALL relevant hypotheses. Will be normalized automatically.
- hypotheses_to_rule_out: IDs you are now confident are NOT the root cause.
- next_experiment_rationale: Why you're calling the next tool, or why you're stopping.

## Hypothesis graph
The graph starts empty — no pre-seeded categories, no prior assumptions.
Your FIRST update_hypothesis_graph call MUST use action="create" to add your
initial hypothesis based on what the first tool result tells you. Add competing
hypotheses with additional action="create" calls as evidence suggests new
mechanisms. Only use action="update" or action="merge" once the target
hypothesis already exists in the graph.

## Tool budget
query_metrics, query_logs, and stop_investigation each cost 1 budget unit.
update_hypothesis_graph is free. Do not query the same service twice unless the first
result was ambiguous.
"""


# ── Graph initialization ───────────────────────────────────────────────────────

def build_initial_graph(
    alert: str,
    budget: int = 8,
    investigation_start: Optional[datetime] = None,
) -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary=alert,
        investigation_start=investigation_start or datetime.now(timezone.utc),
        tool_call_budget=budget,
        hypotheses=[],
        open_questions=[],
    )


def _graph_context(graph: HypothesisGraph) -> str:
    """Serialize the hypothesis graph into a human-readable block for the context window."""
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    ruled_out = [h for h in graph.hypotheses if h.status == HypothesisStatus.RULED_OUT]

    header = [
        "## Current Hypothesis Graph",
        f"Alert: {graph.alert_summary}",
        f"Investigation anchored to: {graph.investigation_start.strftime('%Y-%m-%dT%H:%M:%SZ')} — use this as 'now' for all tool query windows.",
        f"Tool calls used: {graph.tool_calls_used} / {graph.tool_call_budget}",
    ]

    if not graph.hypotheses:
        return "\n".join(header + [
            "",
            'No hypotheses yet — your first update_hypothesis_graph call must use action="create".',
        ])

    lines = header + ["", "### Active Hypotheses"]
    for h in sorted(active, key=lambda x: -x.likelihood):
        cat = h.root_cause_category.value if h.root_cause_category else "–"
        lines.append(
            f"  [{h.id}] {cat}  "
            f"likelihood={h.likelihood:.2f}  severity={h.severity.value}"
        )
        lines.append(f"       {h.description}")
        for ev in h.evidence:
            sign = "SUPPORTS" if ev.supports else "CONTRADICTS"
            lines.append(f"       Evidence ({sign}): {ev.observation}")

    if ruled_out:
        lines += ["", "### Ruled Out"]
        for h in ruled_out:
            cat = h.root_cause_category.value if h.root_cause_category else "–"
            lines.append(f"  [{h.id}] {cat}: {h.description}")

    if graph.established_facts:
        lines += ["", "### Established Facts"]
        lines += [f"  - {f}" for f in graph.established_facts]

    if graph.open_questions:
        lines += ["", "### Open Questions"]
        lines += [f"  - {q}" for q in graph.open_questions]

    return "\n".join(lines)


# ── Top-suspect display ───────────────────────────────────────────────────────

def print_top_hypothesis(graph: HypothesisGraph) -> None:
    """Print the highest-likelihood active hypothesis in a format readable at a glance."""
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    if not active:
        print("\n  [no active hypotheses remaining]")
        return
    top = max(active, key=lambda h: h.likelihood)
    bar = "█" * round(top.likelihood * 20) + "░" * (20 - round(top.likelihood * 20))
    supporting = sum(1 for e in top.evidence if e.supports)

    cat_label = top.root_cause_category.value.upper().replace("_", " ") if top.root_cause_category else "UNKNOWN"
    print(f"\n  {'─' * 60}")
    print(f"  MOST LIKELY ROOT CAUSE")
    print(f"  {'─' * 60}")
    print(f"  {cat_label}  [{bar}]  {top.likelihood:.0%}")
    print(f"  Severity   : {top.severity.value}")
    print(f"  Hypothesis : {top.description}")
    if top.evidence:
        print(f"  Evidence   : {supporting}/{len(top.evidence)} pieces support this")
    print(f"  {'─' * 60}")


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class DiagnosisResult:
    root_cause: str
    diagnosis: str
    confidence: float
    recommended_action: str

    def __str__(self) -> str:
        return (
            f"Root cause : {self.root_cause}\n"
            f"Confidence : {self.confidence:.0%}\n"
            f"Diagnosis  : {self.diagnosis}\n"
            f"Action     : {self.recommended_action}"
        )


# ── ReAct loop ────────────────────────────────────────────────────────────────

def run_investigation(
    alert: str,
    budget: int = 8,
    investigation_start: Optional[datetime] = None,
    verbose: bool = True,
) -> tuple[HypothesisGraph, Optional[DiagnosisResult]]:
    """
    Run a ReAct investigation loop.

    Returns (graph, diagnosis). Diagnosis is None if the budget was exhausted
    before the model called stop_investigation.
    """
    client = anthropic.Anthropic()
    graph = build_initial_graph(alert, budget=budget, investigation_start=investigation_start)
    snapshot_graph(graph)

    messages = [
        {
            "role": "user",
            "content": (
                f"{_graph_context(graph)}\n\n"
                "Begin your investigation. Call tools to gather evidence, "
                "then call stop_investigation with your final diagnosis."
            ),
        }
    ]

    diagnosis: Optional[DiagnosisResult] = None

    while graph.tool_calls_used < graph.tool_call_budget:
        remaining = graph.tool_call_budget - graph.tool_calls_used
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"[Loop] tool calls remaining: {remaining}")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if verbose:
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    print(f"\n[Reasoning]\n{block.text}")
                elif block.type == "tool_use":
                    print(f"\n[Tool call] {block.name}")
                    print(json.dumps(block.input, indent=2))

        if response.stop_reason == "end_turn":
            if verbose:
                print("\n[Loop] model stopped without calling stop_investigation")
            graph.termination_reason = "model_end_turn"
            break

        if response.stop_reason != "tool_use":
            if verbose:
                print(f"\n[Loop] unexpected stop_reason: {response.stop_reason!r}")
            graph.termination_reason = f"unexpected:{response.stop_reason}"
            break

        tool_results = []
        done = False

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            name = block.name
            inputs = block.input

            if name == "update_hypothesis_graph":
                # Bookkeeping — free, does not count against budget.
                try:
                    raw_ev = EvidenceInput(
                        tool_called=inputs["new_evidence"]["tool_called"],
                        observation=inputs["new_evidence"]["observation"],
                        supports=inputs["new_evidence"]["supports"],
                        confidence_delta=inputs["new_evidence"]["confidence_delta"],
                    )
                    graph_update = GraphUpdate(
                        action=inputs.get("action", "update"),
                        current_focus=inputs.get("current_focus"),
                        merge_into_id=inputs.get("merge_into_id"),
                        new_hypothesis_description=inputs.get("new_hypothesis_description"),
                        new_hypothesis_severity=inputs.get("new_hypothesis_severity"),
                        new_hypothesis_initial_likelihood=inputs.get("new_hypothesis_initial_likelihood"),
                        new_evidence=raw_ev,
                        likelihood_changes=inputs.get("likelihood_changes", {}),
                        hypotheses_to_rule_out=inputs.get("hypotheses_to_rule_out", []),
                        new_established_facts=inputs.get("new_established_facts", []),
                        next_experiment_rationale=inputs.get("next_experiment_rationale", ""),
                    )
                except Exception as exc:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"status": "error", "message": str(exc)}),
                    })
                    continue

                validation = validate_graph_update(graph_update, graph)
                if not validation.valid:
                    if verbose:
                        print(f"\n[Graph update REJECTED]\n{validation}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({
                            "status": "validation_error",
                            "errors": validation.errors,
                        }),
                    })
                else:
                    update_graph(graph, graph_update)
                    snapshot_graph(graph)
                    if verbose:
                        print_top_hypothesis(graph)
                    active = sorted(
                        [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE],
                        key=lambda h: -h.likelihood,
                    )
                    likelihoods = ", ".join(f"{h.id}={h.likelihood:.0%}" for h in active)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({
                            "status": "ok",
                            "active_likelihoods": likelihoods,
                            "established_facts_count": len(graph.established_facts),
                        }),
                    })

            elif name == "stop_investigation":
                graph.tool_calls_used += 1
                done = True
                diagnosis = DiagnosisResult(
                    root_cause=inputs["root_cause_category"],
                    diagnosis=inputs["diagnosis"],
                    confidence=inputs["confidence"],
                    recommended_action=inputs["recommended_action"],
                )
                graph.termination_reason = "stop_investigation"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Investigation complete. Diagnosis recorded.",
                })
                if verbose:
                    print(f"\n[stop_investigation]")
                    print(diagnosis)
            else:
                if graph.tool_calls_used >= graph.tool_call_budget:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({
                            "status": "budget_exhausted",
                            "message": (
                                f"Tool call discarded — all {graph.tool_call_budget} budget "
                                "units already consumed this iteration. Call stop_investigation now."
                            ),
                        }),
                    })
                    if verbose:
                        print(f"\n[Budget cap] {name} discarded — budget consumed")
                    continue
                graph.tool_calls_used += 1
                result = dispatch_tool(name, inputs)
                result_json = json.dumps(result, indent=2)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json,
                })
                if verbose:
                    print(f"\n[Tool result: {name}]")
                    print(result_json)

        messages.append({"role": "user", "content": tool_results})

        if done:
            break

    else:
        graph.termination_reason = "budget_exhausted"
        if verbose:
            print("\n[Loop] tool budget exhausted — investigation terminated")

    return graph, diagnosis


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ALERT = (
        "ALERT: Model accuracy dropped from 0.91 to 0.76 over the past 6 hours. "
        "Prediction confidence also degraded. No system errors visible on dashboard."
    )

    final_graph, final_diagnosis = run_investigation(
        ALERT,
        budget=8,
        investigation_start=datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc),
        verbose=True,
    )

    print(f"\n{'═' * 60}")
    print("FINAL GRAPH STATE")
    print(_graph_context(final_graph))

    if final_diagnosis:
        print(f"\n{'═' * 60}")
        print("FINAL DIAGNOSIS")
        print(final_diagnosis)
    else:
        print_top_hypothesis(final_graph)
        print("\n[Investigation ended without a diagnosis — see top suspect above]")
