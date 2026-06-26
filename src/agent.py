"""Hand-rolled ReAct loop against the raw Anthropic API.

Architecture: no frameworks, no LangChain, no MCP abstractions.
Loop: send messages → model reasons + calls tool → dispatch → feed result back → repeat.
Terminates when the model calls stop_investigation or the tool budget is exhausted.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import anthropic

from persistence import snapshot_graph
from hypothesis_graph import (
    FailureCategory,
    HypothesisGraph,
    HypothesisStatus,
    Hypothesis,
    Severity,
)
from tools import TOOL_DEFINITIONS, dispatch_tool

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert ML systems reliability engineer investigating a degraded ML system.

Your job: identify the root cause by querying available tools, then call
stop_investigation with your ranked diagnosis.

## Approach
1. Use query_metrics first to quantify the degradation and rule out infra issues.
2. Use query_logs on the service(s) most consistent with what you find in metrics.
3. Call stop_investigation when one hypothesis is clearly dominant (confidence > 0.60)
   or when tool calls remaining drops to 1.

## Failure categories you can diagnose
- feature_drift: Input distributions shifted from the training distribution
- bad_deployment: A recent code or model deployment caused a regression
- label_pipeline_corruption: Ground-truth labels are corrupted, making model appear worse
- training_serving_skew: Feature transforms differ between training and serving

## Tool budget
Every call (including stop_investigation) counts. Be efficient — do not query
the same service twice unless the first result was ambiguous.
"""


# ── Graph initialization ───────────────────────────────────────────────────────

def build_initial_graph(alert: str, budget: int = 6) -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary=alert,
        investigation_start=datetime.now(timezone.utc),
        tool_call_budget=budget,
        hypotheses=[
            Hypothesis(
                id="H1",
                root_cause_category=FailureCategory.FEATURE_DRIFT,
                description="Input feature distributions have shifted from the training distribution",
                likelihood=0.30,
                severity=Severity.HIGH,
                status=HypothesisStatus.ACTIVE,
            ),
            Hypothesis(
                id="H2",
                root_cause_category=FailureCategory.BAD_DEPLOYMENT,
                description="A recent model or code deployment introduced a regression",
                likelihood=0.30,
                severity=Severity.HIGH,
                status=HypothesisStatus.ACTIVE,
            ),
            Hypothesis(
                id="H3",
                root_cause_category=FailureCategory.LABEL_PIPELINE_CORRUPTION,
                description="The label pipeline is producing corrupted ground-truth labels",
                likelihood=0.25,
                severity=Severity.MEDIUM,
                status=HypothesisStatus.ACTIVE,
            ),
            Hypothesis(
                id="H4",
                root_cause_category=FailureCategory.TRAINING_SERVING_SKEW,
                description="Feature transformations differ between training and serving environments",
                likelihood=0.15,
                severity=Severity.MEDIUM,
                status=HypothesisStatus.ACTIVE,
            ),
        ],
        open_questions=[
            "How large is the accuracy drop, and did prediction confidence also degrade?",
            "Are there any recent deployments that could explain the timing?",
            "Are there errors or anomalies in the feature pipeline or label pipeline?",
        ],
    )


def _graph_context(graph: HypothesisGraph) -> str:
    """Serialize the hypothesis graph into a human-readable block for the context window."""
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    ruled_out = [h for h in graph.hypotheses if h.status == HypothesisStatus.RULED_OUT]

    lines = [
        "## Current Hypothesis Graph",
        f"Alert: {graph.alert_summary}",
        f"Tool calls used: {graph.tool_calls_used} / {graph.tool_call_budget}",
        "",
        "### Active Hypotheses",
    ]
    for h in sorted(active, key=lambda x: -x.likelihood):
        lines.append(
            f"  [{h.id}] {h.root_cause_category.value}  "
            f"likelihood={h.likelihood:.2f}  severity={h.severity.value}"
        )
        lines.append(f"       {h.description}")
        for ev in h.evidence:
            sign = "SUPPORTS" if ev.supports else "CONTRADICTS"
            lines.append(f"       Evidence ({sign}): {ev.observation}")

    if ruled_out:
        lines += ["", "### Ruled Out"]
        for h in ruled_out:
            lines.append(f"  [{h.id}] {h.root_cause_category.value}: {h.description}")

    if graph.established_facts:
        lines += ["", "### Established Facts"]
        lines += [f"  - {f}" for f in graph.established_facts]

    if graph.open_questions:
        lines += ["", "### Open Questions"]
        lines += [f"  - {q}" for q in graph.open_questions]

    return "\n".join(lines)


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
    budget: int = 6,
    verbose: bool = True,
) -> tuple[HypothesisGraph, Optional[DiagnosisResult]]:
    """
    Run a ReAct investigation loop.

    Returns (graph, diagnosis). Diagnosis is None if the budget was exhausted
    before the model called stop_investigation.
    """
    client = anthropic.Anthropic()
    graph = build_initial_graph(alert, budget=budget)
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

            graph.tool_calls_used += 1
            name = block.name
            inputs = block.input

            if name == "stop_investigation":
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

    final_graph, final_diagnosis = run_investigation(ALERT, budget=6, verbose=True)

    print(f"\n{'═' * 60}")
    print("FINAL GRAPH STATE")
    print(_graph_context(final_graph))

    if final_diagnosis:
        print(f"\n{'═' * 60}")
        print("FINAL DIAGNOSIS")
        print(final_diagnosis)
    else:
        print("\n[No diagnosis produced]")
