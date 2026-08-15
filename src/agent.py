"""Hand-rolled ReAct loop against the raw Anthropic API.

Architecture: no frameworks, no LangChain, no MCP abstractions.
Loop: send messages → model reasons + calls tool → dispatch → feed result back → repeat.
Terminates when the model calls stop_investigation or the tool budget is exhausted.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv(Path(__file__).parent.parent / ".env")

_langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    tracing_enabled=bool(os.environ.get("LANGFUSE_PUBLIC_KEY")),
)

from persistence import snapshot_graph
from hypothesis_graph import (
    EvidenceInput,
    GraphUpdate,
    HypothesisGraph,
    HypothesisStatus,
    update_graph,
)
from output_validator import validate_graph_update
from stopping_criteria import should_stop, snapshot_likelihoods
from prompts import (
    BUDGET_EXHAUSTED_MESSAGE,
    CONVERGENCE_MESSAGE,
    INITIAL_USER_MESSAGE,
    SYSTEM_PROMPT,
)
from tools import TOOL_DEFINITIONS, dispatch_tool


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
    alternative_categories: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        alts = ", ".join(self.alternative_categories) if self.alternative_categories else "none"
        return (
            f"Root cause : {self.root_cause}\n"
            f"Alternatives: {alts}\n"
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
    ground_truth: Optional[str] = None,
) -> tuple[HypothesisGraph, Optional[DiagnosisResult]]:
    """
    Run a ReAct investigation loop.

    Returns (graph, diagnosis). Diagnosis is None if the budget was exhausted
    before the model called stop_investigation.
    """
    client = anthropic.Anthropic()
    graph = build_initial_graph(alert, budget=budget, investigation_start=investigation_start)
    snapshot_graph(graph)

    trace = _langfuse.start_observation(
        name="investigation",
        as_type="span",
        input={"alert": alert},
        metadata={
            "budget": budget,
            **({"ground_truth": ground_truth} if ground_truth else {}),
        },
    )

    messages = [
        {
            "role": "user",
            "content": INITIAL_USER_MESSAGE.content.format(
                graph_context=_graph_context(graph)
            ),
        }
    ]

    diagnosis: Optional[DiagnosisResult] = None
    _likelihood_snapshots: list[dict[str, float]] = []
    _turn = 0

    while graph.tool_calls_used < graph.tool_call_budget:
        _turn += 1
        turn_span = trace.start_observation(name=f"react_turn_{_turn}", as_type="span")

        remaining = graph.tool_call_budget - graph.tool_calls_used
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"[Loop] tool calls remaining: {remaining}")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT.content,
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
            turn_span.update(metadata={"stop_reason": "end_turn"})
            turn_span.end()
            break

        if response.stop_reason != "tool_use":
            if verbose:
                print(f"\n[Loop] unexpected stop_reason: {response.stop_reason!r}")
            graph.termination_reason = f"unexpected:{response.stop_reason}"
            turn_span.update(metadata={"stop_reason": response.stop_reason})
            turn_span.end()
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
                    _likelihood_snapshots.append(snapshot_likelihoods(graph))
                    if verbose:
                        print_top_hypothesis(graph)
                    active = sorted(
                        [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE],
                        key=lambda h: -h.likelihood,
                    )
                    likelihoods = ", ".join(f"{h.id}={h.likelihood:.0%}" for h in active)
                    turn_span.start_observation(
                        name="update_hypothesis_graph",
                        as_type="span",
                        input={
                            "action": graph_update.action,
                            "likelihood_changes": graph_update.likelihood_changes,
                        },
                        output={"normalized_likelihoods": likelihoods},
                    ).end()

                    convergence = should_stop(graph, _likelihood_snapshots)
                    if convergence.stop and verbose:
                        print(f"\n[Convergence] {convergence.reason} — "
                              f"top={convergence.top_hypothesis.likelihood:.0%}")

                    payload: dict = {
                        "status": "ok",
                        "active_likelihoods": likelihoods,
                        "established_facts_count": len(graph.established_facts),
                    }
                    if convergence.stop:
                        payload["convergence_signal"] = convergence.reason
                        payload["convergence_message"] = CONVERGENCE_MESSAGE.content.format(
                            reason=convergence.reason
                        )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(payload),
                    })

            elif name == "stop_investigation":
                graph.tool_calls_used += 1
                done = True
                diagnosis = DiagnosisResult(
                    root_cause=inputs["root_cause_category"],
                    diagnosis=inputs["diagnosis"],
                    confidence=inputs["confidence"],
                    recommended_action=inputs["recommended_action"],
                    alternative_categories=inputs.get("alternative_categories", []),
                )
                graph.termination_reason = "stop_investigation"
                if ground_truth is not None:
                    top1 = int(diagnosis.root_cause == ground_truth)
                    top3_categories = [diagnosis.root_cause] + diagnosis.alternative_categories
                    top3 = int(ground_truth in top3_categories)
                    trace.score_trace(name="top1_correct", value=top1)
                    trace.score_trace(name="top3_correct", value=top3)
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
                            "message": BUDGET_EXHAUSTED_MESSAGE.content.format(
                                budget=graph.tool_call_budget
                            ),
                        }),
                    })
                    if verbose:
                        print(f"\n[Budget cap] {name} discarded — budget consumed")
                    continue
                graph.tool_calls_used += 1
                result = dispatch_tool(name, inputs)
                result_json = json.dumps(result, indent=2)
                turn_span.start_observation(
                    name=name,
                    as_type="tool",
                    input=inputs,
                    output=result,
                ).end()
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json,
                })
                if verbose:
                    print(f"\n[Tool result: {name}]")
                    print(result_json)

        messages.append({"role": "user", "content": tool_results})
        turn_span.update(metadata={"stop_reason": response.stop_reason, "done": done})
        turn_span.end()

        if done:
            break

    else:
        graph.termination_reason = "budget_exhausted"
        if verbose:
            print("\n[Loop] tool budget exhausted — investigation terminated")

    trace.update(
        output={
            "termination_reason": graph.termination_reason,
            "tool_calls_used": graph.tool_calls_used,
            **({"root_cause": diagnosis.root_cause, "confidence": diagnosis.confidence}
               if diagnosis else {}),
        }
    )
    trace.end()
    _langfuse.flush()
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
