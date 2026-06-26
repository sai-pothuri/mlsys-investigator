"""Output Validator — gates every GraphUpdate before it touches the graph.

Validates the model's structured output against the current graph state.
Returns a ValidationResult; never raises. The caller decides whether to
hard-fail the session or log and continue (eval signal either way).

Checks performed (all from hypothesis-graph-spec.md open issues §7):
  1. new_evidence.tool_called is a registered tool name
  2. Every key in likelihood_changes is an existing hypothesis ID
  3. Every ID in hypotheses_to_rule_out is an existing hypothesis ID
  4. New hypothesis IDs don't collide with existing ones
  5. New hypotheses arrive with empty evidence
"""

from dataclasses import dataclass, field
from typing import List

from hypothesis_graph import GraphUpdate, HypothesisGraph, TOOL_TO_EVIDENCE_TYPE


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.valid:
            return "PASS"
        return "FAIL\n" + "\n".join(f"  - {e}" for e in self.errors)


def validate_graph_update(update: GraphUpdate, graph: HypothesisGraph) -> ValidationResult:
    errors: List[str] = []
    existing_ids = {h.id for h in graph.hypotheses}

    # 1. tool_called must be registered
    if update.new_evidence.tool_called not in TOOL_TO_EVIDENCE_TYPE:
        errors.append(
            f"new_evidence.tool_called {update.new_evidence.tool_called!r} "
            f"is not a registered tool name"
        )

    # 2. likelihood_changes keys must reference existing hypotheses
    for h_id in update.likelihood_changes:
        if h_id not in existing_ids:
            errors.append(
                f"likelihood_changes references unknown hypothesis ID {h_id!r} "
                f"(hallucinated ID)"
            )

    # 3. hypotheses_to_rule_out must reference existing hypotheses
    for h_id in update.hypotheses_to_rule_out:
        if h_id not in existing_ids:
            errors.append(
                f"hypotheses_to_rule_out references unknown hypothesis ID {h_id!r}"
            )

    # 4 & 5. new_hypotheses: no ID collision, no pre-populated evidence
    for h in update.new_hypotheses:
        if h.id in existing_ids:
            errors.append(
                f"new_hypotheses contains ID {h.id!r} that already exists in the graph"
            )
        if h.evidence:
            errors.append(
                f"new_hypothesis {h.id!r} must be created with empty evidence "
                f"(got {len(h.evidence)} evidence item(s))"
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)
