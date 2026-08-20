"""Output Validator — gates every GraphUpdate before it touches the graph.

Validates the model's structured output against the current graph state.
Returns a ValidationResult; never raises. The caller decides whether to
hard-fail the session or log and continue (eval signal either way).

Checks performed (dispatched by action):
  All actions:
    1. new_evidence.tool_called is a registered tool name
    2. Every key in likelihood_changes is an existing hypothesis ID
    3. Every ID in hypotheses_to_rule_out is an existing hypothesis ID
  action="update":
    4. current_focus is provided and references an existing hypothesis
  action="merge":
    5. merge_into_id is provided and references an existing hypothesis
  action="create":
    6. new_hypothesis_description is non-empty
    7. new_hypothesis_severity (if given) is a valid Severity value
    8. new_hypothesis_initial_likelihood (if given) is in [0, 1]
"""

import math
from dataclasses import dataclass, field
from typing import List

from hypothesis_graph import GraphUpdate, HypothesisGraph, HypothesisStatus, Severity, TOOL_TO_EVIDENCE_TYPE


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
    active_ids = {h.id for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE}

    # Action-specific checks
    if update.action == "update":
        if not update.current_focus:
            errors.append("action='update' requires current_focus to be set")
        elif update.current_focus not in active_ids:
            errors.append(
                f"current_focus {update.current_focus!r} does not reference an ACTIVE hypothesis"
            )

    elif update.action == "merge":
        if not update.merge_into_id:
            errors.append("action='merge' requires merge_into_id to be set")
        elif update.merge_into_id not in active_ids:
            errors.append(
                f"merge_into_id {update.merge_into_id!r} does not reference an ACTIVE hypothesis"
            )

    elif update.action == "create":
        if not update.new_hypothesis_description:
            errors.append("action='create' requires new_hypothesis_description to be non-empty")
        valid_severities = {s.value for s in Severity}
        if update.new_hypothesis_severity and update.new_hypothesis_severity not in valid_severities:
            errors.append(
                f"new_hypothesis_severity {update.new_hypothesis_severity!r} is not valid; "
                f"must be one of {sorted(valid_severities)}"
            )
        if update.new_hypothesis_initial_likelihood is not None:
            if not (0.0 <= update.new_hypothesis_initial_likelihood <= 1.0):
                errors.append(
                    f"new_hypothesis_initial_likelihood must be in [0, 1]; "
                    f"got {update.new_hypothesis_initial_likelihood}"
                )

    # Common checks (all actions)
    if update.new_evidence.tool_called not in TOOL_TO_EVIDENCE_TYPE:
        errors.append(
            f"new_evidence.tool_called {update.new_evidence.tool_called!r} "
            f"is not a registered tool name"
        )

    for h_id, v in update.likelihood_changes.items():
        if h_id not in existing_ids:
            errors.append(
                f"likelihood_changes references unknown hypothesis ID {h_id!r} "
                f"(hallucinated ID)"
            )
        elif not math.isfinite(v):
            errors.append(
                f"likelihood_changes[{h_id!r}] is not finite: {v!r}"
            )

    for h_id in update.hypotheses_to_rule_out:
        if h_id not in existing_ids:
            errors.append(
                f"hypotheses_to_rule_out references unknown hypothesis ID {h_id!r}"
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)
