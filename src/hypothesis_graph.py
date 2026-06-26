"""Pydantic models for the agent's structured belief state (Hypothesis Graph).

Spec: /docs/hypothesis-graph-spec.md
Design decisions: delta-based updates, evidence_type derived never model-set,
likelihoods normalized programmatically, per-update delta capped at ±0.25.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class FailureCategory(str, Enum):
    # Derived from chaos injection taxonomy (single source of truth).
    # TODO: expand to full 15-20 category taxonomy before building eval harness.
    FEATURE_DRIFT = "feature_drift"
    BAD_DEPLOYMENT = "bad_deployment"
    LABEL_PIPELINE_CORRUPTION = "label_pipeline_corruption"
    TRAINING_SERVING_SKEW = "training_serving_skew"


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    RULED_OUT = "ruled_out"
    CONFIRMED = "confirmed"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceType(str, Enum):
    METRICS = "metrics"
    LOGS = "logs"
    DEPLOYMENT_HISTORY = "deployment_history"
    FEATURE_DISTRIBUTIONS = "feature_distributions"
    CODE_DIFFS = "code_diffs"


# Maps tool names to EvidenceType — both must stay in sync (see tool-specs.md §2.4).
TOOL_TO_EVIDENCE_TYPE: Dict[str, EvidenceType] = {
    "query_metrics": EvidenceType.METRICS,
    "query_logs": EvidenceType.LOGS,
    "query_deployment_history": EvidenceType.DEPLOYMENT_HISTORY,
    "query_feature_distributions": EvidenceType.FEATURE_DISTRIBUTIONS,
    "query_code_diffs": EvidenceType.CODE_DIFFS,
}


def derive_evidence_type(tool_called: str) -> EvidenceType:
    try:
        return TOOL_TO_EVIDENCE_TYPE[tool_called]
    except KeyError:
        raise ValueError(f"Unregistered tool name: {tool_called!r}")


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceInput(BaseModel):
    """Model-facing schema — the model never sets evidence_type."""
    tool_called: str
    observation: str
    supports: bool
    confidence_delta: float = Field(ge=-1.0, le=1.0)


class Evidence(BaseModel):
    """Stored representation — evidence_type is always derived, never model-set."""
    tool_called: str
    evidence_type: EvidenceType
    observation: str
    supports: bool
    confidence_delta: float = Field(ge=-1.0, le=1.0)

    @classmethod
    def from_input(cls, raw: EvidenceInput) -> "Evidence":
        return cls(
            **raw.model_dump(),
            evidence_type=derive_evidence_type(raw.tool_called),
        )


# ── Hypothesis ────────────────────────────────────────────────────────────────

class Experiment(BaseModel):
    tool_to_call: str
    parameters: dict
    rationale: str
    expected_if_true: str
    expected_if_false: str


class Hypothesis(BaseModel):
    id: str
    root_cause_category: FailureCategory
    description: str
    likelihood: float = Field(ge=0.0, le=1.0)
    severity: Severity
    status: HypothesisStatus
    evidence: List[Evidence] = []
    distinguishing_experiments: List[Experiment] = []


# ── Graph ─────────────────────────────────────────────────────────────────────

class HypothesisGraph(BaseModel):
    alert_summary: str
    investigation_start: datetime
    tool_calls_used: int = 0
    tool_call_budget: int
    hypotheses: List[Hypothesis]
    established_facts: List[str] = []
    open_questions: List[str] = []
    current_focus: Optional[str] = None
    termination_reason: Optional[str] = None


# ── GraphUpdate (delta applied programmatically, not full-rewrite) ────────────

class GraphUpdate(BaseModel):
    """Structured delta the model emits after each tool call."""
    new_evidence: EvidenceInput
    likelihood_changes: Dict[str, float]
    hypotheses_to_rule_out: List[str] = []
    new_hypotheses: List[Hypothesis] = []
    new_established_facts: List[str] = []
    next_experiment_rationale: str


_MAX_LIKELIHOOD_DELTA = 0.25  # cap per spec §5.3


def update_graph(graph: HypothesisGraph, update: GraphUpdate) -> HypothesisGraph:
    """Apply a GraphUpdate delta in-place. Returns the mutated graph."""
    # Attach evidence to current_focus hypothesis (evidence always belongs to one hypothesis).
    evidence = Evidence.from_input(update.new_evidence)
    for h in graph.hypotheses:
        if h.id == graph.current_focus:
            h.evidence.append(evidence)
            break

    # Apply capped likelihood changes.
    for h_id, delta in update.likelihood_changes.items():
        capped = max(-_MAX_LIKELIHOOD_DELTA, min(_MAX_LIKELIHOOD_DELTA, delta))
        for h in graph.hypotheses:
            if h.id == h_id:
                h.likelihood = max(0.0, min(1.0, h.likelihood + capped))

    # Rule out hypotheses.
    for h_id in update.hypotheses_to_rule_out:
        for h in graph.hypotheses:
            if h.id == h_id:
                h.status = HypothesisStatus.RULED_OUT

    # Insert surprising new hypotheses (evidence must be empty per spec §3.3).
    for new_h in update.new_hypotheses:
        if new_h.evidence:
            raise ValueError(f"New hypothesis {new_h.id} must have empty evidence on creation")
        graph.hypotheses.append(new_h)

    # Accumulate established facts.
    graph.established_facts.extend(update.new_established_facts)

    # Normalize likelihoods over active hypotheses only.
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    total = sum(h.likelihood for h in active)
    if total > 0:
        for h in active:
            h.likelihood = round(h.likelihood / total, 4)

    return graph
