"""Pydantic models for the agent's structured belief state (Hypothesis Graph).

Spec: /docs/hypothesis-graph-spec.md
Design decisions: delta-based updates, evidence_type derived never model-set,
likelihoods normalized programmatically, per-update delta capped at ±0.25.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class FailureCategory(str, Enum):
    # Derived from chaos injection taxonomy (single source of truth).
    # Sync enforced by tests/test_taxonomy_sync.py.
    FEATURE_DRIFT                     = "feature_drift"
    BAD_DEPLOYMENT                    = "bad_deployment"
    UPSTREAM_SCHEMA_CHANGE            = "upstream_schema_change"
    INFRASTRUCTURE_LATENCY_SPIKE      = "infrastructure_latency_spike"
    MODEL_VERSION_ROLLBACK_REGRESSION = "model_version_rollback_regression"
    LABEL_PIPELINE_CORRUPTION         = "label_pipeline_corruption"
    TRAINING_SERVING_SKEW             = "training_serving_skew"
    DATA_FRESHNESS_DEGRADATION        = "data_freshness_degradation"
    FEATURE_ENCODING_BUG              = "feature_encoding_bug"
    GRADUAL_CONCEPT_DRIFT             = "gradual_concept_drift"
    MODEL_CALIBRATION_DRIFT           = "model_calibration_drift"
    SHADOW_MODE_LEAK                  = "shadow_mode_leak"
    FEATURE_PIPELINE_PARTIAL_FAILURE  = "feature_pipeline_partial_failure"
    DELAYED_LABEL_FEEDBACK_SHIFT      = "delayed_label_feedback_shift"
    CASCADING_UPSTREAM_FAILURE        = "cascading_upstream_failure"
    MODEL_STALENESS                   = "model_staleness"
    FEATURE_IMPORTANCE_INVERSION      = "feature_importance_inversion"
    COMPOUND_DRIFT_PLUS_DEPLOYMENT    = "compound_drift_plus_deployment"


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
    root_cause_category: Optional[FailureCategory] = None
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
    """Structured delta the model emits after each tool call.

    The `action` field determines which mutation is performed:
    - "update": attach evidence to an existing hypothesis (`current_focus` required)
    - "create": add a new hypothesis and attach the triggering evidence to it
    - "merge": the proposed new hypothesis is semantically equivalent to an existing
               one — attach evidence to `merge_into_id` instead of creating a duplicate
    """
    action: Literal["create", "update", "merge"]
    # "update" action
    current_focus: Optional[str] = None
    # "merge" action
    merge_into_id: Optional[str] = None
    # "create" action — no FailureCategory; categories are an eval-side concern
    new_hypothesis_description: Optional[str] = None
    new_hypothesis_severity: Optional[str] = None   # plain str avoids type coupling
    new_hypothesis_initial_likelihood: Optional[float] = None
    # Common fields (all actions)
    new_evidence: EvidenceInput
    likelihood_changes: Dict[str, float]
    hypotheses_to_rule_out: List[str] = []
    new_established_facts: List[str] = []
    next_experiment_rationale: str


_MAX_LIKELIHOOD_DELTA = 0.25  # cap per spec §5.3


def _next_hypothesis_id(graph: HypothesisGraph) -> str:
    return f"H{len(graph.hypotheses) + 1}"


def update_graph(graph: HypothesisGraph, update: GraphUpdate) -> HypothesisGraph:
    """Apply a GraphUpdate delta in-place. Returns the mutated graph."""
    existing_ids = {h.id for h in graph.hypotheses}

    if update.action == "create":
        new_id = _next_hypothesis_id(graph)
        severity = Severity(update.new_hypothesis_severity or "medium")
        new_hyp = Hypothesis(
            id=new_id,
            description=update.new_hypothesis_description or "",
            likelihood=max(0.0, min(1.0, update.new_hypothesis_initial_likelihood or 0.15)),
            severity=severity,
            status=HypothesisStatus.ACTIVE,
        )
        graph.hypotheses.append(new_hyp)
        focus_id = new_id
    elif update.action == "update":
        if not update.current_focus or update.current_focus not in existing_ids:
            raise ValueError(f"update action requires a valid current_focus; got {update.current_focus!r}")
        focus_id = update.current_focus
    elif update.action == "merge":
        if not update.merge_into_id or update.merge_into_id not in existing_ids:
            raise ValueError(f"merge action requires a valid merge_into_id; got {update.merge_into_id!r}")
        focus_id = update.merge_into_id
    else:
        raise ValueError(f"Unknown action: {update.action!r}")

    graph.current_focus = focus_id

    # Attach evidence to the focused hypothesis.
    evidence = Evidence.from_input(update.new_evidence)
    for h in graph.hypotheses:
        if h.id == focus_id:
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

    # Accumulate established facts.
    graph.established_facts.extend(update.new_established_facts)

    # Normalize likelihoods over active hypotheses only.
    active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
    total = sum(h.likelihood for h in active)
    if total > 0:
        for h in active:
            h.likelihood = round(h.likelihood / total, 4)

    return graph
