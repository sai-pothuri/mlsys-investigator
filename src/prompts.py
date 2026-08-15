"""Prompt registry for the MLSys Investigator agent.

All LLM-facing strings live here, named and versioned. agent.py and tools.py
import from this module — no raw string literals for prompts elsewhere.

To update a prompt: edit its `content` and bump its `version`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    content: str


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = Prompt(
    name="system_prompt",
    version="v2",
    content="""\
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
  - likelihood_changes MUST be empty ({}) on a create action — the new
    hypothesis does not have an ID yet. Set its weight via
    new_hypothesis_initial_likelihood only.

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
""",
)


# ── User-turn templates ───────────────────────────────────────────────────────

INITIAL_USER_MESSAGE = Prompt(
    name="initial_user_message",
    version="v1",
    content=(
        "{graph_context}\n\n"
        "Begin your investigation. Call tools to gather evidence, "
        "then call stop_investigation with your final diagnosis."
    ),
)

# ── Tool-result messages ──────────────────────────────────────────────────────

CONVERGENCE_MESSAGE = Prompt(
    name="convergence_message",
    version="v1",
    content=(
        "CONVERGENCE ({reason}): the evidence strongly "
        "favours one hypothesis. Call stop_investigation now."
    ),
)

BUDGET_EXHAUSTED_MESSAGE = Prompt(
    name="budget_exhausted_message",
    version="v1",
    content=(
        "Tool call discarded — all {budget} budget units already consumed "
        "this iteration. Call stop_investigation now."
    ),
)

# ── Tool descriptions ─────────────────────────────────────────────────────────

STOP_INVESTIGATION_DESCRIPTION = Prompt(
    name="stop_investigation_description",
    version="v1",
    content=(
        "Terminate the investigation and submit the final diagnosis. "
        "Call when one hypothesis is clearly dominant (likelihood > 0.60) "
        "or when the tool budget is nearly exhausted."
    ),
)

ROOT_CAUSE_CATEGORY_DESCRIPTION = Prompt(
    name="root_cause_category_description",
    version="v2",
    content=(
        "Pick the single most specific category. Definitions:\n"
        "• feature_drift — Input feature value distributions shifted in production; no code or infra change.\n"
        "• bad_deployment — A new deployment (model/code/config) introduced a bug; errors/latency coincide with the deploy event.\n"
        "• upstream_schema_change — An upstream data source changed its schema (field added, removed, renamed, or retyped), breaking the feature pipeline.\n"
        "• infrastructure_latency_spike — Underlying infra degradation (disk I/O saturation, CPU/memory pressure, network congestion) on serving or feature-store hosts; latency spikes with no code change.\n"
        "• model_version_rollback_regression — Rolling back to an older model version caused a regression (e.g., the old model's expected feature schema no longer matches the current pipeline).\n"
        "• label_pipeline_corruption — Ground-truth labels are corrupted, systematically wrong, or arriving with incorrect values.\n"
        "• training_serving_skew — Feature transformations applied at training time differ from those applied at serving time; not caused by a rollback.\n"
        "• data_freshness_degradation — Features arrive stale or with increased end-to-end delay, making predictions on outdated signals.\n"
        "• feature_encoding_bug — A bug in feature encoding/normalization code (wrong scaler, dtype mismatch, missing imputation) introduced by a code change.\n"
        "• gradual_concept_drift — The statistical relationship between features and the target label has shifted over time due to real-world change; no sudden trigger.\n"
        "• model_calibration_drift — Model confidence scores are miscalibrated relative to empirical accuracy; accuracy and calibration diverge without other obvious signals.\n"
        "• shadow_mode_leak — Shadow/canary traffic or logged predictions bleed into production metrics, distorting observed performance.\n"
        "• feature_pipeline_partial_failure — The feature pipeline process itself is degraded (slow, dropping records, returning NaNs/nulls) due to a software or logic issue — distinct from underlying infra saturation.\n"
        "• delayed_label_feedback_shift — The label feedback loop has a new or increased delay, causing evaluation metrics to lag reality.\n"
        "• cascading_upstream_failure — A failure in one upstream service cascades to affect multiple downstream components simultaneously.\n"
        "• model_staleness — The model was trained on old data and the world has changed since training, without a clear concept-drift signal.\n"
        "• feature_importance_inversion — The relative predictive importance of features has inverted; previously important features become noise.\n"
        "• compound_drift_plus_deployment — A deployment and feature/data drift are co-occurring simultaneously, making root cause ambiguous without careful isolation."
    ),
)

ALTERNATIVE_CATEGORIES_DESCRIPTION = Prompt(
    name="alternative_categories_description",
    version="v1",
    content=(
        "Ranked list of up to 2 runner-up category IDs you considered but ruled below "
        "root_cause_category. Omit if only one hypothesis was viable. "
        "Used for Top-3 evaluation."
    ),
)
