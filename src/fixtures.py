"""Canned GraphUpdate fixtures for mock ReAct loop testing.

Four fixtures covering the four cases the Output Validator must handle,
starting from an empty hypothesis graph:
  1. CREATE_FIRST    — first tool result, must use action="create"; likelihood_changes={}
  2. CREATE_SECOND   — second hypothesis added; evidence contradicts it; no same-turn rule_out
  3. CREATE_THEN_PRUNE — third hypothesis (surprising result); H1/H2 already exist so
                          likelihood_changes and hypotheses_to_rule_out can reference them
  4. HALLUCINATED_ID — model invents "H99"; validator must catch and reject

Each fixture carries the simulated thought and observation text so the
mock runner can print a readable thought/action/observation trace.
"""

from dataclasses import dataclass

from hypothesis_graph import EvidenceInput, GraphUpdate


@dataclass(frozen=True)
class Fixture:
    label: str
    thought: str            # model's reasoning before the tool call
    tool_called: str        # which data tool was called
    tool_inputs: dict       # inputs passed to the tool dispatcher
    observation: str        # what the tool returned (human-readable summary)
    current_focus: str      # for display only; action routing is in update.action
    update: GraphUpdate
    expect_valid: bool = True


FIXTURES: list[Fixture] = [

    # ── 1. First tool result — action="create" required (graph is empty) ──────
    # query_metrics returned accuracy -15pp and prediction_confidence -23pp with
    # flat latency and error_rate.  Both quality metrics degraded together with
    # no infrastructure signal — consistent with OOD inputs (feature drift).
    # Because the graph is empty, action must be "create".
    Fixture(
        label="CREATE FIRST HYPOTHESIS",
        thought=(
            "I'll start with metrics to quantify the degradation and look for "
            "signals that distinguish a data issue from an infrastructure failure. "
            "The graph is empty — I'll create H1 from whatever the metrics tell me."
        ),
        tool_called="query_metrics",
        tool_inputs={
            "metric_names": ["accuracy", "prediction_confidence", "latency_p99", "error_rate"],
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "comparison_window": {"start": "2026-06-24T20:00:00Z", "end": "2026-06-25T02:00:00Z"},
        },
        observation=(
            "accuracy: 0.91→0.76 (−0.15), prediction_confidence: 0.84→0.61 (−0.23). "
            "latency_p99 and error_rate are flat. Both quality metrics degraded "
            "simultaneously with no infra signal — consistent with bad input data "
            "rather than a deployment or infrastructure failure."
        ),
        current_focus="H1",  # display only — H1 is being created this turn
        update=GraphUpdate(
            action="create",
            new_hypothesis_description=(
                "Input feature distributions have shifted from the training distribution — "
                "model receives OOD inputs, causing simultaneous accuracy and confidence "
                "degradation with flat error_rate and latency"
            ),
            new_hypothesis_severity="high",
            new_hypothesis_initial_likelihood=0.60,
            new_evidence=EvidenceInput(
                tool_called="query_metrics",
                observation=(
                    "accuracy dropped 0.91→0.76 (−15pp) and prediction_confidence "
                    "0.84→0.61 (−23pp); latency_p99 and error_rate unchanged — "
                    "quality drop with no infra signal points to data quality issue."
                ),
                supports=True,
                confidence_delta=0.20,
            ),
            likelihood_changes={},  # only hypothesis in graph — nothing else to adjust
            new_established_facts=[
                "Accuracy degraded 15pp and prediction_confidence 23pp over 6 hours.",
                "Latency and error_rate flat — no infrastructure degradation.",
            ],
            next_experiment_rationale=(
                "H1 created. Flat error_rate weakens bad_deployment; I'll query "
                "inference_service logs to check for deployments and confirm."
            ),
        ),
        expect_valid=True,
    ),

    # ── 2. Second hypothesis created — evidence against it, no same-turn rule_out ──
    # query_logs on inference_service: clean logs, no deployment in 48h, no errors.
    # This contradicts the bad_deployment hypothesis we're creating.
    # Important: we CANNOT rule out H2 in the same turn we create it — the validator
    # checks existing_ids before the create action runs, so H2 isn't there yet.
    Fixture(
        label="CREATE SECOND HYPOTHESIS (evidence immediately contradicts it)",
        thought=(
            "Flat error_rate weakens bad_deployment. Checking inference_service logs "
            "for recent deployments or serving errors. I'll create a bad_deployment "
            "hypothesis so I have something to track evidence against."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "inference_service: serving model v3.2.1 unchanged, no deployments in "
            "past 48h, throughput 481 req/s, zero errors — serving layer clean."
        ),
        current_focus="H2",  # display only — H2 is being created this turn
        update=GraphUpdate(
            action="create",
            new_hypothesis_description=(
                "A recent model or code deployment introduced a regression — "
                "scaler mismatch, wrong model version, or broken preprocessing"
            ),
            new_hypothesis_severity="high",
            new_hypothesis_initial_likelihood=0.10,
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "inference_service: model v3.2.1 unchanged, no deployments in "
                    "past 48h, throughput 481 req/s, zero errors — serving layer clean."
                ),
                supports=False,
                confidence_delta=-0.25,
            ),
            # H1 exists from Fixture 1 — valid to reference in likelihood_changes
            likelihood_changes={"H1": +0.05},
            new_established_facts=[
                "No model or code deployments in the past 48 hours.",
                "inference_service is healthy — not the failure origin.",
            ],
            next_experiment_rationale=(
                "H2 created but contradicted immediately. H1 is the working hypothesis. "
                "Querying feature_pipeline logs to look for the drift mechanism."
            ),
        ),
        expect_valid=True,
    ),

    # ── 3. Surprising result → third hypothesis, rule_out H2 (now it exists) ──────
    # query_logs on feature_pipeline: schema validation errors + upstream column removal.
    # This is the specific mechanism behind H1 — create H3 for precision.
    # H1 and H2 both exist now, so likelihood_changes and hypotheses_to_rule_out
    # can reference both.
    Fixture(
        label="SURPRISING RESULT — CREATE H3, RULE OUT H2",
        thought=(
            "H1 (feature drift) is highest-likelihood. I'll query feature_pipeline "
            "logs to look for validation errors that explain why features shifted."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "feature_pipeline",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "feature_pipeline logs: schema validation ERRORS — 'feature_age_days' spiked "
            "to 847.3 (expected 0–365). CRITICAL: upstream 'customer_events' table is "
            "MISSING column 'event_date'; pipeline fell back to row insertion_timestamp. "
            "168 validation failures suppressed silently."
        ),
        current_focus="H3",  # display only — H3 is being created this turn
        update=GraphUpdate(
            action="create",
            new_hypothesis_description=(
                "Upstream 'customer_events' table dropped column 'event_date'; "
                "feature_pipeline silently fell back to insertion_timestamp, "
                "producing out-of-range feature_age_days values (up to 847 days)"
            ),
            new_hypothesis_severity="critical",
            new_hypothesis_initial_likelihood=0.40,
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "feature_pipeline: schema validation errors — feature_age_days up "
                    "to 847.3 (expected ≤365); upstream customer_events missing "
                    "event_date column; pipeline silently using insertion_timestamp; "
                    "168 failures suppressed with defaults."
                ),
                supports=True,
                confidence_delta=0.25,
            ),
            # H1 and H2 both exist from Fixtures 1 and 2 — valid references
            likelihood_changes={"H1": +0.25, "H2": -0.25},
            hypotheses_to_rule_out=["H2"],  # H2 exists from Fixture 2 ✓
            new_established_facts=[
                "feature_pipeline has active schema validation errors on feature_age_days.",
                "Upstream customer_events table is missing event_date column.",
                "Pipeline silently using insertion_timestamp as fallback.",
            ],
            next_experiment_rationale=(
                "H3 is now the most specific and best-supported hypothesis. "
                "Evidence strong enough to call stop_investigation."
            ),
        ),
        expect_valid=True,
    ),

    # ── 4. Malformed / hallucinated ID ────────────────────────────────────────
    # The model hallucinates hypothesis ID "H99" in likelihood_changes.
    # H1 exists (created in Fixture 1), so current_focus="H1" is valid.
    # H99 and H88 don't exist in any fixture — validator must reject the whole update.
    Fixture(
        label="MALFORMED — HALLUCINATED HYPOTHESIS ID",
        thought=(
            "I want to update the likelihood of H99 which I believe I created "
            "earlier, and record deployment evidence."
        ),
        tool_called="query_deployment_history",
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-24T08:00:00Z", "end": "2026-06-25T08:00:00Z"},
        },
        observation="(this turn should be rejected before the update is applied)",
        current_focus="H1",
        update=GraphUpdate(
            action="update",
            current_focus="H1",  # H1 exists ✓ — but H99/H88 will cause rejection
            new_evidence=EvidenceInput(
                tool_called="query_deployment_history",
                observation="No deployments found.",
                supports=False,
                confidence_delta=-0.10,
            ),
            likelihood_changes={
                "H1": -0.05,
                "H99": +0.40,  # hallucinated — H99 does not exist
            },
            hypotheses_to_rule_out=["H88"],  # also hallucinated
            next_experiment_rationale="Updating H99 based on deployment check.",
        ),
        expect_valid=False,
    ),
]
