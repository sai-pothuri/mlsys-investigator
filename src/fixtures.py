"""Canned GraphUpdate fixtures for mock ReAct loop testing.

Four fixtures covering the four cases the Output Validator must handle:
  1. NORMAL_EVIDENCE    — straightforward evidence update, likelihoods shift
  2. RULED_OUT          — hypothesis eliminated, likelihood reallocated
  3. SURPRISING_RESULT  — unexpected finding triggers a new hypothesis mid-run
  4. HALLUCINATED_ID    — model invents "H99"; validator must catch and reject

Each fixture also carries the simulated thought and observation text so the
mock runner can print a readable thought/action/observation trace.
"""

from dataclasses import dataclass

from hypothesis_graph import (
    EvidenceInput,
    FailureCategory,
    GraphUpdate,
    Hypothesis,
    HypothesisStatus,
    Severity,
)


@dataclass(frozen=True)
class Fixture:
    label: str
    thought: str            # model's reasoning before the tool call
    tool_called: str        # which data tool was called
    tool_inputs: dict       # inputs passed to the tool dispatcher
    observation: str        # what the tool returned (human-readable summary)
    current_focus: str      # hypothesis ID to set as current_focus before applying
    update: GraphUpdate
    expect_valid: bool = True


FIXTURES: list[Fixture] = [

    # ── 1. Normal evidence update ──────────────────────────────────────────────
    # query_metrics returned accuracy -15pp and prediction_confidence -23pp.
    # Both degradations together are consistent with feature_drift (H1) and
    # inconsistent with bad_deployment (H2) where we'd normally see error_rate
    # spike too.  Latency and error_rate are flat, which weakly contradicts H2.
    Fixture(
        label="NORMAL EVIDENCE UPDATE",
        thought=(
            "I'll start with metrics to quantify the degradation and check "
            "whether the pattern is consistent with a model quality issue "
            "or an infrastructure failure."
        ),
        tool_called="query_metrics",
        tool_inputs={
            "metric_names": ["accuracy", "prediction_confidence", "latency_p99", "error_rate"],
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "comparison_window": {"start": "2026-06-24T20:00:00Z", "end": "2026-06-25T02:00:00Z"},
        },
        observation=(
            "accuracy: 0.91 → 0.76 (−0.15), prediction_confidence: 0.84 → 0.61 (−0.23). "
            "latency_p99 and error_rate are flat. Both quality metrics degraded "
            "simultaneously with no infra signal — consistent with bad input data."
        ),
        current_focus="H1",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_metrics",
                observation=(
                    "accuracy dropped 0.91→0.76 (−15pp) and prediction_confidence "
                    "0.84→0.61 (−23pp) over the 6-hour window; latency_p99 and "
                    "error_rate unchanged — no infra signal."
                ),
                supports=True,
                confidence_delta=0.20,
            ),
            likelihood_changes={"H1": +0.20, "H2": -0.08, "H3": -0.06, "H4": -0.04},
            hypotheses_to_rule_out=[],
            new_hypotheses=[],
            new_established_facts=[
                "Accuracy degraded 15pp and prediction_confidence 23pp over 6 hours.",
                "Latency and error_rate are flat — no infrastructure degradation.",
            ],
            next_experiment_rationale=(
                "Flat error_rate weakens bad_deployment (H2); I'll next query "
                "inference_service logs to check for deployment events and confirm."
            ),
        ),
        expect_valid=True,
    ),

    # ── 2. Hypothesis ruled out ────────────────────────────────────────────────
    # query_logs on inference_service returned clean logs: serving model v3.2.1,
    # no deployment in 48h, no errors.  This definitively rules out H2.
    Fixture(
        label="HYPOTHESIS RULED OUT",
        thought=(
            "The flat error_rate already weakened bad_deployment. Now I'll check "
            "inference_service logs directly to see if there was a recent deployment "
            "or any serving-layer error that could explain the accuracy drop."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning"],
        },
        observation=(
            "inference_service logs are clean: serving model v3.2.1 unchanged, "
            "no deployment events in past 48h, throughput nominal at 481 req/s. "
            "No serving-layer errors whatsoever."
        ),
        current_focus="H2",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "inference_service: model v3.2.1 unchanged, no deployments in "
                    "past 48h, throughput 481 req/s, zero errors — serving layer clean."
                ),
                supports=False,
                confidence_delta=-0.30,
            ),
            likelihood_changes={"H2": -0.25},
            hypotheses_to_rule_out=["H2"],
            new_hypotheses=[],
            new_established_facts=[
                "No model or code deployments in the past 48 hours.",
                "inference_service is healthy — not the failure origin.",
            ],
            next_experiment_rationale=(
                "H2 is ruled out. H1 (feature_drift) and H3 (label_pipeline) are "
                "the remaining contenders. I'll query feature_pipeline logs next — "
                "it's the most direct path to confirming or refuting feature_drift."
            ),
        ),
        expect_valid=True,
    ),

    # ── 3. Surprising result → new hypothesis ─────────────────────────────────
    # query_logs on feature_pipeline returned schema validation errors AND
    # a message revealing the upstream customer_events table lost its event_date
    # column, causing feature_age_days to be computed from insertion_timestamp.
    # This is surprising: it's not generic feature_drift, it's a specific upstream
    # schema breakage.  We add H5 to track the specific mechanism.
    Fixture(
        label="SURPRISING RESULT — NEW HYPOTHESIS",
        thought=(
            "H1 (feature_drift) is the highest-likelihood active hypothesis. "
            "I'll query feature_pipeline logs to look for validation errors or "
            "anomalies that would directly explain why features shifted."
        ),
        tool_called="query_logs",
        tool_inputs={
            "service": "feature_pipeline",
            "time_range": {"start": "2026-06-25T02:00:00Z", "end": "2026-06-25T08:00:00Z"},
            "severity": ["error", "warning", "info"],
        },
        observation=(
            "feature_pipeline logs contain schema validation ERRORS: "
            "'feature_age_days' spiked to 847.3 (expected 0–365), "
            "'feature_transaction_count_30d' has 23% nulls. "
            "Critical: upstream 'customer_events' table is MISSING column 'event_date'; "
            "pipeline fell back to row insertion_timestamp — this produces wildly "
            "wrong feature_age_days values. 168 validation failures suppressed silently."
        ),
        current_focus="H1",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_logs",
                observation=(
                    "feature_pipeline: schema validation errors — feature_age_days "
                    "values up to 847.3 (expected ≤365); upstream customer_events "
                    "table missing event_date column, pipeline silently falling back "
                    "to insertion_timestamp; 168 failures suppressed with defaults."
                ),
                supports=True,
                confidence_delta=0.25,
            ),
            # H5 is a new hypothesis — its initial likelihood is set on the
            # Hypothesis object directly; referencing it here before insertion
            # would be a hallucinated ID and fail validation.
            likelihood_changes={"H1": +0.25, "H3": -0.15},
            hypotheses_to_rule_out=[],
            new_hypotheses=[
                Hypothesis(
                    id="H5",
                    root_cause_category=FailureCategory.FEATURE_DRIFT,
                    description=(
                        "Upstream 'customer_events' table dropped column 'event_date'; "
                        "feature_pipeline silently fell back to insertion_timestamp, "
                        "producing out-of-range feature_age_days values (up to 847 days)"
                    ),
                    likelihood=0.40,
                    severity=Severity.CRITICAL,
                    status=HypothesisStatus.ACTIVE,
                    evidence=[],  # must be empty on creation per spec §3.3
                ),
            ],
            new_established_facts=[
                "feature_pipeline has active schema validation errors on feature_age_days.",
                "Upstream customer_events table is missing event_date column.",
                "Pipeline is silently using insertion_timestamp as fallback, corrupting feature_age_days.",
            ],
            next_experiment_rationale=(
                "H5 is now the most specific and best-supported hypothesis. "
                "Evidence is strong enough to call stop_investigation."
            ),
        ),
        expect_valid=True,
    ),

    # ── 4. Malformed / hallucinated ID ────────────────────────────────────────
    # The model hallucinates hypothesis ID "H99" in likelihood_changes.
    # It also references a non-existent tool "query_deployment_history" which
    # isn't wired in this MVP (would pass in full implementation, but included
    # here to show the validator catches unregistered tool names too).
    # Output Validator must reject this update entirely; graph state must not change.
    Fixture(
        label="MALFORMED — HALLUCINATED HYPOTHESIS ID",
        thought=(
            "I want to update the likelihood of H99 which I believe I created "
            "earlier, and record deployment evidence."
        ),
        tool_called="query_deployment_history",  # not registered in MVP tool set
        tool_inputs={
            "service": "inference_service",
            "time_range": {"start": "2026-06-24T08:00:00Z", "end": "2026-06-25T08:00:00Z"},
        },
        observation="(this turn should be rejected before the update is applied)",
        current_focus="H1",
        update=GraphUpdate(
            new_evidence=EvidenceInput(
                tool_called="query_deployment_history",  # unregistered in MVP
                observation="No deployments found.",
                supports=False,
                confidence_delta=-0.10,
            ),
            likelihood_changes={
                "H1": -0.05,
                "H99": +0.40,  # hallucinated — H99 does not exist
            },
            hypotheses_to_rule_out=["H88"],  # also hallucinated
            new_hypotheses=[],
            new_established_facts=[],
            next_experiment_rationale="Updating H99 based on deployment check.",
        ),
        expect_valid=False,
    ),
]
