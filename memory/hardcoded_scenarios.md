# Hardcoded Investigation Scenarios

Three scenarios are built into the mock runner (`src/scenarios.py`). Each represents a
distinct class of ML system failure with its own alert, metric fingerprint, log signatures,
and agent reasoning path. All three share a fourth fixture — a deliberately malformed
GraphUpdate — that tests the output validator's ability to reject hallucinated hypothesis IDs.

Run any scenario with:
```bash
PYTHONPATH=src python3 src/run_mock.py --scenario <name>
# Names: feature_drift, bad_deployment, label_corruption
```

---

## Scenario 1 — Feature Drift (Upstream Schema Change)

**Alert:** Model accuracy dropped from 0.91 to 0.76 over the past 6 hours. Prediction
confidence also degraded. No system errors visible on dashboard.

### What actually happened

The upstream `customer_events` database table silently dropped its `event_date` column.
The feature pipeline, instead of raising an error, fell back to using the row's
`insertion_timestamp` as a substitute. This produced wildly out-of-range values for the
`feature_age_days` feature — values up to 847 days when the model expects 0–365. The
pipeline ran 14,203 records through with 168 validation failures, but suppressed all of
them and filled with defaults. No alert was raised. The model kept making predictions on
corrupted inputs.

### Metric fingerprint

| Metric | Before | After | Signal |
|---|---|---|---|
| accuracy | 0.91 | 0.76 | ↓ 15pp |
| prediction_confidence | 0.84 | 0.61 | ↓ 23pp |
| latency_p99 | 139ms | 143ms | flat |
| error_rate | 0.002 | 0.002 | flat |

Both quality metrics degrade together. Infrastructure metrics are flat. This rules out
a deployment regression (which spikes error_rate) and label corruption (which leaves
confidence stable). The model is receiving bad inputs — it's trying hard, getting
confidently wrong answers.

### How the agent reasons through it

1. **query_metrics** → accuracy and confidence both down, no infra signal. Feature drift
   and label corruption are still candidates; bad deployment weakens because error_rate is flat.
2. **query_logs (inference_service)** → model v3.2.1 has been serving unchanged for 14 days,
   zero errors. Bad deployment ruled out entirely.
3. **query_logs (feature_pipeline)** → schema validation errors, `feature_age_days` spiked
   to 847.3, upstream `customer_events` missing `event_date`, 168 failures suppressed. Root
   cause confirmed. Agent spawns a refined hypothesis H5 with the specific column name and
   fallback behaviour described.

### Why this is hard

The failure is entirely silent from the outside. The dashboard shows no errors. The
inference service is healthy. Only the feature pipeline logs contain the smoking gun —
and even there, failures were suppressed. The agent has to actively query the right service
rather than react to an obvious alert.

---

## Scenario 2 — Bad Deployment (Feature Normalizer Shape Mismatch)

**Alert:** Model accuracy dropped from 0.91 to 0.73 over the past 7 hours. Error rate has
elevated significantly. A model deployment occurred earlier today.

### What actually happened

Model v3.3.0 was deployed at 14:58 UTC. The new model's feature normalizer (the scaler
that standardises inputs before feeding them to the model) was trained on 14 features.
But the serving pipeline had since grown to 28 features — a schema addition that wasn't
reflected in the training pipeline. When v3.3.0 started serving, every request sent 28
features to a scaler that expected 14. About 2% of requests raise a `ValueError` outright
(these count as errors). The remaining 98% silently receive mis-normalised inputs, causing
the model to produce degraded but non-erroring predictions. Auto-rollback was never
triggered because the error rate (1.9%) stayed below the 5% threshold.

### Metric fingerprint

| Metric | Before | After | Signal |
|---|---|---|---|
| accuracy | 0.91 | 0.73 | ↓ 18pp |
| prediction_confidence | 0.84 | 0.69 | ↓ 15pp |
| latency_p99 | 139ms | 158ms | ↑ 19ms (error handling overhead) |
| error_rate | 0.002 | 0.019 | ↑ 850% — KEY SIGNAL |

The error rate spike is the discriminating signal. Feature drift and label pipeline
failures don't cause serving errors. An 850% spike alongside an accuracy drop means
something in the model or serving code is actively throwing exceptions.

### How the agent reasons through it

1. **query_metrics** → error_rate spike is immediately visible. Bad deployment jumps to
   the top suspect. Feature drift and label corruption weaken but aren't ruled out yet.
2. **query_logs (inference_service)** → deployment event at 14:58 UTC, error rate warning
   4 minutes later, then explicit `ValueError: scaler expected input shape (1, 14), got
   (1, 28)` in the error logs. Deployment timestamp aligns exactly with the start of
   degradation. Feature drift (H1) and label corruption (H3) are ruled out. Bad deployment
   reaches 91% likelihood.
3. **Turn 3 (malformed fixture)** → the fixture tries to update hypotheses H99 and H88,
   which don't exist. The output validator rejects the update. Graph is unchanged at 91%.

### Why this is hard

The auto-rollback didn't fire (1.9% < 5% threshold), so the system kept running in a
degraded state indefinitely. The failure affects two classes of requests differently:
the ~2% that error visibly, and the ~98% that silently receive wrong normalisation. An
engineer looking only at accuracy might not notice the error spike. The agent's first tool
call revealing the error spike is what focuses the investigation.

---

## Scenario 3 — Label Pipeline Corruption (Join Key Misconfiguration)

**Alert:** Model accuracy dropped from 0.91 to 0.74 over the past 8 hours. No
infrastructure errors reported. Engineers unsure if the model degraded or if something
changed in the evaluation pipeline.

### What actually happened

The label pipeline — the service that matches prediction outcomes to ground-truth labels
and computes accuracy — had its join key changed from `session_id` to `request_id`.
The two IDs don't correspond to the same records. As a result, 38.8% of outcome labels
were assigned to the wrong predictions or left unmatched entirely, with unmatched
predictions automatically counted as incorrect. The label alignment rate dropped from ~99%
to 61.2%. The model itself is performing normally and was never changed. The apparent
17pp accuracy drop is a measurement artifact — it reflects how broken the evaluation
pipeline is, not how the model is performing.

### Metric fingerprint

| Metric | Before | After | Signal |
|---|---|---|---|
| accuracy | 0.91 | 0.74 | ↓ 17pp |
| prediction_confidence | 0.84 | 0.83 | **flat — KEY SIGNAL** |
| latency_p99 | 139ms | 140ms | flat |
| error_rate | 0.002 | 0.002 | flat |

The discriminating signal is that **prediction confidence is stable** despite a 17pp
accuracy drop. Confidence is the model's internal certainty about its own output — it
reflects what the model is seeing, not how it's being evaluated. If the model were
receiving bad features (feature drift) or running broken code (bad deployment), confidence
would drop alongside accuracy. Here, the model is as certain as ever. It's being marked
wrong on a test where someone changed the answer key.

### How the agent reasons through it

1. **query_metrics** → accuracy down 17pp, but prediction_confidence barely moved
   (0.84→0.83, −0.01). The agent explicitly notices this divergence in its reasoning:
   "stable confidence + accuracy drop = model is healthy, evaluation labels are wrong."
   Label corruption jumps to the top. Feature drift and bad deployment weaken significantly
   because both of those would cause confidence to drop.
2. **query_logs (label_pipeline)** → log at 07:14 UTC: "config updated to use 'request_id'
   as join key (previously 'session_id')". Warning at 07:22 UTC: 847 records unmatched.
   Error at 07:30 UTC: label alignment rate 61.2% (threshold 95%). Root cause confirmed
   in 2 tool calls. H1, H2, H4 all ruled out.
3. **Turn 3 (malformed fixture)** → same hallucinated-ID fixture as the other scenarios.
   Rejected by the validator.

### Why this is hard

This is the scenario most likely to cause a false alarm that wastes engineering time.
The dashboard shows a real accuracy drop. Without checking the label pipeline, a team
might assume the model regressed and start a root cause investigation of the training
pipeline, recent feature changes, or data quality — all of which are fine. The key
insight the agent uses — that prediction_confidence being stable while accuracy drops is
a fingerprint of evaluation infrastructure failure rather than model failure — requires
understanding the relationship between those two metrics.

---

## The Shared Malformed Fixture

Every scenario ends with a fixture that deliberately produces an invalid `GraphUpdate`:

- `likelihood_changes` references hypothesis ID `H99` (never created in any scenario)
- `hypotheses_to_rule_out` references hypothesis ID `H88` (also hallucinated)

The output validator (`src/output_validator.py`) catches both, returns `valid=False` with
specific error messages for each unknown ID, and the graph is not mutated. This fixture
tests that the validator acts as a reliable gate between model output and graph state —
a hallucinated hypothesis ID in the model's response causes a no-op, not a crash or
silent corruption.
