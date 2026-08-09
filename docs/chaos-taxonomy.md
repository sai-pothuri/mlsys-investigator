# Chaos Injection Taxonomy

Single source of truth for `FailureCategory` enum values and chaos injection scenarios.
Every entry here must have a 1:1 corresponding value in `FailureCategory` (enforce via CI).

---

## Format

Each entry:
- **`id`** — slug → becomes the `FailureCategory` enum value
- **`tier`** — easy / medium / hard
- **`description`** — what failed and how
- **`primary_signal`** — the tool/metric that most directly exposes it
- **`distinguishing_signal`** — what rules out near-miss categories

---

## Easy Tier (~5 scenarios)

Single clear signal; one tool call is usually sufficient to diagnose.

### 1. `feature_drift`
- **Description:** Input feature distributions have shifted significantly from the training distribution, causing model accuracy to drop.
- **Primary signal:** `query_feature_distributions` — high PSI (>0.25) on one or more features; `prediction_confidence` drops alongside `accuracy`.
- **Distinguishing signal:** Both `accuracy` and `prediction_confidence` degrade together (model is uncertain, not just wrong). No deployment event in the window. Latency and error rate unchanged.
- **Injection:** Shift a feature's distribution (e.g. `login_failure_rate` U(0,0.3) → U(0.3,0.7)) for a sub-range.

### 2. `bad_deployment`
- **Description:** A recent model or code deployment introduced a regression — shape mismatch, wrong version, broken preprocessing.
- **Primary signal:** `query_deployment_history` — deploy event coincides with degradation onset; `query_logs` shows hard errors (e.g. `ValueError: scaler shape mismatch`); `error_rate` spikes.
- **Distinguishing signal:** `error_rate` spike (not just accuracy drop). Timestamp of degradation aligns exactly with deploy event. Rolling back resolves it.
- **Injection:** Insert a deployment event with a mis-versioned scaler (14 features vs. 28 in production).

### 3. `upstream_schema_change`
- **Description:** An upstream data source dropped, renamed, or retyped a column; the feature pipeline silently fell back to a proxy value with a very different distribution.
- **Primary signal:** `query_logs` on `feature_pipeline` — schema validation errors, "missing column, falling back to X"; `feature_age_days` or similar shows extreme out-of-range values.
- **Distinguishing signal:** Log errors reference a specific upstream source and column name. No deployment event. `prediction_confidence` drops (model sees garbage inputs).
- **Injection:** Modify the upstream schema (remove `event_date` column) so the pipeline falls back to `insertion_timestamp`.

### 4. `infrastructure_latency_spike`
- **Description:** Serving infrastructure degraded (CPU throttle, disk I/O, network congestion), causing latency to spike and request timeouts to increase.
- **Primary signal:** `query_metrics` — `latency_p99` spikes dramatically; `throughput` drops; `error_rate` increases; `accuracy` is unaffected (surviving requests are fine).
- **Distinguishing signal:** `accuracy` remains at baseline — the model logic is correct but slow. `latency_p99` / `error_rate` diverge from accuracy.
- **Injection:** Insert artificial latency in the generator (requests start timing out, producing error log entries and dropping throughput).

### 5. `model_version_rollback_regression`
- **Description:** A rollback to a previous model version after an incident inadvertently reverted to a version trained on stale data, causing accuracy to drop.
- **Primary signal:** `query_deployment_history` — `is_rollback=True` event; `query_metrics` — accuracy drop starts at rollback timestamp.
- **Distinguishing signal:** Deployment event is a rollback (`is_rollback=True`). Error rate does not spike. Feature distributions are normal.
- **Injection:** Mark a deployment event as a rollback to an older, lower-performing model checkpoint.

---

## Medium Tier (~8 scenarios)

Requires 2–3 tools; some ambiguity between categories; agent must reason across evidence.

### 6. `label_pipeline_corruption`
- **Description:** The label pipeline is producing corrupted ground-truth labels (wrong join key, wrong aggregation window, label leakage), making an accurate model appear inaccurate.
- **Primary signal:** `query_logs` on `label_pipeline` — join alignment rate drops, label match errors; `query_metrics` — `accuracy` drops but `prediction_confidence` stays high (model is confident, inputs are clean).
- **Distinguishing signal:** `prediction_confidence` is stable while `accuracy` drops — the key discriminator. Feature distributions are clean. No deployment event.
- **Injection:** Change label pipeline join key from `session_id` to `request_id`, misaligning 38% of labels.

### 7. `training_serving_skew`
- **Description:** Feature transformations differ between training and serving (different scaler, imputer, encoding), so the model receives inputs outside its training manifold.
- **Primary signal:** `query_feature_distributions` — individual features look fine (reasonable values), but `prediction_confidence` is degraded; `query_code_diffs` — shows divergence in preprocessing code between training pipeline and serving pipeline.
- **Distinguishing signal:** Feature values are individually plausible (low PSI per feature), but model underperforms. Requires code diff to discover the transform divergence. No upstream schema errors.
- **Injection:** Introduce different normalization constants in serving vs. training pipeline.

### 8. `data_freshness_degradation`
- **Description:** A data pipeline is delayed or stalled, causing the feature store to serve stale feature values. Features are technically present but reflect behavior from hours/days ago.
- **Primary signal:** `query_feature_distributions` — timestamps of feature values are lagging; `query_logs` on `feature_pipeline` — "batch job delayed" or "last successful run >Xh ago".
- **Distinguishing signal:** Feature values are stale but not invalid (PSI may be low; the pipeline ran, but slowly). Log messages show scheduling delay, not schema errors.
- **Injection:** Stall the feature pipeline for 12 hours so features are 12h stale when predictions are made.

### 9. `feature_encoding_bug`
- **Description:** A categorical feature is encoded incorrectly (wrong mapping, missing category defaulting to wrong index), causing systematic errors on a subset of users.
- **Primary signal:** `query_feature_distributions` — distribution of encoded feature values looks off (unexpected mode or missing values); `query_logs` — encoding warnings for unseen categories.
- **Distinguishing signal:** Subset of requests are affected (those with the unseen category). Overall accuracy drops moderately. Other features clean.
- **Injection:** Introduce a new category value in `product_category` not present in training encoder; defaults to 0 (wrong).

### 10. `gradual_concept_drift`
- **Description:** The underlying data-generating process has shifted slowly over weeks; the model was trained on old behavior and no longer reflects current user behavior.
- **Primary signal:** `query_metrics` — accuracy decays slowly over days, not a sharp drop; `query_feature_distributions` — moderate PSI (0.1–0.2) across several features; no deployment event.
- **Distinguishing signal:** Gradual trend (not a step function). Multiple features show moderate drift (not one outlier). No upstream schema errors, no deploy event.
- **Injection:** Shift multiple feature distributions gradually across a 7-day window (ramp rather than step).

### 11. `model_calibration_drift`
- **Description:** The model's probability scores are no longer calibrated — predicted probabilities don't match empirical frequencies, causing downstream systems using thresholds to misfire.
- **Primary signal:** `query_metrics` — `prediction_confidence` is high but `accuracy` is lower than expected for that confidence level; calibration curve shows overconfidence.
- **Distinguishing signal:** Confidence is high but accuracy is low (inverted from label corruption where confidence is high and accuracy should be high). Features clean.
- **Injection:** Shift prior probability in training data so the model learned a different base rate than serving sees.

### 12. `shadow_mode_leak`
- **Description:** A shadow model or experiment flag was accidentally promoted to production, replacing the primary model for a subset of traffic.
- **Primary signal:** `query_logs` on `inference_service` — shows two model versions serving requests; `query_metrics` — accuracy bimodal or degraded on specific request subsets.
- **Distinguishing signal:** Log entries show two different model versions (`v3.2.1` and `v4.0.0-shadow`) handling requests. Deployment history shows no official promotion.
- **Injection:** Add log entries showing shadow model handling 20% of traffic; shadow model has lower accuracy.

### 13. `feature_pipeline_partial_failure`
- **Description:** One of multiple feature computation jobs failed silently for a subset of features; those features default to fill values, degrading model performance partially.
- **Primary signal:** `query_feature_distributions` — one or more features show near-zero variance (all-fill-value); `query_logs` — feature pipeline job partial failure for specific features.
- **Distinguishing signal:** Only a subset of features are affected (degenerate distributions), not all. Other features normal. Error rate unchanged.
- **Injection:** Force `support_tickets_90d` feature to constant 0 (fill value) for all requests in a window.

---

## Hard Tier (~5 scenarios)

Requires cross-evidence reasoning, causal chain tracing, or delayed/indirect signals.

### 14. `delayed_label_feedback_shift`
- **Description:** The label pipeline's 24h delay means accuracy metrics reflect yesterday's model performance; a fix deployed today won't show accuracy improvement for 24h, making a correct fix look like it failed.
- **Primary signal:** `query_metrics` — accuracy appears unchanged despite a recent "fix" deployment; `query_deployment_history` — fix was deployed <24h ago.
- **Distinguishing signal:** The degradation event is >24h old but fix is <24h old. Requires reasoning about label delay to conclude the fix hasn't had time to propagate to accuracy metrics.
- **Injection:** Inject a real fix deployment but ensure the accuracy improvement only appears in metrics after a 24h delay.

### 15. `cascading_upstream_failure`
- **Description:** Two unrelated upstream services failed simultaneously; each alone would cause only minor degradation, but their joint failure causes a severe accuracy drop.
- **Primary signal:** Multiple tools needed — `query_logs` shows errors in both `feature_pipeline` and `label_pipeline`; `query_feature_distributions` shows drift; `query_metrics` shows accuracy collapse worse than either failure alone would predict.
- **Distinguishing signal:** No single cause explains the magnitude of the drop. Requires synthesizing evidence from at least 3 tools to reconstruct the causal chain.
- **Injection:** Simultaneously inject feature pipeline schema error and label pipeline join corruption.

### 16. `model_staleness`
- **Description:** The model was trained months ago and user behavior has shifted; no pipeline failure or deployment event, just gradual performance erosion that recently crossed an alert threshold.
- **Primary signal:** `query_metrics` — accuracy degrading over weeks (available in historical window); `query_feature_distributions` — broad moderate drift across most features; `query_deployment_history` — last model retrain was >60 days ago.
- **Distinguishing signal:** No single feature shows extreme PSI. No errors. No deploy events. Historical metrics show slow trend, not a step function. Requires long-window metrics comparison.
- **Injection:** Use a model trained on 90-day-old data against current feature distributions that have evolved.

### 17. `feature_importance_inversion`
- **Description:** A high-importance feature started carrying inverted signal (e.g., high values now predict low-churn users instead of high-churn users) due to a product change that reversed the business meaning of the feature.
- **Primary signal:** `query_feature_distributions` — feature values are in normal range (low PSI) but model's confidence on these examples is low; `query_code_diffs` — shows a product-side change in how the feature is computed.
- **Distinguishing signal:** PSI looks normal (values in range), but model underperforms on examples with high values of the inverted feature. Requires code diff to discover the semantic inversion.
- **Injection:** Flip the sign of `login_failure_rate` computation (now outputs 1-failure_rate instead of failure_rate).

### 18. `compound_drift_plus_deployment`
- **Description:** A feature drift event and a deployment event occurred within hours of each other; the agent must attribute the accuracy drop correctly (deployment is the proximate cause but drift predated it).
- **Primary signal:** All tools required — deployment history shows deploy event; feature distributions show pre-existing drift; logs show neither has hard errors; the deploy is not the root cause (drift was already present).
- **Distinguishing signal:** Drift began before the deployment. The deployment is a red herring. Requires careful timeline reconstruction to identify feature drift as the root cause despite the salient deployment event.
- **Injection:** Start feature drift 6h before a benign deployment event (config tweak); accuracy drop is causally due to drift, not the deploy.

---

## CI Enforcement

Add a test that asserts `FailureCategory` enum values exactly match the `id` fields in this file:

```python
# tests/test_taxonomy_sync.py
import re, pytest
from hypothesis_graph import FailureCategory

TAXONOMY = "docs/chaos-taxonomy.md"

def extract_ids(path):
    ids = re.findall(r"^### \d+\. `([a-z_]+)`", open(path).read(), re.MULTILINE)
    return set(ids)

def test_failure_category_matches_taxonomy():
    taxonomy_ids = extract_ids(TAXONOMY)
    enum_values = {e.value for e in FailureCategory}
    assert enum_values == taxonomy_ids, (
        f"FailureCategory enum out of sync with chaos taxonomy.\n"
        f"  In taxonomy but not enum: {taxonomy_ids - enum_values}\n"
        f"  In enum but not taxonomy: {enum_values - taxonomy_ids}"
    )
```
