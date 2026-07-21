# ML System — What Was Built and How It Works

This document describes the synthetic ML target system in `target-system/`. It
is the **thing being diagnosed**, not the diagnosis agent. The agent's five
tools (`query_metrics`, `query_logs`, etc.) will sit as thin wrappers on top
of the data stores described here.

---

## Overview

The system is a served binary classifier that predicts customer churn from
tabular account data. It was built as a realistic but fully synthetic test
harness with four queryable evidence stores and a real git history — exactly
the evidence sources a diagnosis agent needs to investigate failures.

```
target-system/
├── model/              # XGBoost classifier + training script
├── generator/          # Parameterized data generators (all four stores)
├── inference_service/  # FastAPI /predict endpoint
├── pipeline_repo/      # Nested git repo with 7 commits of pipeline history
├── data/               # Four SQLite databases (generated)
└── setup_pipeline_repo.py
```

---

## The Model

**What it does:** predicts whether a customer will churn (binary: 0 = stays, 1 = churns).

**Algorithm:** XGBoost gradient-boosted trees, trained on 40,000 synthetic
samples, validated on 10,000.

**Performance on normal data:**
- Churn rate: ~27% (realistic class balance)
- Validation accuracy: 86.6%
- Mean prediction confidence: 0.27 (matches base rate, well-calibrated)

**How labels are assigned:** there is a deterministic ground-truth scoring
function that maps features to a probability of churn. This function is shared
between the training script and the data generator — so when features drift
during a chaos injection scenario, model accuracy degrades as a real
downstream consequence of the model being wrong about the new distribution,
not as an independently hardcoded metric change.

### Features (12 total)

| Feature | Distribution | Meaning |
|---|---|---|
| `account_age_days` | Normal(730, 200), clipped ≥ 0 | Days since account opened |
| `monthly_spend` | LogNormal(μ=5.5, σ=0.8) | Monthly USD spend (~$245 avg) |
| `num_transactions_30d` | Normal(42, 15), clipped ≥ 0 | Transaction count last 30 days |
| `avg_transaction_value` | LogNormal(μ=4.0, σ=0.6) | Average transaction USD (~$55 avg) |
| `days_since_last_login` | Normal(3, 2), clipped ≥ 0 | Recency signal |
| `support_tickets_90d` | Normal(1.2, 1.5), clipped ≥ 0 | Support burden |
| `product_category` | Categorical {0, 1, 2} | 0=Basic, 1=Pro, 2=Enterprise |
| `region` | Categorical {0, 1, 2, 3, 4} | Geographic region |
| `device_type` | Categorical {0, 1, 2} | 0=mobile, 1=desktop, 2=tablet |
| `login_failure_rate` | Uniform(0, 0.3) | Fraction of failed login attempts |
| `session_duration_min` | Normal(18, 8), clipped ≥ 0 | Avg session length |
| `referral_source` | Categorical {0, 1, 2, 3} | 0=organic, 1=paid, 2=partner, 3=unknown |

The strongest churn signals are `login_failure_rate` (high failure → churn),
`days_since_last_login` (longer away → churn), and `support_tickets_90d` (more
tickets → churn). `product_category=2` (Enterprise) is strongly anti-churn.

---

## The Inference Service

A FastAPI app in `inference_service/main.py`. Start it with:

```bash
uvicorn inference_service.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok", "model_version": "2.1.1"}` |
| `GET` | `/model/info` | Feature names, training stats, distribution configs |
| `POST` | `/predict` | Single prediction from a feature dict |
| `POST` | `/predict/batch` | Batch predictions (up to 1000) |

**Request body for `/predict`:**
```json
{
  "request_id": "optional-uuid",
  "features": {
    "account_age_days": 720.0,
    "monthly_spend": 245.0,
    "login_failure_rate": 0.05,
    "..."
  }
}
```

**Response:**
```json
{
  "request_id": "...",
  "prediction": 0,
  "probability": 0.12,
  "latency_ms": 1.3
}

### 1. Feature Store (`data/feature_store.db`)

Stores every inference request with its raw feature values. 84,000 rows for a
default 7-day run at 500 req/hour.

**Schema — table `requests`:**

| Column | Type | Notes |
|---|---|---|
| `request_id` | TEXT PK | UUID |
| `timestamp` | REAL | Unix seconds (simulated) |
| `features` | TEXT | JSON object of all 12 feature values |
| `prediction` | INTEGER | 0 or 1; -1 if the request errored |
| `pred_proba` | REAL | P(churn); -1.0 if errored |
| `true_label` | INTEGER | Ground-truth label |
| `label_ts` | REAL | When the label becomes available (`timestamp + 24h`) |
| `service` | TEXT | Always `"inference_service"` |

**Why `label_ts` exists:** real classifiers get feedback labels hours or days
after serving. The `label_ts` column simulates a 24-hour delay. Accuracy
metrics are only computed using rows whose `label_ts` has passed — querying
a window where no labels have arrived yet returns no accuracy data, which is
the correct behavior.

**Query function:**
```python
from generator.schema import query_feature_values

values = query_feature_values(
    db_path="data/feature_store.db",
    features=["login_failure_rate", "days_since_last_login"],
    start_ts=1704067200,   # 2024-01-01 00:00 UTC
    end_ts=1704153600,     # 2024-01-02 00:00 UTC
)
# → {"login_failure_rate": [0.12, 0.27, ...], "days_since_last_login": [2.1, ...]}
```

The return format is raw value arrays per feature, ready for PSI or KS-stat
computation in the tool layer.

---

### 2. Metrics Service (`data/metrics.db`)

Hourly aggregated metrics derived from actual inference results. 1,176 rows
for a 7-day run (168 hours × ~7 metric types).

**Schema — table `metrics`:**

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | REAL | Bucket start (hour boundary) |
| `metric_name` | TEXT | See table below |
| `metric_value` | REAL | |
| `service` | TEXT | `"inference_service"` |
| `tags` | TEXT | JSON (currently `{}`, ready for future dimensions) |

**Available metrics:**

| Name | Description | Normal range |
|---|---|---|
| `throughput` | Requests in the hour | ~500 |
| `latency_p50` | Median latency ms | ~45ms |
| `latency_p99` | 99th percentile latency ms | ~80–300ms (tail events) |
| `error_rate` | Fraction of requests that errored | ~0.005 |
| `prediction_confidence` | Mean P(churn) for non-errored requests | ~0.27 |
| `positive_rate` | Fraction predicted churn=1 | ~0.27 |
| `accuracy` | Fraction correct (only for hours with labels available) | ~0.86 |

**Critical note on accuracy:** accuracy rows appear with a 24-hour offset.
The accuracy in hour H reflects requests from hour H−24, whose labels just
became available. The first 24 hours of a run have no accuracy rows. This is
correct behavior, not a bug.

**Query function:**
```python
from generator.schema import query_metrics

rows = query_metrics(
    db_path="data/metrics.db",
    metric_names=["accuracy", "error_rate", "latency_p99"],
    start_ts=1704067200,
    end_ts=1704672000,  # 7 days later
)
# → [{"timestamp": ..., "metric_name": "accuracy", "metric_value": 0.87, ...}, ...]
```

---

### 3. Log Aggregator (`data/logs.db`)

Structured logs from three services. 5,752 entries for a 7-day run.

**Breakdown by severity:** INFO (4,276), WARNING (875), ERROR (601)

**Breakdown by service:** `inference_service` (5,588), `feature_pipeline` (86), `label_pipeline` (78)

**Schema — table `logs`:**

| Column | Type | Notes |
|---|---|---|
| `log_id` | TEXT PK | UUID |
| `timestamp` | REAL | Unix seconds |
| `severity` | TEXT | `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `service` | TEXT | `inference_service` \| `feature_pipeline` \| `label_pipeline` |
| `message` | TEXT | Full message text; errors include stack traces |
| `context` | TEXT | JSON: `{"request_id": ..., "latency_ms": ...}` |

**What logs look like:**

- **INFO** — request processed, feature lookup timings, cache hits
- **WARNING** — feature value out of expected range, low prediction confidence,
  slow feature store lookups, missing optional features
- **ERROR** — full Python stack traces for validation errors, timeouts, NaN
  inputs, schema mismatches, label pipeline connection failures

**Query function:**
```python
from generator.schema import query_logs

entries, truncated = query_logs(
    db_path="data/logs.db",
    start_ts=1704067200,
    end_ts=1704153600,
    service="inference_service",
    severities=["ERROR", "WARNING"],
    filter_text="timeout",     # substring match on message text
    limit=200,
)
# entries: list of dicts
# truncated: True if more rows existed beyond the limit
```

The `truncated` flag matches the `QueryLogsResult.truncated` field in
`tool-specs.md §4.2`.

---

### 4. Deployment History (`data/deployments.db`)

A log of all deploy events in the simulated window. 4 events for the default
7-day run.

**Default deploy events:**

| Timestamp | Service | Version | Type | What changed |
|---|---|---|---|---|
| 2024-01-01 09:00 | inference_service | v1.3.1→v1.3.2 | config_change | Request timeout 200→250ms; worker pool 4→6 |
| 2024-01-03 14:00 | feature_pipeline | v1.3.2→v1.4.0 | feature_pipeline_change | Added `referral_source`; clipped `session_duration_min` at 120min |
| 2024-01-05 11:00 | inference_service | v2.0.0→v2.1.0 | model_retrain | Retrain on 3-month window; includes new referral_source feature |
| 2024-01-07 08:00 | inference_service | v2.1.0→v2.1.1 | dependency_bump | xgboost 1.7.6 → 2.0.3; no model changes |

**Schema — table `deployments`:**

| Column | Type | Notes |
|---|---|---|
| `deploy_id` | TEXT PK | UUID |
| `timestamp` | REAL | Deploy time |
| `service` | TEXT | Which service was deployed |
| `version_before` | TEXT | e.g. `"v1.3.1"` |
| `version_after` | TEXT | e.g. `"v1.3.2"` |
| `commit_sha` | TEXT | Full SHA in `pipeline_repo/` (valid, inspectable) |
| `change_type` | TEXT | `model_retrain` \| `config_change` \| `feature_pipeline_change` \| `dependency_bump` |
| `changelog` | TEXT | Human-readable description |
| `deployed_by` | TEXT | Actor (`"alice"`, `"mlops-bot"`) |
| `is_rollback` | INTEGER | 0 or 1 |
| `config` | TEXT | JSON of changed config values |

**Query function:**
```python
from generator.schema import query_deployments

events = query_deployments(
    db_path="data/deployments.db",
    start_ts=1704067200,
    end_ts=1704672000,
    service="inference_service",
)
```

**How the agent uses this:** a deployment event's `commit_sha` is a real commit
in `pipeline_repo/`. The typical investigation flow is:
1. `query_deployment_history` finds a suspicious deploy (e.g., the feature pipeline change)
2. `query_code_diffs` uses that `commit_sha` to inspect exactly what changed in `feature_engineering.py`

---

### 5. Code Diffs (`pipeline_repo/`)

A nested git repository (separate `.git` from the main repo) containing the
feature engineering, training, and serving code. It has 7 real commits with
timestamps spread across the simulated window.

**Commit history:**

```
cf1246e  Upgrade xgboost 1.7.6 → 2.0.3              (2024-01-07 08:00)
23c603f  Tune XGBoost hyperparams; retrain on 3-month window  (2024-01-05 10:00)
bee909c  Add referral_source feature; add categorical validation  (2024-01-03 13:00)
44c6ed4  Bump request timeout to 250ms; increase worker pool to 6  (2024-01-01 08:00)
ed74713  Fix session_duration_min clip bound (480 → 120)        (2023-12-20)
82d8c7a  Add feature normalization with clip bounds             (2023-12-10)
11b78fd  Initial commit: churn classifier v1.0                  (2023-12-01)
```

**How to get a diff (what the `query_code_diffs` tool wrapper will do):**
```python
import subprocess

result = subprocess.run(
    ["git", "diff", sha_before, sha_after, "--", "feature_engineering.py"],
    cwd="pipeline_repo/",
    capture_output=True,
    text=True,
)
patch = result.stdout   # unified diff text
```

The commits are real; `git diff`, `git log`, and `git show` all work normally.

---

## How Data Is Generated

Data generation is the core of this system. Understanding it is important
because the chaos injection phase will directly manipulate it.

### The generation pipeline

```
GenerationConfig
    │
    ├─ resolve_sub_ranges()  ← splits window into contiguous sub-ranges
    │                           (normal operation = 1 sub-range for the whole window)
    │
    └─ For each sub-range:
           │
           ├─ generate_feature_matrix(FeatureParams)
           │       ↓
           ├─ model.predict_proba(X)          ← ACTUAL model inference
           │       ↓
           ├─ assign_true_labels(X)           ← ground-truth function
           │       ↓
           ├─ sample_latencies(InferenceParams)
           │       ↓
           └─ write to feature_store.db
                   ↓
           aggregate_metrics() → metrics.db
           generate_logs()     → logs.db
```

**The key causal property:** metrics are not independently sampled. If you
change `FeatureParams` for a sub-range (e.g., shift `login_failure_rate` from
`U(0, 0.3)` to `U(0.2, 0.6)`), the model receives those perturbed features
and produces a different prediction distribution. `prediction_confidence`,
`positive_rate`, and `accuracy` all shift as downstream consequences.

### Running the generator

```bash
cd target-system/

# Default: 7 days, 500 req/hour
python -m generator.generate

# Custom window
python -m generator.generate --days 14 --req-per-hour 1000

# Start fresh
rm -f data/*.db && python -m generator.generate
```

### Simulated time anchor

All timestamps are offsets from `SIM_START = 2024-01-01T00:00:00 UTC`
(Unix: `1704067200`). To get the timestamp for day 3, hour 6:

```python
from generator.defaults import SIM_START
ts = SIM_START + 3 * 86400 + 6 * 3600
```

This is the value you pass to every query function. Never use wall-clock time.

---

## How Chaos Injection Will Work (Not Built Yet)

The generator is structured so a chaos scenario needs only to:

1. Build a modified `FeatureParams` (e.g., shift a distribution)
2. Wrap it in a `SubRangeConfig` covering the failure window
3. Set `config.overrides = [that_sub_range]`
4. Call `generate_data(config, model_dir)` — the rest is automatic

```python
from generator.params import SubRangeConfig, FeatureParams, FeatureConfig, InferenceParams, LogParams
from generator.defaults import make_default_config, SIM_START
from generator.generate import generate_data

config = make_default_config(n_days=7)

# Inject: login_failure_rate spikes on days 4–6
drifted = FeatureParams(
    requests_per_hour=500,
    feature_configs=[
        # ... copy all 12 from defaults, but change login_failure_rate:
        FeatureConfig("login_failure_rate", "uniform", low=0.20, high=0.60),
        # all other features unchanged
    ],
)
config.overrides = [
    SubRangeConfig(
        start_ts=SIM_START + 4 * 86400,
        end_ts=SIM_START + 6 * 86400,
        feature_params=drifted,
        inference_params=config.default_inference_params,
        log_params=config.default_log_params,
    )
]

generate_data(config, model_dir="model/artifacts")
```

After this runs:
- The feature store shows `login_failure_rate` values in 0.20–0.60 for days 4–6
- `query_feature_distributions` comparing day 1–3 vs. day 4–6 will return high PSI for that feature
- `prediction_confidence` and `positive_rate` rise (model sees more churn signals)
- `accuracy` degrades (model was trained on 0–0.3 range; higher values are OOD)
- Everything is causally consistent — no metrics were manually set

Different failure categories map to different generator overrides:
- **Feature drift** → change `FeatureParams` for a sub-range
- **High error rate** → increase `InferenceParams.error_rate`
- **Latency spike** → increase `InferenceParams.tail_prob` and `tail_latency_ms`
- **Deploy event** → add a `DeploymentEvent` to `config.deployment_events`
- **Log storm** → increase `LogParams.background_error_rate`

---

## Full Setup Sequence

```bash
cd target-system/
pip install -r requirements.txt

# 1. Train the XGBoost model (saves model.pkl, scaler.pkl, feature_info.json)
python -m model.train

# 2. Create the pipeline repo with 7 commits of git history
#    (also patches generator/defaults.py with real commit SHAs)
python setup_pipeline_repo.py

# 3. Generate 7 days of synthetic data into data/*.db
python -m generator.generate --days 7

# 4. (Optional) Start the inference service
uvicorn inference_service.main:app --port 8000
```

Step 2 only needs to be run once. Steps 1 and 3 can be re-run to regenerate
from scratch; step 3 is idempotent for a given `--seed`.
