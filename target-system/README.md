# Target ML System — Failure Diagnosis Test Harness

Synthetic but realistic served binary classifier (XGBoost, churn prediction) with
four queryable mock data stores and a real git repo for the model pipeline.
This is the **target system** that the diagnosis agent investigates — not the agent itself.

---

## Directory structure

```
target-system/
├── model/
│   ├── train.py              # trains XGBoost, saves artifacts
│   └── artifacts/            # model.pkl, scaler.pkl, feature_info.json  (generated)
├── generator/
│   ├── params.py             # all parameter dataclasses
│   ├── defaults.py           # default 7-day normal-operation config
│   ├── schema.py             # SQLite DDL + documented query interfaces
│   └── generate.py           # main orchestrator (CLI entry point)
├── inference_service/
│   └── main.py               # FastAPI /predict endpoint
├── pipeline_repo/            # nested git repo (created by setup_pipeline_repo.py)
├── data/                     # SQLite databases (created by generator)
│   ├── feature_store.db
│   ├── metrics.db
│   ├── logs.db
│   └── deployments.db
├── setup_pipeline_repo.py    # creates pipeline_repo with 7 commits of history
└── requirements.txt
```

---

## Setup

```bash
cd target-system/
pip install -r requirements.txt

# 1. Train the model
python -m model.train

# 2. Create the pipeline repo with git history
python setup_pipeline_repo.py

# 3. Generate 7 days of synthetic data
python -m generator.generate --days 7

# 4. (Optional) Start the inference service
uvicorn inference_service.main:app --host 0.0.0.0 --port 8000
```

---

## Simulated time window

All timestamps are on a **controlled simulated clock**, not wall-clock time.

- **Epoch**: `2024-01-01 00:00:00 UTC` (`SIM_START` in `generator/defaults.py`)
- **Default window**: 7 days (`2024-01-01` → `2024-01-08`)
- **Request rate**: 500 req/hour (configurable via `--req-per-hour`)

All four data stores share the same timestamp convention — pass `SIM_START + offset_seconds`
when querying any of them.

---

## Injection-ready design

The generator is structured so a chaos injection phase can override parameters
for a sub-range **without modifying generator code**:

```python
from generator.params import SubRangeConfig, FeatureParams, FeatureConfig, InferenceParams, LogParams
from generator.defaults import make_default_config, SIM_START
from generator.generate import generate_data

config = make_default_config(n_days=7)

# Override days 3–5 with a feature distribution shift
drifted_features = FeatureParams(
    requests_per_hour=500,
    feature_configs=[
        # ... same as default but with login_failure_rate shifted
        FeatureConfig("login_failure_rate", "uniform", low=0.15, high=0.5),  # was 0–0.3
        # ... rest unchanged
    ],
)
config.overrides = [
    SubRangeConfig(
        start_ts=SIM_START + 3 * 86400,
        end_ts=SIM_START + 5 * 86400,
        feature_params=drifted_features,
        inference_params=config.default_inference_params,
        log_params=config.default_log_params,
    )
]

generate_data(config, model_dir="model/artifacts")
```

The generator fills gaps between overrides with default params automatically.
Prediction distribution and accuracy metrics shift as a **downstream consequence**
of running the perturbed features through the actual model — they are not hardcoded.

---

## Data stores — query interface

All four stores live under `data/` (SQLite). Import from `generator.schema` or
query directly with `sqlite3`.

### 1. Feature store (`data/feature_store.db`)

**Table: `requests`**

| Column | Type | Description |
|---|---|---|
| `request_id` | TEXT PK | UUID |
| `timestamp` | REAL | Unix timestamp (simulated) |
| `features` | TEXT | JSON: `{"feature_name": value, ...}` |
| `prediction` | INTEGER | 0 or 1; -1 if request errored |
| `pred_proba` | REAL | P(churn); -1.0 if errored |
| `true_label` | INTEGER | Ground-truth (available immediately in DB; use `label_ts` to simulate delay) |
| `label_ts` | REAL | Timestamp when label became available (= `timestamp + 24h`) |
| `service` | TEXT | `"inference_service"` |

**Python query function:**
```python
from generator.schema import query_feature_values

# Returns {"feature_name": [float, ...], ...}
values = query_feature_values(
    db_path="data/feature_store.db",
    features=["login_failure_rate", "monthly_spend"],
    start_ts=SIM_START + 3 * 86400,
    end_ts=SIM_START + 5 * 86400,
)
# → {"login_failure_rate": [0.12, 0.27, ...], "monthly_spend": [180.4, ...]}
```

PSI calculation: bin the returned arrays and compare baseline vs. comparison window.
The tool layer does this; the feature store returns raw values.

---

### 2. Metrics service (`data/metrics.db`)

**Table: `metrics`** — hourly aggregated rows

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | — |
| `timestamp` | REAL | Hour bucket start (Unix, simulated) |
| `metric_name` | TEXT | See enum below |
| `metric_value` | REAL | — |
| `service` | TEXT | `"inference_service"` |
| `tags` | TEXT | JSON (currently `{}`) |

**Available metric names:**

| Name | Description |
|---|---|
| `throughput` | Requests in the hour |
| `latency_p50` | Median latency (ms) |
| `latency_p99` | 99th percentile latency (ms) |
| `error_rate` | Fraction of requests that errored |
| `prediction_confidence` | Mean P(churn) across non-errored requests |
| `positive_rate` | Fraction predicted as churn=1 |
| `accuracy` | Fraction correct (only present for hours where labels have arrived, i.e., > 24h ago) |

**Python query function:**
```python
from generator.schema import query_metrics

rows = query_metrics(
    db_path="data/metrics.db",
    metric_names=["accuracy", "prediction_confidence", "error_rate"],
    start_ts=SIM_START,
    end_ts=SIM_START + 7 * 86400,
)
# Returns list of dicts: [{timestamp, metric_name, metric_value, service, tags}, ...]
```

---

### 3. Log aggregator (`data/logs.db`)

**Table: `logs`**

| Column | Type | Description |
|---|---|---|
| `log_id` | TEXT PK | UUID |
| `timestamp` | REAL | Unix timestamp (simulated) |
| `severity` | TEXT | `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `service` | TEXT | `inference_service` \| `feature_pipeline` \| `label_pipeline` |
| `message` | TEXT | Full log message (errors include stack traces) |
| `context` | TEXT | JSON: `{"request_id": ..., ...}` |

**Python query function:**
```python
from generator.schema import query_logs

entries, truncated = query_logs(
    db_path="data/logs.db",
    start_ts=SIM_START + 3 * 86400,
    end_ts=SIM_START + 4 * 86400,
    service="inference_service",
    severities=["ERROR", "WARNING"],
    filter_text="timeout",   # substring match on message
    limit=200,
)
# entries: list of dicts [{log_id, timestamp, severity, service, message, context}, ...]
# truncated: True if more rows existed than limit
```

---

### 4. Deployment history (`data/deployments.db`)

**Table: `deployments`**

| Column | Type | Description |
|---|---|---|
| `deploy_id` | TEXT PK | UUID |
| `timestamp` | REAL | Deploy time (Unix, simulated) |
| `service` | TEXT | `inference_service` \| `feature_pipeline` \| `label_pipeline` |
| `version_before` | TEXT | e.g. `"v1.3.1"` |
| `version_after` | TEXT | e.g. `"v1.3.2"` |
| `commit_sha` | TEXT | Full SHA in `pipeline_repo/` |
| `change_type` | TEXT | `model_retrain` \| `config_change` \| `feature_pipeline_change` \| `dependency_bump` |
| `changelog` | TEXT | Human-readable description |
| `deployed_by` | TEXT | Actor |
| `is_rollback` | INTEGER | 0 or 1 |
| `config` | TEXT | JSON of changed config values |

**Python query function:**
```python
from generator.schema import query_deployments

events = query_deployments(
    db_path="data/deployments.db",
    start_ts=SIM_START,
    end_ts=SIM_START + 7 * 86400,
    service="inference_service",
)
# Returns list of dicts [{deploy_id, timestamp, service, version_before,
#                         version_after, commit_sha, change_type, changelog,
#                         deployed_by, is_rollback, config}, ...]
```

---

### 5. Code diffs (`pipeline_repo/`)

A real git repo containing feature engineering, training, and serving code
with 7 commits of realistic history. Use `git diff` to get diffs between any
two commits:

```python
import subprocess

result = subprocess.run(
    ["git", "diff", commit_before, commit_after, "--", "feature_engineering.py"],
    cwd="pipeline_repo/",
    capture_output=True,
    text=True,
)
patch = result.stdout  # unified diff text
```

The `commit_sha` field in deployment events corresponds to commits in this repo.

---

## Regenerating data

The generator is idempotent for a given seed. To start fresh:

```bash
rm -f data/*.db
python -m generator.generate --days 7 --seed 42
```

To change the request volume or window length:

```bash
python -m generator.generate --days 14 --req-per-hour 1000
```

---

## Feature catalog

12 features matching the trained model:

| Feature | Distribution | Notes |
|---|---|---|
| `account_age_days` | Normal(730, 200), clipped ≥ 0 | Days since account opened |
| `monthly_spend` | Lognormal(5.5, 0.8) | USD; mean ≈ $245 |
| `num_transactions_30d` | Normal(42, 15), clipped ≥ 0 | — |
| `avg_transaction_value` | Lognormal(4.0, 0.6) | USD; mean ≈ $55 |
| `days_since_last_login` | Normal(3, 2), clipped ≥ 0 | — |
| `support_tickets_90d` | Normal(1.2, 1.5), clipped ≥ 0 | — |
| `product_category` | Categorical {0,1,2} | 0=Basic, 1=Pro, 2=Enterprise |
| `region` | Categorical {0,1,2,3,4} | Geographic region |
| `device_type` | Categorical {0,1,2} | 0=mobile, 1=desktop, 2=tablet |
| `login_failure_rate` | Uniform(0, 0.3) | Fraction of failed logins |
| `session_duration_min` | Normal(18, 8), clipped ≥ 0 | Minutes |
| `referral_source` | Categorical {0,1,2,3} | 0=organic, 1=paid, 2=partner, 3=unknown |

Normal-operation baseline values match the training distribution exactly —
any PSI > 0.1 against this baseline indicates meaningful drift.
