# Task: Build target ML system for Failure Diagnosis Agent testing

## Context
This is a synthetic but realistic served ML system that will serve as the fixed
target for a failure-diagnosis agent (built in a later phase). This system's
tools/data ARE the evidence sources the agent's ReAct loop will query. Prioritize
correctness of interfaces and realistic data shapes over production robustness —
this is a test harness, not a product.

## System archetype (locked, do not deviate)
A served binary classifier on tabular data (XGBoost or LightGBM), with:
- A feature store
- A metrics service
- A log aggregator
- Deployment history tracking
- Git-diff-accessible model/pipeline code

## Components to build

### 1. The model
- Train a real XGBoost or LightGBM binary classifier on a synthetic tabular
  dataset (10-20 features, mix of numeric/categorical, ~10k-50k rows). Use
  sklearn's make_classification or a public tabular dataset (e.g. a Kaggle
  churn/fraud dataset) — doesn't need to be sophisticated, just real enough
  that drift/degradation is meaningful.
- Wrap it in a simple inference service (FastAPI) exposing a `/predict` endpoint.

### 2. Feature store (mock, queryable)
- Store historical feature values per-request with timestamps (SQLite or
  parquet files are fine — no need for a real feature store product).
- Must support querying feature distributions over a time range, to back
  a `query_feature_distributions` tool later. Return format should support
  PSI calculation (i.e., binned distributions or raw values + timestamp
  the agent's tool layer can bin).

### 3. Metrics service (mock, queryable)
- Emit and store time-series metrics: prediction latency, request volume,
  error rate, prediction distribution (score histogram), and (since this is
  synthetic) ground-truth-derived accuracy/AUC where labels are available
  with a delay (simulates delayed label arrival, realistic for classifiers).
- Must support time-range queries to back a `query_metrics` tool later.

### 4. Log aggregator (mock, queryable)
- Structured logs from the inference service: request-level logs, errors,
  warnings, with timestamps and severity. Include realistic log formats
  (not just "error happened") — e.g. stack traces for exceptions, schema
  validation failures, timeout logs.
- Must support time-range + severity-filtered queries to back a
  `query_logs` tool later.

### 5. Deployment history (mock, queryable)
- A log of deploy events: timestamp, version, what changed (model retrain,
  config change, feature pipeline change, dependency bump), and a
  human-readable changelog entry per deploy.
- Must support time-range queries to back a `query_deployment_history` tool.

### 6. Code diff access
- A real git repo (can be a subdirectory with its own git history) containing
  the feature engineering pipeline and model training/serving code, with
  actual commit history showing realistic changes over time (some benign,
  some failure-inducing — these will be used later for chaos injection).
- Must support diff queries between two commits/timestamps to back a
  `query_code_diffs` tool.

## Critical requirement: deterministic time-range anchoring
All queryable data (metrics, logs, feature distributions, deployment history)
must be timestamped on a **consistent, controllable clock** — not wall-clock
time. Support generating a full run of synthetic "normal operation" data
across a configurable time window (e.g. 7 simulated days), so that a later
chaos-injection phase can insert failures at a specific point in that window
and everything downstream (metrics, logs, feature distributions) reflects it
consistently. This is required for deterministic replay in later evaluation —
don't use real-time data generation.

## Generator design: injection-ready parameterization

The data generation process for each evidence type (metrics, feature
distributions, logs, deployment events) must be structured so a future
chaos-injection phase can alter the underlying generative process for a
bounded time window, without modifying the generator code itself.

Concretely:

- Each generator function should accept a **parameter set** governing the
  "normal" process (e.g., for a feature: distribution family, mean, std;
  for error rate: baseline rate; for latency: baseline + variance) as an
  explicit argument or config object — not inlined as literals in the
  function body.
- Generation should support being run **piecewise over sub-ranges** of the
  full time window, each sub-range with its own parameter set. Normal
  operation is the default parameter set applied across the whole window;
  a future chaos scenario will override the parameter set for a specific
  [start_time, end_time) sub-range and regenerate just that slice.
- Where evidence types are causally linked in reality, structure the
  generator so a single upstream parameter change propagates correctly.
  Example: if feature X's distribution shifts, the model's prediction
  distribution and (with the existing label delay) accuracy/AUC metrics
  should shift as a *downstream consequence* of feeding the perturbed
  feature through the actual trained model at inference time — not as
  independently hardcoded metric changes. This means metrics generation
  should, where feasible, run actual inference through the trained model
  on generated feature data rather than sampling metric values directly
  from an unrelated distribution. This is what will make injected failures
  produce realistic, correlated evidence across multiple tool types later.
- Deployment events and logs are the exception — these can be more directly
  parameterized (e.g., "insert a deploy event with changelog X at time T",
  "insert N error logs of type Y between T1 and T2") since they don't need
  to flow through the model.
- Do not build the chaos injection framework itself (still out of scope for
  this phase) — just ensure the generator's internals make "override
  parameters for a sub-range" a natural, already-supported operation rather
  than something requiring a rewrite later.

## Non-goals (do not build these now)
- Do NOT build the ReAct agent, tool-calling layer, or Hypothesis Graph —
  that's a separate, later phase.
- Do NOT build chaos injection logic — that's a separate, later phase. Just
  make sure the data generation is structured so failures CAN be injected
  later (e.g. feature generation is parameterized so drift can be introduced
  by changing a distribution parameter at a given timestamp).
- Do NOT build the five `query_*` tool functions themselves — just make sure
  each data store above is queryable in a way that a thin tool wrapper can
  sit on top of later. Keep query interfaces simple and directly callable
  (Python functions or REST endpoints, your choice, but document the schema).

## Deliverables
- Working inference service + trained model
- Four queryable mock data stores (feature store, metrics, logs, deployment
  history) with a documented query interface for each (function signature or
  REST endpoint + return schema)
- Git repo with realistic commit history for the pipeline/model code
- A script to generate N days of synthetic "normal operation" data end-to-end
- README documenting: how to run the system, how to regenerate data, and the
  exact query interface for each data store (this becomes the spec for the
  agent's tool layer in the next phase)

## Output structure
Put this under a clearly separated directory `target_system/` since
it's infrastructure for testing, not the diagnosis agent itself.