# MLSys Investigator

An agentic failure-diagnosis system for distributed ML systems. Given a production alert, it autonomously investigates by querying metrics, logs, deployment history, feature distributions, and code diffs — then produces a ranked, evidence-backed root-cause diagnosis.

---

## What problem this solves

Diagnosing ML system failures is slow and expert-heavy. An accuracy drop could mean feature drift, a bad deployment, upstream schema corruption, label pipeline corruption, or a dozen other things. Each hypothesis requires pulling data from different sources and cross-referencing timelines. This system automates that investigation loop.

---

## Architecture

The system is a **hand-rolled ReAct (Reason + Act) agent** built directly on the Anthropic API. No LangChain, no agent frameworks. The loop is ~500 lines of explicit Python.

```
Alert text
    │
    ▼
┌─────────────────────────────────────────┐
│              ReAct Loop                  │
│  1. Send messages → Claude reasons       │
│  2. Claude calls a tool                  │
│  3. Dispatch to evidence source          │
│  4. Feed result back → repeat            │
│  5. Terminate on stop_investigation call │
└───────────────────────┬─────────────────┘
                        │  reads/writes
                        ▼
               Hypothesis Graph (Pydantic)
                        │
               ┌────────┴────────┐
               ▼                 ▼
        Stopping Criteria   Output Validator
        (direct graph read)  (direct graph read)
```

### Six modules

| Module | Responsibility |
|---|---|
| **ReAct Loop** (`agent.py`) | Drives the reason-act cycle; owns session lifecycle |
| **Tool Dispatcher** (`tools.py`) | Routes tool calls to evidence sources; uniform error envelope |
| **Hypothesis Graph** (`hypothesis_graph.py`) | Pydantic belief state — hypotheses, evidence, status |
| **Stopping Criteria** (`stopping_criteria.py`) | Decides when to terminate; reads graph directly |
| **Output Validator** (`output_validator.py`) | Gates graph updates before they mutate state |
| **Observability Layer** | Langfuse tracing; every turn and tool call is a span |

**Key architectural constraint:** Stopping Criteria and Output Validator read the Hypothesis Graph directly, not through the ReAct Loop. This avoids the god-object anti-pattern where a single orchestrator mediates every data access.

---

## Hypothesis Graph — the agent's belief state

The graph is the core design artifact. Rather than tracking belief implicitly in conversation history, the agent maintains explicit structured state at every step.

```python
class HypothesisGraph(BaseModel):
    alert_summary: str
    investigation_start: datetime
    tool_calls_used: int
    tool_call_budget: int
    hypotheses: List[Hypothesis]
    established_facts: List[str]
    open_questions: List[str]
    termination_reason: Optional[str]

class Hypothesis(BaseModel):
    id: str                            # H1, H2, ...
    root_cause_category: FailureCategory
    description: str
    likelihood: float                  # always normalized over active hypotheses
    severity: Severity
    status: HypothesisStatus           # active | ruled_out | confirmed
    evidence: List[Evidence]
```

### Design decisions in the graph

**Delta-based updates, not full rewrites.** The model emits a `GraphUpdate` (action + evidence + likelihood_changes), which is applied programmatically. The model never resubmits the full graph state.

**Likelihood normalization is programmatic.** After every update, active hypothesis likelihoods are re-normalized so they sum to 1.0. The model proposes deltas; the system enforces the invariant.

**Per-update delta cap of ±0.25.** A single piece of evidence cannot swing any hypothesis by more than 25 percentage points. This prevents runaway confidence from a single tool result.

**`evidence_type` is derived, never model-set.** The model records `tool_called`; the system derives `EvidenceType` from a static mapping. This prevents the model from hallucinating evidence type labels.

**Three graph update actions:**
- `create` — new hypothesis, model proposes initial likelihood weight
- `update` — attach evidence to existing hypothesis (most common)
- `merge` — proposed hypothesis is semantically equivalent to an existing one; consolidate rather than duplicate

---

## Tool layer

Five read-only, side-effect-free tools backed by real data sources:

| Tool | Backend | Purpose |
|---|---|---|
| `query_metrics` | SQLite (`metrics.db`) | Time-series accuracy, latency, error rate, throughput |
| `query_logs` | SQLite (`logs.db`) | Structured service logs with severity/keyword filter |
| `query_deployment_history` | SQLite (`deployments.db`) | Deploy events, rollbacks, commit SHAs |
| `query_feature_distributions` | SQLite (`feature_store.db`) | PSI-based drift detection between time windows |
| `query_code_diffs` | `git diff` on `pipeline_repo/` | Unified diffs between commit SHAs |

All tools share a uniform error envelope:
```json
{
  "tool_name": "...",
  "status": "error",
  "error": { "error_type": "...", "message": "...", "retryable": false },
  "query_metadata": { "latency_ms": 0 }
}
```

**Self-correction for feature names:** `query_feature_distributions` returns a `valid_values` field when an unknown feature name is requested, so the model can correct itself without burning a budget unit on an invalid call.

**PSI implementation is custom.** Population Stability Index is computed inline in Python against raw feature value samples from the SQLite store. No external library dependency for this.

**`update_hypothesis_graph` is free.** It does not count against the tool budget. The budget only applies to the five data query tools and `stop_investigation`. This is enforced in the loop, not by trust.

---

## Stopping criteria

Three convergence triggers, checked after every graph update:

1. **Top likelihood > 0.60** — one hypothesis has clear dominance
2. **Dominance ratio > 2.5×** — top hypothesis is more than 2.5× as likely as the runner-up
3. **Stalled** — maximum per-hypothesis likelihood delta across the last 3 snapshots is < 0.05

The convergence signal is returned in the `update_hypothesis_graph` tool result, so the model sees it without requiring an extra round trip.

---

## Budget management

The investigation runs on a configurable budget (default: 8 tool calls). When the budget is exhausted, the loop gives the model one final turn with only `stop_investigation` available and forces a `tool_choice` to ensure a diagnosis is always produced — even if the investigation terminated early.

---

## Evaluation harness

### Target system

A complete synthetic ML system in `target-system/`:
- **Training:** XGBoost churn-prediction model trained on 12 user-behavior features
- **Serving:** FastAPI inference service
- **Data generator:** Produces 7 days of realistic metrics, logs, feature store records, and deployment history into SQLite

### Chaos injection taxonomy

18 failure categories across three difficulty tiers, each with a precise injection mechanism:

**Easy (5 scenarios)** — Single clear signal; one tool call usually sufficient
- `feature_drift`, `bad_deployment`, `upstream_schema_change`, `infrastructure_latency_spike`, `model_version_rollback_regression`

**Medium (8 scenarios)** — 2–3 tools required; ambiguity between categories
- `label_pipeline_corruption`, `training_serving_skew`, `data_freshness_degradation`, `feature_encoding_bug`, `gradual_concept_drift`, `model_calibration_drift`, `shadow_mode_leak`, `feature_pipeline_partial_failure`

**Hard (5 scenarios)** — Cross-evidence reasoning; causal chain tracing; delayed signals
- `delayed_label_feedback_shift`, `cascading_upstream_failure`, `model_staleness`, `feature_importance_inversion`, `compound_drift_plus_deployment`

Each scenario defines exact data overrides (distribution shifts, error rates, log injections, deployment events) that produce the failure signal in the synthetic data.

### Taxonomy sync enforcement

`FailureCategory` enum values in the agent code are CI-enforced to exactly match the chaos taxonomy markdown:

```python
# tests/test_taxonomy_sync.py
def test_failure_category_matches_taxonomy():
    taxonomy_ids = extract_ids("docs/chaos-taxonomy.md")
    enum_values = {e.value for e in FailureCategory}
    assert enum_values == taxonomy_ids
```

This prevents the agent's output space from drifting out of sync with the ground-truth labels.

### Evaluation metrics

- **Top-1 / Top-3 root cause accuracy** — primary signal; logged as Langfuse scores
- **Agent vs. rule-based baseline delta** — comparison against a deterministic baseline
- **Mean tool calls to diagnosis** — efficiency signal
- **Hypothesis calibration** — confidence scores are explicitly elicited from the model, never rank-derived, so calibration curves are meaningful
- **Tool selection efficiency** — did the agent choose the shortest evidence path?
- **LLM-as-judge scoring** with inter-rater reliability

### Running an eval

```bash
cd target-system/

# Generate data for a scenario
python -m evaluation.run_eval --scenario feature_drift

# Generate data and run the agent
python -m evaluation.run_eval --scenario feature_drift --run-agent

# List all scenarios
python -m evaluation.run_eval --list
```

---

## Prompt architecture

All LLM-facing strings are in `src/prompts.py` as versioned `Prompt` objects. No raw string literals for prompts elsewhere in the codebase. This makes prompt versioning explicit and trackable.

The system prompt enforces a strict protocol:
1. Call a query tool
2. Immediately call `update_hypothesis_graph` (free)
3. Repeat until convergence
4. Call `stop_investigation`

The model is told explicitly that `update_hypothesis_graph` does not count against the budget, that it must be called after every query tool before calling another, and that it must reserve 1 budget unit for `stop_investigation`.

---

## HTTP API

The agent is wrapped in a FastAPI server (`src/server.py`) with async job execution:

```
POST /investigate            → 202  { job_id, status: "pending" }
GET  /jobs/{job_id}          → job record with result or error
POST /webhook/alertmanager   → Prometheus Alertmanager receiver
GET  /health                 → liveness probe
```

Investigations run 30–120 seconds (multiple Anthropic API round trips), so they execute in a `ThreadPoolExecutor` and are polled via job ID.

The `/webhook/alertmanager` endpoint accepts the Prometheus Alertmanager v4 webhook schema and auto-converts firing alerts into investigations.

---

## Observability

Every investigation is traced to Langfuse:
- One top-level span per investigation
- One child span per ReAct turn
- One child span per tool call with raw input/output
- `top1_correct` and `top3_correct` scores attached to the trace when `ground_truth` is provided
- Hypothesis graph state (normalized likelihoods) logged after each update

---

## Deployment

Docker:
```bash
docker build -t mlsys-investigator .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=... \
  -e LANGFUSE_PUBLIC_KEY=... \
  -e LANGFUSE_SECRET_KEY=... \
  mlsys-investigator
```

Kubernetes manifests in `k8s/` (namespace, deployment, service, configmap, secret).

---

## Project structure

```
src/
  agent.py              — ReAct loop; main entry point
  hypothesis_graph.py   — Pydantic belief state + graph mutation logic
  tools.py              — Tool definitions (Anthropic API schema) + dispatchers
  stopping_criteria.py  — Three convergence triggers
  output_validator.py   — GraphUpdate validation before mutation
  prompts.py            — All LLM-facing strings, versioned
  server.py             — FastAPI async job server
  persistence.py        — Live JSON snapshot of graph for editor preview

target-system/
  model/                — XGBoost model training (scikit-learn pipeline)
  generator/            — Synthetic data generation (metrics, logs, features, deployments)
  inference_service/    — FastAPI serving layer for the target ML system
  pipeline_repo/        — Git repo for query_code_diffs to diff against
  evaluation/
    chaos_scenarios.py  — 5 implemented scenarios with injection specs
    run_eval.py         — CLI harness: generate data + run agent + score

docs/
  chaos-taxonomy.md     — Single source of truth for 18 failure categories
  hypothesis-graph-spec.md — Full data model specification
  tool-specs.md         — Tool interface contracts
  architecture-proposal.md — Architecture design document

tests/
  unit/                 — Hypothesis graph, stopping criteria, output validator
  integration/          — Graph pipeline, tool dispatch
  e2e/                  — Full scenario runs
  test_taxonomy_sync.py — CI enforcement: enum ↔ taxonomy sync
  test_no_taxonomy_leakage.py — Agent output space matches taxonomy
```

---

## Key design tradeoffs

**Raw Anthropic API vs. an agent framework.** Using the raw API means explicit control over every message, tool definition, and budget accounting — no magic, no hidden prompt injection, no framework version drift. The tradeoff is more boilerplate in the loop itself.

**Structured belief state vs. implicit conversation history.** The Hypothesis Graph makes investigation progress explicit and queryable, which enables principled stopping criteria and calibration measurement. The tradeoff is that the model must maintain coherence between its reasoning and its structured updates — enforced by the Output Validator.

**Delta-based graph updates vs. full-state rewrites.** Deltas keep individual tool-result contributions traceable and allow the per-update cap to prevent confidence runaway. The tradeoff is a more complex graph update schema.

**SQLite for the target system.** All four data stores (metrics, logs, feature store, deployments) are SQLite files. This makes the evaluation harness fully self-contained and reproducible without any external services. The tradeoff is that it doesn't represent real production data infrastructure.

**In-process job store.** The server holds job state in a dict. This is simple and fast but means jobs are lost on restart and multi-replica deployments require sticky routing. The comment in `server.py` acknowledges this: run `replicas=1` or add Redis.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add ANTHROPIC_API_KEY and LANGFUSE_* keys

# Generate target-system data
cd target-system/
python -m generator.generate --days 7 --output-dir data

# Run a single investigation
cd ..
PYTHONPATH=src python src/agent.py

# Run the server
PYTHONPATH=src uvicorn server:app --app-dir src --port 8080

# Run tests
pytest tests/
```

---

## Stack

- **Python 3.11**, **Pydantic v2** for all structured data
- **Anthropic SDK** (`claude-sonnet-4-6`) as reasoning engine
- **FastAPI + uvicorn** for the HTTP layer
- **Langfuse** for tracing
- **SQLite** for the synthetic target-system data stores
- **XGBoost + scikit-learn** for the target ML model
- **Docker + Kubernetes** for deployment
