# Tool Interface Specs

> **Status: reconstructed.** The original tool-specs draft was lost. This
> version is rebuilt from scratch to align with `CLAUDE.md` and
> `hypothesis-graph-spec.md`. Design decisions below are **proposed**, not
> locked — review before treating as final, unlike the hypothesis graph spec.

## 1. Purpose

Defines the interface contract for the five tools the agent can call:
`query_metrics`, `query_logs`, `query_deployment_history`,
`query_feature_distributions`, `query_code_diffs`. The **Tool Dispatcher**
module (see component diagram) is the only code that constructs requests
against these tools and is responsible for: validating model-supplied input
against each tool's schema, executing the call, and wrapping the result in
the unified envelope defined below.

> **Canonical tool names.** These five strings are the single source of
> truth for tool identity across the project:
> `query_metrics`, `query_logs`, `query_deployment_history`,
> `query_feature_distributions`, `query_code_diffs`.
> `TOOL_TO_EVIDENCE_TYPE` in `hypothesis-graph-spec.md` §2.4 must contain
> exactly these keys — if you add a tool here, add it there too.

---

## 2. Shared Conventions

### 2.1 Time Ranges — Always Absolute, Anchored to `investigation_start`

```python
class TimeRange(BaseModel):
    start: datetime
    end: datetime
```

Every `TimeRange` passed to a tool **must be an absolute timestamp**,
resolved by the ReAct loop relative to `HypothesisGraph.investigation_start`
— never a relative string like `"6h"`, and never wall-clock time at the
moment the tool happens to execute.

This matters for one specific reason: **chaos injection replay must be
deterministic.** If "6 hours ago" is resolved against wall-clock time at
call time, the same investigation re-run against the same injected failure
queries different windows depending on how long the agent took to get
there. Resolving relative expressions to absolute timestamps once, at graph
initialization, makes every tool call in a session reproducible regardless
of investigation length. The relative-to-absolute conversion is the ReAct
loop's job, not the tool's — tools only ever see `datetime` values.

### 2.2 Unified Response Envelope

```python
class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class ToolErrorType(str, Enum):
    INVALID_PARAMETER = "invalid_parameter"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    MALFORMED_REQUEST = "malformed_request"


class ToolError(BaseModel):
    error_type: ToolErrorType
    message: str
    retryable: bool
    valid_values: Optional[Dict[str, List[str]]] = None
    # e.g. {"service": ["inference_service", "feature_pipeline", ...]}
    # populated whenever error_type == invalid_parameter and the
    # offending field has a known, finite value space


class QueryMetadata(BaseModel):
    resolved_time_range: Optional[TimeRange] = None
    latency_ms: int
    result_count: Optional[int] = None


class ToolResponse(BaseModel):
    tool_name: str
    status: ToolStatus
    data: Optional[dict] = None    # present iff status == ok
    error: Optional[ToolError] = None  # present iff status == error
    query_metadata: QueryMetadata
```

`data` is typed as `dict` at the envelope level because its shape is
tool-specific — `dict` is the wire format, not a license to put anything in
there. Each tool's section below defines a strongly-typed result model; the
Tool Dispatcher constructs that model internally and calls `.model_dump()`
before attaching it to `data`. Those per-tool models are the actual source
of truth for what's inside `data`, not this envelope.

> **Empty results are not errors.** A tool call that finds no deployments,
> no matching log lines, or no drift is a **successful** response with an
> empty `data` collection — `status: "ok"`, not `status: "error"`. This
> matters because absence of evidence is itself evidence: "no deployments in
> the past 24 hours" is exactly the kind of result that rules out a
> hypothesis (see the H1 example in `hypothesis-graph-spec.md` §6).
> `ToolError` exists only for genuine failures to execute the query — bad
> parameters, the backing service being down, a timeout, or a malformed
> request. Conflating "no data" with "error" would corrupt the calibration
> analysis: a hypothesis correctly ruled out by an empty result looks
> identical to a hypothesis the agent never got real information about.

### 2.3 Validation at the Tool Dispatcher Boundary

The Tool Dispatcher validates the model's tool-call input against the
specific tool's Pydantic input model **before** executing anything. On
validation failure, it returns a `ToolResponse` with `status: "error"` and
`error_type: invalid_parameter` or `malformed_request` — it never lets a
validation exception propagate and crash the ReAct loop. This is the same
"never raise to the loop" principle that governs the Output Validator
elsewhere in the architecture.

### 2.4 Tool Discovery — Resolving the Mean-Tool-Calls Inflation Problem

Flagged as an open issue in `CLAUDE.md`: without a way to know what
parameter values are valid, the agent burns tool calls just discovering the
shape of the system (which services exist, which metrics exist) before it
can investigate anything — inflating mean-tool-calls-to-diagnosis with calls
that aren't actually diagnostic.

**Resolution, split by catalog size:**

- **Small, finite catalogs** (service names, metric names, log severities,
  drift metric choice — on the order of 5–10 values each): enum-constrain
  them directly in the JSON schema passed to the Anthropic API. The model
  never needs to discover these; they're listed in the tool definition it
  already has. No tool call spent, ever.
- **Large or dataset-dependent catalogs** (feature names — could be dozens
  to hundreds, and grows with the dataset): do **not** enum-constrain in the
  schema — that would bloat every tool definition with the full feature list
  on every API call. Instead, accept `features: List[str]` as free text,
  validate against the canonical feature list at the Tool Dispatcher, and on
  a miss, return `ToolError(invalid_parameter, valid_values={"features": [...]})`.
  The agent self-corrects on its *next* call using the values it just
  received — one wasted call in the worst case, not a dedicated discovery
  round-trip every session.

This avoids adding a sixth "list available X" tool purely for discovery,
which would cut against the project's hand-rolled, minimal-surface-area
ethos and would itself need justifying.

### 2.5 Result Size Limits

`query_logs` and `query_code_diffs` are the two tools whose raw output can
be unbounded (log volume, diff size). Both cap results and report
`truncated: bool` in `data` so the agent knows it isn't seeing the full
picture rather than silently assuming it is.

### 2.6 Tools Return Data, Not Interpretation

No tool pre-summarizes its result into natural language. Tools return
structured ground-truth data; the agent is responsible for writing
`Evidence.observation` itself, after reading the raw tool result. Same
separation-of-concerns logic that keeps Stopping Criteria and Output
Validator querying the Hypothesis Graph directly instead of going through
the ReAct loop: the tool layer stays dumb and deterministic, interpretation
stays in the reasoning layer.

---

## 3. Anthropic API Tool Definition (Example)

Concrete shape for how `query_metrics` gets registered in the raw API call's
`tools` parameter — same pattern applies to the other four.

```python
{
    "name": "query_metrics",
    "description": (
        "Query time-series metrics for the ML system. Returns aggregated "
        "values over a time window, with an optional comparison window for "
        "before/after deltas. Use this to check whether accuracy, latency, "
        "prediction confidence, throughput, or error rate have shifted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric_names": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["accuracy", "prediction_confidence",
                             "latency_p50", "latency_p99",
                             "throughput", "error_rate"]
                }
            },
            "time_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date-time"},
                    "end": {"type": "string", "format": "date-time"}
                },
                "required": ["start", "end"]
            },
            "comparison_window": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date-time"},
                    "end": {"type": "string", "format": "date-time"}
                }
            },
            "aggregation": {
                "type": "string",
                "enum": ["mean", "p50", "p95", "p99", "min", "max"],
                "default": "mean"
            }
        },
        "required": ["metric_names", "time_range"]
    }
}
```

---

## 4. Individual Tool Specs

### 4.1 `query_metrics`

Query time-series metrics, optionally comparing two windows.

```python
class MetricName(str, Enum):
    # TODO: finalize against your actual metrics catalog
    ACCURACY = "accuracy"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    LATENCY_P50 = "latency_p50"
    LATENCY_P99 = "latency_p99"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class AggregationType(str, Enum):
    MEAN = "mean"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    MIN = "min"
    MAX = "max"


class QueryMetricsInput(BaseModel):
    metric_names: List[MetricName]
    time_range: TimeRange
    comparison_window: Optional[TimeRange] = None
    aggregation: AggregationType = AggregationType.MEAN
```

| Field | Type | Description |
|---|---|---|
| `metric_names` | `List[MetricName]` | Which metrics to query |
| `time_range` | `TimeRange` | Primary window |
| `comparison_window` | `Optional[TimeRange]` | If set, returns a delta vs. this window |
| `aggregation` | `AggregationType` | Default `mean` |

**Result (`data` when `status == "ok"`):**

```python
class MetricSeriesPoint(BaseModel):
    metric_name: MetricName
    window: str            # "primary" | "comparison"
    aggregated_value: float
    sample_count: int


class QueryMetricsResult(BaseModel):
    series: List[MetricSeriesPoint]
    delta: Optional[Dict[str, float]] = None
    # present only if comparison_window was supplied
    # e.g. {"accuracy": -0.14, "prediction_confidence": -0.23}
```

### 4.2 `query_logs`

Search logs for a service over a time window.

```python
class LogSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ServiceName(str, Enum):
    # TODO: finalize against your actual service topology
    INFERENCE_SERVICE = "inference_service"
    FEATURE_PIPELINE = "feature_pipeline"
    LABEL_PIPELINE = "label_pipeline"
    TRAINING_PIPELINE = "training_pipeline"


class QueryLogsInput(BaseModel):
    service: ServiceName
    time_range: TimeRange
    filter: Optional[str] = None       # substring or regex
    severity: Optional[List[LogSeverity]] = None
    max_results: int = Field(default=50, le=200)
```

| Field | Type | Description |
|---|---|---|
| `service` | `ServiceName` | Which service to search |
| `time_range` | `TimeRange` | — |
| `filter` | `Optional[str]` | Substring/regex applied to message text |
| `severity` | `Optional[List[LogSeverity]]` | Restrict to these levels |
| `max_results` | `int` | Capped at 200 |

**Result:**

```python
class LogEntry(BaseModel):
    timestamp: datetime
    service: ServiceName
    severity: LogSeverity
    message: str


class QueryLogsResult(BaseModel):
    entries: List[LogEntry]
    truncated: bool
```

### 4.3 `query_deployment_history`

Retrieve deployment events for a service.

```python
class QueryDeploymentHistoryInput(BaseModel):
    service: ServiceName
    time_range: TimeRange
```

**Result:**

```python
class DeploymentEvent(BaseModel):
    timestamp: datetime
    service: ServiceName
    version_before: str
    version_after: str
    commit_sha: str
    deployed_by: str
    is_rollback: bool


class QueryDeploymentHistoryResult(BaseModel):
    deployments: List[DeploymentEvent]
```

`commit_sha` from a `DeploymentEvent` is the expected input to
`query_code_diffs` (§4.5) — the typical flow is: find a suspicious
deployment here, then inspect what actually changed there.

### 4.4 `query_feature_distributions`

Compare feature distributions between a baseline and comparison window —
the agent's primary tool for detecting drift.

```python
class DriftMetric(str, Enum):
    PSI = "psi"


class QueryFeatureDistributionsInput(BaseModel):
    features: List[str]          # validated against canonical list,
                                  # not enum-constrained — see §2.4
    baseline_window: TimeRange
    comparison_window: TimeRange
    drift_metric: DriftMetric = DriftMetric.PSI
```

> **Drift metric: PSI (Population Stability Index), fixed as the default.**
> This was flagged as undocumented in `CLAUDE.md`. PSI is the right default
> here over KL divergence or KS statistic because it handles both continuous
> (binned) and categorical features uniformly, and has conventional,
> citable thresholds: `< 0.1` = no significant change, `0.1–0.25` =
> moderate shift, `> 0.25` = significant drift. KS statistic only applies to
> continuous features; KL divergence is undefined at zero-probability bins
> and has no standard interpretable threshold. Pin PSI as the default now —
> if you don't, different runs of the eval harness can silently use
> different drift metrics with different threshold semantics, and your
> calibration curve stops meaning anything across runs.

**Result:**

```python
class FeatureDriftResult(BaseModel):
    feature_name: str
    drift_score: float
    drift_metric_used: DriftMetric
    exceeds_threshold: bool        # drift_score > 0.25 under PSI
    baseline_mean: Optional[float] = None
    comparison_mean: Optional[float] = None


class QueryFeatureDistributionsResult(BaseModel):
    results: List[FeatureDriftResult]
```

### 4.5 `query_code_diffs`

Retrieve the actual diff between two commits — used after
`query_deployment_history` surfaces a suspicious `commit_sha`, to inspect
whether the change plausibly explains the failure.

```python
class QueryCodeDiffsInput(BaseModel):
    commit_before: str
    commit_after: str
    paths: Optional[List[str]] = None   # glob filters, e.g. "src/features/**"
```

**Result:**

```python
class FileDiff(BaseModel):
    path: str
    additions: int
    deletions: int
    patch: str           # unified diff text, truncated if large

class QueryCodeDiffsResult(BaseModel):
    files_changed: List[FileDiff]
    truncated: bool
```

---

## 5. Design Decisions (Proposed)

1. **Time ranges are always absolute, anchored to `investigation_start`.**
   Required for deterministic chaos injection replay (§2.1).

2. **Empty results are successful, not errors.** `ToolError` is reserved for
   genuine execution failures only (§2.2).

3. **All tool calls count against `tool_call_budget`, including ones that
   return errors.** A malformed call is itself a wasted investigative step —
   not counting it would let the tool-selection-efficiency metric hide
   inefficient behavior rather than measure it.

4. **Small catalogs are enum-constrained in the schema; large catalogs are
   validated at the dispatcher with `valid_values` on miss.** Resolves the
   mean-tool-calls inflation issue without adding a discovery-only tool
   (§2.4).

5. **PSI is the fixed default drift metric**, with documented thresholds
   (§4.4).

6. **Transient infra errors (`service_unavailable`, `timeout`) are retried
   once, transparently, inside the Tool Dispatcher — not counted against the
   budget and not shown to the model unless the retry also fails.**
   Rationale: chaos injection is meant to test reasoning under *injected
   failure modes*, not your own tool backend's flakiness. If the retry also
   fails, it's surfaced as a real, budget-counted error — at that point it
   might genuinely be the injected failure (e.g., a "dependency outage"
   scenario), and the agent should reason about it like any other evidence.
   **This one is the least confident of the six — flag if it conflicts with
   how you want infra-flakiness scenarios to be scored.**

---

## 6. Open Issues

- [ ] `MetricName`, `ServiceName`, and the feature catalog are placeholders.
  These need to be finalized against your actual system under test before
  any chaos injection scenario can be built against them — this is a
  blocking dependency for the chaos injection framework, not just a nice-to-have.
- [ ] No retry policy was previously documented; §5.6 above is a proposal,
  not a retrieved decision. Confirm it matches your intent for how
  infra-failure chaos scenarios should be scored.
- [ ] `ToolErrorType` may not be exhaustive — revisit once the Tool
  Dispatcher is actually implemented against real backends and you hit error
  cases this list doesn't cover.
