# MLSys Investigator — Session Handoff

Last updated: 2026-07-22. Branch: `mvp/react-loop-mock-runner`.

---

## What This Project Is

An agentic failure-diagnosis system for distributed ML systems. A hand-rolled ReAct loop
(raw Anthropic API, no frameworks) investigates a degraded ML system by querying metrics,
logs, deployment history, feature distributions, and code diffs, while maintaining a
structured `HypothesisGraph` as its belief state. Produces a ranked root-cause diagnosis.

**Non-negotiables (from CLAUDE.md — do not debate these):**
- Raw Anthropic SDK only. No LangChain, no agent frameworks.
- Pydantic for all structured data.
- `FailureCategory` enum is derived 1:1 from `docs/chaos-taxonomy.md` (single source of truth).
- Langfuse for observability (not yet wired).

---

## Module Status

| File | Status | Notes |
|------|--------|-------|
| `src/hypothesis_graph.py` | **Complete** | All Pydantic models, `update_graph()`, `derive_evidence_type()`, delta capping at ±0.25, normalization over active hypotheses |
| `src/output_validator.py` | **Complete** | `validate_graph_update(update, graph)` → `ValidationResult`; 5 checks (registered tool, existing IDs, no collisions, empty evidence on new hypotheses, current_focus set) |
| `src/persistence.py` | **Complete** | `snapshot_graph(graph)` → writes `hypothesis_graph.json` at project root; VS Code auto-reloads it |
| `src/tools.py` | **Partial** | Definitions for all 4 tools (`query_metrics`, `query_logs`, `update_hypothesis_graph`, `stop_investigation`). Only `query_metrics` and `query_logs` have implementations — both return **hardcoded** static responses. `query_deployment_history`, `query_feature_distributions`, `query_code_diffs` are defined but NOT wired to real backends. |
| `src/agent.py` | **Complete (this session)** | Full ReAct loop with GraphUpdate integration (see below). Graph initialized, threaded, updated, and snapshotted on every turn. |
| `src/run_mock.py` | **Complete** | Drives the loop with canned fixtures (no API calls). Use to verify graph update pipeline without spending tokens. Run: `python -m src.run_mock --scenario feature_drift` |
| `src/scenarios.py` | **Complete** | 3 canned scenarios with canned fixtures: `feature_drift`, `bad_deployment`, `label_corruption`. Ground truth and expected hypothesis IDs documented inside. |
| `src/fixtures.py` | **Complete** | Raw fixture dicts used by `run_mock.py` |
| `docs/chaos-taxonomy.md` | **Complete (this session)** | 18 failure categories across easy/medium/hard tiers. Single source of truth for `FailureCategory` enum. |
| `docs/tool-specs.md` | **Complete (spec)** | Full interface contracts for all 5 query tools + unified error envelope |
| `docs/hypothesis-graph-spec.md` | **Complete (spec)** | Full data model, enums, design decisions |
| `docs/MLSYSTEM.md` | **Complete** | Architecture of the synthetic target ML system |
| `target-system/` | **Complete** | Fully built synthetic ML target (see below) |

---

## What Was Done This Session

### 1. GraphUpdate Integration in the Real Agent Loop (`src/agent.py`, `src/tools.py`)

The mock runner had always proven the full update pipeline worked. This session wired it
into the live Anthropic API loop.

**Changes in `src/tools.py`:**
- Added `update_hypothesis_graph` tool definition with the full `GraphUpdate` schema
  as its `input_schema`. It includes `current_focus` (hypothesis ID), `new_evidence`,
  `likelihood_changes`, `hypotheses_to_rule_out`, `new_established_facts`,
  `next_experiment_rationale`.

**Changes in `src/agent.py`:**
- Added imports: `EvidenceInput`, `GraphUpdate`, `update_graph` from `hypothesis_graph`;
  `validate_graph_update` from `output_validator`.
- Extended system prompt with an explicit **Protocol** section: call a query tool → call
  `update_hypothesis_graph` → repeat → `stop_investigation`. States that
  `update_hypothesis_graph` is free (does not count against the tool budget).
- In the dispatch loop, added an `if name == "update_hypothesis_graph":` branch that:
  1. Sets `graph.current_focus` from the tool input
  2. Constructs a `GraphUpdate` from the tool's parameters
  3. Calls `validate_graph_update(graph_update, graph)` → returns validation errors to
     Claude as a tool result if invalid (so Claude can self-correct)
  4. On success: calls `update_graph(graph, graph_update)`, `snapshot_graph(graph)`,
     `print_top_hypothesis(graph)`
  5. Returns a compact JSON result with current active likelihoods and established facts count
  6. **Does NOT increment `graph.tool_calls_used`** (graph updates are free)

**How the loop now works end-to-end:**
```
[build_initial_graph]   → 4 hypotheses, H1–H4, even prior
[snapshot_graph]        → hypothesis_graph.json written
[first user message]    → includes full _graph_context() render + investigation prompt
[Claude]                → calls query_metrics(...)
[loop]                  → dispatches → tool_calls_used += 1
[Claude]                → calls update_hypothesis_graph(current_focus="H1", ...)
[loop]                  → validates → update_graph → snapshot → print_top_hypothesis
                           → does NOT increment tool_calls_used
[Claude]                → calls query_logs(...)
[loop]                  → dispatches → tool_calls_used += 1
[Claude]                → calls update_hypothesis_graph(current_focus="H1", ...)
[loop]                  → validates → update_graph → snapshot → print_top_hypothesis
[Claude]                → calls stop_investigation(...)
[loop]                  → tool_calls_used += 1 → DiagnosisResult → done
```

### 2. Chaos Taxonomy (`docs/chaos-taxonomy.md`)

Created 18 failure categories across 3 tiers. Each entry has: `id` (becomes enum value),
`tier`, `description`, `primary_signal`, `distinguishing_signal`, `injection` method.

Tiers:
- **Easy (5):** `feature_drift`, `bad_deployment`, `upstream_schema_change`,
  `infrastructure_latency_spike`, `model_version_rollback_regression`
- **Medium (8):** `label_pipeline_corruption`, `training_serving_skew`,
  `data_freshness_degradation`, `feature_encoding_bug`, `gradual_concept_drift`,
  `model_calibration_drift`, `shadow_mode_leak`, `feature_pipeline_partial_failure`
- **Hard (5):** `delayed_label_feedback_shift`, `cascading_upstream_failure`,
  `model_staleness`, `feature_importance_inversion`, `compound_drift_plus_deployment`

Also includes a CI test snippet at the bottom of the taxonomy doc that enforces
`FailureCategory` enum values == taxonomy `id` fields.

---

## What Is NOT Built Yet (Next Steps, Priority Order)

### Next: Expand `FailureCategory` Enum + Add CI Test

`src/hypothesis_graph.py` currently has only 4 values:
```python
class FailureCategory(str, Enum):
    FEATURE_DRIFT = "feature_drift"
    BAD_DEPLOYMENT = "bad_deployment"
    LABEL_PIPELINE_CORRUPTION = "label_pipeline_corruption"
    TRAINING_SERVING_SKEW = "training_serving_skew"
```

Needs to be expanded to all 18 values from `docs/chaos-taxonomy.md`. The enum value
(e.g. `FEATURE_DRIFT = "feature_drift"`) is the slug from the taxonomy doc's `id` field.

Also add `tests/test_taxonomy_sync.py` (template in `docs/chaos-taxonomy.md` § CI Enforcement).

Also update `stop_investigation` tool's `root_cause_category` enum in `src/tools.py`
to include all 18 values (currently only has 4).

Also update `build_initial_graph()` in `src/agent.py` to seed 4–5 hypotheses from the
full 18-category set (pick the most plausible given the alert).

### After: Wire Real Tool Implementations to Target System

Three tools need real backends (contracts in `docs/tool-specs.md`):

**`query_deployment_history`** → `target-system/data/deployments.db`
- Table: `deployments` with columns: `timestamp`, `service`, `version_before`,
  `version_after`, `commit_sha`, `change_type`, `changelog`, `deployed_by`, `is_rollback`
- Query function `query_deployments(time_range, service)` already exists in target-system
- Add implementation in `src/tools.py` that calls this and wraps in unified envelope

**`query_feature_distributions`** → `target-system/data/feature_store.db`
- Table: `requests` with columns: `request_id`, `timestamp`, `features (JSON)`,
  `prediction`, `pred_proba`, `true_label`
- Must compute PSI between baseline and comparison windows
- PSI formula: Σ (P_observed - P_expected) × ln(P_observed / P_expected)
- Thresholds: <0.1 = no drift, 0.1–0.25 = moderate, >0.25 = significant
- Query function `query_feature_values()` already in target-system

**`query_code_diffs`** → `target-system/pipeline_repo/` (nested git repo with 7 commits)
- Takes `commit_before`, `commit_after`, optional `paths[]`
- Run `git diff <before> <after> -- <paths>` inside `pipeline_repo/`
- Returns `FileDiff[]` with patch text, add/delete counts, `truncated` bool
- Cap at ~200 lines per file to avoid context overflow

**`query_metrics`** and **`query_logs`** currently return hardcoded static responses.
Once chaos injection is built, replace them with real SQLite queries against
`target-system/data/metrics.db` and `target-system/data/logs.db`.

### After: Chaos Injection Framework

Use the target system's `SubRangeConfig` mechanism to inject failures:
```python
# In target-system/generator/generate.py
config.overrides = [
    SubRangeConfig(
        start_ts=SIM_START + 3 * 86400,
        end_ts=SIM_START + 5 * 86400,
        feature_params=drifted_features,
    )
]
generate_data(config)  # rewrites all 5 evidence stores causally
```

Build one scenario per taxonomy category (start with easy tier). Each scenario:
1. Specifies the `SubRangeConfig` override that injects the failure
2. Records the ground truth (`FailureCategory`, affected time window)
3. Runs the agent against the injected data
4. Scores top-1 and top-3 accuracy

### After: Stopping Criteria Module

Not yet built. Spec: queries `HypothesisGraph` directly (not through ReAct loop).
Convergence checks to implement:
- Top hypothesis likelihood > 0.60
- Likelihood ratio (top vs. runner-up) > 2.5×
- No likelihood change > 0.05 in the last N updates (investigation stalled)

### After: Langfuse Integration

Not yet built. Add spans at: ReAct loop turns, each tool dispatch, each graph update,
session start/end. Export calibration curve data (confidence vs. accuracy).

### After: Rule-Based Baseline

A heuristic agent (no LLM) for comparison:
- `error_rate` spike + deployment event → `bad_deployment`
- `prediction_confidence` stable + `accuracy` drops → `label_pipeline_corruption`
- `prediction_confidence` drops + no deployment → `feature_drift`
- etc.
Needed to compute "agent vs. baseline delta" evaluation metric.

---

## How to Run

```bash
# Verify graph update pipeline (no API key needed)
cd "src"
python -m run_mock --scenario feature_drift
python -m run_mock --scenario bad_deployment
python -m run_mock --scenario label_corruption
python -m run_mock --list   # show all scenarios with ground truth

# Run the live agent (needs ANTHROPIC_API_KEY)
cd "src"
ANTHROPIC_API_KEY=sk-... python -m agent

# Watch the graph update in real time
# Open hypothesis_graph.json in VS Code — it auto-reloads after each graph update
```

---

## Key File Locations

| What | Where |
|------|-------|
| ReAct loop | `src/agent.py` |
| Hypothesis graph models + update logic | `src/hypothesis_graph.py` |
| Tool definitions + hardcoded impls | `src/tools.py` |
| Output validator | `src/output_validator.py` |
| Graph snapshot to disk | `src/persistence.py` |
| Mock runner (no API) | `src/run_mock.py` |
| Canned scenarios + fixtures | `src/scenarios.py`, `src/fixtures.py` |
| Live graph snapshot | `hypothesis_graph.json` (project root) |
| Chaos taxonomy (source of truth) | `docs/chaos-taxonomy.md` |
| Tool interface contracts | `docs/tool-specs.md` |
| Hypothesis graph spec | `docs/hypothesis-graph-spec.md` |
| Target system architecture | `docs/MLSYSTEM.md` |
| Target system databases | `target-system/data/` (metrics.db, logs.db, deployments.db, feature_store.db) |
| Target system generator | `target-system/generator/generate.py` |
| Target system pipeline repo | `target-system/pipeline_repo/` (nested git repo) |
| Target system tests | `target-system/tests/` |

---

## Important Design Decisions (Don't Revisit)

- **`evidence_type` is never model-set.** Always derived from `tool_called` via
  `TOOL_TO_EVIDENCE_TYPE` lookup in `hypothesis_graph.py`. This prevents hallucinated
  evidence types.
- **Likelihood delta capped at ±0.25 per update.** Prevents the model from collapsing
  to 100% confidence after one supporting observation.
- **Likelihoods normalized over active hypotheses only.** Ruled-out hypotheses don't
  participate in normalization.
- **`update_hypothesis_graph` does not count against tool budget.** It's bookkeeping,
  not investigation. Only the 5 query tools + `stop_investigation` consume budget.
- **New hypotheses must arrive with empty evidence.** Enforced by `output_validator.py`.
- **Stopping Criteria queries graph directly, not through ReAct loop.** Avoids god-object
  anti-pattern (see component diagram in `docs/component-diagram.puml`).
- **Absolute timestamps only in tool calls.** Relative time ("last 6 hours") is resolved
  to absolute at the call site to ensure deterministic chaos replay.
