# QA Plan — ML Investigator

**Version:** 1.0  
**Last updated:** 2026-08-09  
**Scope:** `src/`, `tests/`, `target-system/`, `docs/`

---

## 1. Purpose

This document defines the quality assurance strategy for the ML Investigator — a hand-rolled ReAct agent that diagnoses degraded ML systems without any agent framework. It covers what is tested, how, and what "passing" means at each level.

The goal of the test suite is not just coverage; it is **catching regressions that would silently degrade diagnosis accuracy** — hallucinated hypothesis IDs, taxonomy drift, confidence scores that are rank-derived rather than model-elicited, and tool implementations that diverge from the spec.

---

## 2. Quality Dimensions

| Dimension | What it means for this project |
|---|---|
| **Correctness** | `update_graph` applies deltas faithfully, likelihoods normalize, rules-out propagate |
| **Safety** | `validate_graph_update` always rejects hallucinated IDs and invalid schema |
| **Taxonomy integrity** | `FailureCategory` enum stays 1:1 with `chaos-taxonomy.md` (CI-enforced) |
| **Leakage prevention** | Eval-side taxonomy never appears in agent-facing prompts, schemas, or Pydantic models |
| **Tool contract** | All five tools return the unified error envelope; invalid inputs return errors, not crashes |
| **Scenario fidelity** | Mock runner fixtures reflect realistic agent behavior; malformed updates are always rejected |
| **Observability** | `snapshot_graph` produces valid, parseable JSON at every turn |

---

## 3. Test Architecture

```
tests/
├── conftest.py                        # sys.path setup (adds src/)
├── test_taxonomy_sync.py              # CI guard: enum == taxonomy doc
├── test_no_taxonomy_leakage.py        # CI guard: enum not in agent artifacts
│
├── unit/
│   ├── test_hypothesis_graph.py       # Enums, models, update_graph logic
│   ├── test_output_validator.py       # ValidationResult, all validation checks
│   ├── test_agent_helpers.py          # build_initial_graph, _graph_context, print_top_hypothesis
│   ├── test_tools.py                  # dispatch_tool, tool implementations, PSI
│   └── test_persistence.py           # snapshot_graph JSON round-trip
│
├── integration/
│   ├── test_graph_pipeline.py         # validate → update pipeline, multi-turn sequences
│   └── test_tool_dispatch.py          # Real SQLite + git backends
│
└── e2e/
    └── test_scenarios.py              # Full mock runner across all 3 scenarios
```

**Total tests:** 214  
**External dependencies required:** SQLite databases in `target-system/data/`, git history in `target-system/pipeline_repo/`  
**Anthropic API calls:** None — all tests are offline by design

---

## 4. Test Levels

### 4.1 Unit Tests

**Goal:** Verify each module in isolation with no I/O, no filesystem, no API.

#### `test_hypothesis_graph.py`
Tests the Pydantic data models and `update_graph` mutation logic.

| Area | Tests |
|---|---|
| `derive_evidence_type` | All 5 registered tools map correctly; unregistered tool raises `ValueError` |
| `Evidence.from_input` | `evidence_type` always derived from `tool_called`, never manually set |
| `EvidenceInput` validation | `confidence_delta` outside `[-1.0, 1.0]` rejected by Pydantic |
| `update_graph` — action="create" | New hypothesis appended, sequential ID assigned, evidence attached, severity and likelihood defaults applied |
| `update_graph` — action="update" | Evidence attached to `current_focus`; raises `ValueError` if focus is missing or not in graph |
| `update_graph` — action="merge" | Evidence attached to `merge_into_id`; no new hypothesis created; raises `ValueError` on bad ID |
| Likelihood delta capping | Deltas > `±0.25` capped before application; raw value from update is never used directly |
| Likelihood floor/ceiling | After capping, likelihoods clamped to `[0.0, 1.0]` |
| Normalization | After each update, active hypotheses sum to 1.0 (±rounding to 4 d.p.); ruled-out hypotheses excluded |
| Rule-out | `hypotheses_to_rule_out` sets status to `RULED_OUT`; other hypotheses unaffected |
| Established facts | `new_established_facts` appended to graph; prior facts preserved |
| `FailureCategory` enum | 18 values, all lowercase snake_case strings |

#### `test_output_validator.py`
Tests `validate_graph_update` — the gate that must reject all malformed model output.

| Check | Expected outcome |
|---|---|
| action="update", valid `current_focus` | PASS |
| action="update", `current_focus` missing | FAIL — error mentions `current_focus` |
| action="update", `current_focus` not in graph | FAIL — error quotes the bad ID |
| action="merge", valid `merge_into_id` | PASS |
| action="merge", `merge_into_id` missing or not in graph | FAIL |
| action="create", non-empty description | PASS |
| action="create", empty or `None` description | FAIL |
| action="create", invalid severity string | FAIL |
| action="create", all valid severity values | PASS |
| action="create", `initial_likelihood` outside `[0, 1]` | FAIL |
| Unregistered tool in `new_evidence.tool_called` | FAIL |
| Hallucinated ID in `likelihood_changes` | FAIL |
| Hallucinated ID in `hypotheses_to_rule_out` | FAIL |
| Multiple violations in one update | All errors collected; `errors` list has ≥ 2 entries |
| All 5 registered tools | PASS for each |
| `validate_graph_update` called with any inputs | Never raises — always returns `ValidationResult` |

#### `test_agent_helpers.py`
Tests agent module helpers without calling the Anthropic API.

| Function | Tests |
|---|---|
| `build_initial_graph` | Correct alert, budget, zero tool calls, empty hypotheses, investigation_start |
| `_graph_context` (empty) | Contains "No hypotheses yet" and budget status |
| `_graph_context` (populated) | Active hypotheses sorted by likelihood descending; evidence labeled SUPPORTS/CONTRADICTS; ruled-out section; established facts; open questions; investigation_start timestamp |
| `print_top_hypothesis` | No active hypotheses → fallback message; picks highest-likelihood active (skips ruled-out); shows likelihood %, evidence count, category |
| `DiagnosisResult.__str__` | All four fields present in output |

#### `test_tools.py`
Tests tool implementations and `dispatch_tool`.

| Area | Tests |
|---|---|
| `dispatch_tool` | Unknown tool returns error envelope; routes each known tool |
| `TOOL_DEFINITIONS` | 7 tools, all unique names, required fields present |
| `_query_metrics` | Returns only requested metrics; comparison window adds delta; no comparison → no delta; accuracy shows degradation; latency shows stability |
| `_query_logs` | `feature_pipeline` returns schema errors; `inference_service` returns nominal; severity filter applied; unknown service returns empty |
| `_query_code_diffs` | Missing `commit_before` or `commit_after` → error; invalid commits → error; error envelope has all required fields |
| `_psi` | Empty baseline → 0.0; empty comparison → 0.0; identical distributions → ~0.0; large shift → PSI > 0.25; always returns float; always non-negative |

#### `test_persistence.py`
Tests `snapshot_graph`.

| Check | Tests |
|---|---|
| File creation | File exists after call |
| Valid JSON | Output parses without error |
| Fidelity | `alert_summary`, hypotheses, evidence, `tool_calls_used`, established facts round-trip correctly |
| Overwrite | Existing file replaced cleanly |

---

### 4.2 Integration Tests

**Goal:** Verify that modules interact correctly when composed; tests that would catch contract drift between the validator and the graph, or between tool implementations and their real backends.

#### `test_graph_pipeline.py`
Tests the validate → update pipeline end-to-end.

| Scenario | Assertion |
|---|---|
| Valid update | Accepted by validator and applied to graph |
| Invalid update (missing `current_focus`) | Rejected; graph left unchanged |
| Rejected malformed ID | Validator catches it; likelihood not mutated |
| Create → update (2-turn) | H1 exists after create; evidence attaches on update |
| Create × 2 → rule out one (3-turn) | Ruled-out hypothesis has `RULED_OUT` status; active hypotheses renormalize |
| Normalization after ruling out | Active hypotheses sum to 1.0; ruled-out hypothesis retains its pre-rule-out likelihood |
| Merge does not create new hypothesis | Graph size unchanged; evidence count on target increases |
| 5-turn sequence | 3 hypotheses created; 1 ruled out; normalization consistent at every step |
| Evidence accumulation | 4 updates on H1 → 4 evidence items |
| Fact accumulation | Facts from 3 separate turns all present in graph |

#### `test_tool_dispatch.py`
Tests real-backend tool implementations against the SQLite databases and git repo.

| Tool | Tests |
|---|---|
| `query_deployment_history` | Returns `ok`; `deployments` list present; required fields on each record; narrow time range returns ≤ wide; invalid timestamp returns error; `result_count` consistent; all 4 services queryable |
| `query_feature_distributions` | Unknown feature returns error with `valid_values`; valid feature returns PSI score; PSI non-negative; `exceeds_threshold` consistent with `drift_score > 0.25`; invalid timestamp returns error; mixed valid/invalid features → error; empty window returns score=0 not a crash |
| `query_code_diffs` | Pipeline repo has git history; diff against itself → empty files; `result_count` consistent; records have `path`, `additions`, `deletions`, `patch`; `truncated` flag present; path filter reduces results |

---

### 4.3 End-to-End Tests

**Goal:** Verify the mock runner drives correct agent behavior across all three scenarios without the Anthropic API.

#### `test_scenarios.py`

**Fixture contract tests** (per scenario):

| Check | Assertion |
|---|---|
| All `expect_valid` flags match actual validation outcome | Mismatch = fixture is wrong |
| Last fixture has `expect_valid=False` | The malformed fixture must be at the end of every scenario |
| First fixture uses `action="create"` | Graph starts empty; first update must create |

**Mock runner E2E tests** (per scenario):

| Check | Assertion |
|---|---|
| Runner completes without exception | No unhandled errors |
| At least one fixture applied | Scenario actually exercises the graph |
| At least one fixture rejected | Malformed fixture rejection works |
| Final graph has ≥ 1 hypothesis | Graph is non-empty after the run |
| Final graph has established facts | Facts were accumulated during the run |
| Scenario has a non-empty ground truth | Scenario metadata is complete |

**Scenario-specific correctness:**

| Scenario | Check |
|---|---|
| `feature_drift` | Top active hypothesis description mentions "feature", "drift", "schema", or "distribution" |
| `bad_deployment` | At least one hypothesis description mentions "deploy", "regression", or "version" |
| `label_corruption` | At least one hypothesis description mentions "label", "corruption", or "join" |

**Registry tests:**

| Check | Assertion |
|---|---|
| Exactly 3 scenarios registered | |
| Expected keys: `feature_drift`, `bad_deployment`, `label_corruption` | |
| Each scenario: `id` matches key, non-empty `label`/`alert`/`ground_truth`, ≥1 fixture, callable `dispatch_tool` | |
| `dispatch_tool` accepts all 5 registered tool names | Returns a dict with `status` key |

---

### 4.4 CI Guard Tests (Pre-existing)

These two test files are the highest-priority tests in the suite — they enforce architectural invariants that, if violated, silently corrupt evaluation results.

#### `test_taxonomy_sync.py`
Parses `### N. \`id\`` headings from `docs/chaos-taxonomy.md` and asserts exact equality with `{e.value for e in FailureCategory}`.

**Failure mode it prevents:** Adding a chaos injection scenario without updating the enum (or vice versa), causing the evaluation harness to use a category that doesn't exist in the taxonomy.

#### `test_no_taxonomy_leakage.py`
Asserts that no `FailureCategory` identifier (class name or member name) appears in:
- `agent._SYSTEM_PROMPT`
- Field annotations of `EvidenceInput` and `GraphUpdate`
- JSON serialization of `TOOL_DEFINITIONS`

**Failure mode it prevents:** The model learning about the eval taxonomy through the prompt or tool schemas, making accuracy metrics meaningless (the agent could just enumerate categories rather than reason about evidence).

---

## 5. Evaluation Metrics (Agent Quality, Not Test Pass/Fail)

These are measured by running the agent (with API calls) against chaos injection scenarios and are separate from the test suite.

| Metric | Definition | Target |
|---|---|---|
| **Top-1 accuracy** | Correct root cause is the highest-likelihood active hypothesis when `stop_investigation` is called | ≥ 70% on easy tier |
| **Top-3 accuracy** | Correct root cause is in the top 3 by likelihood | ≥ 85% on easy tier |
| **Agent vs. rule-based delta** | Agent Top-1 accuracy minus rule-based baseline | ≥ +10 pp |
| **Mean tool calls to diagnosis** | Average `tool_calls_used` when `stop_investigation` fires | ≤ 5 |
| **Hypothesis calibration** | Correlation between stated confidence and empirical accuracy across runs | Brier score ≤ 0.25 |
| **Tool selection efficiency** | Fraction of tool calls that directly shift the top hypothesis likelihood by ≥ 0.05 | ≥ 60% |
| **LLM-as-judge score** | External model rates the quality of the diagnosis narrative | ≥ 4/5 average |

> **Confidence elicitation requirement:** Confidence scores submitted via `stop_investigation` must be explicitly requested from the model in the prompt — never derived by rank. This is a hard constraint, not a soft goal. The evaluation script must verify that confidence values are present in the model's reasoning before `stop_investigation`, not computed post-hoc.

---

## 6. What Is NOT Tested

| Area | Reason |
|---|---|
| `run_investigation` (live ReAct loop) | Requires Anthropic API; tested manually or in eval harness |
| Anthropic API error handling | API reliability is Anthropic's SLA, not ours |
| `target-system/` generator and inference service | Covered by separate test suite in `target-system/tests/` |
| Langfuse tracing | Observability layer; integration tested separately |
| Prompt quality / reasoning paths | Cannot be unit tested; covered by LLM-as-judge eval metric |
| Hypothesis calibration curves | Requires many runs; covered by eval harness, not pytest |

---

## 7. Running the Test Suite

```bash
# All tests
PYTHONPATH=src python -m pytest tests/ -q

# Unit tests only (no filesystem I/O)
PYTHONPATH=src python -m pytest tests/unit/ -q

# Integration tests (requires SQLite DBs and pipeline_repo)
PYTHONPATH=src python -m pytest tests/integration/ -q

# End-to-end mock runner
PYTHONPATH=src python -m pytest tests/e2e/ -q

# CI guards only (run first — fast, highest priority)
PYTHONPATH=src python -m pytest tests/test_taxonomy_sync.py tests/test_no_taxonomy_leakage.py -v
```

Expected output: **214 passed** in ≤ 10 seconds (unit + integration + e2e; no API calls).

---

## 8. Adding New Tests

### When adding a new FailureCategory
1. Add the entry to `docs/chaos-taxonomy.md` (with `### N. \`id\`` heading format).
2. Add the corresponding member to `FailureCategory` enum in `src/hypothesis_graph.py`.
3. `test_taxonomy_sync.py` will pass only when both are in sync.
4. Add at least one scenario to `src/scenarios.py` and fixture coverage in `tests/e2e/test_scenarios.py`.

### When adding a new tool
1. Add the tool to `TOOL_DEFINITIONS` in `src/tools.py`.
2. Add it to `TOOL_TO_EVIDENCE_TYPE` in `src/hypothesis_graph.py`.
3. Add dispatch routing in `dispatch_tool`.
4. Add unit tests in `tests/unit/test_tools.py` (mock inputs + error cases).
5. Add integration tests in `tests/integration/test_tool_dispatch.py` if it has a real backend.

### When modifying `validate_graph_update`
Add a test to `tests/unit/test_output_validator.py` for every new check — one test for the valid case, one for each invalid case. The `test_validate_never_raises` test must continue to pass regardless of inputs.

### When modifying `update_graph`
Add tests to `tests/unit/test_hypothesis_graph.py` and verify normalization and fact-accumulation behavior in `tests/integration/test_graph_pipeline.py` with a multi-turn sequence.

---

## 9. Known Gaps

| Gap | Risk | Plan |
|---|---|---|
| `scenario.id` does not map 1:1 to `FailureCategory` values (`label_corruption` vs. `label_pipeline_corruption`) | Evaluation harness could misattribute diagnoses | Standardize scenario IDs to match enum values in a future refactor |
| No test for `_graph_context` round-tripping into the model prompt | Prompt formatting bugs could be invisible | Add a test that parses `_graph_context` output and checks all hypotheses are present |
| No adversarial test for `confidence_delta` at exactly `±1.0` | Edge case in capping logic | Add boundary tests for `_MAX_LIKELIHOOD_DELTA` |
| No test for budget exhaustion path in `run_investigation` | Budget logic not exercised without API | Add a mock for the API client to test the exhaustion branch |
| `query_metrics` and `query_logs` are static — they don't reflect the injected chaos | Wrong scenario could pass with wrong fixture | Wire chaos injection state into tool responses before moving to production eval |
