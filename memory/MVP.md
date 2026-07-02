# MLSys Investigator — MVP Overview

## What was built

A minimal end-to-end skeleton of the agentic failure diagnosis system described in `CLAUDE.md`. No real ML system is connected yet — all tool responses are hardcoded — but every structural piece of the architecture is present and wired together: the ReAct loop, the Hypothesis Graph, the Tool Dispatcher, the Output Validator, and the live JSON view.

The goal was to prove that the core loop works before touching a real Anthropic API call or a real backend: the agent reasons, calls tools, gets results, validates its own structured output, and updates a belief state — all verifiably, with no external dependencies beyond `anthropic` and `pydantic`.

---

## Hardcoded scenarios

Three self-contained scenarios are implemented in `src/scenarios.py`. Each targets a different failure category and is designed around a distinct discriminating signal pattern — the metric or log observation that lets a well-reasoning agent rule out the other three hypotheses efficiently.

| Scenario ID | Label | Key discriminating signal | Correct hypothesis |
|---|---|---|---|
| `feature_drift` | Upstream Schema Change | `accuracy` ↓, `prediction_confidence` ↓, `error_rate` **flat** → `feature_pipeline` logs show schema validation errors and upstream `event_date` column removal | H1/H5 — `feature_drift` |
| `bad_deployment` | Feature Normalizer Shape Mismatch | `accuracy` ↓, `prediction_confidence` ↓, `error_rate` **spikes 850%** (0.002→0.019), `latency_p99` elevated → `inference_service` logs show deployment event + `ValueError: scaler shape mismatch (1,14) vs (1,28)` | H2 — `bad_deployment` |
| `label_corruption` | Join Key Misconfiguration | `accuracy` ↓ but `prediction_confidence` **stable** (0.84→0.83) — model is confident, so inputs are fine; it's being evaluated against wrong labels → `label_pipeline` logs show join key changed `session_id`→`request_id`, label alignment rate 61.2% | H3 — `label_pipeline_corruption` |

The `prediction_confidence` signal is the sharpest discriminator across scenarios: if it drops alongside accuracy, the model is receiving bad inputs (`feature_drift` or `bad_deployment`); if it holds steady while accuracy drops, the model itself is healthy and the evaluation labels are wrong (`label_corruption`).

Run any scenario with:

```bash
PYTHONPATH=src python3 src/run_mock.py --scenario <id>   # feature_drift | bad_deployment | label_corruption
PYTHONPATH=src python3 src/run_mock.py --list             # show all scenarios with ground truth
```

---

## Files

### `src/hypothesis_graph.py`

**What it is:** All Pydantic models for the agent's structured belief state.

**Why it exists:** The agent's reasoning is only as auditable as its structured output. Rather than having the model write free-text conclusions, every belief update is a typed delta (`GraphUpdate`) applied programmatically to a typed graph (`HypothesisGraph`). This makes calibration curves, tool-selection efficiency metrics, and failure mode analysis all computable after the fact — you can reconstruct exactly what the agent believed and why at every step.

**Key models:**

- `HypothesisGraph` — the full session state: alert summary, hypotheses, established facts, open questions, tool budget, and current focus. This is what gets serialized to `hypothesis_graph.json` after every update.
- `Hypothesis` — one candidate root cause. Has a `likelihood` float (0–1), a `status` (active/ruled_out/confirmed), and a list of `Evidence` objects accumulated over the investigation.
- `GraphUpdate` — the structured delta the model emits after each tool call. Contains new evidence, likelihood changes, hypotheses to rule out, surprising new hypotheses, and new established facts. The graph is never rewritten wholesale — only deltas are applied. This produces an audit trail of every belief change.
- `EvidenceInput` vs `Evidence` — two separate schemas. The model produces `EvidenceInput` (no `evidence_type` field). The `evidence_type` is always derived programmatically from `tool_called` via `TOOL_TO_EVIDENCE_TYPE`. This prevents the model from setting `tool_called: "query_metrics"` and `evidence_type: "logs"` simultaneously, which would silently corrupt the tool-selection-efficiency metric.

**Key design decisions baked in:**

- Likelihoods are normalized programmatically after every update (over active hypotheses only), not schema-enforced. Asking the model to emit valid probability distributions directly produces garbage.
- Per-update likelihood deltas are capped at ±0.25. Without this, the model overreacts to weak evidence and terminates early.
- `update_graph()` is the only place `Evidence` objects are constructed. No other code path creates them directly.

---

### `src/tools.py`

**What it is:** Two things in one file — (1) the Anthropic API tool definitions (JSON schema objects passed to `client.messages.create()`), and (2) the hardcoded Python implementations that return static responses.

**Why it exists:** In the full system these would hit real backends (Prometheus, a log store, a deployment history API). For the MVP, they return static data that represents the injected scenario. The interface is identical to what a real implementation would expose — so swapping in real backends later is a drop-in replacement in `dispatch_tool()`.

**The three tools:**

- `query_metrics` — returns Prometheus-style before/after series for accuracy, prediction_confidence, latency_p99, and error_rate. The hardcoded response is crafted so accuracy and confidence are degraded while latency and error_rate are flat. The flat signals are evidence *against* infrastructure failure and bad deployment — they're as important as the degraded signals.
- `query_logs` — returns different static log entries depending on the `service` parameter. `feature_pipeline` returns the schema validation errors and the upstream column removal message. `inference_service` returns clean info-level logs (serving same model version, no errors). This asymmetry is what lets the agent discriminate between hypotheses.
- `stop_investigation` — a control tool, not a data tool. The model calls this when it has a final diagnosis. The ReAct loop intercepts it before dispatching, extracts the diagnosis fields, and terminates.

**`dispatch_tool(name, inputs)`** — routes by name. `stop_investigation` is handled by the loop directly, not here.

---

### `src/agent.py`

**What it is:** The hand-rolled ReAct loop against the raw Anthropic API.

**Why it exists:** The architecture constraint in `CLAUDE.md` is explicit — no LangChain, no agent frameworks, no MCP abstractions. The loop is written directly so its behavior is fully transparent and auditable. Every design choice is visible in the code rather than inherited from a framework.

**How the loop works:**

1. Build an initial `HypothesisGraph` with 4 equal-prior hypotheses and snapshot it to `hypothesis_graph.json`.
2. Serialize the graph into the first user message so the model knows its starting belief state.
3. Call `client.messages.create()` with the three tool definitions.
4. If the model returns `stop_reason: "tool_use"`, iterate over the `tool_use` content blocks:
   - If `stop_investigation`: extract the diagnosis, set `done = True`.
   - Otherwise: call `dispatch_tool()`, collect the result as a `tool_result` content block.
5. Append tool results as a new user message and loop.
6. Terminate when `stop_investigation` is called, the budget is exhausted, or the model returns `end_turn`.

**What's not wired yet:** The model doesn't emit `GraphUpdate` objects in the current loop — it reasons in text and eventually calls `stop_investigation`. The `GraphUpdate` → `update_graph()` → `snapshot_graph()` chain is fully implemented in `hypothesis_graph.py` and `persistence.py`, but the real agent loop doesn't call it yet. The mock runner (`run_mock.py`) exercises that path with canned fixtures.

---

### `src/output_validator.py`

**What it is:** A gate that validates every `GraphUpdate` before it touches the graph. Returns a `ValidationResult` (valid bool + error list). Never raises — the caller decides what to do with a failure.

**Why it exists:** When a real model emits a `GraphUpdate`, it can hallucinate hypothesis IDs that don't exist, reference unregistered tool names, or create new hypotheses with pre-populated evidence (which is forbidden because `evidence_type` can't be set by the model). Without a validator, any of these would silently corrupt the graph state or raise an unhandled exception mid-loop. The validator catches all of these cases before `update_graph()` is called, and returns structured errors that can be fed back to the model as a correction signal or logged as an eval metric (hallucinated IDs are themselves a failure mode worth measuring).

**Checks performed:**

1. `new_evidence.tool_called` must be a key in `TOOL_TO_EVIDENCE_TYPE` — catches hallucinated tool names.
2. Every key in `likelihood_changes` must be an existing hypothesis ID — catches the most common hallucination pattern.
3. Every ID in `hypotheses_to_rule_out` must be an existing hypothesis ID.
4. New hypothesis IDs must not collide with existing ones.
5. New hypotheses must arrive with empty `evidence` — enforces the `EvidenceInput`/`Evidence` separation.

---

### `src/fixtures.py`

**What it is:** Four canned `GraphUpdate` objects that represent what a real model *would* emit after each tool call in the scenario, plus metadata for the mock runner (the simulated thought, the tool that was called, the inputs, and the human-readable observation).

**Why it exists:** To test the graph update machinery and the Output Validator without needing an API key. Each fixture covers a case the validator must handle:

| Fixture | What it tests |
|---|---|
| `NORMAL EVIDENCE UPDATE` | Straightforward metrics evidence, likelihoods shift and normalize correctly |
| `HYPOTHESIS RULED OUT` | `hypotheses_to_rule_out` path, likelihood reallocation over remaining active hypotheses |
| `SURPRISING RESULT — NEW HYPOTHESIS` | `new_hypotheses` insertion mid-investigation; new hypothesis must have empty evidence |
| `MALFORMED — HALLUCINATED HYPOTHESIS ID` | Validator catches `H99` in `likelihood_changes` and `H88` in `hypotheses_to_rule_out`; graph not mutated |

**One design subtlety in fixture 3:** The new hypothesis `H5` has its initial `likelihood` set directly on the `Hypothesis` object. It deliberately does *not* appear in `likelihood_changes` — referencing a new hypothesis's ID in `likelihood_changes` before it's inserted would itself be a hallucinated ID and would fail validation. Initial likelihood belongs on the `Hypothesis`, not in the delta.

---

### `src/run_mock.py`

**What it is:** A mock ReAct runner that drives the loop with the canned fixtures instead of the Anthropic API. Each iteration is one complete ReAct turn: thought → action (tool dispatch) → observation → validate → apply or reject.

**Why it exists:** Two reasons. First, it lets the full loop run without an API key. Second, it provides a deterministic, repeatable test of every component downstream of the model: tool dispatch, output validation, graph update logic, likelihood normalization, and the JSON snapshot. When the real agent loop is wired with `GraphUpdate` handling, these same paths will execute on every real turn — this verifies them in advance.

**Output:** Prints each turn's full trace (thought, tool call, raw tool result, observation, GraphUpdate summary, validation result, graph state after). Also writes `hypothesis_graph.json` after each successful update so the evolving graph is visible in real time.

---

### `src/persistence.py`

**What it is:** One function — `snapshot_graph(graph, path)` — that serializes the current `HypothesisGraph` to `hypothesis_graph.json` at the project root.

**Why it exists:** Without this, the only way to see the graph state is by reading terminal output after the run completes. With it, you can open `hypothesis_graph.json` in VS Code before the run starts and watch it update in real time as each turn completes — likelihoods shifting, hypotheses getting ruled out, evidence accumulating, new hypotheses appearing. VS Code auto-reloads JSON files on disk change. If a turn's update is rejected by the validator, the file doesn't change — which is itself a visible confirmation that the validator blocked it.

`model_dump(mode="json")` is used instead of `model_dump()` so Pydantic serializes `datetime` objects to ISO strings and enums to their string values rather than Python objects, producing valid JSON directly.

---

### `requirements.txt`

```
anthropic>=0.40.0
pydantic>=2.0.0
```

Two dependencies only. `anthropic` for the raw API client. `pydantic` for all structured data models. No agent frameworks, no LangChain, no additional tooling.

---

## How to run

**Mock runner (no API key needed):**
```bash
PYTHONPATH=src python3 src/run_mock.py
```
Open `hypothesis_graph.json` in VS Code before running to watch it update live.

**Real agent (API key required):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=src python3 src/agent.py
```

---

## What's next

The MVP skeleton is complete. The natural next steps, in order:

1. **Wire `GraphUpdate` into `agent.py`** — add an `update_hypothesis_graph` tool (or a post-tool-call prompt) so the real loop calls `validate_graph_update()` → `update_graph()` → `snapshot_graph()` after each data tool call, not just in the mock runner.
2. **Expand `FailureCategory`** — finalize the chaos injection taxonomy (15–20 categories, easy/medium/hard tiers) and derive the enum from it programmatically so the two can't drift.
3. **Build the chaos injection harness** — inject failures into the hardcoded tool responses programmatically, run the agent against each, measure top-1/top-3 accuracy.
4. **Add the rule-based baseline** — a simple heuristic agent that diagnoses without LLM reasoning, for comparison against the ReAct agent.
5. **Wire Langfuse** — add observability tracing around each tool call and graph update for the calibration curve and tool-selection-efficiency metrics.
