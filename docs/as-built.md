# ML Investigator — As-Built Document

**Compares:** [`architecture-proposal.md`](architecture-proposal.md), [`tool-specs.md`](tool-specs.md),
[`hypothesis-graph-spec.md`](hypothesis-graph-spec.md), and `CLAUDE.md`
**Against:** the codebase as of branch `mvp/react-loop-mock-runner`
**Scope:** `src/`, `target-system/`, `tests/`, `docs/`

This document records what was actually built, where it matches the architecture
proposal, where it deliberately diverged (and why, where a reason is recoverable
from the code), and where proposed scope was not built at all. It does not
re-explain the design — see the proposal documents for that context.

---

## 1. Summary

| Area | Verdict |
|---|---|
| System context (5 evidence sources, Claude, chaos framework, Langfuse) | **Matches** |
| Six-module component architecture | **Matches**, one module's responsibility shifted |
| Hypothesis Graph data model | **Matches core design**, several fields/mechanisms diverged from spec |
| Tool interfaces (5 tools, unified envelope) | **Matches**, one proposed policy (retry) not built |
| Chaos injection taxonomy (18 categories, 3 tiers) | **Matches** as a taxonomy document and enum |
| Chaos injection *scenarios* (executable) | **Matches** — all 18 taxonomy categories now have injection code (as of 2026-09-08) |
| Rule-based baseline | **Not built** |
| LLM-as-judge scoring / inter-rater reliability | **Built** (as of 2026-09-08) — `evaluation/llm_judge.py` |
| Calibration curves, tool-selection-efficiency metric | **Not built** |
| Top-1/Top-3 accuracy, tool-call count, per-run scoring | **Built** |
| Batch sweep across all 18 scenarios | **Built** (as of 2026-09-08) — `evaluation/run_sweep.py`; first full run documented in `docs/evaluation-report.md` |
| HTTP API, Alertmanager webhook, Docker/K8s deployment | **Built** — beyond original architecture scope |
| Offline "mock runner" evaluation harness | **Built** — not in the original architecture proposal |

The core control-flow architecture (ReAct loop, tool dispatch, structured belief
state, stopping criteria, output validation, tracing) was built essentially as
proposed. As of 2026-09-08, the chaos injection taxonomy is fully executable
(18/18 categories), LLM-as-judge scoring is built, and a batch sweep
(`evaluation/run_sweep.py`) has actually been run once across the full
taxonomy — see `docs/evaluation-report.md` for the results. The remaining
evaluation gap is narrower than it was: a rule-based baseline, calibration
curves, and a tool-selection-efficiency metric are still unimplemented (§9).

---

## 2. System Context — Matches

The proposed L0 context (Practitioner, five evidence sources, Anthropic API,
Chaos Injection Framework, Langfuse) is present as designed, with one structural
difference: the proposal's evidence sources are framed as pre-existing systems
"that already exist in a typical ML production stack" (§2). In the actual build,
all five are synthetic and self-hosted under `target-system/` — an XGBoost churn
model, a FastAPI inference service, a 7-day synthetic data generator writing to
SQLite, and a real git repo (`pipeline_repo/`) for `query_code_diffs`. The README
calls this out explicitly as a tradeoff: SQLite/self-hosted evidence sources make
the harness reproducible but don't represent real production infrastructure.

---

## 3. Component Architecture — Matches, with one responsibility shift

All six proposed modules exist as separate files, mapped 1:1:

| Proposed module | Built as |
|---|---|
| ReAct Loop | `src/agent.py` |
| Tool Dispatcher | `src/tools.py` |
| Hypothesis Graph Module | `src/hypothesis_graph.py` |
| Stopping Criteria | `src/stopping_criteria.py` |
| Output Validator | `src/output_validator.py` |
| Observability Layer | Langfuse spans inlined in `agent.py` |

The "Stopping Criteria and Output Validator query the graph directly, not
through the ReAct Loop" constraint (proposal §3, `CLAUDE.md`) held: neither
module is invoked by anything except `agent.py` calling into them directly with
the graph object, and neither routes state through a shared orchestrator API.

**Divergence — Output Validator's job changed.** The proposal (§3) describes it
as validating *the final graph* and producing "a well-formed, ranked report" —
i.e., a single pass at the end of a session. What was built instead
(`src/output_validator.py:1–20`) validates *each* `GraphUpdate` the model emits,
**before** it is allowed to mutate the graph, on every turn:

- registered tool name in `new_evidence.tool_called`
- every hypothesis ID referenced in `likelihood_changes` / `hypotheses_to_rule_out` exists
- action-specific structural checks (`current_focus` set for `update`, `merge_into_id` set for `merge`, non-empty description for `create`)

It never raises and always returns a `ValidationResult`; a failed validation is
fed back to the model as a rejected tool result rather than silently corrupting
the graph. The "final ranked report" responsibility that the proposal assigned
to this module ended up living in `agent.py`'s `_make_diagnosis()` /
`DiagnosisResult`, driven by the `stop_investigation` tool's own JSON schema —
there is no distinct module that validates or ranks the finished graph. This is
a real functional shift, not just a naming difference: the module became a
per-turn input gate instead of an end-of-session output gate.

---

## 4. Hypothesis Graph Data Model

### 4.1 Delivered as specified

- `HypothesisStatus`, `Severity`, `EvidenceType` enums — match exactly.
- `Evidence` / `EvidenceInput` split, with `evidence_type` derived via
  `TOOL_TO_EVIDENCE_TYPE` and never model-set — matches spec §2.4/§3.1 exactly,
  including the rejection of a hallucinated tool name via `derive_evidence_type`.
- Likelihood normalization is programmatic, over active hypotheses only —
  matches spec §5.2.
- Structured-delta update mechanism (not full graph rewrite) — matches spec §4.

### 4.2 `FailureCategory` — fully realized, plus an unplanned safeguard

The proposal explicitly left this enum as a placeholder pending the finalized
chaos taxonomy (spec §2.1, "TODO: replace with finalized... DO NOT treat as
final"). The build delivered the full 18-category enum in
`src/hypothesis_graph.py:17–37`, matching `docs/chaos-taxonomy.md` 1:1, enforced
by `tests/test_taxonomy_sync.py` — closing the proposal's own open issue.

Beyond what was asked for: `tests/test_no_taxonomy_leakage.py` asserts that no
`FailureCategory` member name appears in the system prompt, in `EvidenceInput`/
`GraphUpdate` field annotations, or in the serialized `TOOL_DEFINITIONS`. This
prevents the model from ever seeing the eval taxonomy directly (which would let
it "answer by enumeration" rather than reason from evidence) — not called for in
the proposal, but directly serves the proposal's own accuracy-metric integrity.

### 4.3 `RelationType` / `HypothesisRelation` — deliberately not built

`CLAUDE.md`'s module-structure section lists `RelationType` among the enums the
data model should have. The detailed spec (`hypothesis-graph-spec.md` §5,
decision 5) explicitly decided against this: "No explicit cross-hypothesis
relations... don't model it speculatively now." The two proposal documents
disagree with each other; the build followed the more detailed spec's decision.
There is no `RelationType` or `HypothesisRelation` anywhere in
`src/hypothesis_graph.py`.

### 4.4 `Hypothesis.root_cause_category` — made optional, deliberately

The spec's table (§3.3) lists `root_cause_category: FailureCategory` as a
required field, described as "must map to chaos taxonomy." The build instead
made it `Optional[FailureCategory] = None`
(`src/hypothesis_graph.py:116`), and the `GraphUpdate.action == "create"` code
path carries an explicit comment: *"no FailureCategory; categories are an
eval-side concern"* (`hypothesis_graph.py:155`). In the built system, the model
never assigns a `FailureCategory` when creating a hypothesis mid-investigation —
category only enters the picture once, at the very end, as the
`root_cause_category` field on the `stop_investigation` call
(`src/tools.py:337–360`). This is a deliberate scope narrowing from the spec,
not an oversight — it keeps the taxonomy out of the model's hands during
investigation (reinforcing §4.2's leakage guard) at the cost of the graph never
showing a category for a hypothesis while it's still being investigated.

### 4.5 `GraphUpdate` — schema materially redesigned

The spec's `GraphUpdate` (§3.5) has no action discriminator: it always carries
`new_evidence`, `likelihood_changes`, `hypotheses_to_rule_out`,
`new_established_facts`, and separately `new_hypotheses: List[Hypothesis]` (a
list of already-fully-formed `Hypothesis` objects the model could append in one
shot). `current_focus` lives on `HypothesisGraph`, not on `GraphUpdate`.

The built `GraphUpdate` (`src/hypothesis_graph.py:141–164`) is a three-way
discriminated union via `action: Literal["create", "update", "merge"]`:

- `current_focus` moved onto `GraphUpdate` itself, required only for `"update"`.
- `new_hypotheses: List[Hypothesis]` was replaced by three scalar fields
  (`new_hypothesis_description`, `new_hypothesis_severity`,
  `new_hypothesis_initial_likelihood`), which caps each update to creating **at
  most one** new hypothesis rather than a batch.
- A third action, `"merge"`, was added — not present in the spec at all. It
  lets the model attach evidence to an existing hypothesis it judges
  semantically equivalent to a proposed new one, via `merge_into_id`, rather
  than creating a near-duplicate. `README.md`'s "Design decisions in the graph"
  section documents this as deliberate ("merge — proposed hypothesis is
  semantically equivalent to an existing one; consolidate rather than
  duplicate"), addressing a failure mode (hypothesis duplication) the original
  spec didn't anticipate.

### 4.6 Per-update likelihood delta cap — narrower than proposed

Spec decision 3 (`hypothesis-graph-spec.md` §5.3) caps deltas at ±0.25 "unless
evidence is explicitly marked definitive," implying an escape hatch for
decisive evidence. The build enforces `_MAX_LIKELIHOOD_DELTA = 0.25`
unconditionally (`hypothesis_graph.py:167`, `update_graph:210–215`) with no
"definitive evidence" override anywhere in the schema or update logic. Simpler
than proposed, and closes a potential gap where model self-assessment of
"definitive" could have reintroduced the confidence-runaway problem the cap
exists to prevent — but it is a real narrowing of the proposed design.

### 4.7 Initial hypothesis count (4–5) — not code-enforced

Spec decision 1 calls for 4–5 initial hypotheses. Nothing in
`hypothesis_graph.py`, `output_validator.py`, or `agent.py` enforces a minimum
or maximum hypothesis count — this constraint, if it holds at all, is enforced
only by prompt instruction (`src/prompts.py`), the same way every other
"locked" numeric decision in the spec is code-enforced except this one.

---

## 5. Tool Interfaces — Matches closely, one policy dropped

All five tools (`query_metrics`, `query_logs`, `query_deployment_history`,
`query_feature_distributions`, `query_code_diffs`) are implemented with real
backends (live SQLite queries against `target-system/data/*.db`, and real
`git diff` against `pipeline_repo/`), not the "static responses" the docstring
at the top of `src/tools.py` still claims for `query_metrics`/`query_logs` —
that comment is stale relative to the code beneath it.

Matches the spec precisely:
- Unified `ToolResponse` envelope (`tool_name`, `status`, `data`/`error`, `query_metadata`) — every tool implementation follows it.
- Empty results are `status: "ok"`, not errors (spec §2.2) — confirmed in `_query_deployment_history`, `_query_logs`.
- Small catalogs (service, severity, metric names) are enum-constrained in the JSON schema; the large catalog (feature names) is free text validated at dispatch, returning `valid_values` on a miss for self-correction (spec §2.4) — implemented exactly in `_query_feature_distributions` (`tools.py:713–726`).
- PSI is the fixed drift metric with the documented `<0.1 / 0.1–0.25 / >0.25` thresholds (spec §4.4) — implemented in `_psi()` (`tools.py:682–705`), computed inline with no external library, as proposed.
- `query_logs` and `query_code_diffs` cap output and report `truncated: bool` (spec §2.5) — implemented.
- All tool calls, including ones that error, count against the budget (spec §5, decision 3) — confirmed in `agent.py:389` (`tool_calls_used` incremented before dispatch, unconditionally).

**Not built:** spec design decision 6 (§5) proposed transparently retrying
`service_unavailable`/`timeout` errors once inside the Tool Dispatcher, without
charging the budget, before surfacing them to the model. The spec itself flagged
this as its least-confident decision. No retry logic exists anywhere in
`tools.py` — a transient error is surfaced to the model on the first attempt and
charged against budget like any other tool call.

`update_hypothesis_graph` and `stop_investigation` are registered as two
additional entries in `TOOL_DEFINITIONS` (7 total, not 5) — a necessary
consequence of the Anthropic tool-use API, where graph bookkeeping and session
termination have to be modeled as callable tools even though they aren't
"evidence source" tools in the proposal's sense. `update_hypothesis_graph` is
explicitly exempted from the budget counter, as specified in the README.

---

## 6. Evaluation Harness

### 6.1 Chaos taxonomy — fully specified and fully executable

`docs/chaos-taxonomy.md` documents all 18 categories across easy (5) / medium
(8) / hard (5) tiers, exactly as the proposal called for ("15–20 categories...
easy/medium/hard tiers," §7), and the `FailureCategory` enum matches it 1:1
under CI enforcement (§4.2 above).

**Update (2026-09-08):** `target-system/evaluation/chaos_scenarios.py`'s
`SCENARIOS` dict now contains injection mechanics for all 18 categories
(previously only the 5 easy-tier scenarios were implemented — see git history
for the prior state). Closing the remaining 13 required extending the harness
beyond what the original 5 scenarios needed:

- `InjectionScenario.sub_range: SubRangeConfig` was generalized to
  `sub_ranges: list[SubRangeConfig]`, so a scenario can ramp feature
  distributions across several windows instead of stepping once — used for
  `gradual_concept_drift` (4-day ramp) and `model_staleness` (7-day ramp).
- `InjectionScenario.label_delay_hours` was added (default `0.0`, matching
  the original 5 scenarios' behavior) so `delayed_label_feedback_shift` can
  restore the generator's real 24h label-delay mechanic and exploit it
  directly, rather than faking the effect.
- `InjectionScenario.metric_overrides` was added as a direct upsert into
  `metrics.db` (mirroring the pre-existing `diagnostic_logs` injection
  pattern in `run_eval.py`) for the handful of signals the feature-driven
  simulation has no causal path to produce — corrupted labels
  (`label_pipeline_corruption`), inverted confidence/accuracy relationships
  (`model_calibration_drift`), and a blended second model
  (`shadow_mode_leak`). Every other new scenario drives its metric signature
  organically through `FeatureParams`/`InferenceParams`, same as the original
  5.
- Two new commits were added to `pipeline_repo` (`4612fcb`, `5bc3343`) for
  the two scenarios whose primary/distinguishing signal is `query_code_diffs`
  itself (`training_serving_skew`, `feature_importance_inversion`) — verified
  to reproduce identically from a from-scratch `setup_pipeline_repo.py` run,
  since `pipeline_repo/.git` is a locally-bootstrapped artifact, not part of
  the outer repo's tracked history (only 5 plain-file snapshots under
  `pipeline_repo/` are outer-repo-tracked; `.git` itself is regenerated by
  `setup_pipeline_repo.py`, never committed).

All 18 scenario IDs are asserted to match `FailureCategory` 1:1 at import
time (`assert len(SCENARIOS) == 18` in `chaos_scenarios.py`), and each new
scenario was spot-checked by generating its data and querying it through the
real tool layer (`query_metrics`, `query_deployment_history`,
`query_code_diffs`) rather than only checking that generation didn't crash.

### 6.2 Rule-based baseline — not built

Both `CLAUDE.md` and the proposal (§7) call for "a rule-based baseline for
comparison against the agent," and list "agent vs. rule-based baseline delta"
as an evaluation metric. No baseline module exists anywhere in the repository —
grepping for `baseline` across `src/` and `target-system/` turns up only
unrelated uses (SQLite window-comparison naming in `run_eval.py`, config field
names in the generator). This metric cannot currently be computed.

### 6.3 LLM-as-judge / inter-rater reliability — built (2026-09-08)

`target-system/evaluation/llm_judge.py` scores a completed investigation's
diagnosis narrative — independent of whether `root_cause_category` matches
ground truth, which top1/top3 accuracy already covers — against a five-
dimension rubric (evidence grounding, reasoning coherence, actionability,
calibration reasonableness, overall), plus a `hallucination_detected` flag,
elicited via a forced `submit_judgment` tool call in the same pattern
`agent.py` uses for `stop_investigation`'s final synthesis (raw Anthropic
API only, no separate judge framework — consistent with `CLAUDE.md`'s
architecture constraint).

The judge is shown the alert, the *complete* hypothesis graph (all
hypotheses, all evidence, ruled-out included — via `graph.model_dump()`,
not the condensed view `agent.py`'s `_graph_context()` shows the
investigating model mid-session) and the final diagnosis, but is
deliberately never shown the ground-truth category, so it can't rate
"correctness" by proxy.

**Inter-rater reliability, given the single-provider constraint:** since
`CLAUDE.md` scopes the project to one reasoning engine, "raters" here means
`n` independent repeated judge calls against the same transcript
(self-consistency) rather than multiple models or human annotators.
`judge_diagnosis(..., n_raters=3)` runs the rubric prompt `n` times and
reports, per dimension, the fraction of rater *pairs* whose 1-5 scores land
within 1 point of each other (`_pairwise_agreement`), plus the mean score
and hallucination rate. This is a real, load-bearing signal distinct from
the mean — a dimension every rater scores 3 on 3 passes is a materially
different result from one that swings 1-5-3, and the mean alone hides that.

Wired into the CLI via `run_eval.py --run-agent --judge [--judge-raters N]`.
Covered by 20 unit tests (`target-system/tests/test_llm_judge.py`) that mock
the Anthropic client — verifying the aggregation math (agreement,
hallucination rate), that the judge is never handed a ground-truth field,
and that malformed/refused tool responses raise rather than silently
producing bad scores — plus a live smoke-test of the tool-call wiring
against the real API during development.

### 6.4 Calibration curves and tool-selection efficiency — not built

`run_eval.py` scores a single run's top-1/top-3 correctness
(`run_agent_and_score`, lines 121–171) but does not aggregate confidence vs.
outcome across multiple runs into a calibration curve or Brier score, and does
not compute the "fraction of tool calls that shift the top hypothesis
likelihood by ≥0.05" tool-selection-efficiency metric described in
`docs/QA_Plan.md` §5. What *is* elicited correctly — a hard constraint both the
proposal and `docs/QA_Plan.md` insist on — is that confidence is always
explicitly requested from the model via the `stop_investigation` schema
(`tools.py:365–369`), never rank-derived; the infrastructure to turn that into
a calibration curve across many runs doesn't exist yet.

### 6.5 What the built harness does deliver

`target-system/evaluation/run_eval.py` is a working, single-scenario CLI:
generates 7 days of synthetic data for any of the 18 scenarios — failure-window
feature/inference overrides, injected diagnostic logs, and (for the scenarios
that need them) direct metric-row overrides all applied — runs the live agent
end-to-end against it, and reports top-1/top-3 correctness, tool calls used,
termination reason, and elapsed time as JSON. Passing `--judge` additionally
runs the diagnosis through `llm_judge.py` (§6.3) and folds its scores into the
same JSON output.

**Update (2026-09-08):** `target-system/evaluation/run_sweep.py` was added on
top of `run_eval.py`'s per-scenario pipeline — it iterates all 18 scenarios
(or a `--tier`, or an explicit `--scenarios` list), tolerates one scenario's
API error without aborting the rest, and aggregates results into one JSON
report with overall and per-tier top-1/top-3 accuracy, mean tool calls, and
(when `--judge` is passed) aggregated judge scores and hallucination rate.
`--repeats N` runs each scenario N times for variance, though the first real
run used `N=1`. This has now actually been run once, live, across the full
taxonomy — `docs/evaluation-report.md` is the resulting report, and
`target-system/eval_report_full.json` is the raw output. This closes the
"no batch/sweep mode" gap noted in the prior version of this document; the
remaining harness gap is the metrics in §6.2 and §6.4 (rule-based baseline,
calibration curves, tool-selection efficiency), which a sweep report alone
doesn't provide — those still require dedicated computation this harness
doesn't do.

### 6.6 Offline mock runner — built, not in the original proposal

`src/run_mock.py`, `src/scenarios.py`, and `src/fixtures.py` implement a
second, entirely separate evaluation path: three hand-scripted scenarios
(`feature_drift`, `bad_deployment`, `label_corruption`) with pre-recorded tool
responses and expected `GraphUpdate` fixtures, run through the validator/graph
pipeline with **no Anthropic API calls at all**. This exists to regression-test
the graph-mutation and validation logic in CI without spending API budget or
depending on network access — `docs/QA_Plan.md` documents it as part of the
`tests/e2e/` suite. It is not mentioned anywhere in the architecture proposal
and represents infrastructure the project needed but that wasn't scoped
up front. The current branch name, `mvp/react-loop-mock-runner`, reflects that
this mock-runner-driven phase predates the live chaos-injection harness in
`target-system/evaluation/`.

---

## 7. Observability Layer — Matches

Langfuse tracing matches the proposal and README closely: one top-level span
per investigation, one child span per ReAct turn, one child span per tool call
(and per graph update) with raw input/output (`agent.py:194–219, 325–333,
392–397`), and `top1_correct`/`top3_correct` scores attached to the trace when
`ground_truth` is supplied (`_score_diagnosis`, `agent.py:167–172`). This
satisfies the proposal's requirement that confidence/accuracy scoring be
externally attached to traces rather than computed only in-process.

---

## 8. Built Beyond the Proposal's Scope

The architecture proposal is silent on transport and deployment (§2 mentions
only "Practitioner ↔ ML Investigator"). The build added a substantial amount of
productionization that the proposal never scoped:

- **HTTP API** (`src/server.py`): `POST /investigate` (async job), `GET
  /jobs/{id}`, `GET /jobs` (listing — not in the README's endpoint table),
  `POST /webhook/alertmanager` (Prometheus Alertmanager v4 receiver), `GET
  /health`, plus a served static operator UI at `/` (`src/static/index.html`).
- **In-process job store** with an explicit, documented limitation (jobs lost
  on restart, no multi-replica support without Redis/sticky routing) — called
  out directly in both the module docstring and the README's tradeoffs section.
- **Docker + Kubernetes manifests** (`Dockerfile`, `k8s/namespace.yaml`,
  `deployment.yaml`, `service.yaml`, `configmap.yaml`, `secret.yaml`).
- **The target system itself** (`target-system/`): a full synthetic ML system
  (XGBoost training pipeline, FastAPI inference service, 7-day synthetic data
  generator, a real git repo for diffing) had to be built for the five evidence
  sources to have something concrete to query — the proposal treats these as
  already-existing external systems.

None of this contradicts the architecture proposal; it simply wasn't part of
what the proposal document scoped, and it materially increases the surface area
of what "ML Investigator" now means as a deployable system versus a diagnosis
loop.

---

## 9. Recommended Follow-ups

In priority order, based on the gaps that remain after the 2026-09-08 chaos
injection, LLM-as-judge, and batch-sweep work (§6.1, §6.3, §6.5), and on the
actual results in `docs/evaluation-report.md`:

1. **Investigate the hard-tier 0% top-1 accuracy.** `docs/evaluation-report.md`
   §3 shows a clean monotonic drop across tiers (80% → 25% → 0%), but n=1 per
   scenario means this specific number carries no variance estimate yet —
   `run_sweep.py --tier hard --repeats 5` is the natural next run before
   treating 0% as a stable result rather than a single unlucky pass. The
   report's §5 confusion-pattern analysis (`upstream_schema_change` and
   `feature_encoding_bug` both over-predicted as default "something's off"
   answers) is a concrete starting hypothesis for what's driving it.
2. **Investigate the overconfidence-on-misses pattern.** Mean confidence was
   0.958 when correct vs. 0.866 when wrong, with three wrong diagnoses at
   ≥0.93 confidence (`docs/evaluation-report.md` §1, §6). This tracks the
   judge's own lowest-scoring dimension (`calibration_reasonableness`,
   4.09/5 vs. 4.67–4.91 for the others) — two independent measurements
   pointing at the same weakness, worth prompt-level attention before
   trusting `stop_investigation`'s confidence field for downstream automation.
3. **Build the rule-based baseline.** Still not built. Required for the
   "agent vs. baseline delta" metric that both `CLAUDE.md` and the proposal
   treat as core to knowing whether the agent is worth its cost over a
   simpler system — this is now the single largest remaining evaluation gap,
   and the sweep report has no baseline number to compare its 33% top-1
   against.
4. **Reconcile `CLAUDE.md`'s `RelationType` mention** with the spec's decision
   to exclude it (§4.3 above) — one of the two documents should be corrected so
   they stop disagreeing.
5. **Refresh `src/tools.py`'s module docstring**, which still describes
   `query_metrics`/`query_logs` as static placeholders when both are live
   SQLite-backed implementations.
6. **Decide whether `judge_diagnosis`'s inter-rater reliability should ever
   extend beyond self-consistency** (repeated calls to the same model) to a
   genuinely independent second rater — the current design is an honest
   read of `CLAUDE.md`'s single-reasoning-engine constraint, but if that
   constraint is ever relaxed for evaluation-only tooling (as opposed to the
   agent itself), a second model as a rater would be a stronger reliability
   signal than repeated self-consistency passes. Notably, agreement was
   already 100% in the one real sweep run so far (§6.5) — this follow-up is
   about validity, not about a reliability problem the current data shows.
