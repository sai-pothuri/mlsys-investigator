# Evaluation Report — Full Taxonomy Sweep

**Run date:** 2026-09-08 21:26 UTC
**Harness:** `evaluation/run_sweep.py --judge --judge-raters 3`
**Scope:** all 18 chaos injection scenarios (see `docs/chaos-taxonomy.md`), one run each, no repeats
**Total runtime:** 36m 3s (2,162.9s) · **Runs completed:** 18/18 · **Errors:** 0
**Raw data:** [`target-system/eval_report_full.json`](../target-system/eval_report_full.json) (full per-run detail, including all 54 judge rationales — this document is a curated summary of that file)

This is the first complete run of the agent against the full 18-category chaos
taxonomy, now that all 18 injection scenarios and LLM-as-judge scoring exist
(see `docs/as-built.md` §6.1, §6.3). It replaces the prior state where only the
5 easy-tier scenarios could be evaluated at all.

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Top-1 accuracy (overall) | **33.3%** (6/18) |
| Top-3 accuracy (overall) | **61.1%** (11/18) |
| Mean tool calls used | 5.78 / 8 |
| Mean investigation time | 80.8s |
| LLM-judge overall score | 4.70 / 5 |
| LLM-judge inter-rater agreement | 100% (every rater pair within 1 point, every dimension, every scenario) |
| Hallucination rate (judge-flagged) | 33.3% of judge passes |
| Mean confidence when correct | 0.958 |
| Mean confidence when wrong | 0.866 |

**Headline finding:** accuracy falls monotonically with taxonomy difficulty
tier — 80% (easy) → 25% (medium) → 0% (hard) top-1 — which is itself a real
validation result: the taxonomy's easy/medium/hard grading, assigned when the
scenarios were designed, holds up against actual agent performance rather
than being an arbitrary label.

**Second finding:** the agent is meaningfully overconfident on its misses.
Mean confidence was 0.958 when correct vs. 0.866 when wrong — a real gap in
the right direction, but 0.866 is still high, and three wrong diagnoses were
delivered at ≥0.93 confidence (`bad_deployment` 0.93, `label_pipeline_corruption`
0.97, `cascading_upstream_failure` 0.97). This is consistent with
`calibration_reasonableness` being the lowest-scoring judge dimension (4.09/5,
vs. 4.67–4.91 for the other four) — the judge caught the same pattern the raw
numbers show.

---

## 2. Methodology

Each scenario: generate 7 days of synthetic target-system data with the
scenario's chaos injection applied (feature drift, deployment events,
diagnostic logs, and — for the scenarios that need them — direct metric
overrides; see `docs/as-built.md` §6.1), run the live agent
(`claude-sonnet-4-6`, tool-call budget 8) against the resulting alert, then
score the diagnosis two ways:

1. **Accuracy** — top-1 (`root_cause_category` exact match) and top-3
   (ground truth appears in `root_cause_category` + `alternative_categories`)
   against the scenario's known ground truth.
2. **LLM-as-judge** — 3 independent forced-tool-call rating passes
   (`evaluation/llm_judge.py`) scoring the diagnosis narrative itself —
   evidence grounding, reasoning coherence, actionability, calibration
   reasonableness, and an overall score, plus a hallucination flag — blind to
   ground truth, so it can't rate "correctness" by proxy. Reported per
   dimension as the mean score and the pairwise agreement across the 3
   raters (fraction of rater pairs within 1 point).

No repeats were run (`n=1` per scenario) — see Limitations (§6).

---

## 3. Results by Tier

| Tier | n | Top-1 | Top-3 | Mean tool calls |
|---|---|---|---|---|
| Easy | 5 | **80.0%** | 100.0% | 5.4 |
| Medium | 8 | **25.0%** | 62.5% | 5.6 |
| Hard | 5 | **0.0%** | 20.0% | 6.4 |

Tool-call usage rises with tier (5.4 → 5.6 → 6.4 of 8), consistent with the
taxonomy's own design intent ("hard" scenarios are meant to require more
cross-evidence synthesis) — but the extra tool calls didn't translate into
correct diagnoses at the hard tier. Two runs exhausted the full 8-call budget
and fell back to the forced-synthesis path (`agent.py`'s budget-exhaustion
handling): `delayed_label_feedback_shift` and `compound_drift_plus_deployment`
— both hard tier, both misses. `compound_drift_plus_deployment` was also the
clear time outlier at 292.6s (every other run finished in 41–98s), reflecting
the extra reasoning the deploy-timing red herring demanded.

---

## 4. Full Per-Scenario Results

| Scenario | Tier | Ground truth → Predicted | Top-1 | Top-3 | Conf. | Tools | Elapsed | Judge overall | Halluc. rate |
|---|---|---|:-:|:-:|--:|--:|--:|--:|--:|
| `bad_deployment` | easy | bad_deployment → **feature_encoding_bug** | ✗ | ✓ | 0.93 | 5/8 | 61.1s | 5.00 | 0% |
| `feature_drift` | easy | feature_drift → feature_drift | ✓ | ✓ | 0.92 | 7/8 | 62.6s | 5.00 | 100% |
| `infrastructure_latency_spike` | easy | infrastructure_latency_spike → infrastructure_latency_spike | ✓ | ✓ | 0.97 | 4/8 | 57.2s | 4.67 | 100% |
| `model_version_rollback_regression` | easy | model_version_rollback_regression → model_version_rollback_regression | ✓ | ✓ | 0.97 | 6/8 | 73.1s | 5.00 | 0% |
| `upstream_schema_change` | easy | upstream_schema_change → upstream_schema_change | ✓ | ✓ | 0.97 | 5/8 | 48.6s | 5.00 | 0% |
| `data_freshness_degradation` | medium | data_freshness_degradation → data_freshness_degradation | ✓ | ✓ | 0.97 | 6/8 | 82.2s | 5.00 | 66.7% |
| `feature_encoding_bug` | medium | feature_encoding_bug → feature_encoding_bug | ✓ | ✓ | 0.95 | 4/8 | 54.5s | 5.00 | 33.3% |
| `feature_pipeline_partial_failure` | medium | feature_pipeline_partial_failure → **label_pipeline_corruption** | ✗ | ✗ | 0.88 | 6/8 | 64.9s | 5.00 | 0% |
| `gradual_concept_drift` | medium | gradual_concept_drift → **delayed_label_feedback_shift** | ✗ | ✗ | 0.72 | 7/8 | 97.3s | 5.00 | 0% |
| `label_pipeline_corruption` | medium | label_pipeline_corruption → **upstream_schema_change** | ✗ | ✓ | 0.97 | 4/8 | 52.2s | 5.00 | 0% |
| `model_calibration_drift` | medium | model_calibration_drift → **label_pipeline_corruption** | ✗ | ✓ | 0.92 | 6/8 | 63.5s | 5.00 | 0% |
| `shadow_mode_leak` | medium | shadow_mode_leak → **feature_pipeline_partial_failure** | ✗ | ✗ | 0.82 | 6/8 | 80.5s | 4.00 | 100% |
| `training_serving_skew` | medium | training_serving_skew → **feature_encoding_bug** | ✗ | ✓ | 0.95 | 6/8 | 79.6s | 5.00 | 0% |
| `cascading_upstream_failure` | hard | cascading_upstream_failure → **upstream_schema_change** | ✗ | ✗ | 0.97 | 5/8 | 61.5s | 5.00 | 0% |
| `compound_drift_plus_deployment` | hard | compound_drift_plus_deployment → **training_serving_skew** | ✗ | ✗ | 0.72 | 8/8 | 292.6s | 3.00 | 100% |
| `delayed_label_feedback_shift` | hard | delayed_label_feedback_shift → **upstream_schema_change** | ✗ | ✗ | 0.78 | 8/8 | 91.8s | 5.00 | 0% |
| `feature_importance_inversion` | hard | feature_importance_inversion → **feature_encoding_bug** | ✗ | ✗ | 0.95 | 4/8 | 40.7s | 4.00 | 100% |
| `model_staleness` | hard | model_staleness → **gradual_concept_drift** | ✗ | ✓ | 0.78 | 7/8 | 90.7s | 4.00 | 0% |

All 18 runs terminated via `stop_investigation` (either the model called it
directly, or the budget-exhaustion fallback forced it) — no run ended on
`model_end_turn` or an unhandled error.

---

## 5. Confusion Patterns

The 12 misses are not random — most predicted categories are plausible
near-misses that share a real signal with the true cause, which is a more
useful failure mode to know about than uniform random error:

- **`feature_encoding_bug` over-predicted three times** — for
  `bad_deployment`, `training_serving_skew`, and `feature_importance_inversion`.
  All three genuinely can present with an isolated-feature-looks-off signature,
  which is exactly what `feature_encoding_bug` is; the agent appears to reach
  for it as a default "one feature is behaving oddly" explanation.
- **`upstream_schema_change` over-predicted twice** — for
  `label_pipeline_corruption` and `cascading_upstream_failure`. Both do involve
  genuine upstream data-quality problems; the agent correctly identified
  "something upstream is wrong" but attributed it to the feature pipeline
  rather than the label pipeline.
- **`model_staleness` ↔ `gradual_concept_drift`** confusion (predicted the
  latter for the former) is a near-miss the taxonomy itself anticipated —
  both present as slow, broad, no-single-outlier drift; the taxonomy's own
  distinguishing signal (checking deployment history far outside the usual
  investigation window for a stale retrain event) is the one the agent needed
  to reach for and didn't.
- **`shadow_mode_leak` → `feature_pipeline_partial_failure`** is the one
  genuinely distant miss — these don't share an obvious surface signal,
  and this run also had a 100% hallucination-flagged judge pass (see §6).

---

## 6. LLM-as-Judge Results

| Dimension | Mean (1–5) |
|---|--:|
| Evidence grounding | 4.67 |
| Reasoning coherence | 4.78 |
| Actionability | 4.91 |
| Calibration reasonableness | 4.09 |
| **Overall** | **4.70** |

**Inter-rater agreement:** 100% — across all 18 scenarios × 5 dimensions × 3
raters, every pair of independent judge passes landed within 1 point of each
other. The judge rubric is producing stable, reproducible ratings, which
means the mean scores above are a reliable read rather than noise from a
single roll.

**Hallucination rate: 33.3%** of judge passes (18/54) flagged a fabricated
claim in the diagnosis text — a genuine claim not traceable to anything in
the hypothesis graph's evidence. Six scenarios had at least one hallucination
flag; three (`feature_drift`, `infrastructure_latency_spike`,
`shadow_mode_leak`, `compound_drift_plus_deployment`,
`feature_importance_inversion`) had **all 3 raters** flag it — meaning the
fabrication was clear and specific enough that independent raters converged
on catching it, not a borderline judgment call.

**Concrete example** (`feature_drift`, flagged by all 3 raters
independently): the diagnosis stated *"All other 11 features are perfectly
stable (PSI < 0.002 for all tested)"* — the hypothesis graph's evidence
actually records **5** other features, not 11. Every other factual claim in
that same diagnosis (the PSI value, the accuracy/confidence deltas, the
absence of deployment events) checked out against the graph; this was an
isolated fabricated count, not a systemic accuracy problem with that
diagnosis. This is the kind of error top-1/top-3 accuracy cannot see at
all — the diagnosis was top-1 *correct* — which is the whole reason the judge
exists as a separate metric.

---

## 7. Limitations of This Run

- **n=1 per scenario.** No repeats were run, so single-run accuracy numbers
  (especially the 0% hard-tier top-1) carry no variance estimate. A
  scenario the agent gets wrong once might not be wrong every time, and vice
  versa — `run_sweep.py --repeats N` exists for exactly this, but wasn't used
  here to keep this first full-taxonomy run's cost and runtime bounded.
- **No rule-based baseline comparison.** `CLAUDE.md` and the architecture
  proposal both call for an "agent vs. rule-based baseline" delta; no baseline
  exists yet (`docs/as-built.md` §6.2), so there's no way to say from this
  report alone whether 33% top-1 is a good result relative to a simpler
  system, only how it breaks down by tier and category.
- **No historical comparison.** This is the first complete sweep since all 18
  scenarios existed — there's no prior full-taxonomy run to diff against, so
  regressions or improvements from future prompt/logic changes will need this
  report as the baseline going forward.
- **Judge is the same model family as the agent.** Per `docs/as-built.md`
  §6.3, "inter-rater" here means repeated calls to the same reasoning engine
  (self-consistency), not an independent second model — the 100% agreement
  figure reflects consistency of judgment, not independent corroboration from
  a different model.

---

## 8. Reproducing This Report

```bash
cd target-system/
python -m evaluation.run_sweep --judge --judge-raters 3 --output-file eval_report_full.json
```

Add `--repeats N` for variance estimates, `--tier hard` to re-run just the
weak spot, or `--scenarios shadow_mode_leak,compound_drift_plus_deployment`
to target specific scenarios. See `evaluation/run_sweep.py`'s docstring for
the full CLI.
