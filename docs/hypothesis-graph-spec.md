# Hypothesis Graph — Data Model Spec

## 1. Purpose

The Hypothesis Graph is the agent's structured belief state during a diagnosis
session. It is **not a graph in the CS sense** — no nodes/edges/traversal
algorithms. It's a Pydantic model that the agent reads and rewrites after
every tool call, serialized to JSON in the context window.

At any point in an investigation, the graph must answer three questions:
1. What are the possible root causes right now, and how likely is each one?
2. What evidence supports or contradicts each candidate?
3. What should the agent investigate next to reduce uncertainty?

The **Stopping Criteria** and **Output Validator** modules query this graph
directly (not via the ReAct loop) — see component diagram. This spec is their
contract.

---

## 2. Enums

### 2.1 `FailureCategory`

> **MUST be derived 1:1 from the chaos injection taxonomy.** This is the
> single source of truth for the project — divergence between this enum and
> the chaos taxonomy breaks the evaluation harness. Do not add a category
> here without a corresponding chaos injection scenario, and vice versa.

```python
class FailureCategory(str, Enum):
    # TODO: replace with finalized chaos injection taxonomy
    # (15–20 categories, easy/medium/hard tiers). Placeholders below
    # are illustrative only — DO NOT treat as final.
    FEATURE_DRIFT = "feature_drift"
    BAD_DEPLOYMENT = "bad_deployment"
    LABEL_PIPELINE_CORRUPTION = "label_pipeline_corruption"
    TRAINING_SERVING_SKEW = "training_serving_skew"
```

**Action item:** finalize chaos taxonomy first, generate this enum from it
programmatically (or at minimum, diff them in CI) so they can't silently drift.

### 2.2 `HypothesisStatus`

| Value | Meaning |
|---|---|
| `active` | Still a live candidate, under investigation |
| `ruled_out` | Evidence has dropped likelihood below a viability threshold |
| `confirmed` | Selected as the final root cause at termination |

```python
class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    RULED_OUT = "ruled_out"
    CONFIRMED = "confirmed"
```

### 2.3 `Severity`

```python
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

### 2.4 `EvidenceType`

Maps 1:1 to your five tools. Not in the original draft — added so evidence
can be aggregated by category without string-matching `tool_called` at eval
time (needed for the tool selection efficiency metric).

> **`evidence_type` is never set by the model.** It's derived programmatically
> from `tool_called` via a fixed lookup, immediately after parsing the
> model's output and before constructing the stored `Evidence` object (see
> §3.1). If the model set both fields independently, you'd risk
> `tool_called: "query_metrics"` paired with `evidence_type: "logs"` — a
> silent data integrity bug that corrupts the tool-selection-efficiency
> metric without ever surfacing as an error.

```python
class EvidenceType(str, Enum):
    METRICS = "metrics"
    LOGS = "logs"
    DEPLOYMENT_HISTORY = "deployment_history"
    FEATURE_DISTRIBUTIONS = "feature_distributions"
    CODE_DIFFS = "code_diffs"


TOOL_TO_EVIDENCE_TYPE: Dict[str, EvidenceType] = {
    "query_metrics": EvidenceType.METRICS,
    "query_logs": EvidenceType.LOGS,
    "query_deployment_history": EvidenceType.DEPLOYMENT_HISTORY,
    "query_feature_distributions": EvidenceType.FEATURE_DISTRIBUTIONS,
    "query_code_diffs": EvidenceType.CODE_DIFFS,
}


def derive_evidence_type(tool_called: str) -> EvidenceType:
    try:
        return TOOL_TO_EVIDENCE_TYPE[tool_called]
    except KeyError:
        raise ValueError(f"Unregistered tool name: {tool_called!r}")
```

---

## 3. Models

### 3.1 `Evidence` and `EvidenceInput`

Two schemas, not one — a model-facing schema and a stored schema. The split
exists specifically so the model can never set `evidence_type` itself.

**`EvidenceInput`** — what the model actually emits, as part of `GraphUpdate`:

| Field | Type | Constraints | Description |
|---|---|---|---|
| `tool_called` | `str` | must match a registered tool name | Which tool produced this evidence |
| `observation` | `str` | — | Summarized tool output |
| `supports` | `bool` | — | `True` supports the hypothesis, `False` contradicts it |
| `confidence_delta` | `float` | `-1.0 <= x <= 1.0` | Signed shift in likelihood this evidence caused |

```python
class EvidenceInput(BaseModel):
    tool_called: str
    observation: str
    supports: bool
    confidence_delta: float = Field(ge=-1.0, le=1.0)
```

**`Evidence`** — the stored representation inside the graph. Same fields,
plus `evidence_type`, computed by `derive_evidence_type()` (§2.4) at the
moment an `EvidenceInput` is applied to the graph. The model output is never
parsed directly into this type.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `tool_called` | `str` | must match a registered tool name | Which tool produced this evidence |
| `evidence_type` | `EvidenceType` | **derived, not model-set** | Computed from `tool_called` |
| `observation` | `str` | — | Summarized tool output |
| `supports` | `bool` | — | `True` supports the hypothesis, `False` contradicts it |
| `confidence_delta` | `float` | `-1.0 <= x <= 1.0` | Signed shift in likelihood this evidence caused |

```python
class Evidence(BaseModel):
    tool_called: str
    evidence_type: EvidenceType
    observation: str
    supports: bool
    confidence_delta: float = Field(ge=-1.0, le=1.0)

    @classmethod
    def from_input(cls, raw: EvidenceInput) -> "Evidence":
        return cls(
            **raw.model_dump(),
            evidence_type=derive_evidence_type(raw.tool_called),
        )
```

### 3.2 `Experiment`

A proposed next action attached to a hypothesis — the agent's plan for
distinguishing this hypothesis from others.

| Field | Type | Description |
|---|---|---|
| `tool_to_call` | `str` | Tool name |
| `parameters` | `dict` | Arguments to pass |
| `rationale` | `str` | Why this call discriminates between hypotheses |
| `expected_if_true` | `str` | Expected result if this hypothesis is correct |
| `expected_if_false` | `str` | Expected result if this hypothesis is wrong |

```python
class Experiment(BaseModel):
    tool_to_call: str
    parameters: dict
    rationale: str
    expected_if_true: str
    expected_if_false: str
```

### 3.3 `Hypothesis`

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | `str` | e.g. `"H1"` | Stable identifier within a session |
| `root_cause_category` | `FailureCategory` | — | Must map to chaos taxonomy |
| `description` | `str` | must name specific metrics/services/features — not vague | One-sentence explanation |
| `likelihood` | `float` | `0.0 <= x <= 1.0` | Current probability; normalized post-update, not schema-enforced |
| `severity` | `Severity` | — | Impact if confirmed |
| `status` | `HypothesisStatus` | — | Lifecycle state |
| `evidence` | `List[Evidence]` | default `[]` | Accumulated evidence |
| `distinguishing_experiments` | `List[Experiment]` | default `[]` | Candidate next actions |

```python
class Hypothesis(BaseModel):
    id: str
    root_cause_category: FailureCategory
    description: str
    likelihood: float = Field(ge=0.0, le=1.0)
    severity: Severity
    status: HypothesisStatus
    evidence: List[Evidence] = []
    distinguishing_experiments: List[Experiment] = []
```

> **Constraint:** when the model creates a `Hypothesis` directly — either
> during initialization or via `GraphUpdate.new_hypotheses` — `evidence` must
> be `[]`. The model has no legitimate way to populate `Evidence.evidence_type`
> (same problem as §3.1), so any hypothesis it creates from scratch starts
> with no evidence; evidence gets attached on a later `GraphUpdate` like
> everything else. Enforce this in the Output Validator: reject any
> model-created `Hypothesis` with non-empty `evidence`.

### 3.4 `HypothesisGraph`

| Field | Type | Description |
|---|---|---|
| `alert_summary` | `str` | What triggered the investigation |
| `investigation_start` | `datetime` | Session start timestamp |
| `tool_calls_used` | `int` | Running count against budget |
| `tool_call_budget` | `int` | Max tool calls allowed (e.g. 10) |
| `hypotheses` | `List[Hypothesis]` | All hypotheses, active and resolved |
| `established_facts` | `List[str]` | Confirmed true regardless of root cause |
| `open_questions` | `List[str]` | Still unresolved |
| `current_focus` | `Optional[str]` | ID of hypothesis under active investigation |
| `termination_reason` | `Optional[str]` | Set by Stopping Criteria on exit |

```python
class HypothesisGraph(BaseModel):
    alert_summary: str
    investigation_start: datetime
    tool_calls_used: int = 0
    tool_call_budget: int
    hypotheses: List[Hypothesis]
    established_facts: List[str] = []
    open_questions: List[str] = []
    current_focus: Optional[str] = None
    termination_reason: Optional[str] = None
```

### 3.5 `GraphUpdate`

The model's output after each tool call. The graph itself is updated
**programmatically** by applying this structured delta — the model never
rewrites the full graph directly. This is the chosen approach (see §4).

| Field | Type | Description |
|---|---|---|
| `new_evidence` | `EvidenceInput` | What was just found — *not* `Evidence`; the model never sets `evidence_type` |
| `likelihood_changes` | `Dict[str, float]` | `{"H1": +0.15, "H2": -0.10}` |
| `hypotheses_to_rule_out` | `List[str]` | IDs now implausible |
| `new_hypotheses` | `List[Hypothesis]` | Only if the result was surprising; `evidence` must be `[]` (see §3.3) |
| `new_established_facts` | `List[str]` | — |
| `next_experiment_rationale` | `str` | Why the agent is picking what it picks next |

```python
class GraphUpdate(BaseModel):
    new_evidence: EvidenceInput
    likelihood_changes: Dict[str, float]
    hypotheses_to_rule_out: List[str] = []
    new_hypotheses: List[Hypothesis] = []
    new_established_facts: List[str] = []
    next_experiment_rationale: str
```

`update_graph()` is the only place `Evidence` objects get constructed — it
converts `update.new_evidence` (`EvidenceInput`) into a stored `Evidence` via
`Evidence.from_input()` before appending it to the relevant hypothesis. No
other code path should construct `Evidence` directly.

---

## 4. Update Mechanism: Structured Delta, Not Full Rewrite

Two options were considered:

- **Full rewrite** — prompt the model with the current graph + new evidence,
  ask it to return an entire updated `HypothesisGraph`. Flexible, but
  likelihood updates are inconsistent and there's no audit trail of *why*
  something changed.
- **Structured delta (chosen)** — prompt the model to return a `GraphUpdate`
  specifying exactly what changed and why. The graph is updated by applying
  the delta programmatically.

The structured delta is the better tradeoff: it produces an audit trail of
every likelihood change, which is necessary for the calibration curve and
failure mode analysis in the eval writeup, and it's more debuggable when the
agent does something wrong.

---

## 5. Design Decisions (Locked)

1. **Initial hypothesis count: 4–5.** 3 risks missing the correct root cause
   in ambiguous alerts; 8 causes incoherent likelihood maintenance. New
   hypotheses can still be inserted mid-investigation via
   `GraphUpdate.new_hypotheses` if a tool result is surprising.

2. **Likelihoods are normalized programmatically, not schema-enforced.**
   Asking the model to emit probability-valid likelihoods directly produces
   garbage (e.g., four hypotheses each at 0.3). Normalize over active
   hypotheses after every update instead of validating at the schema level.

3. **Per-update likelihood delta is capped at ±0.25** unless evidence is
   explicitly marked definitive. Without this, the model overreacts to weak
   evidence and terminates investigations early (observed in first-iteration
   testing). Enforce this in the update-application logic, not via prompting
   alone.

4. **Confidence scores are explicitly elicited, never rank-derived.** The
   model must output a likelihood value as part of `GraphUpdate`, not infer
   one from hypothesis ordering. This is required for the calibration curve
   metric to be meaningful.

5. **No explicit cross-hypothesis relations.** No `RelationType` /
   `HypothesisRelation` enum. If a concrete need emerges (e.g., enforcing
   mutual exclusivity, or one hypothesis subsuming another), add it as a
   follow-up — don't model it speculatively now.

6. **`evidence_type` is derived, never model-set.** The model's output
   schema (`EvidenceInput`, `GraphUpdate`) has no `evidence_type` field at
   all. It's computed from `tool_called` via a fixed lookup
   (`TOOL_TO_EVIDENCE_TYPE`) inside `update_graph()`, when constructing the
   stored `Evidence` object. Same constraint applies to any `Hypothesis` the
   model creates directly — it must have empty `evidence`.

---

## 6. Example: Mid-Investigation State

Alert: model accuracy dropped 15% over 6 hours. Two tool calls made. This
shows the **stored** graph state — `evidence_type` is present because it was
already derived by `update_graph()`; the model never produced it directly.

```json
{
  "alert_summary": "Model accuracy dropped from 0.91 to 0.77 over last 6 hours. No errors in logs.",
  "tool_calls_used": 2,
  "tool_call_budget": 10,
  "established_facts": [
    "Infrastructure metrics are normal (CPU, memory, latency all stable)",
    "No deployment events in the past 24 hours"
  ],
  "open_questions": [
    "Have input feature distributions changed?",
    "Is the label pipeline functioning correctly?"
  ],
  "current_focus": "H2",
  "hypotheses": [
    {
      "id": "H1",
      "root_cause_category": "bad_deployment",
      "description": "A recent model deployment introduced a regression",
      "likelihood": 0.05,
      "status": "ruled_out",
      "evidence": [
        {
          "tool_called": "query_deployment_history",
          "evidence_type": "deployment_history",
          "observation": "No deployments in past 24 hours",
          "supports": false,
          "confidence_delta": -0.30
        }
      ],
      "distinguishing_experiments": []
    },
    {
      "id": "H2",
      "root_cause_category": "feature_drift",
      "description": "Input feature distributions have shifted from training distribution",
      "likelihood": 0.55,
      "status": "active",
      "evidence": [
        {
          "tool_called": "query_metrics",
          "evidence_type": "metrics",
          "observation": "Prediction confidence scores dropped from mean 0.84 to mean 0.61 over same window",
          "supports": true,
          "confidence_delta": 0.15
        }
      ],
      "distinguishing_experiments": [
        {
          "tool_to_call": "query_feature_distributions",
          "parameters": {"before": "48h_ago", "after": "now"},
          "rationale": "Directly tests whether input distributions have shifted",
          "expected_if_true": "One or more features show significant divergence from baseline",
          "expected_if_false": "Feature distributions are stable across the window"
        }
      ]
    },
    {
      "id": "H3",
      "root_cause_category": "label_pipeline_corruption",
      "description": "The label pipeline is producing incorrect ground-truth labels, making the model appear to perform worse than it is",
      "likelihood": 0.30,
      "status": "active",
      "evidence": [],
      "distinguishing_experiments": [
        {
          "tool_to_call": "query_logs",
          "parameters": {"service": "label_pipeline", "time_range": "6h", "filter": "error|warning|anomaly"},
          "rationale": "Label pipeline errors would indicate measurement corruption rather than model degradation",
          "expected_if_true": "Anomalies or errors in label computation logs",
          "expected_if_false": "Clean label pipeline logs"
        }
      ]
    },
    {
      "id": "H4",
      "root_cause_category": "training_serving_skew",
      "description": "Feature transformations differ between training and serving environments",
      "likelihood": 0.10,
      "status": "active",
      "evidence": [],
      "distinguishing_experiments": []
    }
  ]
}
```

The agent's next action is unambiguous from the graph alone: call
`query_feature_distributions` for H2, since it's the highest-likelihood
active hypothesis with a ready distinguishing experiment.

---

## 7. Open Issues

- [ ] `FailureCategory` enum values are placeholders — populate from
  finalized chaos injection taxonomy, then add a CI check (or codegen step)
  that fails if the two diverge.
- [ ] No handling yet specified for what happens if `GraphUpdate.likelihood_changes`
  references a hypothesis ID that doesn't exist (model hallucinated an ID) —
  needs explicit error handling in the Output Validator.
- [ ] `derive_evidence_type()` raises on an unregistered tool name. Decide
  whether that should hard-fail the session or get caught and logged as a
  malformed-tool-call eval signal — probably the latter, since a model
  hallucinating a tool name is itself a failure mode worth measuring, not
  just an exception to swallow.
