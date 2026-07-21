**ML Investigator**

Software Architecture Proposal

*Agentic Failure Diagnosis System for Distributed ML Systems*

# 1. Purpose & Scope

ML Investigator is an agentic system that diagnoses failures in
distributed ML systems. Given an alert, it autonomously investigates by
querying metrics, logs, deployment history, feature distributions, and
code diffs, reasoning over the evidence it collects, and producing a
ranked set of root-cause hypotheses for a practitioner to review.

This document describes the system\'s architecture: its boundary and
external dependencies, its internal components, the control flow of a
single diagnosis session, the structure of its core belief-state
representation, and the interfaces through which it gathers evidence. It
is intentionally scoped to architecture --- detailed evaluation
methodology and chaos injection design are covered in a separate
document.












# 2. System Context

At the system boundary, ML Investigator sits between a practitioner and
five evidence sources that already exist in a typical ML production
stack. It uses the Anthropic API as its reasoning engine. Two supporting
components sit outside the core diagnosis loop: a chaos injection
framework, used only during evaluation to generate ground-truth labeled
failures, and Langfuse, used in all sessions to capture execution
traces.

*(L0 Context Diagram — see [component-diagram.puml](component-diagram.puml))*

*L0 Context Diagram*

*Shows ML Investigator, the Practitioner, the five evidence sources, the
Anthropic API, the Chaos Injection Framework (evaluation-only), and
Langfuse.*

## External Entities

  **Entity**                      **Role**
  ------------------------------- -----------------------------------------------------------------------
  **Practitioner**                Submits an investigation request and receives the final ranked report
  **Metrics Store**               Source of time-series metrics
  **Log Aggregator**              Source of service logs
  **Deployment History**          Source of deploy / rollback / config-change events
  **Feature Store**               Source of feature distribution data
  **Code Repository**             Source of commit history and diffs
  **Anthropic API (Claude)**      Reasoning engine for the ReAct loop
  **Chaos Injection Framework**   Generates ground-truth labeled failures (evaluation only)
  **Langfuse**                    Receives execution traces for every session














# 3. Component Architecture

Internally, ML Investigator is composed of six modules. The ReAct Loop
is the single orchestrator of a session\'s control flow, but it is not
the sole owner of system state: the Hypothesis Graph is shared state
that other modules query directly when they need it, rather than routing
every read through the orchestrator. This keeps the orchestrator from
becoming a god object that mediates all data access in the system.

*(Component Diagram — see [component-diagram.puml](component-diagram.puml))*

*Component Diagram*

*Shows the six internal modules, their dependencies on each other, and
their external dependencies on the Anthropic API, evidence sources, and
Langfuse.*

## Modules

  **Module**                    **Responsibility**
  ----------------------------- -------------------------------------------------------------------------------------------------------------------------------------
  **ReAct Loop**                Drives the reason-act cycle: prompts Claude, receives tool calls, triggers graph updates, and owns the session lifecycle end to end
  **Tool Dispatcher**           Validates and routes tool calls to the correct evidence-source handler; returns results and errors in a uniform format
  **Hypothesis Graph Module**   Owns the structured belief state --- hypotheses, evidence, and relations --- shared by the modules below
  **Stopping Criteria**         Decides when an investigation has gathered enough evidence to terminate, by reading the graph\'s current state directly
  **Output Validator**          Validates the final graph and produces a well-formed, ranked report, also by reading the graph directly
  **Observability Layer**       Captures execution traces from every module and exports them to Langfuse
















# 4. Diagnosis Session Flow

A diagnosis session begins when a practitioner submits an investigation
request, typically triggered by an alert. The ReAct Loop then repeats a
reason-act cycle: it asks Claude to reason over the current state and
choose a tool call, dispatches that call to the relevant evidence
source, and updates the Hypothesis Graph with the result. After each
cycle, Stopping Criteria checks whether the investigation has converged.
Once it has, the Output Validator finalizes the graph into a ranked
report, which is returned to the practitioner. Every step along the way
emits a trace to the Observability Layer.

*(Sequence Diagram — see [sequence-diagram.puml](sequence-diagram.puml))*

*Sequence Diagram*

*Shows one full diagnosis session from the practitioner\'s initial
request through the ReAct iteration loop to the final ranked report.*






# 5. Hypothesis Graph

The Hypothesis Graph is the agent\'s structured belief state for a
session. It holds candidate root causes, the evidence gathered in
support of or against them, and the relationships between them.
Representing belief state explicitly, rather than implicitly in
conversation history, is what lets Stopping Criteria and the Output
Validator reason about investigation progress directly.

  **Type**                 **Description**
  ------------------------ --------------------------------------------------------------------------------------------------------------------------------------------------------
  **Hypothesis**           A candidate root cause, with a failure category, an explicitly elicited confidence score, a status, and links to supporting and contradicting evidence
  **Evidence**             A single piece of retrieved evidence --- a metric, log, deployment event, feature distribution, or code diff --- with a model-generated summary
  **HypothesisRelation**   A directed edge between two hypotheses, e.g. one causes another, two are mutually exclusive, or one supersedes another
  **HypothesisGraph**      The container holding all hypotheses, evidence, and relations for one investigation session

Two properties of this model are central to keeping the evaluation
harness valid: a hypothesis\'s failure category is drawn from the same
enum used by the chaos injection framework\'s ground-truth labels, and
confidence scores are elicited explicitly from the model rather than
derived from rank order, so that calibration can be measured
meaningfully.

# 6. Tool Interfaces

Each evidence source is exposed to the agent as a tool, registered
against the Anthropic API and routed through the Tool Dispatcher. All
five tools are read-only, side-effect-free, and deterministic for a
given input, and all return errors in the same uniform format so the
ReAct Loop can reason about failures generically rather than handling
each tool\'s errors as a special case.

  **Tool**                          **Evidence Source**   **Purpose**
  --------------------------------- --------------------- ----------------------------------------------------------------------------
  **query_metrics**                 Metrics Store         Retrieve time-series metric data for a service over a time range
  **query_logs**                    Log Aggregator        Retrieve log entries matching a service, level, and/or keyword filter
  **query_deployment_history**      Deployment History    Retrieve deploy, rollback, and config-change events
  **query_feature_distributions**   Feature Store         Compare feature distributions between a reference and an incident window
  **query_code_diffs**              Code Repository       Retrieve commit and diff history for a repository over a time or SHA range

# 7. Evaluation Context

Evaluation is driven by a chaos injection framework that triggers
labeled failures across 15--20 categories at easy, medium, and hard
difficulty tiers, giving every session a ground-truth root cause to
score against. The agent\'s diagnoses are compared against a rule-based
baseline, and assessed on root-cause accuracy, calibration, tool-use
efficiency, and LLM-as-judge ratings with inter-rater reliability.
Detailed evaluation methodology is documented separately.
