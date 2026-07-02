# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ML Investigator — Project Context

## What this is
An agentic failure diagnosis system for distributed ML systems. A hand-rolled
ReAct agent investigates degraded ML systems by reasoning across metrics, logs,
deployment history, feature distributions, and code diffs to produce ranked
root cause hypotheses.

## Non-negotiable architecture (do not suggest alternatives)
- Raw Anthropic API as the reasoning engine — NO LangChain, NO MCP framework
  abstractions, NO agent frameworks. Hand-rolled ReAct loop only.
- Pydantic-based hypothesis graph as the agent's structured belief state
- Chaos injection as the ground-truth evaluation harness (15–20 failure
  categories across easy/medium/hard tiers)
- Rule-based baseline for comparison against the agent
- Langfuse for tracing and observability

## Module structure (see /docs/component-diagram.puml)
- ReAct Loop
- Tool Dispatcher
- Hypothesis Graph Module
- Stopping Criteria — queries Hypothesis Graph directly (NOT via ReAct loop,
  to avoid god-object anti-pattern)
- Output Validator — same direct-query pattern as Stopping Criteria
- Observability Layer

## Tools (see /docs/tool-specs.md for full interface contracts)
query_metrics, query_logs, query_deployment_history,
query_feature_distributions, query_code_diffs — all share a unified error
envelope (see spec).

## Data model
See /docs/hypothesis-graph-spec.md for FailureCategory, HypothesisStatus,
EvidenceType, RelationType enums and the Evidence/Hypothesis/
HypothesisRelation/HypothesisGraph field tables.

CRITICAL: FailureCategory enum MUST be derived from the chaos injection
taxonomy as single source of truth. Do not let these drift independently.

## Evaluation metrics (what "done" means for any agent change)
- Top-1 / Top-3 root cause accuracy
- Agent vs. rule-based baseline delta
- Mean tool calls to diagnosis
- Hypothesis calibration curves (confidence scores must be explicitly
  elicited from the model, NEVER rank-derived)
- Tool selection efficiency
- LLM-as-judge scoring with inter-rater reliability

## Conventions
- Python, Pydantic for all structured data
- Prefer targeted line-level fixes over full file regeneration when debugging
- Tasks tracked in GitHub Projects (see project board)
