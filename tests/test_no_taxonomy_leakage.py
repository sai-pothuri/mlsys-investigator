"""
FailureCategory leakage guard.

FailureCategory is exclusively a chaos-injection/eval-side contract.  It must
never appear in the agent's system prompt, any Pydantic model on the agent I/O
path, or the tool definition schemas sent to the model.

These tests will fail loudly (with a precise location) if that invariant is
broken, so it can't silently regress as the prompt or schemas evolve.
"""

import json
import pytest

from hypothesis_graph import EvidenceInput, FailureCategory, GraphUpdate
from agent import _SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS

# Every identifier that must never appear in agent-facing artefacts.
# Includes the class name and every Python member name (uppercase).
_FORBIDDEN = {"FailureCategory"} | {m.name for m in FailureCategory}

# Pydantic models that live on the model-facing I/O path.
_AGENT_MODELS = [EvidenceInput, GraphUpdate]


# ── 1. System prompt ──────────────────────────────────────────────────────────

def test_system_prompt_no_taxonomy_leakage():
    for name in sorted(_FORBIDDEN):
        assert name not in _SYSTEM_PROMPT, (
            f"FailureCategory identifier '{name}' found in _SYSTEM_PROMPT (agent.py). "
            "FailureCategory is an eval-side contract and must never appear in agent prompts."
        )


# ── 2. Agent-facing Pydantic models ──────────────────────────────────────────

def _check_annotation(annotation, location: str) -> None:
    """Recursively assert that annotation is not FailureCategory (or a generic wrapping it)."""
    if annotation is FailureCategory:
        pytest.fail(
            f"{location} is typed as FailureCategory. "
            "FailureCategory must not appear in agent-facing Pydantic schemas."
        )
    args = getattr(annotation, "__args__", None)
    if args:
        for arg in args:
            _check_annotation(arg, location)


def test_pydantic_models_no_taxonomy_leakage():
    for model in _AGENT_MODELS:
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            _check_annotation(annotation, f"{model.__name__}.{field_name}")


# ── 3. Tool definitions ───────────────────────────────────────────────────────

def test_tool_definitions_no_taxonomy_leakage():
    schema_str = json.dumps(TOOL_DEFINITIONS)
    for name in sorted(_FORBIDDEN):
        assert name not in schema_str, (
            f"FailureCategory identifier '{name}' found in TOOL_DEFINITIONS (tools.py). "
            "FailureCategory is an eval-side contract and must never appear in tool schemas."
        )
