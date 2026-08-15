"""Convergence checks for the ReAct loop.

Queries HypothesisGraph directly — NOT routed through the ReAct loop — to
avoid the god-object anti-pattern (see CLAUDE.md architecture constraints).

Three triggers, checked in priority order:
  1. top_likelihood   — top active hypothesis likelihood > 0.60
  2. dominance_ratio  — top / runner-up ratio > 2.5×
  3. stalled          — max per-hypothesis delta across last N snapshots < 0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hypothesis_graph import Hypothesis, HypothesisGraph, HypothesisStatus

_LIKELIHOOD_THRESHOLD = 0.60
_DOMINANCE_RATIO      = 2.5
_STALL_DELTA          = 0.05
_STALL_WINDOW         = 3   # number of consecutive snapshots with no meaningful move


@dataclass(frozen=True)
class StoppingDecision:
    stop: bool
    reason: Optional[str]           # "top_likelihood" | "dominance_ratio" | "stalled" | None
    top_hypothesis: Optional[Hypothesis]


def should_stop(
    graph: HypothesisGraph,
    likelihood_snapshots: list[dict[str, float]],
    stall_window: int = _STALL_WINDOW,
) -> StoppingDecision:
    """Return a StoppingDecision based on the three convergence triggers.

    Args:
        graph: current hypothesis graph
        likelihood_snapshots: rolling list of {hypothesis_id: likelihood} dicts,
            one entry appended after each update_graph() call. The caller owns
            this list and passes it in on every check.
        stall_window: how many consecutive snapshots must all show max delta
            < _STALL_DELTA before the stall trigger fires.
    """
    active = sorted(
        [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE],
        key=lambda h: -h.likelihood,
    )

    if not active:
        return StoppingDecision(stop=False, reason=None, top_hypothesis=None)

    top = active[0]

    # Trigger 1: absolute threshold
    if top.likelihood > _LIKELIHOOD_THRESHOLD:
        return StoppingDecision(stop=True, reason="top_likelihood", top_hypothesis=top)

    # Trigger 2: dominance ratio over runner-up
    if len(active) >= 2:
        runner_up = active[1]
        if runner_up.likelihood > 0 and top.likelihood / runner_up.likelihood > _DOMINANCE_RATIO:
            return StoppingDecision(stop=True, reason="dominance_ratio", top_hypothesis=top)

    # Trigger 3: stalled — no meaningful delta in the last stall_window snapshots
    if len(likelihood_snapshots) >= stall_window:
        window = likelihood_snapshots[-stall_window:]
        max_delta = _max_delta_in_window(window)
        if max_delta < _STALL_DELTA:
            return StoppingDecision(stop=True, reason="stalled", top_hypothesis=top)

    return StoppingDecision(stop=False, reason=None, top_hypothesis=top)


def _max_delta_in_window(window: list[dict[str, float]]) -> float:
    """Return the largest absolute likelihood change any hypothesis saw across the window."""
    if len(window) < 2:
        return float("inf")

    max_delta = 0.0
    all_ids = set().union(*window)
    for h_id in all_ids:
        vals = [snap.get(h_id, 0.0) for snap in window]
        max_delta = max(max_delta, max(vals) - min(vals))
    return max_delta


def snapshot_likelihoods(graph: HypothesisGraph) -> dict[str, float]:
    """Convenience: extract the current {id: likelihood} map for all active hypotheses."""
    return {
        h.id: h.likelihood
        for h in graph.hypotheses
        if h.status == HypothesisStatus.ACTIVE
    }
