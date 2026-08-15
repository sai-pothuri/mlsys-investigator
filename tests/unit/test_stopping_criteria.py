"""Unit tests for stopping_criteria.py."""

import pytest

from hypothesis_graph import (
    FailureCategory,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
)
from stopping_criteria import (
    _DOMINANCE_RATIO,
    _LIKELIHOOD_THRESHOLD,
    _STALL_DELTA,
    _STALL_WINDOW,
    StoppingDecision,
    _max_delta_in_window,
    should_stop,
    snapshot_likelihoods,
)
from datetime import datetime, timezone


# ── helpers ───────────────────────────────────────────────────────────────────

def _graph(*likelihoods: float, statuses: list | None = None) -> HypothesisGraph:
    """Build a minimal HypothesisGraph with active hypotheses at the given likelihoods."""
    hypotheses = []
    for i, lk in enumerate(likelihoods):
        status = HypothesisStatus.ACTIVE
        if statuses and i < len(statuses):
            status = statuses[i]
        hypotheses.append(Hypothesis(
            id=f"H{i + 1}",
            description=f"Hypothesis {i + 1}",
            likelihood=lk,
            severity=Severity.MEDIUM,
            status=status,
        ))
    return HypothesisGraph(
        alert_summary="test alert",
        investigation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        tool_call_budget=8,
        hypotheses=hypotheses,
    )


def _snaps(*likelihood_maps: dict) -> list[dict[str, float]]:
    return list(likelihood_maps)


# ── StoppingDecision dataclass ────────────────────────────────────────────────

class TestStoppingDecision:
    def test_frozen(self):
        d = StoppingDecision(stop=True, reason="top_likelihood", top_hypothesis=None)
        with pytest.raises((AttributeError, TypeError)):
            d.stop = False  # type: ignore[misc]

    def test_stop_false_reason_none(self):
        d = StoppingDecision(stop=False, reason=None, top_hypothesis=None)
        assert not d.stop
        assert d.reason is None


# ── Trigger 1: top_likelihood ─────────────────────────────────────────────────

class TestTopLikelihood:
    def test_fires_above_threshold(self):
        g = _graph(0.65, 0.35)
        result = should_stop(g, [])
        assert result.stop
        assert result.reason == "top_likelihood"

    def test_fires_at_exact_boundary(self):
        # > 0.60, not >=
        g = _graph(0.601, 0.399)
        result = should_stop(g, [])
        assert result.stop

    def test_does_not_fire_at_threshold(self):
        g = _graph(0.60, 0.40)
        result = should_stop(g, [])
        assert not result.stop or result.reason != "top_likelihood"

    def test_does_not_fire_below_threshold(self):
        g = _graph(0.55, 0.45)
        result = should_stop(g, [])
        # might still stop on ratio, but not on top_likelihood
        if result.stop:
            assert result.reason != "top_likelihood"

    def test_top_hypothesis_is_returned(self):
        g = _graph(0.70, 0.30)
        result = should_stop(g, [])
        assert result.top_hypothesis is not None
        assert result.top_hypothesis.id == "H1"
        assert result.top_hypothesis.likelihood == pytest.approx(0.70)


# ── Trigger 2: dominance_ratio ────────────────────────────────────────────────

class TestDominanceRatio:
    def test_fires_above_ratio(self):
        # 0.72 / 0.25 = 2.88 > 2.5, but 0.72 < 0.60 after normalisation…
        # Use raw values that will be < 0.60 after normalization: 0.50 / 0.10 = 5.0
        # But normalization happens inside update_graph, not here.
        # Build graph with already-normalized values:
        g = _graph(0.51, 0.20)  # 0.51/0.20 = 2.55 > 2.5, top < 0.60
        result = should_stop(g, [])
        assert result.stop
        assert result.reason == "dominance_ratio"

    def test_does_not_fire_below_ratio(self):
        g = _graph(0.55, 0.30)  # 0.55/0.30 = 1.83 < 2.5
        result = should_stop(g, [])
        if result.stop:
            assert result.reason != "dominance_ratio"

    def test_single_hypothesis_no_ratio_trigger(self):
        g = _graph(0.55)  # only one active hypothesis
        result = should_stop(g, [])
        if result.stop:
            assert result.reason != "dominance_ratio"

    def test_zero_runner_up_no_division_error(self):
        g = _graph(0.55, 0.0)
        # runner_up.likelihood == 0 → ratio guard prevents ZeroDivisionError
        result = should_stop(g, [])
        assert isinstance(result, StoppingDecision)

    def test_ruled_out_hypotheses_excluded(self):
        g = _graph(0.55, 0.30, statuses=[HypothesisStatus.ACTIVE, HypothesisStatus.RULED_OUT])
        result = should_stop(g, [])
        # Only one active hypothesis — ratio trigger can't fire
        if result.stop:
            assert result.reason != "dominance_ratio"


# ── Trigger 3: stalled ────────────────────────────────────────────────────────

class TestStalled:
    def test_fires_when_all_deltas_below_threshold(self):
        g = _graph(0.52, 0.48)  # below likelihood and ratio thresholds
        snaps = [
            {"H1": 0.52, "H2": 0.48},
            {"H1": 0.521, "H2": 0.479},
            {"H1": 0.522, "H2": 0.478},
        ]
        result = should_stop(g, snaps, stall_window=3)
        assert result.stop
        assert result.reason == "stalled"

    def test_does_not_fire_when_delta_large_enough(self):
        g = _graph(0.52, 0.48)
        snaps = [
            {"H1": 0.40, "H2": 0.60},
            {"H1": 0.46, "H2": 0.54},
            {"H1": 0.52, "H2": 0.48},
        ]
        result = should_stop(g, snaps, stall_window=3)
        if result.stop:
            assert result.reason != "stalled"

    def test_does_not_fire_with_fewer_snaps_than_window(self):
        g = _graph(0.52, 0.48)
        snaps = [{"H1": 0.52, "H2": 0.48}, {"H1": 0.521, "H2": 0.479}]
        result = should_stop(g, snaps, stall_window=3)
        if result.stop:
            assert result.reason != "stalled"

    def test_fires_with_exactly_window_snaps(self):
        g = _graph(0.52, 0.48)
        snaps = [
            {"H1": 0.52, "H2": 0.48},
            {"H1": 0.521, "H2": 0.479},
            {"H1": 0.522, "H2": 0.478},
        ]
        result = should_stop(g, snaps, stall_window=3)
        assert result.stop
        assert result.reason == "stalled"

    def test_uses_only_last_n_snaps(self):
        # First 3 have large delta; last 3 are flat — should trigger stall
        g = _graph(0.52, 0.48)
        snaps = [
            {"H1": 0.30, "H2": 0.70},
            {"H1": 0.40, "H2": 0.60},
            {"H1": 0.50, "H2": 0.50},
            {"H1": 0.52, "H2": 0.48},
            {"H1": 0.521, "H2": 0.479},
            {"H1": 0.522, "H2": 0.478},
        ]
        result = should_stop(g, snaps, stall_window=3)
        assert result.stop
        assert result.reason == "stalled"


# ── No active hypotheses ──────────────────────────────────────────────────────

class TestNoActiveHypotheses:
    def test_empty_graph_returns_no_stop(self):
        g = _graph()
        result = should_stop(g, [])
        assert not result.stop
        assert result.reason is None
        assert result.top_hypothesis is None

    def test_all_ruled_out_returns_no_stop(self):
        g = _graph(0.70, 0.30, statuses=[HypothesisStatus.RULED_OUT, HypothesisStatus.RULED_OUT])
        result = should_stop(g, [])
        assert not result.stop


# ── snapshot_likelihoods ──────────────────────────────────────────────────────

class TestSnapshotLikelihoods:
    def test_returns_active_only(self):
        g = _graph(0.60, 0.40, statuses=[HypothesisStatus.ACTIVE, HypothesisStatus.RULED_OUT])
        snap = snapshot_likelihoods(g)
        assert "H1" in snap
        assert "H2" not in snap

    def test_values_match_likelihoods(self):
        g = _graph(0.65, 0.35)
        snap = snapshot_likelihoods(g)
        assert snap["H1"] == pytest.approx(0.65)
        assert snap["H2"] == pytest.approx(0.35)

    def test_empty_graph_returns_empty_dict(self):
        g = _graph()
        assert snapshot_likelihoods(g) == {}


# ── _max_delta_in_window ──────────────────────────────────────────────────────

class TestMaxDeltaInWindow:
    def test_single_snapshot_returns_inf(self):
        assert _max_delta_in_window([{"H1": 0.5}]) == float("inf")

    def test_empty_returns_inf(self):
        assert _max_delta_in_window([]) == float("inf")

    def test_identical_snapshots_returns_zero(self):
        snaps = [{"H1": 0.5, "H2": 0.5}, {"H1": 0.5, "H2": 0.5}]
        assert _max_delta_in_window(snaps) == pytest.approx(0.0)

    def test_computes_max_over_all_hypotheses(self):
        snaps = [
            {"H1": 0.40, "H2": 0.60},
            {"H1": 0.45, "H2": 0.55},
        ]
        # H1 moved 0.05, H2 moved 0.05 — max = 0.05
        assert _max_delta_in_window(snaps) == pytest.approx(0.05)

    def test_hypothesis_appearing_only_in_one_snapshot(self):
        snaps = [{"H1": 0.5}, {"H1": 0.5, "H2": 0.3}]
        # H2: max=0.3, min=0.0 (absent treated as 0.0) → delta=0.3
        assert _max_delta_in_window(snaps) == pytest.approx(0.3)
