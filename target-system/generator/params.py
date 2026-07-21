"""
Parameter dataclasses for all data generators.

Design: every generator accepts an explicit param object. Chaos injection
overrides these for a sub-range without touching generator code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    name: str
    distribution: Literal["normal", "lognormal", "uniform", "categorical"]
    # normal / lognormal
    mean: float = 0.0
    std: float = 1.0
    # uniform
    low: float = 0.0
    high: float = 1.0
    # categorical
    categories: list = field(default_factory=list)
    probs: list = field(default_factory=list)


@dataclass
class FeatureParams:
    feature_configs: list[FeatureConfig]
    requests_per_hour: float = 500.0


# ---------------------------------------------------------------------------
# Inference / latency / error
# ---------------------------------------------------------------------------

@dataclass
class InferenceParams:
    baseline_latency_ms: float = 45.0
    latency_std_ms: float = 8.0
    # p99 is drawn from an exponential tail added to the baseline
    tail_prob: float = 0.02        # fraction of requests that hit the tail
    tail_latency_ms: float = 300.0 # tail latency value
    error_rate: float = 0.005      # fraction of requests that error
    timeout_rate: float = 0.001    # subset of errors that are timeouts


# ---------------------------------------------------------------------------
# Log generation
# ---------------------------------------------------------------------------

@dataclass
class LogParams:
    # Fraction of non-error requests that emit an extra INFO log entry
    info_log_rate: float = 0.05
    # Fraction of non-error requests that emit a WARNING
    warning_rate: float = 0.01
    # Additional ERROR logs unrelated to request errors (e.g. background jobs)
    background_error_rate: float = 0.002


# ---------------------------------------------------------------------------
# Sub-range config (used for chaos injection overrides later)
# ---------------------------------------------------------------------------

@dataclass
class SubRangeConfig:
    """Parameters governing a contiguous slice of the simulation time window."""
    start_ts: float   # Unix timestamp (simulated clock)
    end_ts: float
    feature_params: FeatureParams
    inference_params: InferenceParams
    log_params: LogParams


# ---------------------------------------------------------------------------
# Deployment events
# ---------------------------------------------------------------------------

@dataclass
class DeploymentEvent:
    timestamp: float
    version_before: str
    version_after: str
    service: str   # matches ServiceName enum in tool-specs.md
    change_type: Literal[
        "model_retrain",
        "config_change",
        "feature_pipeline_change",
        "dependency_bump",
    ]
    changelog: str
    commit_sha: str
    deployed_by: str
    is_rollback: bool = False
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level generation config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """
    Describes a full simulation run.

    For normal operation: leave `overrides` empty — the default params apply
    across the entire window.

    For chaos injection (future): add SubRangeConfig entries to `overrides`
    that cover the failure window. The generator fills gaps with default
    params automatically.
    """
    start_ts: float               # simulated window start (Unix timestamp)
    end_ts: float                 # simulated window end
    default_feature_params: FeatureParams
    default_inference_params: InferenceParams
    default_log_params: LogParams
    overrides: list[SubRangeConfig] = field(default_factory=list)
    deployment_events: list[DeploymentEvent] = field(default_factory=list)
    label_delay_hours: float = 24.0
    seed: int = 42
    output_dir: str = "data"
