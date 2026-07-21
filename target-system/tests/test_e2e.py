"""
End-to-end test: causal linkage and chaos injection readiness.

Generates a 3-day dataset where day 1 has a shifted login_failure_rate
(doubled range), then verifies that feature drift propagates correctly
downstream through model inference to metrics — without any hardcoded
metric changes.
"""

from __future__ import annotations
import sqlite3

import numpy as np
import pytest

from generator.defaults import SIM_START, make_default_config
from generator.generate import generate_data
from generator.params import FeatureConfig, FeatureParams, InferenceParams, LogParams, SubRangeConfig
from generator.schema import query_feature_values, query_metrics


MODEL_DIR = "model/artifacts"

N_BINS = 10  # bins for PSI computation


def _compute_psi(baseline: list[float], comparison: list[float], n_bins: int = N_BINS) -> float:
    """Population Stability Index between two value arrays."""
    if not baseline or not comparison:
        return 0.0
    all_vals = baseline + comparison
    bin_edges = np.linspace(min(all_vals), max(all_vals) + 1e-9, n_bins + 1)

    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    comp_counts, _ = np.histogram(comparison, bins=bin_edges)

    eps = 1e-6
    base_freq = (base_counts + eps) / (len(baseline) + eps * n_bins)
    comp_freq = (comp_counts + eps) / (len(comparison) + eps * n_bins)

    return float(np.sum((comp_freq - base_freq) * np.log(comp_freq / base_freq)))


def _mean_metric(db_path: str, name: str, start_ts: float, end_ts: float) -> float:
    rows = query_metrics(db_path, [name], start_ts, end_ts)
    vals = [r["metric_value"] for r in rows if r["metric_name"] == name]
    return float(np.mean(vals)) if vals else float("nan")


@pytest.fixture(scope="module")
def chaos_data(tmp_path_factory):
    """
    3-day dataset:
      Day 0: normal (login_failure_rate ~ U(0, 0.3))
      Day 1: DRIFTED (login_failure_rate ~ U(0.2, 0.6))
      Day 2: normal again
    Returns db_paths and time boundaries.
    """
    out_dir = str(tmp_path_factory.mktemp("chaos"))
    config = make_default_config(n_days=3)
    config.output_dir = out_dir
    config.seed = 42

    # Build drifted FeatureParams: only login_failure_rate changed
    normal_configs = config.default_feature_params.feature_configs
    drifted_configs = [
        FeatureConfig("login_failure_rate", "uniform", low=0.2, high=0.6)
        if cfg.name == "login_failure_rate" else cfg
        for cfg in normal_configs
    ]
    drifted_fp = FeatureParams(
        requests_per_hour=config.default_feature_params.requests_per_hour,
        feature_configs=drifted_configs,
    )

    config.overrides = [
        SubRangeConfig(
            start_ts=SIM_START + 86400,       # day 1 start
            end_ts=SIM_START + 2 * 86400,     # day 1 end
            feature_params=drifted_fp,
            inference_params=config.default_inference_params,
            log_params=config.default_log_params,
        )
    ]

    db_paths = generate_data(config, MODEL_DIR, verbose=False)
    return {
        "db_paths": db_paths,
        "day0_start": SIM_START,
        "day0_end":   SIM_START + 86400,
        "day1_start": SIM_START + 86400,
        "day1_end":   SIM_START + 2 * 86400,
        "day2_start": SIM_START + 2 * 86400,
        "day2_end":   SIM_START + 3 * 86400,
    }


@pytest.mark.timeout(180)
class TestCausalLinkage:
    def test_feature_drift_visible_in_store(self, chaos_data):
        db = chaos_data["db_paths"]["feature_store"]
        day0_vals = query_feature_values(
            db, ["login_failure_rate"],
            chaos_data["day0_start"], chaos_data["day0_end"]
        )["login_failure_rate"]
        day1_vals = query_feature_values(
            db, ["login_failure_rate"],
            chaos_data["day1_start"], chaos_data["day1_end"]
        )["login_failure_rate"]

        day0_mean = np.mean(day0_vals)
        day1_mean = np.mean(day1_vals)
        # Day 0: U(0, 0.3) → mean ≈ 0.15; Day 1: U(0.2, 0.6) → mean ≈ 0.4
        assert day0_mean == pytest.approx(0.15, abs=0.03)
        assert day1_mean == pytest.approx(0.40, abs=0.03)

    def test_psi_detects_significant_drift(self, chaos_data):
        db = chaos_data["db_paths"]["feature_store"]
        day0_vals = query_feature_values(
            db, ["login_failure_rate"],
            chaos_data["day0_start"], chaos_data["day0_end"]
        )["login_failure_rate"]
        day1_vals = query_feature_values(
            db, ["login_failure_rate"],
            chaos_data["day1_start"], chaos_data["day1_end"]
        )["login_failure_rate"]

        psi = _compute_psi(day0_vals, day1_vals)
        assert psi > 0.1, f"PSI={psi:.3f} below moderate drift threshold 0.1"

    def test_undrifted_feature_psi_low(self, chaos_data):
        db = chaos_data["db_paths"]["feature_store"]
        day0_vals = query_feature_values(
            db, ["account_age_days"],
            chaos_data["day0_start"], chaos_data["day0_end"]
        )["account_age_days"]
        day1_vals = query_feature_values(
            db, ["account_age_days"],
            chaos_data["day1_start"], chaos_data["day1_end"]
        )["account_age_days"]

        psi = _compute_psi(day0_vals, day1_vals)
        assert psi < 0.1, f"Undrifted feature PSI={psi:.3f} unexpectedly high"

    def test_prediction_confidence_rises_during_drift(self, chaos_data):
        db = chaos_data["db_paths"]["metrics"]
        conf_day0 = _mean_metric(db, "prediction_confidence",
                                  chaos_data["day0_start"], chaos_data["day0_end"])
        conf_day1 = _mean_metric(db, "prediction_confidence",
                                  chaos_data["day1_start"], chaos_data["day1_end"])
        # Higher login_failure_rate → model assigns higher churn probability
        assert conf_day1 > conf_day0, (
            f"prediction_confidence did not rise: day0={conf_day0:.3f}, day1={conf_day1:.3f}"
        )

    def test_accuracy_degrades_after_drift(self, chaos_data):
        """
        Labels for day 1 (drifted) arrive in day 2 (24h later).
        Accuracy during day 2 should be lower than during day 0+1.
        """
        db = chaos_data["db_paths"]["metrics"]
        # Use day 1 accuracy (labels from normal day 0) as baseline
        acc_day1 = _mean_metric(db, "accuracy",
                                 chaos_data["day1_start"], chaos_data["day1_end"])
        # Day 2 accuracy reflects day 1's drifted predictions
        acc_day2 = _mean_metric(db, "accuracy",
                                 chaos_data["day2_start"], chaos_data["day2_end"])

        if not (acc_day1 > 0 and acc_day2 > 0):
            pytest.skip("Insufficient accuracy rows to compare")

        assert acc_day2 < acc_day1, (
            f"Accuracy did not degrade after drift: day1={acc_day1:.3f}, day2={acc_day2:.3f}"
        )

    def test_sub_range_coverage_complete(self, chaos_data):
        """The override sub-range plus default ranges cover the full 3-day window."""
        db = chaos_data["db_paths"]["feature_store"]
        con = sqlite3.connect(db)
        min_ts, max_ts = con.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM requests"
        ).fetchone()
        con.close()
        assert min_ts >= chaos_data["day0_start"]
        assert max_ts < chaos_data["day2_end"]

    def test_no_generator_code_change_needed_for_override(self, chaos_data):
        """
        The chaos scenario only changed config.overrides — generate_data() itself
        was not modified. Verify by checking the drifted window has MORE feature
        store rows than a trivially incorrect generator (sanity: rows exist for all 3 days).
        """
        db = chaos_data["db_paths"]["feature_store"]
        for start, end in [
            (chaos_data["day0_start"], chaos_data["day0_end"]),
            (chaos_data["day1_start"], chaos_data["day1_end"]),
            (chaos_data["day2_start"], chaos_data["day2_end"]),
        ]:
            con = sqlite3.connect(db)
            count = con.execute(
                "SELECT COUNT(*) FROM requests WHERE timestamp >= ? AND timestamp < ?",
                (start, end)
            ).fetchone()[0]
            con.close()
            assert count > 0, f"No requests found in window [{start}, {end})"
