"""
Unit and integration tests for generator/generate.py.
"""

from __future__ import annotations
import json
import sqlite3

import numpy as np
import pytest

from generator.defaults import SIM_START, make_default_config
from generator.generate import (
    aggregate_metrics,
    generate_data,
    generate_sub_range,
    resolve_sub_ranges,
)
from generator.params import (
    FeatureConfig,
    FeatureParams,
    InferenceParams,
    LogParams,
    SubRangeConfig,
)
from generator.schema import query_metrics


from pathlib import Path
MODEL_DIR = str(Path(__file__).resolve().parent.parent / "model" / "artifacts")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sub_range(start, end, requests_per_hour=100.0, error_rate=0.01):
    fp = FeatureParams(
        requests_per_hour=requests_per_hour,
        feature_configs=[
            FeatureConfig("account_age_days",      "normal",      mean=730,  std=200),
            FeatureConfig("monthly_spend",         "lognormal",   mean=5.5,  std=0.8),
            FeatureConfig("num_transactions_30d",  "normal",      mean=42,   std=15),
            FeatureConfig("avg_transaction_value", "lognormal",   mean=4.0,  std=0.6),
            FeatureConfig("days_since_last_login", "normal",      mean=3,    std=2),
            FeatureConfig("support_tickets_90d",   "normal",      mean=1.2,  std=1.5),
            FeatureConfig("product_category",      "categorical", categories=[0, 1, 2],       probs=[0.50, 0.35, 0.15]),
            FeatureConfig("region",                "categorical", categories=[0, 1, 2, 3, 4], probs=[0.20, 0.30, 0.25, 0.15, 0.10]),
            FeatureConfig("device_type",           "categorical", categories=[0, 1, 2],       probs=[0.55, 0.35, 0.10]),
            FeatureConfig("login_failure_rate",    "uniform",     low=0.0,   high=0.3),
            FeatureConfig("session_duration_min",  "normal",      mean=18,   std=8),
            FeatureConfig("referral_source",       "categorical", categories=[0, 1, 2, 3],    probs=[0.25, 0.35, 0.25, 0.15]),
        ],
    )
    ip = InferenceParams(error_rate=error_rate, timeout_rate=error_rate * 0.2)
    lp = LogParams()
    return SubRangeConfig(start_ts=start, end_ts=end, feature_params=fp,
                          inference_params=ip, log_params=lp)


# ---------------------------------------------------------------------------
# resolve_sub_ranges
# ---------------------------------------------------------------------------

class TestResolveSubRanges:
    def test_no_overrides_single_range(self):
        config = make_default_config(n_days=1)
        config.overrides = []
        ranges = resolve_sub_ranges(config)
        assert len(ranges) == 1
        assert ranges[0].start_ts == config.start_ts
        assert ranges[0].end_ts == config.end_ts

    def test_override_in_middle_produces_three_ranges(self):
        config = make_default_config(n_days=3)
        mid = _make_sub_range(SIM_START + 86400, SIM_START + 2 * 86400)
        config.overrides = [mid]
        ranges = resolve_sub_ranges(config)
        assert len(ranges) == 3
        assert ranges[0].start_ts == config.start_ts
        assert ranges[0].end_ts == SIM_START + 86400
        assert ranges[1] is mid
        assert ranges[2].start_ts == SIM_START + 2 * 86400
        assert ranges[2].end_ts == config.end_ts

    def test_override_at_start_two_ranges(self):
        config = make_default_config(n_days=2)
        ov = _make_sub_range(SIM_START, SIM_START + 86400)
        config.overrides = [ov]
        ranges = resolve_sub_ranges(config)
        assert len(ranges) == 2
        assert ranges[0] is ov
        assert ranges[1].start_ts == SIM_START + 86400

    def test_override_at_end_two_ranges(self):
        config = make_default_config(n_days=2)
        ov = _make_sub_range(SIM_START + 86400, SIM_START + 2 * 86400)
        config.overrides = [ov]
        ranges = resolve_sub_ranges(config)
        assert len(ranges) == 2
        assert ranges[0].end_ts == SIM_START + 86400
        assert ranges[1] is ov

    def test_multiple_overrides(self):
        config = make_default_config(n_days=4)
        ov1 = _make_sub_range(SIM_START + 86400, SIM_START + 2 * 86400)
        ov2 = _make_sub_range(SIM_START + 3 * 86400, SIM_START + 4 * 86400)
        config.overrides = [ov1, ov2]
        ranges = resolve_sub_ranges(config)
        assert len(ranges) == 4  # gap-ov1-gap-ov2 (leading default + ov1 + gap + ov2)

    def test_full_window_coverage(self):
        config = make_default_config(n_days=3)
        ov = _make_sub_range(SIM_START + 86400, SIM_START + 2 * 86400)
        config.overrides = [ov]
        ranges = resolve_sub_ranges(config)
        assert ranges[0].start_ts == config.start_ts
        assert ranges[-1].end_ts == config.end_ts
        for i in range(len(ranges) - 1):
            assert ranges[i].end_ts == ranges[i + 1].start_ts


# ---------------------------------------------------------------------------
# generate_sub_range
# ---------------------------------------------------------------------------

class TestGenerateSubRange:
    @pytest.fixture(scope="class")
    def sub_range_output(self, loaded_model):
        model, scaler, info = loaded_model
        feature_names = info["feature_names"]
        gt_weights = np.array(info["ground_truth_weights"])
        gt_bias = float(info["ground_truth_bias"])
        sr = _make_sub_range(SIM_START, SIM_START + 3600, requests_per_hour=500, error_rate=0.01)
        rng = np.random.default_rng(42)
        req_rows, log_rows = generate_sub_range(
            model, scaler, feature_names, gt_weights, gt_bias, sr, 24.0, rng
        )
        return req_rows, log_rows, sr

    def test_request_count_approximate(self, sub_range_output):
        req_rows, _, sr = sub_range_output
        expected = sr.feature_params.requests_per_hour * 1  # 1 hour
        assert abs(len(req_rows) - expected) / expected < 0.05  # within 5%

    def test_timestamps_sorted(self, sub_range_output):
        req_rows, _, _ = sub_range_output
        ts = [r["timestamp"] for r in req_rows]
        assert ts == sorted(ts)

    def test_timestamps_in_range(self, sub_range_output):
        req_rows, _, sr = sub_range_output
        for r in req_rows:
            assert sr.start_ts <= r["timestamp"] < sr.end_ts

    def test_error_sentinel_values(self, sub_range_output):
        req_rows, _, _ = sub_range_output
        error_rows = [r for r in req_rows if r["_is_error"]]
        for r in error_rows:
            assert r["prediction"] == -1
            assert r["pred_proba"] == -1.0

    def test_valid_rows_have_correct_prediction(self, sub_range_output):
        req_rows, _, _ = sub_range_output
        valid_rows = [r for r in req_rows if not r["_is_error"]]
        for r in valid_rows:
            assert r["prediction"] in (0, 1)
            assert 0.0 <= r["pred_proba"] <= 1.0

    def test_label_ts_equals_timestamp_plus_delay(self, sub_range_output):
        req_rows, _, _ = sub_range_output
        delay = 24.0 * 3600
        for r in req_rows:
            assert r["label_ts"] == pytest.approx(r["timestamp"] + delay)

    def test_features_json_has_all_12_names(self, sub_range_output, feature_names):
        req_rows, _, _ = sub_range_output
        for r in req_rows[:20]:  # spot-check first 20
            feat = json.loads(r["features"])
            assert set(feat.keys()) == set(feature_names)

    def test_error_rate_within_bounds(self, sub_range_output):
        req_rows, _, sr = sub_range_output
        actual = sum(1 for r in req_rows if r["_is_error"]) / len(req_rows)
        expected = sr.inference_params.error_rate
        # Allow 3 sigma: sqrt(p*(1-p)/n)
        n = len(req_rows)
        sigma = (expected * (1 - expected) / n) ** 0.5
        assert abs(actual - expected) < 4 * sigma

    def test_timeout_only_when_error(self, sub_range_output):
        req_rows, _, _ = sub_range_output
        for r in req_rows:
            if r["_is_timeout"]:
                assert r["_is_error"]

    def test_reproducible_with_same_seed(self, loaded_model):
        model, scaler, info = loaded_model
        feature_names = info["feature_names"]
        gt_weights = np.array(info["ground_truth_weights"])
        gt_bias = float(info["ground_truth_bias"])
        sr = _make_sub_range(SIM_START, SIM_START + 3600, requests_per_hour=50)

        rows_a, _ = generate_sub_range(model, scaler, feature_names, gt_weights, gt_bias,
                                        sr, 24.0, np.random.default_rng(99))
        rows_b, _ = generate_sub_range(model, scaler, feature_names, gt_weights, gt_bias,
                                        sr, 24.0, np.random.default_rng(99))
        assert len(rows_a) == len(rows_b)
        for a, b in zip(rows_a, rows_b):
            # request_id is uuid4() — not seeded by numpy RNG, so not compared
            assert a["timestamp"] == pytest.approx(b["timestamp"])
            assert a["pred_proba"] == pytest.approx(b["pred_proba"])
            assert a["prediction"] == b["prediction"]
            assert a["true_label"] == b["true_label"]


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def _make_request_row(self, ts, proba=0.3, prediction=0, true_label=0,
                           error=False, latency=50.0, label_delay=24 * 3600):
        return {
            "timestamp": float(ts),
            "pred_proba": proba if not error else -1.0,
            "prediction": prediction if not error else -1,
            "true_label": true_label,
            "label_ts": float(ts + label_delay),
            "_latency_ms": latency,
            "_is_error": error,
            "_is_timeout": False,
        }

    def test_throughput_counts_all_requests(self):
        base = SIM_START
        rows = [self._make_request_row(base + i * 10) for i in range(10)]
        rows.append(self._make_request_row(base + 5, error=True))
        metrics = aggregate_metrics(rows, 24.0, base, base + 3600)
        tp = next(m for m in metrics if m["metric_name"] == "throughput")
        assert tp["metric_value"] == 11.0

    def test_error_rate_excludes_valid(self):
        base = SIM_START
        rows = [self._make_request_row(base + i * 10) for i in range(9)]
        rows.append(self._make_request_row(base + 5, error=True))
        metrics = aggregate_metrics(rows, 24.0, base, base + 3600)
        er = next(m for m in metrics if m["metric_name"] == "error_rate")
        assert er["metric_value"] == pytest.approx(0.1)

    def test_accuracy_appears_in_label_bucket_not_request_bucket(self):
        delay_hours = 24.0
        base = SIM_START
        # Request at hour 0, label arrives at hour 24
        rows = [self._make_request_row(base + i * 10, prediction=0, true_label=0,
                                        label_delay=delay_hours * 3600) for i in range(10)]
        metrics = aggregate_metrics(rows, delay_hours, base, base + 25 * 3600)

        acc_rows = [m for m in metrics if m["metric_name"] == "accuracy"]
        assert len(acc_rows) > 0
        # Accuracy bucket timestamp must be >= base + 24h
        for ar in acc_rows:
            assert ar["timestamp"] >= base + delay_hours * 3600

    def test_no_accuracy_in_first_24h(self):
        delay_hours = 24.0
        base = SIM_START
        rows = [self._make_request_row(base + i * 10) for i in range(100)]
        metrics = aggregate_metrics(rows, delay_hours, base, base + 2 * 3600)

        acc_rows = [m for m in metrics if m["metric_name"] == "accuracy"
                    and m["timestamp"] < base + delay_hours * 3600]
        assert acc_rows == []

    def test_six_non_accuracy_metrics_per_bucket(self):
        base = SIM_START
        rows = [self._make_request_row(base + i * 10) for i in range(20)]
        metrics = aggregate_metrics(rows, 24.0, base, base + 3600)

        hour_bucket_metrics = {m["metric_name"] for m in metrics
                                if m["timestamp"] == base}
        expected = {"throughput", "latency_p50", "latency_p99",
                    "error_rate", "prediction_confidence", "positive_rate"}
        assert expected.issubset(hour_bucket_metrics)

    def test_prediction_confidence_excludes_errors(self):
        base = SIM_START
        rows = [self._make_request_row(base + i * 10, proba=0.8) for i in range(5)]
        rows += [self._make_request_row(base + 50, error=True)]  # error has proba=-1
        metrics = aggregate_metrics(rows, 24.0, base, base + 3600)
        pc = next(m for m in metrics if m["metric_name"] == "prediction_confidence")
        assert pc["metric_value"] == pytest.approx(0.8)

    def test_accurate_predictions_yield_high_accuracy(self):
        delay_hours = 1.0  # short delay so accuracy appears within 2h window
        base = SIM_START
        # All predictions correct: prediction=0, true_label=0
        rows = [self._make_request_row(base + i * 10, prediction=0, true_label=0,
                                        label_delay=delay_hours * 3600) for i in range(50)]
        metrics = aggregate_metrics(rows, delay_hours, base, base + 4 * 3600)
        acc_rows = [m for m in metrics if m["metric_name"] == "accuracy"]
        assert len(acc_rows) > 0
        assert all(m["metric_value"] == pytest.approx(1.0) for m in acc_rows)


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    @pytest.mark.timeout(120)
    def test_generate_data_populates_all_stores(self, tmp_path):
        config = make_default_config(n_days=1)
        config.output_dir = str(tmp_path)
        config.seed = 42
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        # Feature store row count
        con = sqlite3.connect(db_paths["feature_store"])
        n_requests = con.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        con.close()
        expected = config.default_feature_params.requests_per_hour * 24
        assert abs(n_requests - expected) / expected < 0.05

    @pytest.mark.timeout(120)
    def test_all_seven_metric_names_present(self, tmp_path):
        config = make_default_config(n_days=1)
        config.output_dir = str(tmp_path)
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        con = sqlite3.connect(db_paths["metrics"])
        names = {r[0] for r in con.execute("SELECT DISTINCT metric_name FROM metrics").fetchall()}
        con.close()
        expected = {"throughput", "latency_p50", "latency_p99", "error_rate",
                    "prediction_confidence", "positive_rate", "accuracy"}
        assert expected == names

    @pytest.mark.timeout(120)
    def test_accuracy_rows_start_after_24h(self, tmp_path):
        config = make_default_config(n_days=2)
        config.output_dir = str(tmp_path)
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        con = sqlite3.connect(db_paths["metrics"])
        early_acc = con.execute(
            "SELECT COUNT(*) FROM metrics WHERE metric_name='accuracy' AND timestamp < ?",
            (config.start_ts + 86400,)
        ).fetchone()[0]
        con.close()
        assert early_acc == 0

    @pytest.mark.timeout(120)
    def test_log_severities_all_present(self, tmp_path):
        config = make_default_config(n_days=1)
        config.output_dir = str(tmp_path)
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        con = sqlite3.connect(db_paths["logs"])
        severities = {r[0] for r in con.execute("SELECT DISTINCT severity FROM logs").fetchall()}
        con.close()
        assert {"INFO", "WARNING", "ERROR"}.issubset(severities)

    @pytest.mark.timeout(120)
    def test_deployment_count(self, tmp_path):
        config = make_default_config(n_days=1)
        config.output_dir = str(tmp_path)
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        con = sqlite3.connect(db_paths["deployments"])
        n = con.execute("SELECT COUNT(*) FROM deployments").fetchone()[0]
        con.close()
        assert n == len(config.deployment_events)

    @pytest.mark.timeout(120)
    def test_all_deploy_shas_non_empty(self, tmp_path):
        config = make_default_config(n_days=1)
        config.output_dir = str(tmp_path)
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        con = sqlite3.connect(db_paths["deployments"])
        shas = [r[0] for r in con.execute("SELECT commit_sha FROM deployments").fetchall()]
        con.close()
        assert all(sha and len(sha) > 0 for sha in shas)
