"""
Main data generator. Produces N days of synthetic normal-operation data,
stored in four SQLite databases.

Usage:
    cd target-system/
    python -m generator.generate --days 7 --output-dir data

Injection-ready design:
  - All generators accept an explicit param object.
  - The time window is split into sub-ranges; each range has independent params.
  - A future chaos scenario supplies SubRangeConfig overrides that cover the
    failure window; this file runs unchanged.
"""

from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import joblib
import numpy as np

# Resolve imports whether run as module or script
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.params import (
    DeploymentEvent,
    FeatureConfig,
    FeatureParams,
    GenerationConfig,
    InferenceParams,
    LogParams,
    SubRangeConfig,
)
from generator.schema import setup_databases
from generator.defaults import make_default_config, SIM_START


# ---------------------------------------------------------------------------
# Sub-range resolution
# ---------------------------------------------------------------------------

def resolve_sub_ranges(config: GenerationConfig) -> list[SubRangeConfig]:
    """
    Fill the full [start_ts, end_ts) window with sub-ranges.
    Overrides take priority; gaps are filled with default params.
    """
    overrides = sorted(config.overrides, key=lambda r: r.start_ts)
    ranges: list[SubRangeConfig] = []
    cursor = config.start_ts

    for ov in overrides:
        if ov.start_ts > cursor:
            ranges.append(SubRangeConfig(
                start_ts=cursor,
                end_ts=ov.start_ts,
                feature_params=config.default_feature_params,
                inference_params=config.default_inference_params,
                log_params=config.default_log_params,
            ))
        ranges.append(ov)
        cursor = ov.end_ts

    if cursor < config.end_ts:
        ranges.append(SubRangeConfig(
            start_ts=cursor,
            end_ts=config.end_ts,
            feature_params=config.default_feature_params,
            inference_params=config.default_inference_params,
            log_params=config.default_log_params,
        ))
    return ranges


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def _sample_feature(cfg: FeatureConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = cfg.distribution
    if dist == "normal":
        vals = rng.normal(cfg.mean, cfg.std, n)
    elif dist == "lognormal":
        vals = np.exp(rng.normal(cfg.mean, cfg.std, n))
    elif dist == "uniform":
        vals = rng.uniform(cfg.low, cfg.high, n)
    elif dist == "categorical":
        vals = rng.choice(cfg.categories, size=n, p=cfg.probs).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {dist!r}")

    # Clip non-negative for known non-negative features
    if dist in ("normal", "lognormal") and cfg.mean >= 0 and cfg.std > 0:
        # Only clip if it makes sense (mean > 0 implies non-negative domain)
        if cfg.mean - 3 * cfg.std < 0 and cfg.mean >= 0:
            vals = np.clip(vals, 0, None)
    return vals


def generate_feature_matrix(fp: FeatureParams, n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Returns (X, feature_names)."""
    cols = [_sample_feature(cfg, n, rng) for cfg in fp.feature_configs]
    names = [cfg.name for cfg in fp.feature_configs]
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Ground-truth labels (must match model/train.py exactly)
# ---------------------------------------------------------------------------

def assign_true_labels(X: np.ndarray, feature_names: list[str], weights: np.ndarray, bias: float, rng: np.random.Generator) -> np.ndarray:
    score = X @ weights + bias + rng.normal(0, 0.3, len(X))
    return (score > 0).astype(int)


# ---------------------------------------------------------------------------
# Latency sampling
# ---------------------------------------------------------------------------

def sample_latencies(ip: InferenceParams, n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.abs(rng.normal(ip.baseline_latency_ms, ip.latency_std_ms, n))
    is_tail = rng.random(n) < ip.tail_prob
    tail = rng.exponential(ip.tail_latency_ms, n)
    return np.where(is_tail, base + tail, base)


# ---------------------------------------------------------------------------
# Log message templates
# ---------------------------------------------------------------------------

_INFO_TEMPLATES = [
    "Request processed successfully for account {account_id}",
    "Feature lookup completed in {lookup_ms:.1f}ms",
    "Prediction cached; returning stored result for request {request_id}",
    "Batch prediction complete: {n} records processed",
]

_WARNING_TEMPLATES = [
    "Feature 'days_since_last_login' out of expected range [{lo}, {hi}]: value={val:.2f}",
    "Prediction confidence below threshold (0.55): score={score:.3f} for request {request_id}",
    "High feature correlation detected between monthly_spend and avg_transaction_value",
    "Slow feature store lookup: {lookup_ms:.1f}ms exceeds SLA of 30ms",
    "Input record missing optional feature 'referral_source'; using default=0",
]

_ERROR_TEMPLATES = [
    (
        "ValidationError: field 'account_age_days' expected float, got null\n"
        "  File 'feature_engineering.py', line 42, in validate_features\n"
        "    raise ValidationError(f'Missing required field: {field}')\n"
        "ValidationError: Missing required field: account_age_days"
    ),
    (
        "TimeoutError: Feature store lookup timed out after 200ms\n"
        "  File 'serve.py', line 87, in get_features\n"
        "    features = feature_store.lookup(request_id, timeout=0.2)\n"
        "TimeoutError: Connection to feature-store:6379 timed out"
    ),
    (
        "XGBoostError: Prediction failed for malformed input\n"
        "  File 'serve.py', line 102, in predict\n"
        "    return model.predict_proba(X)\n"
        "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')"
    ),
    (
        "SchemaValidationError: Unexpected feature count (got 11, expected 12)\n"
        "  File 'feature_engineering.py', line 78, in build_feature_vector\n"
        "    assert len(features) == EXPECTED_FEATURE_COUNT\n"
        "AssertionError"
    ),
]

_BACKGROUND_ERROR_TEMPLATES = [
    (
        "LabelPipelineError: Failed to fetch labels for batch 2024-01-{day:02d}\n"
        "  File 'label_pipeline.py', line 55, in fetch_batch_labels\n"
        "    response = requests.get(label_url, timeout=10)\n"
        "ConnectionError: HTTPSConnectionPool(host='labels.internal', port=443): Max retries exceeded"
    ),
    "MetricsFlushError: Failed to write hourly metrics to metrics.db: database is locked",
    "HealthCheckWarning: Feature store connection pool at 85% capacity (34/40 connections in use)",
]


def _make_log_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Request generation for one sub-range
# ---------------------------------------------------------------------------

def generate_sub_range(
    model,
    scaler,
    feature_names: list[str],
    gt_weights: np.ndarray,
    gt_bias: float,
    sub_range: SubRangeConfig,
    label_delay_hours: float,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generate requests, metrics rows, and log entries for one time sub-range.

    Causal chain: features → model inference → prediction distribution
    Labels are assigned by the ground-truth function (not the model) and
    become available after label_delay_hours.

    Returns: (request_rows, log_rows, []) — metrics are aggregated later
    """
    fp = sub_range.feature_params
    ip = sub_range.inference_params
    lp = sub_range.log_params

    duration_hours = (sub_range.end_ts - sub_range.start_ts) / 3600.0
    n = max(1, int(duration_hours * fp.requests_per_hour))

    # Random timestamps within range (sorted)
    timestamps = np.sort(rng.uniform(sub_range.start_ts, sub_range.end_ts, n))

    # Generate feature matrix
    X, _ = generate_feature_matrix(fp, n, rng)

    # True labels (arrive with delay)
    true_labels = assign_true_labels(X, feature_names, gt_weights, gt_bias, rng)

    # Model inference (causal link: features → predictions)
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    # Latency
    latencies = sample_latencies(ip, n, rng)

    # Errors
    is_error = rng.random(n) < ip.error_rate
    is_timeout = is_error & (rng.random(n) < (ip.timeout_rate / ip.error_rate if ip.error_rate > 0 else 0))

    # Build request rows
    request_rows = []
    for i in range(n):
        rid = str(uuid.uuid4())
        ts = float(timestamps[i])
        label_ts = ts + label_delay_hours * 3600

        feat_dict = {name: float(X[i, j]) for j, name in enumerate(feature_names)}

        row = {
            "request_id": rid,
            "timestamp":  ts,
            "features":   json.dumps(feat_dict),
            "prediction": int(predictions[i]) if not is_error[i] else -1,
            "pred_proba": float(probas[i]) if not is_error[i] else -1.0,
            "true_label": int(true_labels[i]),
            "label_ts":   label_ts,
            "service":    "inference_service",
            "_latency_ms": float(latencies[i]),
            "_is_error":   bool(is_error[i]),
            "_is_timeout": bool(is_timeout[i]),
        }
        request_rows.append(row)

    # Build log rows
    log_rows = []
    for i, row in enumerate(request_rows):
        rid = row["request_id"]
        ts = row["timestamp"]

        if row["_is_error"]:
            msg = _ERROR_TEMPLATES[rng.integers(0, len(_ERROR_TEMPLATES))]
            if row["_is_timeout"]:
                msg = _ERROR_TEMPLATES[1]  # timeout template
            log_rows.append({
                "log_id":    _make_log_id(),
                "timestamp": ts + float(latencies[i]),
                "severity":  "ERROR",
                "service":   "inference_service",
                "message":   msg,
                "context":   json.dumps({"request_id": rid, "latency_ms": float(latencies[i])}),
            })
        else:
            # Occasional INFO log
            if rng.random() < lp.info_log_rate:
                tmpl = _INFO_TEMPLATES[rng.integers(0, len(_INFO_TEMPLATES))]
                msg = tmpl.format(
                    account_id=rid[:8],
                    lookup_ms=rng.uniform(2, 25),
                    request_id=rid[:8],
                    n=rng.integers(1, 100),
                )
                log_rows.append({
                    "log_id":    _make_log_id(),
                    "timestamp": ts,
                    "severity":  "INFO",
                    "service":   "inference_service",
                    "message":   msg,
                    "context":   json.dumps({"request_id": rid}),
                })

            # Occasional WARNING log
            if rng.random() < lp.warning_rate:
                tmpl = _WARNING_TEMPLATES[rng.integers(0, len(_WARNING_TEMPLATES))]
                feat_val = X[i, 0]
                msg = tmpl.format(
                    lo=600, hi=1200, val=feat_val,
                    score=float(probas[i]),
                    request_id=rid[:8],
                    lookup_ms=rng.uniform(30, 80),
                )
                log_rows.append({
                    "log_id":    _make_log_id(),
                    "timestamp": ts,
                    "severity":  "WARNING",
                    "service":   "inference_service",
                    "message":   msg,
                    "context":   json.dumps({"request_id": rid}),
                })

        # Background errors (not tied to a specific request)
        if rng.random() < lp.background_error_rate:
            tmpl = _BACKGROUND_ERROR_TEMPLATES[rng.integers(0, len(_BACKGROUND_ERROR_TEMPLATES))]
            msg = tmpl if isinstance(tmpl, str) else tmpl.format(day=1)
            log_rows.append({
                "log_id":    _make_log_id(),
                "timestamp": ts + rng.uniform(0, 3600),
                "severity":  "ERROR",
                "service":   rng.choice(["label_pipeline", "feature_pipeline"]),
                "message":   msg,
                "context":   json.dumps({}),
            })

    return request_rows, log_rows


# ---------------------------------------------------------------------------
# Hourly metric aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(request_rows: list[dict], label_delay_hours: float, start_ts: float, end_ts: float) -> list[dict]:
    """
    Aggregate per-request data into hourly metric rows.

    Accuracy is computed for requests whose label arrived during the bucket
    (label_ts ∈ [bucket_start, bucket_start + 3600)). This simulates the
    realistic 24h label delay: at hour H we know labels for requests from
    hour H - 24h, not the current hour's requests.
    """
    if not request_rows:
        return []

    # Index by request hour
    req_buckets: dict[float, list[dict]] = {}
    # Index by label-arrival hour (for accuracy computation)
    label_buckets: dict[float, list[dict]] = {}

    for row in request_rows:
        hour_req = start_ts + int((row["timestamp"] - start_ts) / 3600) * 3600
        req_buckets.setdefault(hour_req, []).append(row)
        hour_lbl = start_ts + int((row["label_ts"] - start_ts) / 3600) * 3600
        label_buckets.setdefault(hour_lbl, []).append(row)

    all_buckets = sorted(set(req_buckets) | set(label_buckets))

    metric_rows = []
    base = {"service": "inference_service", "tags": "{}"}

    for bucket_ts in all_buckets:
        rows = req_buckets.get(bucket_ts, [])
        if rows:
            n = len(rows)
            latencies = [r["_latency_ms"] for r in rows]
            errors = sum(1 for r in rows if r["_is_error"])
            valid_rows = [r for r in rows if not r["_is_error"]]
            probas = [r["pred_proba"] for r in valid_rows] if valid_rows else [0.0]
            preds = [r["prediction"] for r in valid_rows] if valid_rows else []
            lat_arr = np.array(latencies)

            metric_rows += [
                {**base, "timestamp": bucket_ts, "metric_name": "throughput",            "metric_value": float(n)},
                {**base, "timestamp": bucket_ts, "metric_name": "latency_p50",           "metric_value": float(np.percentile(lat_arr, 50))},
                {**base, "timestamp": bucket_ts, "metric_name": "latency_p99",           "metric_value": float(np.percentile(lat_arr, 99))},
                {**base, "timestamp": bucket_ts, "metric_name": "error_rate",            "metric_value": float(errors / n)},
                {**base, "timestamp": bucket_ts, "metric_name": "prediction_confidence", "metric_value": float(np.mean(probas))},
                {**base, "timestamp": bucket_ts, "metric_name": "positive_rate",         "metric_value": float(np.mean(preds)) if preds else 0.0},
            ]

        # Accuracy: requests whose label arrived in this bucket (i.e., from 24h ago)
        labeled = label_buckets.get(bucket_ts, [])
        valid_labeled = [r for r in labeled if not r["_is_error"]]
        if valid_labeled:
            true_labels = np.array([r["true_label"] for r in valid_labeled])
            pred_labels = np.array([r["prediction"] for r in valid_labeled])
            accuracy = float((pred_labels == true_labels).mean())
            metric_rows.append({**base, "timestamp": bucket_ts, "metric_name": "accuracy", "metric_value": accuracy})

    return metric_rows


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------

def _insert_requests(db_path: str, rows: list[dict], batch_size: int = 5000):
    if not rows:
        return
    # Strip internal fields before insert
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT OR IGNORE INTO requests "
        "(request_id, timestamp, features, prediction, pred_proba, true_label, label_ts, service) "
        "VALUES (:request_id, :timestamp, :features, :prediction, :pred_proba, :true_label, :label_ts, :service)",
        clean,
    )
    con.commit()
    con.close()


def _insert_metrics(db_path: str, rows: list[dict]):
    if not rows:
        return
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT INTO metrics (timestamp, metric_name, metric_value, service, tags) "
        "VALUES (:timestamp, :metric_name, :metric_value, :service, :tags)",
        rows,
    )
    con.commit()
    con.close()


def _insert_logs(db_path: str, rows: list[dict]):
    if not rows:
        return
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT OR IGNORE INTO logs (log_id, timestamp, severity, service, message, context) "
        "VALUES (:log_id, :timestamp, :severity, :service, :message, :context)",
        rows,
    )
    con.commit()
    con.close()


def _insert_deployments(db_path: str, events: list[DeploymentEvent]):
    if not events:
        return
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT OR IGNORE INTO deployments "
        "(deploy_id, timestamp, service, version_before, version_after, commit_sha, "
        " change_type, changelog, deployed_by, is_rollback, config) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                str(uuid.uuid4()), e.timestamp, e.service,
                e.version_before, e.version_after, e.commit_sha,
                e.change_type, e.changelog, e.deployed_by,
                int(e.is_rollback), json.dumps(e.config),
            )
            for e in events
        ],
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def load_model_artifacts(model_dir: str):
    model  = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "feature_info.json")) as f:
        info = json.load(f)
    return model, scaler, info


def generate_data(
    config: GenerationConfig,
    model_dir: str,
    verbose: bool = True,
):
    def log(msg: str):
        if verbose:
            print(msg)

    log("Loading model artifacts...")
    model, scaler, feature_info = load_model_artifacts(model_dir)
    feature_names = feature_info["feature_names"]
    gt_weights = np.array(feature_info["ground_truth_weights"])
    gt_bias = float(feature_info["ground_truth_bias"])

    log(f"Setting up databases in {config.output_dir!r}...")
    db_paths = setup_databases(config.output_dir)

    log("Resolving time sub-ranges...")
    sub_ranges = resolve_sub_ranges(config)
    log(f"  {len(sub_ranges)} sub-range(s) covering "
        f"{(config.end_ts - config.start_ts) / 86400:.1f} simulated days")

    rng = np.random.default_rng(config.seed)

    all_request_rows: list[dict] = []
    all_log_rows: list[dict] = []

    for idx, sr in enumerate(sub_ranges):
        hours = (sr.end_ts - sr.start_ts) / 3600
        n_est = int(hours * sr.feature_params.requests_per_hour)
        log(f"  Sub-range {idx+1}/{len(sub_ranges)}: "
            f"{hours:.1f}h @ {sr.feature_params.requests_per_hour:.0f} req/h "
            f"≈ {n_est} requests")

        req_rows, log_rows = generate_sub_range(
            model, scaler, feature_names, gt_weights, gt_bias,
            sr, config.label_delay_hours, rng,
        )
        all_request_rows.extend(req_rows)
        all_log_rows.extend(log_rows)

    log(f"Generated {len(all_request_rows)} requests, {len(all_log_rows)} log entries")

    log("Writing feature store...")
    _insert_requests(db_paths["feature_store"], all_request_rows)

    log("Aggregating hourly metrics...")
    metric_rows = aggregate_metrics(all_request_rows, config.label_delay_hours, config.start_ts, config.end_ts)
    log(f"  {len(metric_rows)} metric rows")
    _insert_metrics(db_paths["metrics"], metric_rows)

    log("Writing logs...")
    _insert_logs(db_paths["logs"], all_log_rows)

    log("Writing deployment events...")
    _insert_deployments(db_paths["deployments"], config.deployment_events)

    log("Done.")
    return db_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ML system data")
    parser.add_argument("--days",       type=int,   default=7,               help="Number of simulated days")
    parser.add_argument("--output-dir", type=str,   default="data",          help="Directory for SQLite databases")
    parser.add_argument("--model-dir",  type=str,   default="model/artifacts", help="Directory containing model artifacts")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--req-per-hour", type=float, default=500.0,         help="Requests per hour (normal operation)")
    args = parser.parse_args()

    config = make_default_config(n_days=args.days)
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.default_feature_params.requests_per_hour = args.req_per_hour

    generate_data(config, model_dir=args.model_dir)
