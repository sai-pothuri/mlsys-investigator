"""
Unit tests for generator/schema.py — all four query functions.
Uses only temporary SQLite databases with hand-crafted rows (no generator).
"""

from __future__ import annotations
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from generator.schema import (
    query_deployments,
    query_feature_values,
    query_logs,
    query_metrics,
    setup_databases,
)
from generator.defaults import SIM_START


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_request(db_path, request_id, ts, features, prediction=0, pred_proba=0.3,
                    true_label=0, label_ts=None):
    if label_ts is None:
        label_ts = ts + 86400
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, ts, json.dumps(features), prediction, pred_proba, true_label, label_ts, "inference_service"),
    )
    con.commit()
    con.close()


def _insert_metric(db_path, ts, name, value, service="inference_service"):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO metrics (timestamp, metric_name, metric_value, service, tags) VALUES (?, ?, ?, ?, ?)",
        (ts, name, value, service, "{}"),
    )
    con.commit()
    con.close()


def _insert_log(db_path, ts, severity, service, message, context=None):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO logs (log_id, timestamp, severity, service, message, context) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), ts, severity, service, message, json.dumps(context or {})),
    )
    con.commit()
    con.close()


def _insert_deployment(db_path, ts, service, v_before, v_after, sha, change_type,
                       changelog, deployed_by, is_rollback=False, config=None):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO deployments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), ts, service, v_before, v_after, sha,
         change_type, changelog, deployed_by, int(is_rollback), json.dumps(config or {})),
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# setup_databases
# ---------------------------------------------------------------------------

class TestSetupDatabases:
    def test_creates_all_four_files(self, tmp_path):
        paths = setup_databases(str(tmp_path))
        for name in ("feature_store", "metrics", "logs", "deployments"):
            assert name in paths
            assert Path(paths[name]).exists()

    def test_tables_exist(self, tmp_path):
        paths = setup_databases(str(tmp_path))
        expected = {
            "feature_store": "requests",
            "metrics": "metrics",
            "logs": "logs",
            "deployments": "deployments",
        }
        for db_name, table in expected.items():
            con = sqlite3.connect(paths[db_name])
            result = con.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            con.close()
            assert result is not None, f"Table {table!r} missing in {db_name}.db"

    def test_idempotent(self, tmp_path):
        setup_databases(str(tmp_path))
        setup_databases(str(tmp_path))  # must not raise


# ---------------------------------------------------------------------------
# query_feature_values
# ---------------------------------------------------------------------------

class TestQueryFeatureValues:
    def test_happy_path(self, tmp_db_paths):
        db = tmp_db_paths["feature_store"]
        ts = SIM_START + 3600
        _insert_request(db, "r1", ts, {"login_failure_rate": 0.1, "account_age_days": 700.0})
        _insert_request(db, "r2", ts + 100, {"login_failure_rate": 0.2, "account_age_days": 730.0})

        result = query_feature_values(db, ["login_failure_rate", "account_age_days"],
                                      SIM_START, SIM_START + 7200)
        assert set(result.keys()) == {"login_failure_rate", "account_age_days"}
        assert sorted(result["login_failure_rate"]) == pytest.approx([0.1, 0.2])
        assert sorted(result["account_age_days"]) == pytest.approx([700.0, 730.0])

    def test_empty_window(self, tmp_db_paths):
        db = tmp_db_paths["feature_store"]
        _insert_request(db, "r1", SIM_START + 3600, {"login_failure_rate": 0.1})

        result = query_feature_values(db, ["login_failure_rate"],
                                      SIM_START + 10000, SIM_START + 20000)
        assert result["login_failure_rate"] == []

    def test_feature_absent_from_json(self, tmp_db_paths):
        db = tmp_db_paths["feature_store"]
        _insert_request(db, "r1", SIM_START + 100, {"account_age_days": 700.0})

        result = query_feature_values(db, ["missing_feature"], SIM_START, SIM_START + 3600)
        assert result["missing_feature"] == []

    def test_strict_upper_bound(self, tmp_db_paths):
        db = tmp_db_paths["feature_store"]
        # Row exactly at end_ts — should NOT be included
        _insert_request(db, "r_at_boundary", SIM_START + 3600, {"login_failure_rate": 0.9})

        result = query_feature_values(db, ["login_failure_rate"],
                                      SIM_START, SIM_START + 3600)
        assert 0.9 not in result["login_failure_rate"]

    def test_start_ts_included(self, tmp_db_paths):
        db = tmp_db_paths["feature_store"]
        # Row exactly at start_ts — should be included
        _insert_request(db, "r_at_start", SIM_START, {"login_failure_rate": 0.5})

        result = query_feature_values(db, ["login_failure_rate"],
                                      SIM_START, SIM_START + 1)
        assert 0.5 in result["login_failure_rate"]


# ---------------------------------------------------------------------------
# query_metrics
# ---------------------------------------------------------------------------

class TestQueryMetrics:
    def test_happy_path(self, tmp_db_paths):
        db = tmp_db_paths["metrics"]
        _insert_metric(db, SIM_START + 3600, "accuracy", 0.87)
        _insert_metric(db, SIM_START + 7200, "accuracy", 0.85)

        rows = query_metrics(db, ["accuracy"], SIM_START, SIM_START + 10000)
        assert len(rows) == 2
        assert all(r["metric_name"] == "accuracy" for r in rows)
        assert all(k in rows[0] for k in ("timestamp", "metric_name", "metric_value", "service", "tags"))

    def test_metric_name_filter(self, tmp_db_paths):
        db = tmp_db_paths["metrics"]
        _insert_metric(db, SIM_START + 100, "accuracy", 0.87)
        _insert_metric(db, SIM_START + 100, "error_rate", 0.005)

        rows = query_metrics(db, ["accuracy"], SIM_START, SIM_START + 1000)
        names = {r["metric_name"] for r in rows}
        assert names == {"accuracy"}

    def test_service_filter(self, tmp_db_paths):
        db = tmp_db_paths["metrics"]
        con = sqlite3.connect(db)
        con.execute("INSERT INTO metrics (timestamp, metric_name, metric_value, service, tags) VALUES (?, ?, ?, ?, ?)",
                    (SIM_START + 100, "throughput", 500.0, "inference_service", "{}"))
        con.execute("INSERT INTO metrics (timestamp, metric_name, metric_value, service, tags) VALUES (?, ?, ?, ?, ?)",
                    (SIM_START + 100, "throughput", 10.0, "feature_pipeline", "{}"))
        con.commit()
        con.close()

        rows = query_metrics(db, ["throughput"], SIM_START, SIM_START + 1000,
                             service="inference_service")
        assert all(r["service"] == "inference_service" for r in rows)
        assert len(rows) == 1

    def test_ordered_by_timestamp(self, tmp_db_paths):
        db = tmp_db_paths["metrics"]
        for offset in [7200, 3600, 100]:
            _insert_metric(db, SIM_START + offset, "throughput", 500.0)

        rows = query_metrics(db, ["throughput"], SIM_START, SIM_START + 10000)
        timestamps = [r["timestamp"] for r in rows]
        assert timestamps == sorted(timestamps)

    def test_empty_window(self, tmp_db_paths):
        db = tmp_db_paths["metrics"]
        _insert_metric(db, SIM_START + 100, "accuracy", 0.87)

        rows = query_metrics(db, ["accuracy"], SIM_START + 50000, SIM_START + 60000)
        assert rows == []


# ---------------------------------------------------------------------------
# query_logs
# ---------------------------------------------------------------------------

class TestQueryLogs:
    def test_happy_path(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        _insert_log(db, SIM_START + 100, "INFO", "inference_service", "Request OK")
        _insert_log(db, SIM_START + 200, "ERROR", "inference_service", "Timeout hit")

        entries, truncated = query_logs(db, SIM_START, SIM_START + 1000)
        assert len(entries) == 2
        assert truncated is False

    def test_truncation_flag_true(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        for i in range(201):
            _insert_log(db, SIM_START + i, "INFO", "inference_service", f"msg {i}")

        entries, truncated = query_logs(db, SIM_START, SIM_START + 10000, limit=200)
        assert len(entries) == 200
        assert truncated is True

    def test_truncation_flag_false(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        for i in range(50):
            _insert_log(db, SIM_START + i, "INFO", "inference_service", f"msg {i}")

        _, truncated = query_logs(db, SIM_START, SIM_START + 10000, limit=200)
        assert truncated is False

    def test_severity_filter(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        _insert_log(db, SIM_START + 100, "INFO", "inference_service", "info msg")
        _insert_log(db, SIM_START + 200, "ERROR", "inference_service", "error msg")
        _insert_log(db, SIM_START + 300, "WARNING", "inference_service", "warn msg")

        entries, _ = query_logs(db, SIM_START, SIM_START + 1000, severities=["ERROR"])
        assert all(e["severity"] == "ERROR" for e in entries)
        assert len(entries) == 1

    def test_service_filter(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        _insert_log(db, SIM_START + 100, "INFO", "inference_service", "svc A")
        _insert_log(db, SIM_START + 200, "INFO", "feature_pipeline", "svc B")

        entries, _ = query_logs(db, SIM_START, SIM_START + 1000, service="inference_service")
        assert all(e["service"] == "inference_service" for e in entries)

    def test_text_filter(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        _insert_log(db, SIM_START + 100, "ERROR", "inference_service", "Connection timeout exceeded")
        _insert_log(db, SIM_START + 200, "ERROR", "inference_service", "NaN in input features")

        entries, _ = query_logs(db, SIM_START, SIM_START + 1000, filter_text="timeout")
        assert len(entries) == 1
        assert "timeout" in entries[0]["message"].lower()

    def test_context_is_dict(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        _insert_log(db, SIM_START + 100, "INFO", "inference_service", "msg",
                    context={"request_id": "abc123", "latency_ms": 45.0})

        entries, _ = query_logs(db, SIM_START, SIM_START + 1000)
        assert isinstance(entries[0]["context"], dict)
        assert entries[0]["context"]["request_id"] == "abc123"

    def test_empty_result(self, tmp_db_paths):
        db = tmp_db_paths["logs"]
        entries, truncated = query_logs(db, SIM_START, SIM_START + 1000)
        assert entries == []
        assert truncated is False


# ---------------------------------------------------------------------------
# query_deployments
# ---------------------------------------------------------------------------

class TestQueryDeployments:
    def test_happy_path(self, tmp_db_paths):
        db = tmp_db_paths["deployments"]
        _insert_deployment(db, SIM_START + 3600, "inference_service",
                           "v1.0", "v1.1", "abc123", "config_change",
                           "Bumped timeout", "mlops-bot")

        events = query_deployments(db, SIM_START, SIM_START + 10000)
        assert len(events) == 1
        expected_keys = {"deploy_id", "timestamp", "service", "version_before",
                         "version_after", "commit_sha", "change_type", "changelog",
                         "deployed_by", "is_rollback", "config"}
        assert set(events[0].keys()) == expected_keys

    def test_is_rollback_is_bool(self, tmp_db_paths):
        db = tmp_db_paths["deployments"]
        _insert_deployment(db, SIM_START + 100, "inference_service",
                           "v1.1", "v1.0", "abc123", "config_change",
                           "Rolled back", "alice", is_rollback=True)

        events = query_deployments(db, SIM_START, SIM_START + 1000)
        assert isinstance(events[0]["is_rollback"], bool)
        assert events[0]["is_rollback"] is True

    def test_config_is_dict(self, tmp_db_paths):
        db = tmp_db_paths["deployments"]
        _insert_deployment(db, SIM_START + 100, "inference_service",
                           "v1.0", "v1.1", "abc", "config_change",
                           "Changed timeout", "bot", config={"timeout_ms": 250})

        events = query_deployments(db, SIM_START, SIM_START + 1000)
        assert isinstance(events[0]["config"], dict)
        assert events[0]["config"]["timeout_ms"] == 250

    def test_service_filter(self, tmp_db_paths):
        db = tmp_db_paths["deployments"]
        _insert_deployment(db, SIM_START + 100, "inference_service",
                           "v1", "v2", "sha1", "model_retrain", "retrain", "alice")
        _insert_deployment(db, SIM_START + 200, "feature_pipeline",
                           "v1", "v2", "sha2", "feature_pipeline_change", "add feat", "bob")

        events = query_deployments(db, SIM_START, SIM_START + 1000,
                                   service="feature_pipeline")
        assert len(events) == 1
        assert events[0]["service"] == "feature_pipeline"

    def test_ordered_by_timestamp(self, tmp_db_paths):
        db = tmp_db_paths["deployments"]
        for offset in [7200, 100, 3600]:
            _insert_deployment(db, SIM_START + offset, "inference_service",
                               "v1", "v2", "sha", "config_change", "msg", "bot")

        events = query_deployments(db, SIM_START, SIM_START + 10000)
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)
