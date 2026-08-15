"""
SQLite schema setup and documented query interfaces for all four data stores.

Each store lives in a separate .db file under the configured output_dir.
The query functions here are the thin interface layer that the agent's
tool wrappers will sit on top of — they return plain dicts / lists of dicts.
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

FEATURE_STORE_DDL = """
CREATE TABLE IF NOT EXISTS requests (
    request_id   TEXT    PRIMARY KEY,
    timestamp    REAL    NOT NULL,   -- Unix timestamp (simulated clock)
    features     TEXT    NOT NULL,   -- JSON object: {feature_name: value, ...}
    prediction   INTEGER NOT NULL,   -- 0 or 1
    pred_proba   REAL    NOT NULL,   -- P(class=1)
    true_label   INTEGER,            -- NULL until label_delay has passed
    label_ts     REAL,               -- timestamp when label became available
    service      TEXT    NOT NULL DEFAULT 'inference_service'
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp);
"""

METRICS_DDL = """
CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,   -- bucket start (1-hour buckets)
    metric_name  TEXT    NOT NULL,
    metric_value REAL    NOT NULL,
    service      TEXT    NOT NULL DEFAULT 'inference_service',
    tags         TEXT    NOT NULL DEFAULT '{}'  -- JSON
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts   ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
"""

LOGS_DDL = """
CREATE TABLE IF NOT EXISTS logs (
    log_id      TEXT  PRIMARY KEY,
    timestamp   REAL  NOT NULL,
    severity    TEXT  NOT NULL,  -- INFO | WARNING | ERROR | CRITICAL
    service     TEXT  NOT NULL,
    message     TEXT  NOT NULL,
    context     TEXT  NOT NULL DEFAULT '{}'  -- JSON: request_id, feature_name, etc.
);
CREATE INDEX IF NOT EXISTS idx_logs_ts       ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_severity ON logs(severity);
CREATE INDEX IF NOT EXISTS idx_logs_service  ON logs(service);
"""

DEPLOYMENTS_DDL = """
CREATE TABLE IF NOT EXISTS deployments (
    deploy_id      TEXT  PRIMARY KEY,
    timestamp      REAL  NOT NULL,
    service        TEXT  NOT NULL,
    version_before TEXT  NOT NULL,
    version_after  TEXT  NOT NULL,
    commit_sha     TEXT  NOT NULL,
    change_type    TEXT  NOT NULL,
    changelog      TEXT  NOT NULL,
    deployed_by    TEXT  NOT NULL,
    is_rollback    INTEGER NOT NULL DEFAULT 0,
    config         TEXT  NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_deployments_ts ON deployments(timestamp);
"""


_TABLE_NAMES = {
    "feature_store": ["requests"],
    "metrics":       ["metrics"],
    "logs":          ["logs"],
    "deployments":   ["deployments"],
}


def setup_databases(output_dir: str) -> dict[str, str]:
    """Create (or recreate) all SQLite databases. Returns mapping of name → path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_store": str(base / "feature_store.db"),
        "metrics":       str(base / "metrics.db"),
        "logs":          str(base / "logs.db"),
        "deployments":   str(base / "deployments.db"),
    }
    ddls = {
        "feature_store": FEATURE_STORE_DDL,
        "metrics":       METRICS_DDL,
        "logs":          LOGS_DDL,
        "deployments":   DEPLOYMENTS_DDL,
    }
    for name, path in paths.items():
        con = sqlite3.connect(path)
        con.executescript(ddls[name])
        for table in _TABLE_NAMES[name]:
            con.execute(f"DELETE FROM {table}")
        con.commit()
        con.close()
    return paths


def _con(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# Feature store queries
#
# query_feature_distributions(features, baseline_start, baseline_end,
#                             comparison_start, comparison_end) -> dict
#
# Returns per-feature raw value arrays for both windows so the tool layer
# can compute PSI (or any other drift metric) without re-querying the store.
#
# Schema of returned dict:
# {
#   "feature_name": {
#     "baseline":    [float, ...],   # raw values in baseline window
#     "comparison":  [float, ...],   # raw values in comparison window
#   },
#   ...
# }
# ---------------------------------------------------------------------------

def query_feature_values(
    db_path: str,
    features: list[str],
    start_ts: float,
    end_ts: float,
) -> dict[str, list]:
    """
    Return raw feature values for a time window.
    Used by the tool layer to compute drift metrics (PSI etc.).
    """
    con = _con(db_path)
    rows = con.execute(
        "SELECT features FROM requests WHERE timestamp >= ? AND timestamp < ?",
        (start_ts, end_ts),
    ).fetchall()
    con.close()

    result: dict[str, list] = {f: [] for f in features}
    for row in rows:
        fdict = json.loads(row["features"])
        for f in features:
            if f in fdict:
                result[f].append(fdict[f])
    return result


# ---------------------------------------------------------------------------
# Metrics queries
#
# query_metrics(metric_names, start_ts, end_ts, aggregation) -> list[dict]
#
# Returns:
# [
#   {"metric_name": str, "timestamp": float, "value": float, "service": str},
#   ...
# ]
# (hourly rows; caller can aggregate further)
# ---------------------------------------------------------------------------

def query_metrics(
    db_path: str,
    metric_names: list[str],
    start_ts: float,
    end_ts: float,
    service: Optional[str] = None,
) -> list[dict]:
    """Return hourly metric rows within the time window."""
    con = _con(db_path)
    placeholders = ",".join("?" * len(metric_names))
    params: list = [start_ts, end_ts, *metric_names]
    sql = (
        "SELECT timestamp, metric_name, metric_value, service, tags "
        f"FROM metrics WHERE timestamp >= ? AND timestamp < ? "
        f"AND metric_name IN ({placeholders})"
    )
    if service:
        sql += " AND service = ?"
        params.append(service)
    sql += " ORDER BY timestamp"
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Log queries
#
# query_logs(service, start_ts, end_ts, severities, filter_text, limit) -> list[dict]
#
# Returns:
# [
#   {"log_id": str, "timestamp": float, "severity": str,
#    "service": str, "message": str, "context": dict},
#   ...
# ]
# ---------------------------------------------------------------------------

def query_logs(
    db_path: str,
    start_ts: float,
    end_ts: float,
    service: Optional[str] = None,
    severities: Optional[list[str]] = None,
    filter_text: Optional[str] = None,
    limit: int = 200,
) -> tuple[list[dict], bool]:
    """
    Return log entries. Returns (entries, truncated) — truncated=True means
    there were more rows than `limit`.
    """
    con = _con(db_path)
    params: list = [start_ts, end_ts]
    sql = "SELECT * FROM logs WHERE timestamp >= ? AND timestamp < ?"
    if service:
        sql += " AND service = ?"
        params.append(service)
    if severities:
        placeholders = ",".join("?" * len(severities))
        sql += f" AND severity IN ({placeholders})"
        params.extend(severities)
    if filter_text:
        sql += " AND message LIKE ?"
        params.append(f"%{filter_text}%")
    sql += " ORDER BY timestamp LIMIT ?"
    params.append(limit + 1)

    rows = con.execute(sql, params).fetchall()
    con.close()

    truncated = len(rows) > limit
    entries = []
    for r in rows[:limit]:
        d = dict(r)
        d["context"] = json.loads(d["context"])
        entries.append(d)
    return entries, truncated


# ---------------------------------------------------------------------------
# Deployment history queries
#
# query_deployments(service, start_ts, end_ts) -> list[dict]
#
# Returns:
# [
#   {"deploy_id": str, "timestamp": float, "service": str,
#    "version_before": str, "version_after": str, "commit_sha": str,
#    "change_type": str, "changelog": str, "deployed_by": str,
#    "is_rollback": bool, "config": dict},
#   ...
# ]
# ---------------------------------------------------------------------------

def query_deployments(
    db_path: str,
    start_ts: float,
    end_ts: float,
    service: Optional[str] = None,
) -> list[dict]:
    """Return deployment events in the time window."""
    con = _con(db_path)
    params: list = [start_ts, end_ts]
    sql = "SELECT * FROM deployments WHERE timestamp >= ? AND timestamp < ?"
    if service:
        sql += " AND service = ?"
        params.append(service)
    sql += " ORDER BY timestamp"
    rows = con.execute(sql, params).fetchall()
    con.close()
    result = []
    for r in rows:
        d = dict(r)
        d["is_rollback"] = bool(d["is_rollback"])
        d["config"] = json.loads(d["config"])
        result.append(d)
    return result
