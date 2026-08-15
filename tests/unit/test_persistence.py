"""Unit tests for persistence.py — snapshot_graph."""

import json
import pathlib
import tempfile
from datetime import datetime, timezone

from hypothesis_graph import (
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisGraph,
    HypothesisStatus,
    Severity,
)
from persistence import snapshot_graph


def _graph() -> HypothesisGraph:
    return HypothesisGraph(
        alert_summary="accuracy dropped 15pp",
        investigation_start=datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc),
        tool_call_budget=8,
        tool_calls_used=2,
        hypotheses=[
            Hypothesis(
                id="H1",
                description="feature drift hypothesis",
                likelihood=0.75,
                severity=Severity.HIGH,
                status=HypothesisStatus.ACTIVE,
                evidence=[
                    Evidence(
                        tool_called="query_metrics",
                        evidence_type=EvidenceType.METRICS,
                        observation="accuracy -15pp",
                        supports=True,
                        confidence_delta=0.20,
                    )
                ],
            )
        ],
        established_facts=["error_rate is flat"],
    )


class TestSnapshotGraph:
    def test_writes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_writes_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            assert isinstance(data, dict)
        finally:
            path.unlink(missing_ok=True)

    def test_json_contains_alert_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            assert data["alert_summary"] == "accuracy dropped 15pp"
        finally:
            path.unlink(missing_ok=True)

    def test_json_contains_hypotheses(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            assert len(data["hypotheses"]) == 1
            assert data["hypotheses"][0]["id"] == "H1"
        finally:
            path.unlink(missing_ok=True)

    def test_json_contains_evidence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            h = data["hypotheses"][0]
            assert len(h["evidence"]) == 1
            assert h["evidence"][0]["tool_called"] == "query_metrics"
        finally:
            path.unlink(missing_ok=True)

    def test_overwrites_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            path.write_text("old content")
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            assert "alert_summary" in data
        finally:
            path.unlink(missing_ok=True)

    def test_round_trips_tool_calls_used(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            g = _graph()
            g.tool_calls_used = 5
            snapshot_graph(g, path=path)
            data = json.loads(path.read_text())
            assert data["tool_calls_used"] == 5
        finally:
            path.unlink(missing_ok=True)

    def test_round_trips_established_facts(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = pathlib.Path(f.name)
        try:
            snapshot_graph(_graph(), path=path)
            data = json.loads(path.read_text())
            assert "error_rate is flat" in data["established_facts"]
        finally:
            path.unlink(missing_ok=True)
