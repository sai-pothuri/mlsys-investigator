"""FastAPI HTTP server wrapping the hand-rolled ReAct investigation loop.

Investigations run 30–120 s (multiple Anthropic API round trips), so the
API is async:

  POST /investigate            → 202  { job_id, status: "pending" }
  GET  /jobs/{job_id}          → job record (status / result / error)
  POST /webhook/alertmanager   → Prometheus Alertmanager receiver (202)
  GET  /health                 → 200 liveness / readiness probe

Job state is in-process. With >1 replica, sticky routing or Redis is
required for /jobs lookups to work reliably — run replicas=1 until then.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import os

from agent import run_investigation
from hypothesis_graph import HypothesisStatus

app = FastAPI(title="MLSys Investigator", version="1.0.0")

# Allow the UI page to call the API even when opened from a different origin
# (e.g. the URL field is pointed at a different host than the page was served from).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKER_THREADS", "4")))
_STATIC_DIR = Path(__file__).parent / "static"


# ── Schema ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class InvestigateRequest(BaseModel):
    alert: str = Field(..., description="Alert text to investigate")
    budget: int = Field(8, ge=1, le=20, description="Max tool calls")
    investigation_start: Optional[datetime] = Field(
        None, description="Anchor timestamp for tool queries (defaults to now)"
    )
    ground_truth: Optional[str] = Field(
        None, description="Known root cause for eval scoring (optional)"
    )


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    alert: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# Alertmanager webhook payload types (v4 schema)
class AlertmanagerAlert(BaseModel):
    status: str
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}
    startsAt: str = ""
    endsAt: str = ""
    generatorURL: str = ""


class AlertmanagerPayload(BaseModel):
    version: str = "4"
    status: str = "firing"
    receiver: str = ""
    groupLabels: dict[str, str] = {}
    commonLabels: dict[str, str] = {}
    commonAnnotations: dict[str, str] = {}
    alerts: list[AlertmanagerAlert] = []


# ── In-process job store ──────────────────────────────────────────────────────

_jobs: dict[str, JobRecord] = {}


def _run_sync(alert: str, budget: int, investigation_start: Optional[datetime], ground_truth: Optional[str]) -> dict:
    graph, diagnosis = run_investigation(
        alert=alert,
        budget=budget,
        investigation_start=investigation_start,
        verbose=False,
        ground_truth=ground_truth,
    )
    result: dict[str, Any] = {
        "termination_reason": graph.termination_reason,
        "tool_calls_used": graph.tool_calls_used,
        "hypothesis_count": len(graph.hypotheses),
        "established_facts": graph.established_facts,
    }
    if diagnosis:
        result["diagnosis"] = {
            "root_cause": diagnosis.root_cause,
            "diagnosis": diagnosis.diagnosis,
            "confidence": diagnosis.confidence,
            "recommended_action": diagnosis.recommended_action,
            "alternative_categories": diagnosis.alternative_categories,
        }
    else:
        active = [h for h in graph.hypotheses if h.status == HypothesisStatus.ACTIVE]
        if active:
            top = max(active, key=lambda h: h.likelihood)
            result["top_suspect"] = {
                "category": top.root_cause_category.value if top.root_cause_category else None,
                "likelihood": top.likelihood,
                "description": top.description,
            }
    return result


async def _launch_job(job_id: str, alert: str, budget: int, investigation_start: Optional[datetime], ground_truth: Optional[str]) -> None:
    loop = asyncio.get_event_loop()
    try:
        _jobs[job_id].status = JobStatus.running
        result = await loop.run_in_executor(
            _executor, _run_sync, alert, budget, investigation_start, ground_truth
        )
        _jobs[job_id].status = JobStatus.completed
        _jobs[job_id].result = result
        _jobs[job_id].completed_at = datetime.now(timezone.utc)
    except Exception as exc:
        _jobs[job_id].status = JobStatus.failed
        _jobs[job_id].error = str(exc)
        _jobs[job_id].completed_at = datetime.now(timezone.utc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "jobs": len(_jobs)}


@app.post("/investigate", status_code=202)
async def start_investigation(req: InvestigateRequest) -> dict:
    """Enqueue an investigation. Poll GET /jobs/{job_id} for the result."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobRecord(
        job_id=job_id,
        status=JobStatus.pending,
        alert=req.alert,
        created_at=datetime.now(timezone.utc),
    )
    asyncio.create_task(
        _launch_job(job_id, req.alert, req.budget, req.investigation_start, req.ground_truth)
    )
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}", response_model=JobRecord)
async def get_job(job_id: str) -> JobRecord:
    """Return the current state of an investigation job."""
    record = _jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return record


@app.get("/jobs", response_model=list[JobRecord])
async def list_jobs(limit: int = 50) -> list[JobRecord]:
    """List recent jobs (most recent first, up to limit)."""
    all_jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return all_jobs[:limit]


@app.post("/webhook/alertmanager", status_code=202)
async def alertmanager_webhook(payload: AlertmanagerPayload) -> dict:
    """Prometheus Alertmanager webhook receiver.

    Converts each firing alert into an investigation. Resolved alerts are
    acknowledged but not investigated.
    """
    fired = [a for a in payload.alerts if a.status == "firing"]
    if not fired:
        return {"message": "no firing alerts — nothing to investigate", "job_ids": []}

    job_ids = []
    for alert in fired:
        alertname = alert.labels.get("alertname", "unknown")
        summary = alert.annotations.get("summary", "")
        description = alert.annotations.get("description", "")
        severity = alert.labels.get("severity", "")

        parts = [f"ALERT: {alertname}"]
        if severity:
            parts.append(f"[{severity.upper()}]")
        if summary:
            parts.append(summary)
        if description:
            parts.append(description)
        alert_text = " — ".join(parts)

        job_id = str(uuid.uuid4())
        _jobs[job_id] = JobRecord(
            job_id=job_id,
            status=JobStatus.pending,
            alert=alert_text,
            created_at=datetime.now(timezone.utc),
        )
        asyncio.create_task(
            _launch_job(job_id, alert_text, 8, None, None)
        )
        job_ids.append(job_id)

    return {"message": f"investigating {len(job_ids)} alert(s)", "job_ids": job_ids}
