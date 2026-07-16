"""FastAPI application — REST interface for the AlphaAgents pipeline.

Endpoints:
  POST /research          — submit a query, receive a job_id
  GET  /research/{job_id} — poll job status and state snapshot
  GET  /research/{job_id}/note — retrieve approved final note
  POST /research/{job_id}/hitl — record human decision
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import aiosqlite
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from agents.memory import store_note_in_memory
from config.logging import configure_logging
from graph.pipeline import pipeline
from graph.state import ResearchNote, ResearchState
from llm.tracing import create_trace_id

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./alpha_agents.db")
_DB_PATH: str = DATABASE_URL.replace("sqlite:///", "")


async def _init_db() -> None:
    """Create the jobs table if it does not already exist."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                state_json  TEXT NOT NULL
            )
            """
        )
        await db.commit()


async def _upsert_job(job_id: str, status: str, state: ResearchState) -> None:
    """Insert or replace a job record in SQLite.

    Args:
        job_id: Unique job identifier.
        status: Current status string (e.g. 'running', 'awaiting_hitl', 'done').
        state: Full ResearchState to serialise as JSON.
    """
    now = datetime.now(UTC).isoformat()

    def _default(obj: object) -> object:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    state_json = json.dumps(dict(state), default=_default)

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, updated_at, state_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status     = excluded.status,
                updated_at = excluded.updated_at,
                state_json = excluded.state_json
            """,
            (job_id, status, now, now, state_json),
        )
        await db.commit()


async def _get_job(job_id: str) -> dict | None:
    """Fetch a single job record by job_id.

    Args:
        job_id: Unique job identifier.

    Returns:
        A dict with keys job_id, status, created_at, updated_at, state_json,
        or None if not found.
    """
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT job_id, status, created_at, updated_at, state_json FROM jobs WHERE job_id = ?",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def _list_approved_jobs() -> list[dict]:
    """Return all jobs with status='done' for the history tab.

    Returns:
        List of dicts with job_id, status, created_at, updated_at.
    """
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT job_id, status, created_at, updated_at FROM jobs WHERE status = 'done' ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def _run_pipeline(job_id: str, query: str) -> None:
    """Run the LangGraph pipeline for a job in a background task.

    Creates a LangFuse trace, invokes the compiled pipeline, and persists
    the resulting state to SQLite. On any unhandled exception, records the
    error in SQLite so GET /research/{job_id} surfaces it.

    Args:
        job_id: Unique job identifier.
        query: User's research query.
    """
    trace_id = create_trace_id()
    initial_state: ResearchState = {
        "query": query,
        "job_id": job_id,
        "revision_count": 0,
        "hitl_decision": None,
        "final_note": None,
        "error": None,
        "langfuse_trace_id": trace_id,
    }

    try:
        final_state: ResearchState = await pipeline.ainvoke(initial_state)
        status = "awaiting_hitl" if final_state.get("error") is None else "error"
        await _upsert_job(job_id, status, final_state)
        logger.info("Pipeline complete for job %s. Status: %s", job_id, status)
    except Exception as exc:
        error_msg = f"Pipeline crashed: {exc}"
        logger.error("job %s — %s", job_id, error_msg)
        error_state: ResearchState = {**initial_state, "error": error_msg}
        await _upsert_job(job_id, "error", error_state)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the database and logging on startup."""
    configure_logging()
    await _init_db()
    logger.info("Database initialised.")
    yield


app = FastAPI(
    title="AlphaAgents API",
    description="Agentic equity research analyst — multi-agent LangGraph pipeline.",
    version="0.1.0",
    lifespan=_lifespan,
)


class ResearchRequest(BaseModel):
    """Request body for POST /research."""

    query: str


class ResearchSubmitResponse(BaseModel):
    """Response for POST /research."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response for GET /research/{job_id}."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    has_draft_note: bool = False
    has_critique: bool = False
    revision_count: int = 0


class HITLRequest(BaseModel):
    """Request body for POST /research/{job_id}/hitl."""

    decision: str
    edited_note: ResearchNote | None = None


@app.post("/research", response_model=ResearchSubmitResponse, status_code=202)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchSubmitResponse:
    """Accept a research query, create a job, and run the pipeline in the background.

    Args:
        request: JSON body with ``query`` field.
        background_tasks: FastAPI background task manager.

    Returns:
        Job ID and status="running".
    """
    job_id = str(uuid.uuid4())
    initial_state: ResearchState = {
        "query": request.query,
        "job_id": job_id,
        "revision_count": 0,
        "hitl_decision": None,
        "final_note": None,
        "error": None,
        "langfuse_trace_id": None,
    }
    await _upsert_job(job_id, "running", initial_state)
    background_tasks.add_task(_run_pipeline, job_id, request.query)
    logger.info("Job %s created for query: %s", job_id, request.query)
    return ResearchSubmitResponse(job_id=job_id, status="running")


@app.get("/research/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    logger.debug("Status check for job %s", job_id)
    """Return the current status and a lightweight snapshot of job state.

    Args:
        job_id: UUID of the research job.

    Returns:
        JobStatusResponse with status, error (if any), and presence flags.

    Raises:
        HTTPException 404: If job_id does not exist in the database.
    """
    row = await _get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    state: dict = json.loads(row["state_json"])
    return JobStatusResponse(
        job_id=job_id,
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=state.get("error"),
        has_draft_note="draft_note" in state and state["draft_note"] is not None,
        has_critique=bool(state.get("critique")),
        revision_count=state.get("revision_count", 0),
    )


@app.get("/research/{job_id}/note", response_model=ResearchNote)
async def get_final_note(job_id: str) -> ResearchNote:
    """Return the approved final research note.

    Args:
        job_id: UUID of the research job.

    Returns:
        The validated ResearchNote.

    Raises:
        HTTPException 404: Job not found.
        HTTPException 409: Job not yet approved (hitl_decision != "approved").
    """
    row = await _get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    state: dict = json.loads(row["state_json"])

    if state.get("hitl_decision") != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} has not been approved yet. "
            f"Current decision: {state.get('hitl_decision')}",
        )

    final_note = state.get("final_note")
    if final_note is None:
        raise HTTPException(
            status_code=500,
            detail=f"Job {job_id} is approved but final_note is missing.",
        )

    return ResearchNote.model_validate(final_note)


@app.post("/research/{job_id}/hitl", status_code=200)
async def record_hitl_decision(job_id: str, request: HITLRequest) -> dict:
    """Record the human-in-the-loop decision for a job.

    If decision is "approved", stores the note in ChromaDB for future memory retrieval.
    If "edited", uses the edited_note provided by the human.
    If "rejected", clears the final_note.

    Args:
        job_id: UUID of the research job.
        request: Decision ("approved"/"edited"/"rejected") and optional edited note.

    Returns:
        Confirmation dict with job_id and recorded decision.

    Raises:
        HTTPException 404: Job not found.
        HTTPException 400: Invalid decision value or missing edited_note when decision="edited".
    """
    valid_decisions = {"approved", "edited", "rejected"}
    if request.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{request.decision}'. Must be one of {valid_decisions}.",
        )

    row = await _get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    state: dict = json.loads(row["state_json"])

    if request.decision == "edited":
        if request.edited_note is None:
            raise HTTPException(
                status_code=400,
                detail="edited_note must be provided when decision='edited'.",
            )
        state["final_note"] = request.edited_note.model_dump(mode="json")

    elif request.decision == "rejected":
        state["final_note"] = None

    state["hitl_decision"] = request.decision
    new_status = "done" if request.decision in ("approved", "edited") else "rejected"

    if request.decision in ("approved", "edited") and state.get("final_note"):
        final_note_dict = state["final_note"]
        try:
            store_note_in_memory(
                note_text=final_note_dict.get("full_text", ""),
                metadata={
                    "job_id": job_id,
                    "ticker": final_note_dict.get("ticker", ""),
                    "company_name": final_note_dict.get("company_name", ""),
                    "date_generated": final_note_dict.get("date_generated", ""),
                },
            )
        except Exception as exc:
            logger.error("Failed to store note in memory for job %s: %s", job_id, exc)

    typed_state: ResearchState = state
    await _upsert_job(job_id, new_status, typed_state)

    logger.info("HITL decision '%s' recorded for job %s.", request.decision, job_id)
    return {"job_id": job_id, "decision": request.decision, "status": new_status}


@app.get("/research/{job_id}/state")
async def get_job_raw_state(job_id: str) -> dict:
    """Return the full raw state dict for a job (used by the Streamlit UI).

    Exposes draft_note and critique in addition to the lightweight status fields.

    Args:
        job_id: UUID of the research job.

    Returns:
        Parsed state_json dict.

    Raises:
        HTTPException 404: Job not found.
    """
    row = await _get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return json.loads(row["state_json"])


@app.get("/research", response_model=list[dict])
async def list_approved_jobs() -> list[dict]:
    """Return all completed (approved/edited) research jobs for the history tab.

    Returns:
        List of job summary dicts with job_id, status, created_at, updated_at.
    """
    return await _list_approved_jobs()
