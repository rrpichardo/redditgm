"""Durable job system for redditgm long-running work.

Jobs are tracked via JSON status files under runtime/<tag>/jobs/<job_id>.json.
Subprocess workers write heartbeats; status reads reconcile dead PIDs and stale
heartbeats to 'failed' or 'interrupted' so callers always see consistent state.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Job states ──────────────────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
INTERRUPTED = "interrupted"

TERMINAL_STATES = {DONE, FAILED, INTERRUPTED}

# Seconds without a heartbeat before a running job is considered stale.
STALE_HEARTBEAT_SECONDS = 120

# Valid job kinds — each maps to one subprocess script.
KIND_CLASSIFY = "classify"
KIND_PDF_EXPORT = "pdf_export"
KIND_TREND = "trend"
KIND_FAISS_QA = "faiss_qa"

VALID_KINDS = {KIND_CLASSIFY, KIND_PDF_EXPORT, KIND_TREND, KIND_FAISS_QA}


# ── Path helpers ─────────────────────────────────────────────────────────────

def job_dir(tag: str, runtime_dir: str = "runtime") -> Path:
    """Return the directory that holds all job status files for a tag."""
    return Path(runtime_dir) / tag / "jobs"


def job_path(tag: str, job_id: str, runtime_dir: str = "runtime") -> Path:
    return job_dir(tag, runtime_dir) / f"{job_id}.json"


# ── Time helpers ─────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_since(iso_str: str | None) -> float | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except ValueError:
        return None


# ── Job metadata factory ─────────────────────────────────────────────────────

def make_job_id() -> str:
    return uuid.uuid4().hex


def new_job(tag: str, kind: str, *, pid: int | None = None, total: int = 0) -> dict[str, Any]:
    """Return a fresh job metadata dict. Call write_job() to persist it."""
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown job kind: {kind!r}. Valid: {sorted(VALID_KINDS)}")
    now = _now_iso()
    return {
        "job_id": make_job_id(),
        "tag": tag,
        "kind": kind,
        "state": RUNNING if pid else PENDING,
        "pid": pid,
        "processed": 0,
        "total": total,
        "errors": 0,
        "started_at": now,
        "heartbeat_at": now if pid else None,
        "updated_at": now,
        "completed_at": None,
        "artifact_paths": [],
    }


# ── Atomic file write ─────────────────────────────────────────────────────────

def write_job(job: dict[str, Any], runtime_dir: str = "runtime") -> Path:
    """Atomically write job status to disk. Returns the path written."""
    path = job_path(job["tag"], job["job_id"], runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(job, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


# ── Stale / dead PID reconciliation ──────────────────────────────────────────

def is_pid_alive(pid: int | None) -> bool:
    """Return True if the OS reports the process is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to another user — counts as alive.
        return True


def reconcile_stale(job: dict[str, Any]) -> dict[str, Any]:
    """Return the job dict with state corrected if the process is dead or stale.

    A running job is marked:
    - 'failed'      if the PID is gone
    - 'interrupted' if the PID is alive but heartbeat is older than STALE_HEARTBEAT_SECONDS
    """
    if job.get("state") != RUNNING:
        return job

    pid = job.get("pid")
    if pid is None or not is_pid_alive(pid):
        # Process is gone; mark failed so callers don't wait forever.
        job = dict(job)
        job["state"] = FAILED
        job["updated_at"] = _now_iso()
        return job

    # PID is alive — check heartbeat freshness.
    stale_secs = _seconds_since(job.get("heartbeat_at"))
    if stale_secs is not None and stale_secs > STALE_HEARTBEAT_SECONDS:
        job = dict(job)
        job["state"] = INTERRUPTED
        job["updated_at"] = _now_iso()

    return job


# ── Read ─────────────────────────────────────────────────────────────────────

def read_job(tag: str, job_id: str, runtime_dir: str = "runtime") -> dict[str, Any] | None:
    """Read job status from disk, applying stale reconciliation. Returns None if not found."""
    path = job_path(tag, job_id, runtime_dir)
    if not path.exists():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return reconcile_stale(job)


# ── List / latest ─────────────────────────────────────────────────────────────

def list_jobs(tag: str, kind: str | None = None, runtime_dir: str = "runtime") -> list[dict[str, Any]]:
    """Return all jobs for a tag, newest first, with stale reconciliation applied."""
    directory = job_dir(tag, runtime_dir)
    if not directory.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        job = reconcile_stale(job)
        if kind is None or job.get("kind") == kind:
            jobs.append(job)
    return jobs


def latest_job(tag: str, kind: str, runtime_dir: str = "runtime") -> dict[str, Any] | None:
    """Return the most-recently-started job of the given kind, or None."""
    jobs = list_jobs(tag, kind=kind, runtime_dir=runtime_dir)
    if not jobs:
        return None
    # Sort by started_at descending; list_jobs already sorts by filename but
    # started_at is the canonical ordering.
    return max(jobs, key=lambda j: j.get("started_at", ""), default=None)


# ── Heartbeat update (called from inside a subprocess job) ───────────────────

def update_heartbeat(
    tag: str,
    job_id: str,
    *,
    processed: int | None = None,
    total: int | None = None,
    errors: int | None = None,
    extra: dict[str, Any] | None = None,
    runtime_dir: str = "runtime",
) -> None:
    """Read-modify-write the job status file, updating heartbeat and progress.

    Safe to call from a subprocess worker every few seconds. Uses atomic write
    to avoid readers seeing partial JSON.
    """
    path = job_path(tag, job_id, runtime_dir)
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return  # nothing we can do if the file vanished

    now = _now_iso()
    job["heartbeat_at"] = now
    job["updated_at"] = now
    if processed is not None:
        job["processed"] = processed
    if total is not None:
        job["total"] = total
    if errors is not None:
        job["errors"] = errors
    if extra:
        job.update(extra)

    write_job(job, runtime_dir)


def finish_job(
    tag: str,
    job_id: str,
    *,
    state: str = DONE,
    artifact_paths: list[str] | None = None,
    processed: int | None = None,
    errors: int | None = None,
    runtime_dir: str = "runtime",
) -> None:
    """Mark a job as terminal (done / failed / interrupted). Called at job end."""
    path = job_path(tag, job_id, runtime_dir)
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    now = _now_iso()
    job["state"] = state
    job["completed_at"] = now
    job["updated_at"] = now
    if artifact_paths is not None:
        job["artifact_paths"] = artifact_paths
    if processed is not None:
        job["processed"] = processed
    if errors is not None:
        job["errors"] = errors

    write_job(job, runtime_dir)


# ── Single-job-per-kind guard ─────────────────────────────────────────────────

def has_active_job(tag: str, kind: str, runtime_dir: str = "runtime") -> bool:
    """Return True if a non-terminal job of this kind already exists for the tag."""
    job = latest_job(tag, kind, runtime_dir=runtime_dir)
    if job is None:
        return False
    return job.get("state") not in TERMINAL_STATES


def launch_job(
    tag: str,
    kind: str,
    command: list[str],
    *,
    total: int = 0,
    cwd: str | Path | None = None,
    runtime_dir: str = "runtime",
    allow_concurrent: bool = False,
) -> dict[str, Any]:
    """Start a subprocess job, write its status file, and return the job dict.

    Raises RuntimeError if a non-terminal job of the same kind is already running
    and allow_concurrent is False (default).
    """
    if not allow_concurrent and has_active_job(tag, kind, runtime_dir=runtime_dir):
        existing = latest_job(tag, kind, runtime_dir=runtime_dir)
        raise RuntimeError(
            f"A {kind!r} job is already active for tag {tag!r} "
            f"(job_id={existing['job_id']!r}, state={existing['state']!r}). "
            "Wait for it to finish or let it fail/interrupt before starting another."
        )

    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from parent's signal group
    )

    job = new_job(tag, kind, pid=proc.pid, total=total)
    write_job(job, runtime_dir)
    return job


# subprocess is only needed by launch_job; import here so the rest of the
# module remains importable in environments where we only read status files.
import subprocess  # noqa: E402 — intentional late import
