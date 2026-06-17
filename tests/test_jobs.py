"""Tests for src/jobs.py — durable job system.

Covers: job creation, atomic write/read, stale-job reconciliation (dead PID →
failed, stale heartbeat → interrupted), has_active_job guard, finish_job
transitions, and list_jobs ordering.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# All tests use a temp directory as the runtime root so they never touch disk.
from src.jobs import (
    DONE,
    FAILED,
    INTERRUPTED,
    PENDING,
    RUNNING,
    STALE_HEARTBEAT_SECONDS,
    finish_job,
    has_active_job,
    is_pid_alive,
    job_path,
    latest_job,
    list_jobs,
    new_job,
    read_job,
    reconcile_stale,
    update_heartbeat,
    write_job,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rt(tmp_path: Path) -> str:
    """Return a temp runtime_dir string."""
    return str(tmp_path)


# ── new_job ───────────────────────────────────────────────────────────────────

def test_new_job_fields():
    job = new_job("my_tag", "classify", pid=1234, total=100)
    assert job["tag"] == "my_tag"
    assert job["kind"] == "classify"
    assert job["state"] == RUNNING
    assert job["pid"] == 1234
    assert job["total"] == 100
    assert job["processed"] == 0
    assert job["errors"] == 0
    assert job["started_at"] is not None
    assert job["heartbeat_at"] is not None
    assert job["completed_at"] is None
    assert job["artifact_paths"] == []


def test_new_job_without_pid_is_pending():
    job = new_job("t", "pdf_export")
    assert job["state"] == PENDING
    assert job["pid"] is None
    assert job["heartbeat_at"] is None


def test_new_job_invalid_kind():
    with pytest.raises(ValueError, match="Unknown job kind"):
        new_job("t", "bogus_kind")


# ── write_job / read_job ──────────────────────────────────────────────────────

def test_write_and_read_roundtrip(rt):
    job = new_job("tag1", "classify", pid=99, total=50)
    write_job(job, runtime_dir=rt)

    loaded = read_job("tag1", job["job_id"], runtime_dir=rt)
    assert loaded is not None
    assert loaded["job_id"] == job["job_id"]
    assert loaded["tag"] == "tag1"
    assert loaded["total"] == 50


def test_read_job_missing_returns_none(rt):
    assert read_job("no_such_tag", "deadbeef", runtime_dir=rt) is None


def test_write_is_atomic(rt):
    """write_job should produce a valid JSON file even if we read mid-write."""
    job = new_job("t", "trend", pid=1, total=0)
    write_job(job, runtime_dir=rt)
    path = job_path("t", job["job_id"], rt)
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["job_id"] == job["job_id"]


# ── reconcile_stale ───────────────────────────────────────────────────────────

def test_reconcile_done_job_unchanged():
    job = new_job("t", "classify")
    job["state"] = DONE
    result = reconcile_stale(job)
    assert result["state"] == DONE


def test_reconcile_dead_pid_becomes_failed():
    job = new_job("t", "classify", pid=999_999_999)  # almost certainly dead
    # Ensure the PID is actually treated as dead.
    with patch("src.jobs.is_pid_alive", return_value=False):
        result = reconcile_stale(job)
    assert result["state"] == FAILED


def test_reconcile_stale_heartbeat_becomes_interrupted():
    stale_time = (
        datetime.now(timezone.utc) - timedelta(seconds=STALE_HEARTBEAT_SECONDS + 30)
    ).isoformat()
    job = new_job("t", "classify", pid=os.getpid())  # our own PID — alive
    job["heartbeat_at"] = stale_time
    with patch("src.jobs.is_pid_alive", return_value=True):
        result = reconcile_stale(job)
    assert result["state"] == INTERRUPTED


def test_reconcile_fresh_heartbeat_stays_running():
    job = new_job("t", "classify", pid=os.getpid())
    # heartbeat_at was just set in new_job so it's fresh.
    with patch("src.jobs.is_pid_alive", return_value=True):
        result = reconcile_stale(job)
    assert result["state"] == RUNNING


# ── is_pid_alive ──────────────────────────────────────────────────────────────

def test_own_pid_is_alive():
    assert is_pid_alive(os.getpid()) is True


def test_none_pid_not_alive():
    assert is_pid_alive(None) is False


def test_dead_pid_not_alive():
    assert is_pid_alive(999_999_999) is False


# ── update_heartbeat ──────────────────────────────────────────────────────────

def test_update_heartbeat_modifies_file(rt):
    job = new_job("t", "classify", pid=1, total=100)
    write_job(job, runtime_dir=rt)

    update_heartbeat("t", job["job_id"], processed=42, total=100, errors=1, runtime_dir=rt)

    loaded = json.loads(job_path("t", job["job_id"], rt).read_text())
    assert loaded["processed"] == 42
    assert loaded["errors"] == 1
    assert loaded["heartbeat_at"] is not None


def test_update_heartbeat_missing_file_is_noop(rt):
    # Should not raise, just silently return.
    update_heartbeat("t", "nonexistent_id", processed=5, runtime_dir=rt)


# ── finish_job ────────────────────────────────────────────────────────────────

def test_finish_job_done(rt):
    job = new_job("t", "pdf_export", pid=1, total=1)
    write_job(job, runtime_dir=rt)

    finish_job("t", job["job_id"], state=DONE, artifact_paths=["out/report.pdf"],
               processed=1, errors=0, runtime_dir=rt)

    loaded = read_job("t", job["job_id"], runtime_dir=rt)
    assert loaded["state"] == DONE
    assert loaded["artifact_paths"] == ["out/report.pdf"]
    assert loaded["completed_at"] is not None


def test_finish_job_failed(rt):
    job = new_job("t", "faiss_qa", pid=1, total=5)
    write_job(job, runtime_dir=rt)

    finish_job("t", job["job_id"], state=FAILED, processed=2, errors=3, runtime_dir=rt)

    loaded = read_job("t", job["job_id"], runtime_dir=rt)
    assert loaded["state"] == FAILED
    assert loaded["processed"] == 2
    assert loaded["errors"] == 3


# ── list_jobs / latest_job ────────────────────────────────────────────────────

def test_list_jobs_empty(rt):
    assert list_jobs("no_tag", runtime_dir=rt) == []


def test_list_jobs_returns_all(rt):
    for _ in range(3):
        job = new_job("t", "classify", pid=1, total=10)
        write_job(job, runtime_dir=rt)

    jobs = list_jobs("t", runtime_dir=rt)
    assert len(jobs) == 3


def test_list_jobs_kind_filter(rt):
    j1 = new_job("t", "classify", pid=1)
    j2 = new_job("t", "trend", pid=1)
    write_job(j1, runtime_dir=rt)
    write_job(j2, runtime_dir=rt)

    classify_jobs = list_jobs("t", kind="classify", runtime_dir=rt)
    assert len(classify_jobs) == 1
    assert classify_jobs[0]["kind"] == "classify"


def test_latest_job_none_when_empty(rt):
    assert latest_job("t", "classify", runtime_dir=rt) is None


def test_latest_job_returns_most_recent(rt):
    j1 = new_job("t", "classify", pid=1)
    time.sleep(0.01)  # ensure distinct started_at
    j2 = new_job("t", "classify", pid=1)
    write_job(j1, runtime_dir=rt)
    write_job(j2, runtime_dir=rt)

    latest = latest_job("t", "classify", runtime_dir=rt)
    assert latest is not None
    assert latest["job_id"] == j2["job_id"]


# ── has_active_job ────────────────────────────────────────────────────────────

def test_has_active_job_true_for_running(rt):
    job = new_job("t", "classify", pid=os.getpid(), total=10)
    write_job(job, runtime_dir=rt)
    with patch("src.jobs.is_pid_alive", return_value=True):
        assert has_active_job("t", "classify", runtime_dir=rt) is True


def test_has_active_job_false_after_done(rt):
    job = new_job("t", "classify", pid=1, total=10)
    write_job(job, runtime_dir=rt)
    finish_job("t", job["job_id"], state=DONE, runtime_dir=rt)
    assert has_active_job("t", "classify", runtime_dir=rt) is False


def test_has_active_job_false_when_none(rt):
    assert has_active_job("t", "classify", runtime_dir=rt) is False
