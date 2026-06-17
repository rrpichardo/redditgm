"""classify_job.py — subprocess worker that classifies evidence and writes heartbeats.

Invoked by launch_job() via src/jobs.py. Never call this directly from the UI;
use the launch endpoint which creates the job record first.

Usage:
  python scripts/classify_job.py --tag gm_vehicle_on_demand --job-id <hex> [--source-type post] [--limit 200] [--workers 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure the project root is on the path so we can import from src/ and top-level modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from classify_evidence import classify_one, fetch_unlabeled, upsert_label
from run_config import RunPaths
from settings import classifier_prompt, load_settings
from src.jobs import DONE, FAILED, finish_job, update_heartbeat, write_job, job_path


# Write a heartbeat every N rows processed.
HEARTBEAT_EVERY = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify evidence units as a durable subprocess job.")
    p.add_argument("--tag", required=True)
    p.add_argument("--job-id", required=True, dest="job_id")
    p.add_argument("--source-type", choices=["post", "comment"], default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--runtime-dir", default="runtime")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag
    job_id = args.job_id
    runtime_dir = args.runtime_dir

    # Confirm the job record exists before doing any work.
    path = job_path(tag, job_id, runtime_dir)
    if not path.exists():
        print(f"[classify_job] ERROR: job file not found: {path}", flush=True)
        sys.exit(1)

    settings = load_settings()
    run = RunPaths.resolve(tag, settings=settings)

    if not run.db_path.exists():
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[classify_job] ERROR: DB not found at {run.db_path}", flush=True)
        sys.exit(1)

    con = duckdb.connect(str(run.db_path))
    unlabeled = fetch_unlabeled(con, limit=args.limit, source_type=args.source_type)
    total = len(unlabeled)

    # Record total in the job file before long work begins.
    update_heartbeat(tag, job_id, processed=0, total=total, runtime_dir=runtime_dir)

    if total == 0:
        con.close()
        finish_job(tag, job_id, state=DONE, processed=0, errors=0, runtime_dir=runtime_dir)
        return

    system_prompt = classifier_prompt(settings)
    confidence_default = float(settings["confidence_default"])

    evidence_rows = [
        {"evidence_id": row[0], "subreddit": row[1], "title": row[2], "text": row[3]}
        for row in unlabeled
    ]

    processed = 0
    errors = 0
    last_heartbeat_at = time.monotonic()

    def maybe_heartbeat() -> None:
        nonlocal last_heartbeat_at
        now = time.monotonic()
        if now - last_heartbeat_at >= 15:
            update_heartbeat(
                tag, job_id,
                processed=processed, total=total, errors=errors,
                runtime_dir=runtime_dir,
            )
            last_heartbeat_at = now

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    classify_one,
                    evidence,
                    system_prompt=system_prompt,
                    confidence_default=confidence_default,
                    log_failures=True,
                ): evidence
                for evidence in evidence_rows
            }
            for future in as_completed(futures):
                label = future.result()
                if label:
                    upsert_label(con, label)
                    processed += 1
                else:
                    errors += 1
                maybe_heartbeat()
    except Exception as exc:
        con.close()
        finish_job(tag, job_id, state=FAILED, processed=processed, errors=errors, runtime_dir=runtime_dir)
        print(f"[classify_job] FATAL: {exc}", flush=True)
        sys.exit(1)

    con.close()
    finish_job(
        tag, job_id,
        state=DONE,
        processed=processed,
        errors=errors,
        runtime_dir=runtime_dir,
    )
    print(f"[classify_job] done: processed={processed} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
