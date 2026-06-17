"""faiss_qa_job.py — subprocess worker for building a FAISS Q&A index.

Phase 6 will fill in src/qa_retrieval.py. This script provides the durable job
scaffold so the endpoint infrastructure and tests can be wired in advance.

Usage:
  python scripts/faiss_qa_job.py --tag gm_vehicle_on_demand --job-id <hex>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_config import RunPaths
from settings import load_settings
from src.jobs import DONE, FAILED, finish_job, update_heartbeat, job_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS Q&A index as a durable subprocess job.")
    p.add_argument("--tag", required=True)
    p.add_argument("--job-id", required=True, dest="job_id")
    p.add_argument("--runtime-dir", default="runtime")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag
    job_id = args.job_id
    runtime_dir = args.runtime_dir

    path = job_path(tag, job_id, runtime_dir)
    if not path.exists():
        print(f"[faiss_qa_job] ERROR: job file not found: {path}", flush=True)
        sys.exit(1)

    settings = load_settings()
    run = RunPaths.resolve(tag, settings=settings)

    if not run.db_path.exists():
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[faiss_qa_job] ERROR: DB not found at {run.db_path}", flush=True)
        sys.exit(1)

    update_heartbeat(tag, job_id, processed=0, total=1, runtime_dir=runtime_dir)

    try:
        artifacts = _build_index(run)
    except NotImplementedError as exc:
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[faiss_qa_job] NOT IMPLEMENTED: {exc}", flush=True)
        sys.exit(2)
    except Exception as exc:
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[faiss_qa_job] FATAL: {exc}", flush=True)
        sys.exit(1)

    finish_job(
        tag, job_id,
        state=DONE,
        processed=1,
        artifact_paths=[str(a) for a in artifacts],
        runtime_dir=runtime_dir,
    )
    print(f"[faiss_qa_job] done: artifacts={artifacts}", flush=True)


def _build_index(run: RunPaths) -> list[Path]:
    """Build FAISS Q&A index and return output artifact paths.

    Phase 6 will replace this with actual src.qa_retrieval calls.
    """
    raise NotImplementedError(
        "FAISS Q&A index build requires Phase 6 src/qa_retrieval.py — not yet implemented."
    )


if __name__ == "__main__":
    main()
