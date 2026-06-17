"""pdf_export_job.py — subprocess worker for PDF chart/briefing exports.

Phase 3 will fill in the actual rendering logic (src/charts.py,
src/pdf_export.py). This script provides the durable job scaffold so the
endpoint infrastructure and tests can be wired before the renderers exist.

Usage:
  python scripts/pdf_export_job.py --tag gm_vehicle_on_demand --job-id <hex> --kind charts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_config import RunPaths
from settings import load_settings
from src.jobs import DONE, FAILED, finish_job, update_heartbeat, job_path

# Accepted PDF export sub-kinds.
EXPORT_KINDS = {"charts", "briefing"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export charts or briefing PDF as a durable subprocess job.")
    p.add_argument("--tag", required=True)
    p.add_argument("--job-id", required=True, dest="job_id")
    p.add_argument("--export-kind", choices=list(EXPORT_KINDS), default="charts", dest="export_kind",
                   help="'charts' for chart-only PDF; 'briefing' for narrative PDF with embedded charts.")
    p.add_argument("--runtime-dir", default="runtime")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag
    job_id = args.job_id
    runtime_dir = args.runtime_dir

    path = job_path(tag, job_id, runtime_dir)
    if not path.exists():
        print(f"[pdf_export_job] ERROR: job file not found: {path}", flush=True)
        sys.exit(1)

    settings = load_settings()
    run = RunPaths.resolve(tag, settings=settings)

    if not run.db_path.exists():
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[pdf_export_job] ERROR: DB not found at {run.db_path}", flush=True)
        sys.exit(1)

    # Heartbeat: 1 step = generating the PDF.
    update_heartbeat(tag, job_id, processed=0, total=1, runtime_dir=runtime_dir)

    try:
        artifact = _run_export(run, args.export_kind)
    except NotImplementedError as exc:
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[pdf_export_job] NOT IMPLEMENTED: {exc}", flush=True)
        sys.exit(2)
    except Exception as exc:
        finish_job(tag, job_id, state=FAILED, runtime_dir=runtime_dir)
        print(f"[pdf_export_job] FATAL: {exc}", flush=True)
        sys.exit(1)

    finish_job(
        tag, job_id,
        state=DONE,
        processed=1,
        artifact_paths=[str(artifact)],
        runtime_dir=runtime_dir,
    )
    print(f"[pdf_export_job] done: artifact={artifact}", flush=True)


def _run_export(run: RunPaths, export_kind: str) -> Path:
    """Render the PDF and return the output path.

    Phase 3 will replace this with actual src.charts / src.pdf_export calls.
    """
    # src.charts and src.pdf_export are not yet implemented (Phase 3).
    raise NotImplementedError(
        f"PDF export ({export_kind!r}) requires Phase 3 src/charts.py and src/pdf_export.py — not yet implemented."
    )


if __name__ == "__main__":
    main()
