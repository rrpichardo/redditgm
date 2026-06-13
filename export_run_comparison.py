#!/usr/bin/env python3
"""Export historical-vs-new CSVs from additive collector outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_config import RunPaths
from settings import load_settings


CSV_FILES = [
    "gm_posts.csv",
    "gm_comments.csv",
    "gm_posts_with_comments.csv",
]


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Export historical and new rows for a selected run_id."
    )
    parser.add_argument("--tag", default=settings["active_tag"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID to export. Defaults to the most recent run summary in --runs-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where comparison files should be written.",
    )
    return parser.parse_args()


def load_run_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_summary(runs_dir: Path) -> dict[str, Any]:
    summaries = []
    for path in runs_dir.glob("*.json"):
        if path.name == "run_manifest.json":
            continue
        try:
            summary = load_run_summary(path)
        except json.JSONDecodeError:
            continue
        summaries.append((summary.get("completed_at") or summary.get("started_at") or "", path, summary))

    if not summaries:
        raise FileNotFoundError(f"No run summary JSON files found in {runs_dir}")

    summaries.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return summaries[0][2]


def row_subreddit(row: dict[str, str]) -> str:
    return row.get("subreddit") or row.get("post_subreddit") or ""


def row_timestamp(row: dict[str, str]) -> str:
    return row.get("created_at") or row.get("comment_created_at") or row.get("downloaded_at") or ""


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    run_ids = sorted({row.get("run_id", "") for row in rows if row.get("run_id", "")})
    subreddits = Counter(row_subreddit(row) for row in rows if row_subreddit(row))
    timestamps = sorted(row_timestamp(row) for row in rows if row_timestamp(row))
    return {
        "rows": len(rows),
        "run_ids": run_ids,
        "subreddits": dict(sorted(subreddits.items())),
        "earliest_timestamp": timestamps[0] if timestamps else "",
        "latest_timestamp": timestamps[-1] if timestamps else "",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_csv_by_run_id(source: Path, output_root: Path, run_id: str) -> dict[str, Any]:
    empty_summary = {
        "total": summarize_rows([]),
        "historical": summarize_rows([]),
        "new": summarize_rows([]),
        "source_exists": source.exists(),
    }
    if not source.exists():
        return empty_summary

    with source.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not fieldnames:
        return empty_summary

    new_rows = [row for row in rows if row.get("run_id") == run_id]
    historical_rows = [row for row in rows if row.get("run_id") != run_id]

    write_csv(output_root / "new" / source.name, fieldnames, new_rows)
    write_csv(output_root / "historical" / source.name, fieldnames, historical_rows)
    write_csv(output_root / "all" / source.name, fieldnames, rows)

    return {
        "total": summarize_rows(rows),
        "historical": summarize_rows(historical_rows),
        "new": summarize_rows(new_rows),
        "source_exists": True,
    }


def main() -> None:
    args = parse_args()
    run = RunPaths.resolve(args.tag, data_dir=args.data_dir)
    data_dir = run.data_dir
    runs_dir = Path(args.runs_dir) if args.runs_dir else run.runs_dir
    out_dir = Path(args.out_dir) if args.out_dir else run.root / "reports" / "comparison"
    run_id = args.run_id
    selected_summary = None

    if not run_id:
        selected_summary = latest_run_summary(runs_dir)
        run_id = str(selected_summary["run_id"])
    else:
        summary_path = runs_dir / f"{run_id}.json"
        if summary_path.exists():
            selected_summary = load_run_summary(summary_path)

    output_root = out_dir / run_id
    file_summaries = {}
    for filename in CSV_FILES:
        file_summaries[filename] = split_csv_by_run_id(data_dir / filename, output_root, run_id)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_run_id": run_id,
        "data_dir": str(data_dir),
        "runs_dir": str(runs_dir),
        "output_dir": str(output_root),
        "target_run_summary": selected_summary or {},
        "files": file_summaries,
        "notes": [
            "new contains rows collected in target_run_id",
            "historical contains all additive master rows whose run_id is not target_run_id",
            "all is a snapshot copy of the master CSVs at export time",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Exported historical-vs-new comparison for run_id={run_id} to {output_root}")
    for filename in CSV_FILES:
        summary = file_summaries[filename]
        print(
            f"{filename}: "
            f"historical={summary['historical']['rows']} "
            f"new={summary['new']['rows']} "
            f"total={summary['total']['rows']}"
        )


if __name__ == "__main__":
    main()
