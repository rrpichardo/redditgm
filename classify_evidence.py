"""
classify_evidence.py - Label evidence_units with structured tags via LLM.

Reads unlabeled rows from evidence_units, calls the configured LLM per item,
and writes results to labels. Existing labels are skipped, so the job is safe
to stop and restart.

Run:
  .venv311/bin/python classify_evidence.py --tag gm_vehicle_on_demand --source-type post
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
from json_repair import repair_json
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_core import chat, current_generation_model
from run_config import RunPaths
from settings import classifier_prompt, load_settings

console = Console()


def build_prompt(evidence: dict) -> str:
    title = evidence.get("title", "")
    text = evidence.get("text", "")
    sub = evidence.get("subreddit", "")
    return f"r/{sub}\nTitle: {title}\n\n{text[:2000]}"


def _coerce_confidence(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def classify_one(
    evidence: dict,
    *,
    system_prompt: str,
    confidence_default: float,
    retries: int = 2,
    log_failures: bool = True,
) -> dict | None:
    """Call LLM and return parsed label dict. Returns None on repeated failure."""
    prompt = build_prompt(evidence)
    for attempt in range(retries + 1):
        try:
            raw = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            parsed = json.loads(repair_json(raw))
            return {
                "evidence_id": evidence["evidence_id"],
                "brand": str(parsed.get("brand", "unknown"))[:50],
                "model": str(parsed.get("model", "unknown"))[:100],
                "powertrain": str(parsed.get("powertrain", "unknown"))[:20],
                "is_pain_point": bool(parsed.get("is_pain_point", False)),
                "pain_theme": parsed.get("pain_theme") or None,
                "is_delight": bool(parsed.get("is_delight", False)),
                "delight_theme": parsed.get("delight_theme") or None,
                "sentiment": str(parsed.get("sentiment", "neutral"))[:20],
                "confidence": _coerce_confidence(parsed.get("confidence"), confidence_default),
            }
        except Exception as exc:
            if attempt >= retries:
                if log_failures:
                    console.print(
                        f"[dim yellow]! classify failed for {evidence['evidence_id']}: {exc}[/]"
                    )
                return None
            time.sleep(2 ** attempt)
    return None


def classify_one_with_retry(evidence: dict) -> dict | None:
    """Retry up to MAX_RETRIES times with exponential backoff on failure."""
    for attempt in range(MAX_RETRIES):
        result = classify_one(evidence)
        if result is not None:
            return result
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # 1s, 2s back-off
    return None


def upsert_label(con: duckdb.DuckDBPyConnection, label: dict) -> None:
    con.execute("""
        INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (evidence_id) DO UPDATE SET
            brand=excluded.brand, model=excluded.model, powertrain=excluded.powertrain,
            is_pain_point=excluded.is_pain_point, pain_theme=excluded.pain_theme,
            is_delight=excluded.is_delight, delight_theme=excluded.delight_theme,
            sentiment=excluded.sentiment, confidence=excluded.confidence
    """, [
        label["evidence_id"], label["brand"], label["model"], label["powertrain"],
        label["is_pain_point"], label["pain_theme"],
        label["is_delight"], label["delight_theme"],
        label["sentiment"], label["confidence"],
    ])


def fetch_unlabeled(
    con: duckdb.DuckDBPyConnection,
    *,
    limit: int | None,
    source_type: str | None,
) -> list[tuple]:
    type_filter = "AND source_type = ?" if source_type else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    params = [source_type] if source_type else []
    return con.execute(f"""
        SELECT e.evidence_id, e.subreddit, e.title, e.text
        FROM evidence_units e
        LEFT JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.evidence_id IS NULL
          AND length(COALESCE(e.text, '')) > 10
          {type_filter}
        ORDER BY e.created_at NULLS LAST, e.evidence_id
        {limit_clause}
    """, params).fetchall()


def estimate_classification(total: int, settings: dict, workers: int) -> dict[str, float]:
    estimates = settings["classification_estimates"]
    input_tokens = total * estimates["prompt_tokens_per_item"]
    output_tokens = total * estimates["completion_tokens_per_item"]
    estimated_cost = (
        input_tokens / 1_000_000 * estimates["usd_per_1m_input_tokens"]
        + output_tokens / 1_000_000 * estimates["usd_per_1m_output_tokens"]
    )
    serial_seconds = total * estimates["seconds_per_item_serial"]
    parallel_seconds = serial_seconds / max(workers, 1)
    return {
        "calls": float(total),
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "estimated_cost_usd": estimated_cost,
        "estimated_minutes": parallel_seconds / 60,
    }


def emit_progress(enabled: bool, **payload) -> None:
    if enabled:
        print(json.dumps(payload, sort_keys=True), flush=True)


def run_classify(
    db_path: Path,
    *,
    limit: int | None,
    source_type: str | None,
    workers: int,
    jsonl_progress: bool = False,
    estimate_only: bool = False,
) -> dict[str, int]:
    settings = load_settings()
    if not db_path.exists():
        message = f"Database not found at {db_path}. Run build_analytics_db.py first."
        if jsonl_progress:
            emit_progress(True, step="error", message=message)
        else:
            console.print(f"[red]x[/] {message}")
        sys.exit(1)

    con = duckdb.connect(str(db_path))
    unlabeled = fetch_unlabeled(con, limit=limit, source_type=source_type)

    if not unlabeled:
        if jsonl_progress:
            emit_progress(True, step="classify", completed=0, total=0, status="done")
        else:
            console.print("[green]+[/] All evidence already labeled. Nothing to do.")
        con.close()
        return {"labeled": 0, "failed": 0, "total": 0}

    total = len(unlabeled)
    estimates = estimate_classification(total, settings, workers)

    if jsonl_progress:
        emit_progress(True, step="estimate", **estimates)
    else:
        console.print(Panel.fit(
            f"[bold cyan]Classifying {total:,} evidence units[/]\n"
            f"[dim]Model: {current_generation_model()}[/]\n"
            f"[dim]Workers: {workers}[/]\n"
            f"[dim]Estimated calls: {total:,}; cost: ${estimates['estimated_cost_usd']:.2f}; "
            f"time: {estimates['estimated_minutes']:.1f} min[/]",
            border_style="cyan",
        ))

    if estimate_only:
        con.close()
        return {"labeled": 0, "failed": 0, "total": total}

    system_prompt = classifier_prompt(settings)
    confidence_default = float(settings["confidence_default"])
    successes = 0
    failures = 0

    evidence_rows = [
        {
            "evidence_id": row[0],
            "subreddit": row[1],
            "title": row[2],
            "text": row[3],
        }
        for row in unlabeled
    ]

    def submit(pool: ThreadPoolExecutor, evidence: dict):
        return pool.submit(
            classify_one,
            evidence,
            system_prompt=system_prompt,
            confidence_default=confidence_default,
            log_failures=not jsonl_progress,
        )

    if jsonl_progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {submit(pool, evidence): evidence for evidence in evidence_rows}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                label = future.result()
                if label:
                    upsert_label(con, label)
                    successes += 1
                    status = "labeled"
                else:
                    failures += 1
                    status = "failed"
                emit_progress(
                    True,
                    step="classify",
                    completed=completed,
                    total=total,
                    evidence_id=futures[future]["evidence_id"],
                    status=status,
                )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Labeling...", total=total)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {submit(pool, evidence): evidence for evidence in evidence_rows}
                for future in as_completed(futures):
                    label = future.result()
                    if label:
                        upsert_label(con, label)
                        successes += 1
                    else:
                        failures += 1
                    progress.advance(task)

    con.close()

    if not jsonl_progress:
        table = Table(title="Classification Summary", border_style="dim")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="bold")
        table.add_row("Labeled", f"{successes:,}", style="green")
        table.add_row("Failed", f"{failures:,}", style="red" if failures else "dim")
        table.add_row("Total", f"{total:,}")
        console.print(table)

    return {"labeled": successes, "failed": failures, "total": total}


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Label evidence_units via LLM into labels table.")
    p.add_argument("--tag", default=settings["active_tag"])
    p.add_argument("--db-path", default=None)
    p.add_argument("--limit", type=int, default=settings["default_classify_limit"], help="Cap labeling cost")
    p.add_argument("--source-type", choices=["post", "comment"], default=None)
    p.add_argument("--workers", type=int, default=settings["classification_workers"])
    p.add_argument("--jsonl-progress", action="store_true", help="Emit one JSON object per progress event")
    p.add_argument("--estimate-only", action="store_true", help="Print estimate and exit without labeling")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    run = RunPaths.resolve(args.tag, db_path=args.db_path)
    run_classify(
        run.db_path,
        limit=args.limit,
        source_type=args.source_type,
        workers=args.workers,
        jsonl_progress=args.jsonl_progress,
        estimate_only=args.estimate_only,
    )
