"""
classify_evidence.py — Label each evidence_unit with structured tags via LLM.

Reads unlabeled rows from evidence_units, classifies concurrently via LLM,
writes results to the labels table. Idempotent: already-labeled rows are skipped.

Run:
  .venv311/bin/python classify_evidence.py --tag gm
  .venv311/bin/python classify_evidence.py --tag gm --limit 100 --source-type post
  .venv311/bin/python classify_evidence.py --tag gm --jsonl-progress
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from rag_core import chat
from run_config import Run

console = Console()

MAX_WORKERS = 8
MAX_RETRIES = 3


def _classifier_prompt() -> str:
    """Return the classifier system prompt from current settings taxonomy."""
    from settings import get_settings
    return get_settings().classifier_prompt


def build_prompt(evidence: dict) -> str:
    title = evidence.get("title", "")
    text = evidence.get("text", "")
    sub = evidence.get("subreddit", "")
    return f"r/{sub}\nTitle: {title}\n\n{text[:2000]}"


def classify_one(evidence: dict) -> dict | None:
    """Call LLM and return parsed label dict. Returns None on failure."""
    prompt = build_prompt(evidence)
    try:
        raw = chat(
            messages=[
                {"role": "system", "content": _classifier_prompt()},
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
            "confidence": float(parsed.get("confidence", 0.5)),
        }
    except Exception as exc:
        console.print(f"[dim yellow]⚠ classify failed for {evidence['evidence_id']}: {exc}[/]")
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


def estimate_run(total: int, model: str) -> str:
    """Human-readable cost/time estimate shown before starting classification."""
    input_tokens = total * 300
    output_tokens = total * 100
    pricing: dict[str, tuple[float, float]] = {
        "openai/gpt-4o-mini": (0.15, 0.60),
        "gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4o": (2.50, 10.00),
        "anthropic/claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
    }
    in_price, out_price = pricing.get(model, (1.0, 3.0))
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    minutes = (total / MAX_WORKERS) * 0.5 / 60
    return f"~{total:,} LLM calls | ~${cost:.2f} | ~{minutes:.0f} min at {MAX_WORKERS} workers"


def run_classify(
    db_path: Path,
    limit: int | None,
    source_type: str | None,
    jsonl_progress: bool = False,
) -> None:
    if not db_path.exists():
        msg = f"Database not found at {db_path}. Run build_analytics_db.py first."
        if jsonl_progress:
            print(json.dumps({"error": msg}), flush=True)
        else:
            console.print(f"[red]✗[/] {msg}")
        sys.exit(1)

    con = duckdb.connect(str(db_path))

    type_filter = f"AND source_type = '{source_type}'" if source_type else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    unlabeled = con.execute(f"""
        SELECT e.evidence_id, e.subreddit, e.title, e.text
        FROM evidence_units e
        LEFT JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.evidence_id IS NULL
          AND length(e.text) > 10
          {type_filter}
        {limit_clause}
    """).fetchall()

    if not unlabeled:
        msg = "All evidence already labeled. Nothing to do."
        if jsonl_progress:
            print(json.dumps({"done": True, "message": msg}), flush=True)
        else:
            console.print(f"[green]✓[/] {msg}")
        con.close()
        return

    total = len(unlabeled)
    evidence_list = [
        {"evidence_id": row[0], "subreddit": row[1], "title": row[2], "text": row[3]}
        for row in unlabeled
    ]

    from settings import get_settings
    model = get_settings().generation_model
    estimate = estimate_run(total, model)

    if not jsonl_progress:
        console.print(Panel.fit(
            f"[bold cyan]Classifying {total:,} evidence units[/]\n"
            f"[dim]Model: {model}[/]\n"
            f"[dim]{estimate}[/]",
            border_style="cyan",
        ))

    successes = 0
    failures = 0

    if jsonl_progress:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(classify_one_with_retry, e): e for e in evidence_list}
            for i, future in enumerate(as_completed(futures), 1):
                label = future.result()
                if label:
                    upsert_label(con, label)
                    successes += 1
                else:
                    failures += 1
                print(json.dumps({
                    "progress": i,
                    "total": total,
                    "evidence_id": futures[future]["evidence_id"],
                    "ok": label is not None,
                }), flush=True)
    else:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        ) as progress:
            task = progress.add_task("Labeling...", total=total)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(classify_one_with_retry, e): e for e in evidence_list}
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
        table.add_row("Labeled", str(successes), style="green")
        table.add_row("Failed", str(failures), style="red" if failures else "dim")
        table.add_row("Total", str(total))
        console.print(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label evidence_units via LLM → labels table.")
    p.add_argument("--tag", default="gm_vehicle_on_demand", help="Analysis run tag")
    p.add_argument("--db-path", default=None, help="Override DuckDB path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source-type", choices=["post", "comment"], default=None)
    p.add_argument("--jsonl-progress", action="store_true",
                   help="Emit one JSON line per item (for Streamlit pipeline page)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run = Run(args.tag)
    db_path = Path(args.db_path) if args.db_path else run.db_path
    run_classify(db_path, args.limit, args.source_type, jsonl_progress=args.jsonl_progress)
