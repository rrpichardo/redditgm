"""
classify_evidence.py — Label each evidence_unit with structured tags via LLM.

Reads unlabeled rows from evidence_units, calls the LLM per item using the
Lab 2 analyze_review JSON shape, writes results to the labels table.

Fixed theme taxonomy keeps themes comparable across runs:
  transmission | reliability | dealer_service | pricing | infotainment |
  battery_range | charging | build_quality | recall | warranty |
  performance | comfort | other

Run:
  .venv311/bin/python classify_evidence.py [--limit N] [--source-type post]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
from json_repair import repair_json
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from rag_core import chat

console = Console()
DB_PATH = Path("analytics/redditgm.duckdb")

# Fixed theme enum so counts stay comparable (Lab 2 pattern)
THEME_ENUM = [
    "transmission", "reliability", "dealer_service", "pricing",
    "infotainment", "battery_range", "charging", "build_quality",
    "recall", "warranty", "performance", "comfort", "other",
]

# System prompt — Lab 2 analyze_review JSON shape
CLASSIFIER_SYSTEM = """You are a vehicle-brand sentiment analyst for GM (General Motors).

For each Reddit post/comment, output a single JSON object with exactly these fields:
{
  "brand": "Chevy|GMC|Cadillac|Buick|GM|unknown",
  "model": "Silverado|Equinox|Tahoe|Sierra|Blazer|Escalade|Corvette|Camaro|<model>|unknown",
  "powertrain": "EV|ICE|PHEV|unknown",
  "is_pain_point": true|false,
  "pain_theme": "transmission|reliability|dealer_service|pricing|infotainment|battery_range|charging|build_quality|recall|warranty|performance|comfort|other|null",
  "is_delight": true|false,
  "delight_theme": "performance|comfort|value|technology|design|safety|dealer_service|reliability|other|null",
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.0-1.0
}

Rules:
- pain_theme must be one of the listed values or null (if is_pain_point is false)
- delight_theme must be one of the listed values or null (if is_delight is false)
- powertrain: EV if mentions electric/EV/battery/Bolt/Lyriq/Blazer EV; PHEV if plug-in hybrid; ICE otherwise
- confidence: your certainty this is actually about a GM vehicle (0.0 = not sure, 1.0 = certain)
- Output ONLY the JSON object — no explanation, no markdown fences
"""


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
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        # json-repair handles minor JSON malformations (Lab 2 parse_json_analysis)
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


def run_classify(db_path: Path, limit: int | None, source_type: str | None) -> None:
    if not db_path.exists():
        console.print(f"[red]✗[/] Database not found at {db_path}. Run build_analytics_db.py first.")
        sys.exit(1)

    con = duckdb.connect(str(db_path))

    # Only fetch rows that haven't been labeled yet
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
        console.print("[green]✓[/] All evidence already labeled. Nothing to do.")
        con.close()
        return

    total = len(unlabeled)
    console.print(Panel.fit(
        f"[bold cyan]Classifying {total} evidence units[/]\n"
        f"[dim]Model: {__import__('rag_core').GENERATION_MODEL}[/]",
        border_style="cyan"
    ))

    successes = 0
    failures = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Labeling...", total=total)

        for row in unlabeled:
            evidence = {
                "evidence_id": row[0],
                "subreddit": row[1],
                "title": row[2],
                "text": row[3],
            }
            label = classify_one(evidence)
            if label:
                upsert_label(con, label)
                successes += 1
            else:
                failures += 1
            progress.advance(task)

    con.close()

    # Summary
    table = Table(title="Classification Summary", border_style="dim")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    table.add_row("Labeled", str(successes), style="green")
    table.add_row("Failed", str(failures), style="red" if failures else "dim")
    table.add_row("Total", str(total))
    console.print(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label evidence_units via LLM → labels table.")
    p.add_argument("--db-path", default=str(DB_PATH))
    p.add_argument("--limit", type=int, default=None, help="Cap labeling cost; omit for all")
    p.add_argument("--source-type", choices=["post", "comment"], default=None,
                   help="Label only posts or only comments (default: both)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_classify(Path(args.db_path), args.limit, args.source_type)
