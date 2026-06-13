"""
build_analytics_db.py — Load gm_posts.csv + gm_comments.csv into DuckDB.

Creates two tables:
  evidence_units  — one row per post OR comment (deduped at source level)
  labels          — one row per evidence_id (written by classify_evidence.py)

Run:
  .venv311/bin/python build_analytics_db.py --data-dir data/gm_vehicle_on_demand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()
DB_PATH = Path("analytics/redditgm.duckdb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load Reddit CSVs into DuckDB for analytics.")
    p.add_argument("--data-dir", default="data/gm_vehicle_on_demand",
                   help="Directory containing gm_posts.csv and gm_comments.csv")
    p.add_argument("--db-path", default=str(DB_PATH), help="Path to DuckDB database file")
    p.add_argument("--reset", action="store_true", help="Drop and rebuild from scratch")
    return p.parse_args()


def build_db(data_dir: Path, db_path: Path, reset: bool = False) -> dict:
    """Load CSVs into DuckDB. Returns counts for display."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    posts_path = data_dir / "gm_posts.csv"
    comments_path = data_dir / "gm_comments.csv"

    if not posts_path.exists():
        console.print(f"[red]✗[/] gm_posts.csv not found in {data_dir}")
        sys.exit(1)

    con = duckdb.connect(str(db_path))

    if reset:
        con.execute("DROP TABLE IF EXISTS labels")
        con.execute("DROP TABLE IF EXISTS evidence_units")

    # evidence_units: one row per post + one row per comment (NOT the combined file)
    con.execute("""
        CREATE TABLE IF NOT EXISTS evidence_units (
            evidence_id  VARCHAR PRIMARY KEY,
            source_type  VARCHAR,  -- 'post' or 'comment'
            run_id       VARCHAR,
            subreddit    VARCHAR,
            post_id      VARCHAR,
            comment_id   VARCHAR,
            author       VARCHAR,
            created_at   TIMESTAMP,
            title        VARCHAR,
            text         VARCHAR,
            permalink    VARCHAR,
            score        INTEGER
        )
    """)

    # labels: written separately by classify_evidence.py; schema mirrors Lab 2 JSON shape
    con.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            evidence_id    VARCHAR PRIMARY KEY REFERENCES evidence_units(evidence_id),
            brand          VARCHAR,
            model          VARCHAR,
            powertrain     VARCHAR,  -- 'EV', 'ICE', 'PHEV', 'unknown'
            is_pain_point  BOOLEAN,
            pain_theme     VARCHAR,
            is_delight     BOOLEAN,
            delight_theme  VARCHAR,
            sentiment      VARCHAR,  -- 'positive', 'negative', 'neutral', 'mixed'
            confidence     FLOAT
        )
    """)

    # Load posts — use DuckDB's native CSV reader for speed
    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT
            'post_' || id              AS evidence_id,
            'post'                     AS source_type,
            run_id,
            subreddit,
            id                         AS post_id,
            ''                         AS comment_id,
            author,
            TRY_CAST(created_at AS TIMESTAMP) AS created_at,
            title,
            COALESCE(selftext, content, '') AS text,
            permalink,
            TRY_CAST(score AS INTEGER)
        FROM read_csv_auto('{posts_path}', header=true, ignore_errors=true)
        WHERE id IS NOT NULL AND id != ''
    """)

    post_count = con.execute("SELECT COUNT(*) FROM evidence_units WHERE source_type='post'").fetchone()[0]

    # Load comments
    if comments_path.exists():
        con.execute(f"""
            INSERT OR IGNORE INTO evidence_units
            SELECT
                'comment_' || comment_id  AS evidence_id,
                'comment'                 AS source_type,
                run_id,
                post_subreddit            AS subreddit,
                post_id,
                comment_id,
                author,
                NULL                      AS created_at,
                post_title                AS title,
                body                      AS text,
                comment_permalink         AS permalink,
                TRY_CAST(score AS INTEGER)
            FROM read_csv_auto('{comments_path}', header=true, ignore_errors=true)
            WHERE comment_id IS NOT NULL AND comment_id != ''
        """)

    comment_count = con.execute("SELECT COUNT(*) FROM evidence_units WHERE source_type='comment'").fetchone()[0]
    label_count = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    con.close()
    return {"posts": post_count, "comments": comment_count, "labels": label_count}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)

    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics — Build DB[/]\n"
        f"Source: [dim]{data_dir}[/]\n"
        f"Target: [dim]{db_path}[/]",
        border_style="cyan"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading CSVs into DuckDB...", total=None)
        counts = build_db(data_dir, db_path, reset=args.reset)
        progress.update(task, completed=1, total=1, description="Done")

    # Summary table
    table = Table(title="Evidence Units Loaded", border_style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    table.add_row("Posts", str(counts["posts"]))
    table.add_row("Comments", str(counts["comments"]))
    table.add_row("Labels (existing)", str(counts["labels"]))
    console.print(table)

    console.print(f"\n[green]✓[/] Database ready at [bold]{db_path}[/]")
    console.print("[dim]Next: run classify_evidence.py to label evidence units[/]")


if __name__ == "__main__":
    main()
