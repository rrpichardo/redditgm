"""
build_analytics_db.py — Load Reddit CSVs into DuckDB.

Accepts EITHER:
  • split files: gm_posts.csv + gm_comments.csv  (preferred)
  • combined file: gm_posts_with_comments.csv     (collector headline output)

For the combined file, evidence_units are derived by deduplication:
  post rows  → DISTINCT post_id   → evidence_id = 'post_<id>'
  comment rows → DISTINCT comment_id → evidence_id = 'comment_<id>'

Run:
  .venv311/bin/python build_analytics_db.py --tag gm
  .venv311/bin/python build_analytics_db.py --data-dir data/gm_vehicle_on_demand --tag gm
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

from run_config import Run

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load Reddit CSVs into DuckDB for analytics.")
    p.add_argument("--tag", default="gm_vehicle_on_demand", help="Analysis run tag")
    p.add_argument("--data-dir", default=None,
                   help="Directory with CSVs (defaults to data/<tag>)")
    p.add_argument("--db-path", default=None, help="Override DuckDB path (defaults to run layout)")
    p.add_argument("--reset", action="store_true", help="Drop and rebuild from scratch")
    return p.parse_args()


def _create_tables(con: duckdb.DuckDBPyConnection, reset: bool) -> None:
    if reset:
        con.execute("DROP TABLE IF EXISTS labels")
        con.execute("DROP TABLE IF EXISTS evidence_units")

    con.execute("""
        CREATE TABLE IF NOT EXISTS evidence_units (
            evidence_id  VARCHAR PRIMARY KEY,
            source_type  VARCHAR,
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
    con.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            evidence_id    VARCHAR PRIMARY KEY REFERENCES evidence_units(evidence_id),
            brand          VARCHAR,
            model          VARCHAR,
            powertrain     VARCHAR,
            is_pain_point  BOOLEAN,
            pain_theme     VARCHAR,
            is_delight     BOOLEAN,
            delight_theme  VARCHAR,
            sentiment      VARCHAR,
            confidence     FLOAT
        )
    """)


def _load_split_csvs(con: duckdb.DuckDBPyConnection, data_dir: Path) -> tuple[int, int]:
    """Load gm_posts.csv + gm_comments.csv (separate files)."""
    posts_path = data_dir / "gm_posts.csv"
    comments_path = data_dir / "gm_comments.csv"

    if not posts_path.exists():
        console.print(f"[red]✗[/] gm_posts.csv not found in {data_dir}")
        sys.exit(1)

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
    post_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='post'"
    ).fetchone()[0]

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
                TRY_CAST(created_at AS TIMESTAMP) AS created_at,
                post_title                AS title,
                body                      AS text,
                comment_permalink         AS permalink,
                TRY_CAST(score AS INTEGER)
            FROM read_csv_auto('{comments_path}', header=true, ignore_errors=true)
            WHERE comment_id IS NOT NULL AND comment_id != ''
        """)

    comment_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='comment'"
    ).fetchone()[0]
    return post_count, comment_count


def _load_combined_csv(con: duckdb.DuckDBPyConnection, combined_path: Path) -> tuple[int, int]:
    """
    Load gm_posts_with_comments.csv (combined format: each row is a post+comment pair).
    Deduplicates posts by post_id and comments by comment_id.
    """
    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT DISTINCT ON (id)
            'post_' || id              AS evidence_id,
            'post'                     AS source_type,
            COALESCE(run_id, '')       AS run_id,
            subreddit,
            id                         AS post_id,
            ''                         AS comment_id,
            author,
            TRY_CAST(created_at AS TIMESTAMP) AS created_at,
            title,
            COALESCE(selftext, content, '') AS text,
            permalink,
            TRY_CAST(score AS INTEGER)
        FROM read_csv_auto('{combined_path}', header=true, ignore_errors=true)
        WHERE id IS NOT NULL AND id != ''
    """)
    post_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='post'"
    ).fetchone()[0]

    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT DISTINCT ON (comment_id)
            'comment_' || comment_id  AS evidence_id,
            'comment'                 AS source_type,
            COALESCE(run_id, '')      AS run_id,
            post_subreddit            AS subreddit,
            post_id,
            comment_id,
            comment_author            AS author,
            TRY_CAST(comment_created_at AS TIMESTAMP) AS created_at,
            post_title                AS title,
            comment_body              AS text,
            comment_permalink         AS permalink,
            TRY_CAST(comment_score AS INTEGER)
        FROM read_csv_auto('{combined_path}', header=true, ignore_errors=true)
        WHERE comment_id IS NOT NULL AND comment_id != ''
    """)

    comment_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='comment'"
    ).fetchone()[0]
    return post_count, comment_count


def build_db(data_dir: Path, db_path: Path, reset: bool = False) -> dict:
    """Load CSVs into DuckDB. Returns counts for display."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    _create_tables(con, reset)

    combined_path = data_dir / "gm_posts_with_comments.csv"
    if combined_path.exists():
        post_count, comment_count = _load_combined_csv(con, combined_path)
    else:
        post_count, comment_count = _load_split_csvs(con, data_dir)

    label_count = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    con.close()
    return {"posts": post_count, "comments": comment_count, "labels": label_count}


def main() -> None:
    args = parse_args()
    run = Run(args.tag)
    data_dir = Path(args.data_dir) if args.data_dir else run.data_dir
    db_path = Path(args.db_path) if args.db_path else run.db_path

    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics — Build DB[/]\n"
        f"Source: [dim]{data_dir}[/]\n"
        f"Target: [dim]{db_path}[/]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Loading CSVs into DuckDB...", total=None)
        counts = build_db(data_dir, db_path, reset=args.reset)
        progress.update(task, completed=1, total=1, description="Done")

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
