"""
build_analytics_db.py - Load Reddit CSV exports into DuckDB.

Accepts either:
  - split gm_posts.csv + gm_comments.csv
  - combined gm_posts_with_comments.csv

Evidence counts are always derived from distinct post/comment IDs in the input,
never from fixed planning numbers.

Run:
  .venv311/bin/python build_analytics_db.py --tag gm_vehicle_on_demand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from run_config import RunPaths
from settings import load_settings

console = Console()


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Load Reddit CSVs into DuckDB for analytics.")
    p.add_argument("--tag", default=settings["active_tag"], help="Dataset/run tag")
    p.add_argument("--data-dir", default=None, help="Directory containing Reddit CSV export files")
    p.add_argument("--db-path", default=None, help="Path to DuckDB database file")
    p.add_argument("--reset", action="store_true", help="Drop and rebuild from scratch")
    return p.parse_args()


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _create_tables(con: duckdb.DuckDBPyConnection) -> None:
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


def inspect_input_counts(data_dir: Path) -> dict[str, int | str]:
    """Return input mode and distinct source counts without loading labels."""
    posts_path = data_dir / "gm_posts.csv"
    comments_path = data_dir / "gm_comments.csv"
    combined_path = data_dir / "gm_posts_with_comments.csv"

    con = duckdb.connect(":memory:")
    try:
        if posts_path.exists():
            posts = con.execute(f"""
                SELECT COUNT(DISTINCT id)
                FROM read_csv_auto('{_sql_path(posts_path)}', header=true, ignore_errors=true)
                WHERE id IS NOT NULL AND id != ''
            """).fetchone()[0]
            comments = 0
            if comments_path.exists():
                comments = con.execute(f"""
                    SELECT COUNT(DISTINCT comment_id)
                    FROM read_csv_auto('{_sql_path(comments_path)}', header=true, ignore_errors=true)
                    WHERE comment_id IS NOT NULL AND comment_id != ''
                """).fetchone()[0]
            return {"mode": "split", "posts": posts, "comments": comments}

        if combined_path.exists():
            posts = con.execute(f"""
                SELECT COUNT(DISTINCT post_id)
                FROM read_csv_auto('{_sql_path(combined_path)}', header=true, ignore_errors=true)
                WHERE post_id IS NOT NULL AND post_id != ''
            """).fetchone()[0]
            comments = con.execute(f"""
                SELECT COUNT(DISTINCT comment_id)
                FROM read_csv_auto('{_sql_path(combined_path)}', header=true, ignore_errors=true)
                WHERE comment_id IS NOT NULL AND comment_id != ''
            """).fetchone()[0]
            return {"mode": "combined", "posts": posts, "comments": comments}
    finally:
        con.close()

    return {"mode": "missing", "posts": 0, "comments": 0}


def _load_split(con: duckdb.DuckDBPyConnection, posts_path: Path, comments_path: Path) -> None:
    posts_sql = _sql_path(posts_path)
    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT
            'post_' || id                 AS evidence_id,
            'post'                        AS source_type,
            run_id,
            subreddit,
            id                            AS post_id,
            ''                            AS comment_id,
            author,
            TRY_CAST(created_at AS TIMESTAMP) AS created_at,
            title,
            COALESCE(selftext, content, '') AS text,
            permalink,
            TRY_CAST(score AS INTEGER)
        FROM read_csv_auto('{posts_sql}', header=true, ignore_errors=true)
        WHERE id IS NOT NULL AND id != ''
    """)
    post_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='post'"
    ).fetchone()[0]

    if not comments_path.exists():
        return

    comments_sql = _sql_path(comments_path)
    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT
            'comment_' || c.comment_id AS evidence_id,
            'comment'                  AS source_type,
            c.run_id,
            c.post_subreddit           AS subreddit,
            c.post_id,
            c.comment_id,
            c.author,
            COALESCE(
                TRY_CAST(p.created_at AS TIMESTAMP),
                TRY_CAST(c.downloaded_at AS TIMESTAMP)
            ) AS created_at,
            c.post_title               AS title,
            c.body                     AS text,
            c.comment_permalink        AS permalink,
            TRY_CAST(c.score AS INTEGER)
        FROM read_csv_auto('{comments_sql}', header=true, ignore_errors=true) c
        LEFT JOIN read_csv_auto('{posts_sql}', header=true, ignore_errors=true) p
          ON c.post_id = p.id
        WHERE c.comment_id IS NOT NULL AND c.comment_id != ''
    """)


def _load_combined(con: duckdb.DuckDBPyConnection, combined_path: Path) -> None:
    combined_sql = _sql_path(combined_path)
    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        WITH ranked_posts AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY post_id
                    ORDER BY TRY_CAST(comment_rank AS INTEGER) NULLS LAST
                ) AS rn
            FROM read_csv_auto('{combined_sql}', header=true, ignore_errors=true)
            WHERE post_id IS NOT NULL AND post_id != ''
        )
        SELECT
            'post_' || post_id AS evidence_id,
            'post'             AS source_type,
            run_id,
            post_subreddit     AS subreddit,
            post_id,
            ''                 AS comment_id,
            post_author        AS author,
            TRY_CAST(post_created_at AS TIMESTAMP) AS created_at,
            post_title         AS title,
            COALESCE(post_selftext, post_content, '') AS text,
            post_permalink     AS permalink,
            TRY_CAST(post_score AS INTEGER) AS score
        FROM ranked_posts
        WHERE rn = 1
    """)

    con.execute(f"""
        INSERT OR IGNORE INTO evidence_units
        SELECT
            'comment_' || comment_id AS evidence_id,
            'comment'                AS source_type,
            run_id,
            post_subreddit           AS subreddit,
            post_id,
            comment_id,
            comment_author           AS author,
            TRY_CAST(post_created_at AS TIMESTAMP) AS created_at,
            post_title               AS title,
            comment_body             AS text,
            comment_permalink        AS permalink,
            TRY_CAST(comment_score AS INTEGER) AS score
        FROM read_csv_auto('{combined_sql}', header=true, ignore_errors=true)
        WHERE comment_id IS NOT NULL AND comment_id != ''
    """)


def build_db(data_dir: Path, db_path: Path, reset: bool = False) -> dict:
    """Load CSVs into DuckDB. Returns derived counts for display."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    posts_path = data_dir / "gm_posts.csv"
    comments_path = data_dir / "gm_comments.csv"
    combined_path = data_dir / "gm_posts_with_comments.csv"

    input_counts = inspect_input_counts(data_dir)
    if input_counts["mode"] == "missing":
        console.print(
            f"[red]x[/] Expected gm_posts.csv or gm_posts_with_comments.csv in {data_dir}"
        )
        sys.exit(1)

    con = duckdb.connect(str(db_path))
    if reset:
        con.execute("DROP TABLE IF EXISTS labels")
        con.execute("DROP TABLE IF EXISTS evidence_units")
    _create_tables(con)

    if posts_path.exists():
        _load_split(con, posts_path, comments_path)
    else:
        _load_combined(con, combined_path)

    post_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='post'"
    ).fetchone()[0]
    comment_count = con.execute(
        "SELECT COUNT(*) FROM evidence_units WHERE source_type='comment'"
    ).fetchone()[0]
    label_count = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    con.close()

    return {
        "mode": input_counts["mode"],
        "input_posts": input_counts["posts"],
        "input_comments": input_counts["comments"],
        "posts": post_count,
        "comments": comment_count,
        "labels": label_count,
    }


def main() -> None:
    args = parse_args()
    run = RunPaths.resolve(args.tag, data_dir=args.data_dir, db_path=args.db_path)

    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics - Build DB[/]\n"
        f"Tag: [dim]{run.tag}[/]\n"
        f"Source: [dim]{run.data_dir}[/]\n"
        f"Target: [dim]{run.db_path}[/]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Loading CSVs into DuckDB...", total=None)
        counts = build_db(run.data_dir, run.db_path, reset=args.reset)
        progress.update(task, completed=1, total=1, description="Done")

    table = Table(title="Evidence Units Loaded", border_style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    table.add_row("Input mode", str(counts["mode"]))
    table.add_row("Distinct posts in input", f"{counts['input_posts']:,}")
    table.add_row("Distinct comments in input", f"{counts['input_comments']:,}")
    table.add_row("Posts in DB", f"{counts['posts']:,}")
    table.add_row("Comments in DB", f"{counts['comments']:,}")
    table.add_row("Labels (existing)", f"{counts['labels']:,}")
    console.print(table)

    console.print(f"\n[green]+[/] Database ready at [bold]{run.db_path}[/]")
    console.print("[dim]Next: run classify_evidence.py to label evidence units[/]")


if __name__ == "__main__":
    main()
