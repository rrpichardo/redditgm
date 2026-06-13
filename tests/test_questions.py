"""
tests/test_questions.py — Offline verification of the 4 guaranteed analytics questions.

No API keys required. Uses synthetic fixture data loaded into a throwaway DuckDB.
Tests the same SQL logic the ask.py router is prompted to generate.

Run:
  .venv311/bin/pytest tests/test_questions.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EVIDENCE_CSV = FIXTURES_DIR / "evidence_fixture.csv"
LABELS_CSV = FIXTURES_DIR / "labels_fixture.csv"


@pytest.fixture
def db():
    """Throwaway in-memory DuckDB loaded with fixture data. No files written."""
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE evidence_units (
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
        CREATE TABLE labels (
            evidence_id    VARCHAR PRIMARY KEY,
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

    # Load evidence rows from CSV
    with EVIDENCE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            con.execute("""
                INSERT INTO evidence_units VALUES (?, ?, ?, ?, ?, ?, ?, TRY_CAST(? AS TIMESTAMP), ?, ?, ?, TRY_CAST(? AS INTEGER))
            """, [
                row["evidence_id"], row["source_type"], row["run_id"],
                row["subreddit"], row["post_id"], row["comment_id"],
                row["author"], row["created_at"], row["title"],
                row["text"], row["permalink"], row["score"],
            ])

    # Load label rows from CSV
    with LABELS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            con.execute("""
                INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["evidence_id"], row["brand"], row["model"], row["powertrain"],
                row["is_pain_point"].lower() == "true",
                row["pain_theme"] or None,
                row["is_delight"].lower() == "true",
                row["delight_theme"] or None,
                row["sentiment"],
                float(row["confidence"]),
            ])

    yield con
    con.close()


# ---------------------------------------------------------------------------
# Q1: "How many unique authors complained about Silverado pain points?"
# Tests: COUNT(DISTINCT), filter on model + is_pain_point, exclude [deleted]
# Expected: 10 distinct non-deleted authors across all Silverado pain posts
# ---------------------------------------------------------------------------
def test_q1_unique_silverado_pain_authors(db):
    result = db.execute("""
        SELECT COUNT(DISTINCT e.author) AS unique_authors
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.model = 'Silverado'
          AND l.is_pain_point = TRUE
          AND e.author != '[deleted]'
    """).fetchone()

    # Fixture has: user_alice(x2), user_bob, user_grace, user_henry, user_iris,
    #              user_jack, user_kate, user_leo, user_peter, user_quinn = 10 unique
    assert result[0] == 10, f"Expected 10 unique Silverado pain authors, got {result[0]}"


# ---------------------------------------------------------------------------
# Q2: "What share of GMC posts are negative?"
# Tests: conditional aggregation / percentage by brand, sentiment grouping
# Expected: 40% of GMC posts are negative (2 negative out of 5 GMC posts)
# ---------------------------------------------------------------------------
def test_q2_gmc_negative_share(db):
    result = db.execute("""
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN l.sentiment = 'negative' THEN 1 ELSE 0 END)
                / COUNT(*),
                1
            ) AS negative_pct
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.brand = 'GMC'
    """).fetchone()

    # Fixture has 5 GMC rows: 2 negative, 2 positive, 1 neutral → 40.0%
    assert result[0] == pytest.approx(40.0, abs=0.1), (
        f"Expected 40.0% GMC negative, got {result[0]}"
    )


# ---------------------------------------------------------------------------
# Q3: "Top 3 pain-point themes for Chevy"
# Tests: GROUP BY theme, ORDER BY count DESC, LIMIT 3
# Expected: transmission(3) > reliability(2) > battery_range(1)
# Uses secondary sort (pain_theme ASC) for determinism at count=1 ties.
# ---------------------------------------------------------------------------
def test_q3_top3_chevy_pain_themes(db):
    rows = db.execute("""
        SELECT l.pain_theme, COUNT(*) AS cnt
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.brand = 'Chevy'
          AND l.is_pain_point = TRUE
          AND l.pain_theme IS NOT NULL
        GROUP BY l.pain_theme
        ORDER BY cnt DESC, l.pain_theme ASC
        LIMIT 3
    """).fetchall()

    assert len(rows) == 3, f"Expected 3 theme rows, got {len(rows)}"

    themes = [r[0] for r in rows]
    counts = [r[1] for r in rows]

    # Top theme must be transmission (count=3)
    assert themes[0] == "transmission", f"Expected 'transmission' first, got '{themes[0]}'"
    assert counts[0] == 3

    # Second must be reliability (count=2)
    assert themes[1] == "reliability", f"Expected 'reliability' second, got '{themes[1]}'"
    assert counts[1] == 2

    # Third is one of the count=1 themes (battery_range first alphabetically)
    assert counts[2] == 1
    assert themes[2] in {"battery_range", "build_quality", "charging", "dealer_service"}


# ---------------------------------------------------------------------------
# Q4: "EV vs ICE pain-point counts"
# Tests: GROUP BY powertrain WHERE is_pain_point, open-ended comparison
# Expected: EV=3, ICE=14
# ---------------------------------------------------------------------------
def test_q4_ev_vs_ice_pain_counts(db):
    rows = db.execute("""
        SELECT l.powertrain, COUNT(*) AS cnt
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE l.is_pain_point = TRUE
          AND l.powertrain IN ('EV', 'ICE', 'PHEV')
        GROUP BY l.powertrain
        ORDER BY cnt DESC
    """).fetchall()

    pain_by_powertrain = {r[0]: r[1] for r in rows}

    # Fixture: 3 EV pain posts (post_015, 016, 017), 14 ICE pain posts
    assert pain_by_powertrain.get("EV") == 3, (
        f"Expected EV pain count=3, got {pain_by_powertrain.get('EV')}"
    )
    assert pain_by_powertrain.get("ICE") == 14, (
        f"Expected ICE pain count=14, got {pain_by_powertrain.get('ICE')}"
    )

    # ICE should have more pain points than EV in the fixture
    assert pain_by_powertrain["ICE"] > pain_by_powertrain["EV"]


def test_sql_guard_rejects_file_reads_and_unknown_tables():
    from ask import is_safe_sql

    assert is_safe_sql("""
        SELECT COUNT(*)
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
    """)
    assert is_safe_sql("""
        WITH pain AS (
            SELECT evidence_id FROM labels WHERE is_pain_point = TRUE
        )
        SELECT COUNT(*) FROM pain
    """)
    assert not is_safe_sql("SELECT * FROM read_csv_auto('.env')")
    assert not is_safe_sql("SELECT * FROM evidence_units; DROP TABLE labels")
    assert not is_safe_sql("SELECT * FROM private_table")


def test_combined_csv_ingestion_uses_distinct_post_and_comment_ids(tmp_path):
    from build_analytics_db import build_db

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    combined_path = data_dir / "gm_posts_with_comments.csv"
    fields = [
        "run_id",
        "downloaded_at",
        "source_tool",
        "category",
        "time_filter",
        "post_id",
        "post_subreddit",
        "post_title",
        "post_author",
        "post_score",
        "post_comment_count",
        "post_created_at",
        "post_type",
        "post_content_source",
        "post_content",
        "post_selftext",
        "post_outbound_url",
        "post_domain",
        "post_permalink",
        "comment_rank",
        "comment_id",
        "comment_author",
        "comment_score",
        "comment_body",
        "comment_reply_count",
        "comment_permalink",
    ]
    rows = [
        {
            "run_id": "run_1",
            "downloaded_at": "2026-06-01T00:00:00+00:00",
            "source_tool": "redditwarp",
            "category": "new",
            "time_filter": "",
            "post_id": "p1",
            "post_subreddit": "Silverado",
            "post_title": "Transmission issue",
            "post_author": "user_a",
            "post_score": "5",
            "post_comment_count": "2",
            "post_created_at": "2026-06-01T00:00:00+00:00",
            "post_type": "self",
            "post_content_source": "selftext",
            "post_content": "Truck is slipping.",
            "post_selftext": "Truck is slipping.",
            "post_outbound_url": "",
            "post_domain": "",
            "post_permalink": "/r/Silverado/comments/p1/",
            "comment_rank": "1",
            "comment_id": "c1",
            "comment_author": "user_b",
            "comment_score": "2",
            "comment_body": "Same issue here.",
            "comment_reply_count": "0",
            "comment_permalink": "/r/Silverado/comments/p1/c1/",
        },
        {
            "run_id": "run_1",
            "downloaded_at": "2026-06-01T00:00:00+00:00",
            "source_tool": "redditwarp",
            "category": "new",
            "time_filter": "",
            "post_id": "p1",
            "post_subreddit": "Silverado",
            "post_title": "Transmission issue",
            "post_author": "user_a",
            "post_score": "5",
            "post_comment_count": "2",
            "post_created_at": "2026-06-01T00:00:00+00:00",
            "post_type": "self",
            "post_content_source": "selftext",
            "post_content": "Truck is slipping.",
            "post_selftext": "Truck is slipping.",
            "post_outbound_url": "",
            "post_domain": "",
            "post_permalink": "/r/Silverado/comments/p1/",
            "comment_rank": "2",
            "comment_id": "c2",
            "comment_author": "user_c",
            "comment_score": "1",
            "comment_body": "Dealer fixed mine.",
            "comment_reply_count": "0",
            "comment_permalink": "/r/Silverado/comments/p1/c2/",
        },
        {
            "run_id": "run_1",
            "downloaded_at": "2026-06-01T00:00:00+00:00",
            "source_tool": "redditwarp",
            "category": "new",
            "time_filter": "",
            "post_id": "p2",
            "post_subreddit": "Chevy",
            "post_title": "No comments post",
            "post_author": "user_d",
            "post_score": "3",
            "post_comment_count": "0",
            "post_created_at": "2026-06-02T00:00:00+00:00",
            "post_type": "self",
            "post_content_source": "selftext",
            "post_content": "Just bought one.",
            "post_selftext": "Just bought one.",
            "post_outbound_url": "",
            "post_domain": "",
            "post_permalink": "/r/Chevy/comments/p2/",
            "comment_rank": "",
            "comment_id": "",
            "comment_author": "",
            "comment_score": "",
            "comment_body": "",
            "comment_reply_count": "",
            "comment_permalink": "",
        },
    ]
    with combined_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = build_db(data_dir, tmp_path / "analytics.duckdb", reset=True)

    assert counts["mode"] == "combined"
    assert counts["input_posts"] == 2
    assert counts["input_comments"] == 2
    assert counts["posts"] == 2
    assert counts["comments"] == 2


def _combined_upload_row(post_id: str, comment_id: str = "", *, title: str | None = None) -> dict[str, str]:
    return {
        "run_id": "run_1",
        "post_id": post_id,
        "post_subreddit": "Silverado",
        "post_title": title or f"Post {post_id}",
        "post_author": f"author_{post_id}",
        "post_score": "5",
        "post_created_at": "2026-06-01T00:00:00+00:00",
        "post_content": f"Content for {post_id}",
        "post_selftext": f"Content for {post_id}",
        "post_permalink": f"/r/Silverado/comments/{post_id}/",
        "comment_rank": "1" if comment_id else "",
        "comment_id": comment_id,
        "comment_author": f"author_{comment_id}" if comment_id else "",
        "comment_score": "2" if comment_id else "",
        "comment_body": f"Comment {comment_id}" if comment_id else "",
        "comment_permalink": f"/r/Silverado/comments/{post_id}/{comment_id}/" if comment_id else "",
    }


def test_cumulative_combined_upload_dedupes_overlapping_windows(tmp_path):
    from build_analytics_db import build_db
    from csv_store import merge_upload_frames

    data_dir = tmp_path / "data"
    db_path = tmp_path / "analytics.duckdb"

    first_window = pd.DataFrame([
        _combined_upload_row("p1", "c1"),
        _combined_upload_row("p2"),
    ])
    first_stats = merge_upload_frames(
        {"combined": ("day_1.csv", b"", first_window)},
        data_dir,
        reset=False,
    )
    first_counts = build_db(data_dir, db_path, reset=True)

    second_window = pd.DataFrame([
        _combined_upload_row("p1", "c1", title="Post p1 updated"),
        _combined_upload_row("p1", "c2"),
        _combined_upload_row("p3"),
    ])
    second_stats = merge_upload_frames(
        {"combined": ("last_3_days.csv", b"", second_window)},
        data_dir,
        reset=False,
    )
    second_counts = build_db(data_dir, db_path, reset=False)
    stored = pd.read_csv(data_dir / "gm_posts_with_comments.csv", dtype=str, keep_default_na=False)

    assert first_stats[0]["stored_rows"] == 2
    assert first_counts["posts"] == 2
    assert first_counts["comments"] == 1

    assert second_stats[0]["existing_rows"] == 2
    assert second_stats[0]["incoming_rows"] == 3
    assert second_stats[0]["stored_rows"] == 4
    assert second_stats[0]["duplicate_rows_removed"] == 1
    assert len(stored) == 4
    assert second_counts["posts"] == 3
    assert second_counts["comments"] == 2


def test_lab_style_model_names_route_to_openrouter_by_default():
    from rag_core import _provider_for_model
    from settings import load_settings

    settings = load_settings()
    settings["generation_provider"] = "auto"

    assert _provider_for_model("llama-4-scout", settings) == "openrouter"
    assert _provider_for_model("gpt-oss-120b", settings) == "openrouter"
    assert _provider_for_model("google/gemma-4-31b-it", settings) == "openrouter"
