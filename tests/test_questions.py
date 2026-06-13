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
