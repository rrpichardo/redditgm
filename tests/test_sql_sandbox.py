"""Tests for ask.is_safe_sql() — blocks injection, allows safe queries."""
from __future__ import annotations

import pytest
from ask import is_safe_sql


# --- Queries that must be BLOCKED ---

@pytest.mark.parametrize("sql", [
    # DuckDB file-reading functions
    "SELECT * FROM read_csv_auto('.env')",
    "SELECT * FROM read_csv('.env', header=true)",
    "SELECT * FROM read_parquet('secrets.parquet')",
    "SELECT * FROM read_json_auto('config.json')",
    "SELECT * FROM read_text('/etc/passwd')",
    "SELECT * FROM glob('**/*.env')",
    "SELECT * FROM parquet_scan('data.parquet')",
    "SELECT * FROM scan_csv('data.csv')",
    # Non-SELECT statements
    "DROP TABLE evidence_units",
    "DELETE FROM labels",
    "UPDATE labels SET confidence=1.0",
    "INSERT INTO labels VALUES ('x','y','z',false,null,false,null,'neutral',1.0)",
    # COPY exfiltration
    "COPY (SELECT * FROM labels) TO '/tmp/leak.csv'",
    # Mixed case / whitespace tricks
    "  select * FROM read_csv_auto('secrets')",
])
def test_dangerous_sql_blocked(sql: str):
    assert is_safe_sql(sql) is False, f"Expected BLOCKED but got ALLOWED: {sql!r}"


# --- Queries that must be ALLOWED ---

@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM evidence_units",
    "SELECT l.model, COUNT(*) FROM evidence_units e JOIN labels l ON e.evidence_id = l.evidence_id WHERE l.is_pain_point = TRUE GROUP BY l.model ORDER BY 2 DESC LIMIT 10",
    "SELECT COUNT(DISTINCT e.author) FROM evidence_units e JOIN labels l ON e.evidence_id = l.evidence_id WHERE l.model = 'Silverado' AND e.author != '[deleted]'",
])
def test_safe_sql_allowed(sql: str):
    assert is_safe_sql(sql) is True, f"Expected ALLOWED but got BLOCKED: {sql!r}"
