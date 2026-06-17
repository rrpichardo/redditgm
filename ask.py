"""
ask.py - Natural-language question -> DuckDB SQL and/or RAG evidence.

The reusable service entrypoint is answer(question, ...). The CLI is only a
thin rendering wrapper around that function.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from rag_core import chat, load_index, make_context, retrieve
from run_config import RunPaths
from settings import load_settings

console = Console()


def schema_description(settings: dict[str, Any] | None = None) -> str:
    loaded = settings or load_settings()
    pain_themes = " | ".join(loaded["taxonomy"]["pain"])
    return f"""
Tables in the DuckDB database:

evidence_units (
    evidence_id  VARCHAR,   -- PK; format 'post_<id>' or 'comment_<id>'
    source_type  VARCHAR,   -- 'post' or 'comment'
    run_id       VARCHAR,
    subreddit    VARCHAR,
    post_id      VARCHAR,
    comment_id   VARCHAR,
    author       VARCHAR,   -- '[deleted]' means removed
    created_at   TIMESTAMP,
    title        VARCHAR,
    text         VARCHAR,
    permalink    VARCHAR,
    score        INTEGER
)

labels (
    evidence_id    VARCHAR,  -- FK to evidence_units
    brand          VARCHAR,  -- 'Chevy', 'GMC', 'Cadillac', 'Buick', 'GM', 'unknown'
    model          VARCHAR,  -- 'Silverado', 'Tahoe', 'Sierra', etc.
    powertrain     VARCHAR,  -- 'EV', 'ICE', 'PHEV', 'unknown'
    is_pain_point  BOOLEAN,
    pain_theme     VARCHAR,
    is_delight     BOOLEAN,
    delight_theme  VARCHAR,
    sentiment      VARCHAR,  -- 'positive', 'negative', 'neutral', 'mixed'
    confidence     FLOAT
)

PAIN THEMES: {pain_themes}

Rules you MUST follow when writing SQL:
- Always JOIN labels l ON e.evidence_id = l.evidence_id when label fields are needed
- To count unique people, use COUNT(DISTINCT e.author)
- Filter out deleted authors when counting people: e.author != '[deleted]'
- Use only evidence_units and labels; do not read files or external tables
- Use only columns listed above; do not invent column names
- Output only the SQL query, no explanation, no markdown fences
"""


@dataclass
class AnswerResult:
    question: str
    intent: str
    answer_text: str
    sql: str | None = None
    rows: list[tuple] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sql_system_prompt(settings: dict[str, Any]) -> str:
    return settings["prompts"]["sql"].format(schema=schema_description(settings))


def classify_intent(question: str, settings: dict[str, Any] | None = None) -> str:
    loaded = settings or load_settings()
    raw = chat(
        messages=[
            {"role": "system", "content": loaded["prompts"]["router"]},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=10,
    ).strip().upper()
    if raw not in ("COUNT", "QUALITATIVE", "BOTH"):
        return "BOTH"
    return raw


def generate_sql(question: str, settings: dict[str, Any] | None = None) -> str:
    loaded = settings or load_settings()
    raw = chat(
        messages=[
            {"role": "system", "content": _sql_system_prompt(loaded)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=400,
    ).strip()
    return re.sub(r"```sql|```", "", raw).strip()


def _strip_sql_comments(sql: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--.*?$", " ", without_block, flags=re.MULTILINE)


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"('([^']|'')*'|\"([^\"]|\"\")*\")", "''", sql)


def unsafe_sql_reason(sql: str) -> str | None:
    """Return None when SQL is allowed, otherwise explain the rejection."""
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        return "empty SQL"

    if re.search(r";\s*\S", cleaned.rstrip()):
        return "multiple statements are not allowed"
    cleaned = cleaned.rstrip(";").strip()
    scrubbed = _strip_string_literals(cleaned)

    first_word = scrubbed.split()[0].upper() if scrubbed.split() else ""
    if first_word not in {"SELECT", "WITH"}:
        return "only SELECT queries are allowed"

    blocked_terms = [
        "ATTACH", "CALL", "COPY", "CREATE", "DELETE", "DETACH", "DROP", "EXPORT",
        "IMPORT", "INSERT", "INSTALL", "LOAD", "PRAGMA", "SET", "UPDATE",
    ]
    blocked_functions = [
        "FROM_CSV_AUTO", "GLOB", "PARQUET_SCAN", "POSTGRES_SCAN", "QUERY_TABLE",
        "READ_BLOB", "READ_CSV", "READ_CSV_AUTO", "READ_JSON", "READ_PARQUET",
        "READ_TEXT", "SQLITE_SCAN",
    ]
    for term in blocked_terms + blocked_functions:
        if re.search(rf"\b{term}\b", scrubbed, flags=re.IGNORECASE):
            return f"blocked SQL term: {term.lower()}"

    if re.search(r"\bFROM\s+['\"]", scrubbed, flags=re.IGNORECASE):
        return "file paths are not allowed in FROM clauses"

    cte_names = {
        match.group(1).lower()
        for match in re.finditer(r"(?:WITH|,)\s+([A-Za-z_]\w*)\s+AS\s*\(", scrubbed, re.IGNORECASE)
    }
    allowed_tables = {"evidence_units", "labels"} | cte_names
    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)",
        scrubbed,
        flags=re.IGNORECASE,
    )
    for ref in table_refs:
        table = ref.split(".")[-1].lower()
        if table not in allowed_tables:
            return f"table is not allowed: {ref}"

    if re.search(r"\bFROM\b", scrubbed, flags=re.IGNORECASE) and not table_refs:
        return "could not verify FROM clause"

    return None


def is_safe_sql(sql: str) -> bool:
    return unsafe_sql_reason(sql) is None


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[tuple], list[str]]:
    result = con.execute(sql)
    rows = result.fetchall()
    cols = [d[0] for d in result.description]
    return rows, cols


def sql_rows_to_text(rows: list[tuple], cols: list[str]) -> str:
    if not rows:
        return "No data found."
    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)


def qualitative_answer(question: str, context: str, settings: dict[str, Any]) -> str:
    return chat(
        messages=[
            {"role": "system", "content": settings["prompts"]["answer"]},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
        temperature=settings["temperature"],
        max_tokens=600,
    )


def full_answer(question: str, sql_summary: str, context: str, settings: dict[str, Any]) -> str:
    return chat(
        messages=[
            {"role": "system", "content": settings["prompts"]["answer"]},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"SQL Results:\n{sql_summary}\n\n"
                f"Reddit Evidence:\n{context}"
            )},
        ],
        temperature=settings["temperature"],
        max_tokens=800,
    )


def answer(
    question: str,
    *,
    tag: str | None = None,
    db_path: str | Path | None = None,
    index_path: str | Path | None = None,
    top_k: int | None = None,
) -> AnswerResult:
    settings = load_settings()
    run = RunPaths.resolve(tag, db_path=db_path, index_path=index_path, settings=settings)
    resolved_top_k = top_k if top_k is not None else int(settings["default_ask_top_k"])
    intent = classify_intent(question, settings)

    sql = None
    rows: list[tuple] = []
    cols: list[str] = []
    sql_summary = ""
    context = ""
    retrieved: list[dict[str, Any]] = []
    warnings: list[str] = []

    if intent in ("COUNT", "BOTH"):
        if not run.db_path.exists():
            raise FileNotFoundError(f"Database not found at {run.db_path}. Run build_analytics_db.py first.")
        sql = generate_sql(question, settings)
        reason = unsafe_sql_reason(sql)
        if reason:
            raise ValueError(f"Guardrail rejected SQL: {reason}\n{sql}")

        con = duckdb.connect(str(run.db_path), read_only=True)
        try:
            rows, cols = run_sql(con, sql)
        finally:
            con.close()
        sql_summary = sql_rows_to_text(rows, cols)

    if intent in ("QUALITATIVE", "BOTH"):
        index_file = run.index_path / "index.faiss"
        chunks_file = run.index_path / "chunks.pkl"
        if not index_file.exists() or not chunks_file.exists():
            warnings.append(f"RAG index not found at {run.index_path}; no retrieved evidence available.")
        else:
            index, chunks = load_index(run.index_path)
            retrieved = retrieve(question, index, chunks, top_k=resolved_top_k)
            context = make_context(retrieved)
            if not context:
                warnings.append("RAG index returned no matching evidence.")

    if intent == "QUALITATIVE" and not context:
        answer_text = "No retrieved Reddit evidence is available for that question yet. Build the RAG index first, then ask again."
    elif intent == "COUNT" and not context:
        answer_text = full_answer(question, sql_summary, "", settings)
    elif intent == "BOTH" and not context:
        no_evidence = "No retrieved Reddit evidence is available. Do not add narrative examples."
        answer_text = full_answer(question, sql_summary, no_evidence, settings)
    elif intent == "QUALITATIVE":
        answer_text = qualitative_answer(question, context, settings)
    else:
        answer_text = full_answer(question, sql_summary, context, settings)

    sources = [
        {
            "subreddit": chunk.get("subreddit", ""),
            "post_id": chunk.get("post_id", ""),
            "permalink": chunk.get("permalink", ""),
            "title": chunk.get("title", ""),
        }
        for chunk in retrieved
    ]

    return AnswerResult(
        question=question,
        intent=intent,
        answer_text=answer_text,
        sql=sql,
        rows=rows,
        cols=cols,
        sources=sources,
        warnings=warnings,
    )


def display_sql_results(sql: str, rows: list[tuple], cols: list[str]) -> None:
    console.print(Syntax(sql, "sql", theme="monokai", word_wrap=True))
    if not rows:
        console.print("[dim]Query returned no rows.[/]")
        return

    table = Table(border_style="dim", show_header=True, header_style="bold cyan")
    for col in cols:
        table.add_column(col)
    for row in rows[:20]:
        table.add_row(*[str(v) for v in row])
    console.print(table)


def main() -> None:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Ask a plain-English question about GM Reddit data.")
    p.add_argument("question", nargs="?", help="Your question in plain English")
    p.add_argument("--tag", default=settings["active_tag"], help="Run/dataset tag")
    p.add_argument("--db-path", default=None)
    p.add_argument("--index-path", default=None)
    p.add_argument("--top-k", type=int, default=settings["default_ask_top_k"], help="RAG chunks to retrieve")
    args = p.parse_args()


    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics - Ask[/]\n"
        f"[white]{args.question}[/]",
        border_style="cyan",
    ))

    try:
        result = answer(
            args.question,
            tag=args.tag,
            db_path=args.db_path,
            index_path=args.index_path,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]x[/] {exc}")
        sys.exit(1)

    console.print(f"[dim]Intent:[/] [bold]{result.intent}[/]\n")
    if result.sql:
        display_sql_results(result.sql, result.rows, result.cols)
        console.print()

    for warning in result.warnings:
        console.print(f"[yellow]![/] {warning}")

    console.print(Panel(
        Markdown(result.answer_text),
        title="[bold green]Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))

    if result.sources:
        source_table = Table(title="Sources", border_style="dim", show_header=True)
        source_table.add_column("Subreddit", style="cyan")
        source_table.add_column("Permalink", style="dim")
        for source in result.sources[:5]:
            source_table.add_row(
                f"r/{source.get('subreddit', '')}",
                source.get("permalink", ""),
            )
        console.print(source_table)


def main() -> None:
    p = argparse.ArgumentParser(description="Ask a plain-English question about GM Reddit data.")
    p.add_argument("question", nargs="?", help="Your question in plain English")
    p.add_argument("--tag", default="gm_vehicle_on_demand", help="Analysis run tag")
    args = p.parse_args()

    if not args.question:
        console.print("[yellow]Usage:[/] python ask.py \"Your question here\"")
        sys.exit(1)

    run = Run(args.tag)

    console.print(Panel.fit(
        f"[bold cyan]GM Reddit Analytics — Ask[/]\n[white]{args.question}[/]",
        border_style="cyan",
    ))

    try:
        with console.status("[cyan]Analyzing...[/]"):
            result = answer(args.question, run)
        _display_result(result)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/] {exc}")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[red]✗ Guardrail:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
