"""
ask.py — Natural-language question → DuckDB SQL (text-to-SQL) + RAG evidence.

Library API:
  from ask import answer, AnswerResult
  result = answer("How many authors...", Run("gm"))

CLI:
  .venv311/bin/python ask.py "How many unique authors complained about Silverado pain points?"
  .venv311/bin/python ask.py --tag gm "What are people saying about EV range?"
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from rag_core import chat, load_index, retrieve, make_context
from run_config import Run

console = Console()

# Regex of DuckDB functions that can read from disk or shell — blocked by guardrail
_DANGEROUS_FN = re.compile(
    r'\b(read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|'
    r'read_text|read_blob|glob|from_csv_auto|parquet_scan|scan_csv|copy)\b',
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    """Allow only SELECT statements that don't use DuckDB file/system functions."""
    stripped = sql.strip()
    if not stripped:
        return False
    first_word = stripped.split()[0].upper()
    if first_word != "SELECT":
        return False
    if _DANGEROUS_FN.search(stripped):
        return False
    return True


def classify_intent(question: str) -> str:
    # Prompts come from settings, not module-level constants
    from settings import get_settings
    raw = chat(
        messages=[
            {"role": "system", "content": get_settings().router_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=10,
    ).strip().upper()
    if raw not in ("COUNT", "QUALITATIVE", "BOTH"):
        return "BOTH"
    return raw


def generate_sql(question: str) -> str:
    from settings import get_settings
    raw = chat(
        messages=[
            {"role": "system", "content": get_settings().sql_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=400,
    ).strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"```sql|```", "", raw).strip()
    return raw


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list, list[str]]:
    """Execute SQL and return (rows, column_names)."""
    result = con.execute(sql)
    rows = result.fetchall()
    cols = [d[0] for d in result.description]
    return rows, cols


def qualitative_answer(question: str, context: str) -> str:
    from settings import get_settings
    return chat(
        messages=[
            {"role": "system", "content": get_settings().answer_prompt},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
        temperature=0.3,
        max_tokens=600,
    )


def full_answer(question: str, sql_summary: str, context: str) -> str:
    from settings import get_settings
    return chat(
        messages=[
            {"role": "system", "content": get_settings().answer_prompt},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"SQL Results:\n{sql_summary}\n\n"
                f"Reddit Evidence:\n{context}"
            )},
        ],
        temperature=0.3,
        max_tokens=800,
    )


def _rows_to_summary(rows: Optional[list], cols: Optional[list[str]]) -> str:
    """Convert SQL result rows into a compact text table for the LLM."""
    if not rows or not cols:
        return "No data found."
    lines = [" | ".join(cols)]
    lines.extend(" | ".join(str(v) for v in row) for row in rows)
    return "\n".join(lines)


@dataclass
class AnswerResult:
    question: str
    intent: str           # COUNT | QUALITATIVE | BOTH
    sql: Optional[str]
    rows: Optional[list]
    cols: Optional[list[str]]
    answer_text: str
    sources: list[dict] = field(default_factory=list)


def answer(question: str, run: Run) -> AnswerResult:
    """
    Route question through intent classifier → SQL and/or RAG → synthesize.
    Returns a structured AnswerResult for programmatic use (Streamlit, CLI, tests).
    Raises FileNotFoundError if the database is needed but missing.
    Raises ValueError if the generated SQL fails the safety guardrail.
    """
    from settings import get_settings
    s = get_settings()

    intent = classify_intent(question)
    sql: Optional[str] = None
    rows: Optional[list] = None
    cols: Optional[list[str]] = None
    context = ""
    sources: list[dict] = []

    # COUNT path: generate SQL and execute against DuckDB
    if intent in ("COUNT", "BOTH"):
        if not run.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {run.db_path}. Run build_analytics_db.py first."
            )
        sql = generate_sql(question)
        if not is_safe_sql(sql):
            raise ValueError(f"SQL guardrail rejected: {sql!r}")
        con = duckdb.connect(str(run.db_path), read_only=True)
        rows, cols = run_sql(con, sql)
        con.close()

    # QUALITATIVE path: FAISS retrieve → make_context
    if intent in ("QUALITATIVE", "BOTH"):
        if not run.index_path.exists():
            # Graceful fallback — no crash, just empty context
            context = ""
        else:
            index, chunks = load_index(run.index_path)
            retrieved = retrieve(question, index, chunks, top_k=s.ask_top_k)
            context = make_context(retrieved)
            sources = retrieved

    # Synthesize final answer based on what data we have
    if intent == "COUNT" and not context:
        answer_text = full_answer(question, _rows_to_summary(rows, cols), "")
    elif intent == "QUALITATIVE":
        if not context:
            answer_text = "No RAG index found. Run build_rag_index.py to enable evidence retrieval."
        else:
            answer_text = qualitative_answer(question, context)
    else:
        answer_text = full_answer(question, _rows_to_summary(rows, cols), context)

    return AnswerResult(
        question=question,
        intent=intent,
        sql=sql,
        rows=rows,
        cols=cols,
        answer_text=answer_text,
        sources=sources,
    )


def _display_result(result: AnswerResult) -> None:
    """Print an AnswerResult to the terminal using Rich."""
    console.print(f"[dim]Intent:[/] [bold]{result.intent}[/]\n")

    if result.sql:
        console.print(Panel("SQL Query", border_style="dim", padding=(0, 1)))
        console.print(Syntax(result.sql, "sql", theme="monokai", word_wrap=True))

        if result.rows is not None:
            if not result.rows:
                console.print("[dim]Query returned no rows.[/]")
            else:
                table = Table(border_style="dim", show_header=True, header_style="bold cyan")
                for col in (result.cols or []):
                    table.add_column(col)
                for row in result.rows[:20]:  # cap display at 20 rows
                    table.add_row(*[str(v) for v in row])
                console.print(table)

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
        for chunk in result.sources[:5]:
            source_table.add_row(
                f"r/{chunk.get('subreddit', '')}",
                chunk.get("permalink", ""),
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
