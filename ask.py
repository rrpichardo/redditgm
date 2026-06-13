"""
ask.py — Natural-language question → DuckDB SQL (text-to-SQL) + RAG evidence.

Architecture:
  1. Router LLM: classifies intent as COUNT | QUALITATIVE | BOTH
  2. COUNT path: LLM writes DuckDB SQL → executed against labels + evidence_units
  3. QUALITATIVE path: FAISS retrieve → make_context → LLM synthesizes
  4. Answer = SQL numbers + RAG permalink evidence + caveats

Guardrails:
  - SELECT-only (rejects any non-read SQL)
  - Schema passed verbatim in prompt (no hallucinated columns)
  - LLM explains but never invents numbers

Run:
  .venv311/bin/python ask.py "How many unique authors complained about Silverado pain points?"
  .venv311/bin/python ask.py --tag gm_vehicle_on_demand "What are people saying about EV range?"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from rag_core import chat, load_index, retrieve, make_context

console = Console()
DB_PATH = Path("analytics/redditgm.duckdb")

# Schema passed verbatim to LLM — guardrail against column hallucination
SCHEMA_DESCRIPTION = """
Tables in the DuckDB database:

evidence_units (
    evidence_id  VARCHAR,   -- PK; format 'post_<id>' or 'comment_<id>'
    source_type  VARCHAR,   -- 'post' or 'comment'
    run_id       VARCHAR,
    subreddit    VARCHAR,   -- e.g. 'Silverado', 'GMC', 'Chevy'
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
    evidence_id    VARCHAR,  -- FK → evidence_units
    brand          VARCHAR,  -- 'Chevy', 'GMC', 'Cadillac', 'Buick', 'GM', 'unknown'
    model          VARCHAR,  -- 'Silverado', 'Tahoe', 'Sierra', etc.
    powertrain     VARCHAR,  -- 'EV', 'ICE', 'PHEV', 'unknown'
    is_pain_point  BOOLEAN,
    pain_theme     VARCHAR,  -- see THEME_ENUM below
    is_delight     BOOLEAN,
    delight_theme  VARCHAR,
    sentiment      VARCHAR,  -- 'positive', 'negative', 'neutral', 'mixed'
    confidence     FLOAT
)

THEME_ENUM: transmission | reliability | dealer_service | pricing | infotainment |
            battery_range | charging | build_quality | recall | warranty |
            performance | comfort | other

Rules you MUST follow when writing SQL:
- Always JOIN labels l ON e.evidence_id = l.evidence_id
- To count unique people, use COUNT(DISTINCT e.author)
- Always filter out deleted authors: WHERE e.author != '[deleted]'
- Use only columns listed above — do not invent column names
- Output only the SQL query, no explanation, no markdown fences
"""

ROUTER_SYSTEM = """You classify the intent of a question about Reddit vehicle data.

Output ONE of these three tokens only:
  COUNT       — the answer is a number (how many, what share, top N, compare counts)
  QUALITATIVE — the answer is narrative (what are people saying, summarize themes)
  BOTH        — needs a number AND narrative examples

Output only the token, nothing else."""

SQL_SYSTEM = f"""You write DuckDB SQL queries for a Reddit vehicle analytics database.

{SCHEMA_DESCRIPTION}

Output only the SQL query. No markdown. No explanation."""

ANSWER_SYSTEM = """You synthesize analytics results for a business audience.
Present findings clearly. Lead with the number, then interpret it, then cite evidence.
Be concise and honest — never invent numbers."""


def classify_intent(question: str) -> str:
    raw = chat(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=10,
    ).strip().upper()
    if raw not in ("COUNT", "QUALITATIVE", "BOTH"):
        return "BOTH"
    return raw


def generate_sql(question: str) -> str:
    raw = chat(
        messages=[
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=400,
    ).strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"```sql|```", "", raw).strip()
    return raw


def is_safe_sql(sql: str) -> bool:
    """Reject anything that isn't a SELECT statement (guardrail)."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word == "SELECT"


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list, list[str]]:
    """Execute SQL and return (rows, column_names)."""
    result = con.execute(sql)
    rows = result.fetchall()
    cols = [d[0] for d in result.description]
    return rows, cols


def display_sql_results(sql: str, rows: list, cols: list[str]) -> str:
    """Pretty-print SQL and results; return a text summary."""
    console.print(Syntax(sql, "sql", theme="monokai", word_wrap=True))

    if not rows:
        console.print("[dim]Query returned no rows.[/]")
        return "No data found."

    table = Table(border_style="dim", show_header=True, header_style="bold cyan")
    for col in cols:
        table.add_column(col)
    for row in rows[:20]:  # cap display at 20
        table.add_row(*[str(v) for v in row])
    console.print(table)

    # Build a compact text summary for the answer LLM
    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)


def qualitative_answer(question: str, context: str) -> str:
    return chat(
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
        temperature=0.3,
        max_tokens=600,
    )


def full_answer(question: str, sql_summary: str, context: str) -> str:
    return chat(
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"SQL Results:\n{sql_summary}\n\n"
                f"Reddit Evidence:\n{context}"
            )},
        ],
        temperature=0.3,
        max_tokens=800,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Ask a plain-English question about GM Reddit data.")
    p.add_argument("question", nargs="?", help="Your question in plain English")
    p.add_argument("--db-path", default=str(DB_PATH))
    p.add_argument("--tag", default="gm_vehicle_on_demand", help="RAG index tag")
    p.add_argument("--top-k", type=int, default=5, help="RAG chunks to retrieve")
    args = p.parse_args()

    if not args.question:
        console.print("[yellow]Usage:[/] python ask.py \"Your question here\"")
        sys.exit(1)

    db_path = Path(args.db_path)
    index_path = Path("state") / "rag" / args.tag
    question = args.question

    console.print(Panel.fit(
        f"[bold cyan]GM Reddit Analytics — Ask[/]\n"
        f"[white]{question}[/]",
        border_style="cyan"
    ))

    # Step 1: classify intent
    with console.status("[cyan]Analyzing question...[/]"):
        intent = classify_intent(question)
    console.print(f"[dim]Intent:[/] [bold]{intent}[/]\n")

    sql_summary = ""
    context = ""

    # Step 2a: COUNT path — text-to-SQL
    if intent in ("COUNT", "BOTH"):
        if not db_path.exists():
            console.print(f"[red]✗[/] Database not found at {db_path}. Run build_analytics_db.py first.")
            sys.exit(1)

        console.print(Panel("Generating SQL...", border_style="dim", padding=(0, 1)))
        with console.status("[cyan]Writing SQL...[/]"):
            sql = generate_sql(question)

        if not is_safe_sql(sql):
            console.print(f"[red]✗ Guardrail: rejected non-SELECT SQL:[/]\n{sql}")
            sys.exit(1)

        with console.status("[cyan]Executing query...[/]"):
            con = duckdb.connect(str(db_path), read_only=True)
            rows, cols = run_sql(con, sql)
            con.close()

        sql_summary = display_sql_results(sql, rows, cols)

    # Step 2b: QUALITATIVE path — RAG retrieval
    if intent in ("QUALITATIVE", "BOTH"):
        if not index_path.exists():
            console.print(f"[yellow]⚠[/] RAG index not found at {index_path}. Skipping evidence retrieval.")
        else:
            with console.status("[cyan]Retrieving evidence...[/]"):
                index, chunks = load_index(index_path)
                retrieved = retrieve(question, index, chunks, top_k=args.top_k)
                context = make_context(retrieved)

            console.print(f"[dim]Retrieved {len(retrieved)} evidence chunks[/]")

    # Step 3: synthesize answer
    console.print()
    with console.status("[cyan]Synthesizing answer...[/]"):
        if intent == "COUNT" and not context:
            answer = full_answer(question, sql_summary, "")
        elif intent == "QUALITATIVE":
            answer = qualitative_answer(question, context)
        else:
            answer = full_answer(question, sql_summary, context)

    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # Show source permalinks
    if context and retrieved:
        source_table = Table(title="Sources", border_style="dim", show_header=True)
        source_table.add_column("Subreddit", style="cyan")
        source_table.add_column("Permalink", style="dim")
        for chunk in retrieved[:5]:
            source_table.add_row(
                f"r/{chunk.get('subreddit', '')}",
                chunk.get("permalink", ""),
            )
        console.print(source_table)


if __name__ == "__main__":
    main()
