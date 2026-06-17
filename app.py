"""Streamlit app for redditgm analytics.

The app is intentionally run-centric: every count, filter, and chart is derived
from the selected run's DuckDB database or uploaded CSVs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ask import answer
from build_analytics_db import build_db, inspect_input_counts
from classify_evidence import estimate_classification, fetch_unlabeled
from csv_store import merge_upload_frames
from run_config import RunPaths
from settings import load_settings, save_settings


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

SAMPLE_DB_PATH = Path("analytics/redditgm_demo.duckdb")
PYTHON_BIN = Path(".venv311/bin/python") if Path(".venv311/bin/python").exists() else Path(sys.executable)
APP_TZ = ZoneInfo("America/Detroit")

MODEL_PRESETS = {
    "Custom": None,
    "Lab 2 - Llama 4 Scout (OpenRouter)": ("openrouter", "llama-4-scout"),
    "Lab 2/3 - GPT OSS 120B (OpenRouter)": ("openrouter", "gpt-oss-120b"),
    "Lab 2 - Gemma (OpenRouter)": ("openrouter", "google/gemma-4-31b-it"),
    "OpenAI direct - GPT-4o mini": ("openai", "gpt-4o-mini"),
    "Anthropic - Claude Haiku": ("anthropic", "claude-3-5-haiku-20241022"),
    "Lab 1 - Llama 4 Scout (Jetstream)": ("jetstream", "llama-4-scout"),
}


st.set_page_config(
    page_title="redditgm",
    page_icon="GM",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --rgm-ink: #171717;
        --rgm-muted: #60646c;
        --rgm-line: #d9dde3;
        --rgm-panel: #ffffff;
        --rgm-soft: #f6f7f9;
        --rgm-green: #168a5b;
        --rgm-red: #c2413b;
        --rgm-gold: #a36d12;
        --rgm-teal: #0f766e;
      }
      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }
      h1, h2, h3 {
        letter-spacing: 0;
      }
      div[data-testid="stMetric"] {
        background: var(--rgm-panel);
        border: 1px solid var(--rgm-line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
      }
      div[data-testid="stMetricLabel"] {
        color: var(--rgm-muted);
      }
      .rgm-status {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--rgm-line);
        border-radius: 999px;
        padding: 0.2rem 0.55rem;
        font-size: 0.78rem;
        color: var(--rgm-muted);
        background: var(--rgm-soft);
        margin-right: 0.35rem;
      }
      .rgm-ok {
        color: var(--rgm-green);
        border-color: rgba(22, 138, 91, 0.35);
        background: rgba(22, 138, 91, 0.08);
      }
      .rgm-warn {
        color: var(--rgm-gold);
        border-color: rgba(163, 109, 18, 0.35);
        background: rgba(163, 109, 18, 0.08);
      }
      .rgm-bad {
        color: var(--rgm-red);
        border-color: rgba(194, 65, 59, 0.35);
        background: rgba(194, 65, 59, 0.08);
      }
      .rgm-small {
        color: var(--rgm-muted);
        font-size: 0.86rem;
      }
      .rgm-panel {
        border: 1px solid var(--rgm-line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: var(--rgm-panel);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_int(value: Any) -> str:
    if value is None:
        return "0"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "0.0%"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def status_badge(label: str, state: str) -> str:
    css = {"done": "rgm-ok", "partial": "rgm-warn", "missing": "", "failed": "rgm-bad"}.get(state, "")
    return f'<span class="rgm-status {css}">{label}</span>'


def available_tags(settings: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    active = settings.get("active_tag")
    if active:
        tags.append(active)

    runtime_dir = Path(settings.get("runtime_dir", "runtime"))
    if runtime_dir.exists():
        for child in sorted(runtime_dir.iterdir()):
            if child.is_dir() and (child / "analytics.duckdb").exists():
                tags.append(child.name)

    if SAMPLE_DB_PATH.exists():
        tags.append("demo")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped or ["gm_vehicle_on_demand"]


def resolve_app_run(tag: str, settings: dict[str, Any]) -> RunPaths:
    if tag == "demo":
        return RunPaths(
            tag="demo",
            root=Path("reports/demo"),
            data_dir=Path("runtime/demo/data"),
            db_path=SAMPLE_DB_PATH,
            index_path=Path("runtime/demo/rag_index"),
            report_dir=Path("reports/demo/analytics"),
            state_file=Path("runtime/demo/state/seen_posts.json"),
            runs_dir=Path("runtime/demo/runs"),
        )
    return RunPaths.resolve(tag, settings=settings)


def open_db(run: RunPaths, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(run.db_path), read_only=read_only)


def fetch_df(run: RunPaths, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    if not run.db_path.exists():
        return pd.DataFrame()
    con = open_db(run)
    try:
        return con.execute(sql, params or []).fetchdf()
    finally:
        con.close()


def scalar(run: RunPaths, sql: str, params: list[Any] | None = None, default: Any = 0) -> Any:
    if not run.db_path.exists():
        return default
    con = open_db(run)
    try:
        row = con.execute(sql, params or []).fetchone()
    finally:
        con.close()
    if not row:
        return default
    return row[0]


def run_status(run: RunPaths) -> dict[str, Any]:
    exists = run.db_path.exists()
    evidence = scalar(run, "SELECT COUNT(*) FROM evidence_units", default=0) if exists else 0
    labels = scalar(run, "SELECT COUNT(*) FROM labels", default=0) if exists else 0
    index_done = (run.index_path / "index.faiss").exists() and (run.index_path / "chunks.pkl").exists()
    report_done = (run.report_dir / "dashboard.html").exists() or (run.report_dir / "report.md").exists()
    if not exists:
        classify_state = "missing"
    elif labels == 0:
        classify_state = "missing"
    elif labels < evidence:
        classify_state = "partial"
    else:
        classify_state = "done"
    return {
        "db": "done" if exists else "missing",
        "classify": classify_state,
        "index": "done" if index_done else "missing",
        "report": "done" if report_done else "missing",
        "evidence": evidence,
        "labels": labels,
    }


def distinct_values(run: RunPaths, expression: str, where: str = "") -> list[str]:
    if not run.db_path.exists():
        return []
    sql = f"""
        SELECT DISTINCT {expression} AS value
        FROM evidence_units e
        LEFT JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE {expression} IS NOT NULL
          AND CAST({expression} AS VARCHAR) != ''
          {where}
        ORDER BY value
    """
    df = fetch_df(run, sql)
    if df.empty:
        return []
    return [str(value) for value in df["value"].dropna().tolist()]


def date_bounds(run: RunPaths) -> tuple[date | None, date | None]:
    df = fetch_df(
        run,
        """
        SELECT
          MIN(CAST(created_at AS DATE)) AS min_date,
          MAX(CAST(created_at AS DATE)) AS max_date
        FROM evidence_units
        WHERE created_at IS NOT NULL
        """,
    )
    if df.empty or pd.isna(df.loc[0, "min_date"]) or pd.isna(df.loc[0, "max_date"]):
        return None, None
    return pd.to_datetime(df.loc[0, "min_date"]).date(), pd.to_datetime(df.loc[0, "max_date"]).date()


def add_in_filter(clauses: list[str], params: list[Any], expression: str, values: list[str]) -> None:
    if not values:
        return
    placeholders = ", ".join(["?"] * len(values))
    clauses.append(f"{expression} IN ({placeholders})")
    params.extend(values)


def build_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []

    add_in_filter(clauses, params, "l.brand", filters.get("brand", []))
    add_in_filter(clauses, params, "l.model", filters.get("model", []))
    add_in_filter(clauses, params, "e.subreddit", filters.get("subreddit", []))
    add_in_filter(clauses, params, "e.source_type", filters.get("source_type", []))
    add_in_filter(clauses, params, "l.sentiment", filters.get("sentiment", []))

    if filters.get("date_start") and filters.get("date_end"):
        clauses.append("CAST(e.created_at AS DATE) BETWEEN ? AND ?")
        params.extend([filters["date_start"], filters["date_end"]])

    confidence = filters.get("confidence")
    if confidence is not None:
        clauses.append("(l.evidence_id IS NULL OR l.confidence >= ?)")
        params.append(float(confidence))

    search = (filters.get("search") or "").strip()
    if search:
        clauses.append("(e.title ILIKE ? OR e.text ILIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    return " AND ".join(clauses), params


def filter_bar(run: RunPaths, settings: dict[str, Any]) -> dict[str, Any]:
    with st.sidebar:
        st.divider()
        st.caption("Filters")
        brand = st.multiselect("Brand", distinct_values(run, "l.brand", "AND l.brand != 'unknown'"))
        model = st.multiselect("Model", distinct_values(run, "l.model", "AND l.model != 'unknown'"))
        subreddit = st.multiselect("Subreddit", distinct_values(run, "e.subreddit"))
        source_type = st.multiselect("Source type", ["post", "comment"])
        sentiment = st.multiselect("Sentiment", distinct_values(run, "l.sentiment"))
        min_date, max_date = date_bounds(run)
        selected_dates: tuple[date, date] | tuple[()] = ()
        if min_date and max_date:
            selected_dates = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
        confidence = st.slider(
            "Confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.get("confidence_default", 0.5)),
            step=0.05,
        )
        search = st.text_input("Text search", "")

    date_start = date_end = None
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        date_start, date_end = selected_dates

    return {
        "brand": brand,
        "model": model,
        "subreddit": subreddit,
        "source_type": source_type,
        "sentiment": sentiment,
        "date_start": date_start,
        "date_end": date_end,
        "confidence": confidence,
        "search": search,
    }


def render_empty_run(run: RunPaths) -> None:
    st.info(f"No DuckDB database found for `{run.tag}` at `{run.db_path}`.")


def dashboard_page(run: RunPaths, settings: dict[str, Any]) -> None:
    st.title("GM Reddit Analytics")
    st.caption(f"Run `{run.tag}`")

    if not run.db_path.exists():
        render_empty_run(run)
        return

    filters = filter_bar(run, settings)
    where_sql, params = build_where(filters)

    summary = fetch_df(
        run,
        f"""
        SELECT
          COUNT(DISTINCT e.evidence_id) AS evidence_units,
          COUNT(DISTINCT l.evidence_id) AS labeled,
          SUM(CASE WHEN l.is_pain_point THEN 1 ELSE 0 END) AS pain_points,
          SUM(CASE WHEN l.is_delight THEN 1 ELSE 0 END) AS delights,
          COUNT(DISTINCT CASE WHEN l.is_pain_point THEN e.author END) AS pain_authors
        FROM evidence_units e
        LEFT JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE {where_sql}
        """,
        params,
    )
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    labeled = row.get("labeled") or 0
    pain = row.get("pain_points") or 0
    delight = row.get("delights") or 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Evidence units", fmt_int(row.get("evidence_units")))
    c2.metric("Labeled", fmt_int(labeled))
    c3.metric("Pain rate", fmt_pct((pain / labeled * 100) if labeled else 0))
    c4.metric("Delight rate", fmt_pct((delight / labeled * 100) if labeled else 0))
    c5.metric("Pain authors", fmt_int(row.get("pain_authors")))

    trend = fetch_df(
        run,
        f"""
        SELECT
          DATE_TRUNC('week', e.created_at) AS week,
          SUM(CASE WHEN l.is_pain_point THEN 1 ELSE 0 END) AS pain_points,
          SUM(CASE WHEN l.is_delight THEN 1 ELSE 0 END) AS delights,
          COUNT(l.evidence_id) AS labeled
        FROM evidence_units e
        LEFT JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE {where_sql}
          AND e.created_at IS NOT NULL
        GROUP BY week
        ORDER BY week
        """,
        params,
    )
    st.subheader("Movement")
    if trend.empty:
        st.info("No dated evidence matched the current filters.")
    else:
        trend["week"] = pd.to_datetime(trend["week"])
        st.line_chart(trend.set_index("week")[["pain_points", "delights", "labeled"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment")
        sentiment = fetch_df(
            run,
            f"""
            SELECT l.sentiment, COUNT(*) AS evidence_units
            FROM evidence_units e
            JOIN labels l ON e.evidence_id = l.evidence_id
            WHERE {where_sql}
              AND l.sentiment IS NOT NULL
            GROUP BY l.sentiment
            ORDER BY evidence_units DESC
            """,
            params,
        )
        if sentiment.empty:
            st.info("No labeled sentiment matched the filters.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.pie(
                sentiment["evidence_units"],
                labels=sentiment["sentiment"],
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops={"width": 0.45},
                colors=["#168a5b", "#c2413b", "#60646c", "#a36d12", "#0f766e"],
            )
            ax.set_title("Overall sentiment")
            st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Powertrain")
        powertrain = fetch_df(
            run,
            f"""
            SELECT
              l.powertrain,
              SUM(CASE WHEN l.is_pain_point THEN 1 ELSE 0 END) AS pain_points,
              SUM(CASE WHEN l.is_delight THEN 1 ELSE 0 END) AS delights
            FROM evidence_units e
            JOIN labels l ON e.evidence_id = l.evidence_id
            WHERE {where_sql}
              AND l.powertrain IN ('EV', 'ICE', 'PHEV')
            GROUP BY l.powertrain
            ORDER BY l.powertrain
            """,
            params,
        )
        if powertrain.empty:
            st.info("No EV/ICE/PHEV evidence matched the filters.")
        else:
            st.bar_chart(powertrain.set_index("powertrain")[["pain_points", "delights"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Pain themes")
        pain_themes = fetch_df(
            run,
            f"""
            SELECT l.pain_theme, COUNT(*) AS evidence_units
            FROM evidence_units e
            JOIN labels l ON e.evidence_id = l.evidence_id
            WHERE {where_sql}
              AND l.is_pain_point = TRUE
              AND l.pain_theme IS NOT NULL
            GROUP BY l.pain_theme
            ORDER BY evidence_units DESC
            LIMIT 12
            """,
            params,
        )
        if pain_themes.empty:
            st.info("No pain themes matched the filters.")
        else:
            st.bar_chart(pain_themes.set_index("pain_theme")["evidence_units"])

    with right:
        st.subheader("Delight themes")
        delight_themes = fetch_df(
            run,
            f"""
            SELECT l.delight_theme, COUNT(*) AS evidence_units
            FROM evidence_units e
            JOIN labels l ON e.evidence_id = l.evidence_id
            WHERE {where_sql}
              AND l.is_delight = TRUE
              AND l.delight_theme IS NOT NULL
            GROUP BY l.delight_theme
            ORDER BY evidence_units DESC
            LIMIT 12
            """,
            params,
        )
        if delight_themes.empty:
            st.info("No delight themes matched the filters.")
        else:
            st.bar_chart(delight_themes.set_index("delight_theme")["evidence_units"])

    st.subheader("Brand sentiment")
    brand_sentiment = fetch_df(
        run,
        f"""
        SELECT l.brand, l.sentiment, COUNT(*) AS evidence_units
        FROM evidence_units e
        JOIN labels l ON e.evidence_id = l.evidence_id
        WHERE {where_sql}
          AND l.brand NOT IN ('unknown', 'GM')
          AND l.sentiment IS NOT NULL
        GROUP BY l.brand, l.sentiment
        ORDER BY l.brand, l.sentiment
        """,
        params,
    )
    if brand_sentiment.empty:
        st.info("No brand sentiment matched the filters.")
    else:
        pivot = brand_sentiment.pivot_table(
            index="brand",
            columns="sentiment",
            values="evidence_units",
            aggfunc="sum",
            fill_value=0,
        )
        st.bar_chart(pivot)

    with st.expander("Matched evidence"):
        detail = fetch_df(
            run,
            f"""
            SELECT
              e.created_at,
              e.source_type,
              e.subreddit,
              l.brand,
              l.model,
              l.sentiment,
              l.is_pain_point,
              l.pain_theme,
              e.title,
              e.permalink
            FROM evidence_units e
            LEFT JOIN labels l ON e.evidence_id = l.evidence_id
            WHERE {where_sql}
            ORDER BY e.created_at DESC NULLS LAST
            LIMIT 100
            """,
            params,
        )
        st.dataframe(detail, width="stretch", hide_index=True)


def classify_upload(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    if {"post_id", "post_title"}.issubset(cols) and (
        "post_author" in cols or "post_created_at" in cols or "comment_body" in cols
    ):
        return "combined"
    if {"id", "title", "subreddit"}.issubset(cols):
        return "posts"
    if {"comment_id", "post_id"}.issubset(cols) and ("body" in cols or "comment_body" in cols):
        return "comments"
    return None


def uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str, keep_default_na=False)


def validate_uploads(uploaded_files) -> tuple[dict[str, tuple[str, bytes, pd.DataFrame]], list[dict[str, Any]], list[str], list[str]]:
    files: dict[str, tuple[str, bytes, pd.DataFrame]] = {}
    summary: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for uploaded in uploaded_files or []:
        try:
            df = uploaded_csv(uploaded)
        except Exception as exc:
            errors.append(f"{uploaded.name}: could not read CSV ({exc})")
            continue

        kind = classify_upload(df)
        if not kind:
            errors.append(f"{uploaded.name}: schema did not match combined, posts, or comments CSV.")
            continue

        if kind in files:
            errors.append(f"{uploaded.name}: another {kind} CSV was already uploaded.")
            continue

        files[kind] = (uploaded.name, uploaded.getvalue(), df)
        summary.append({"file": uploaded.name, "kind": kind, "rows": len(df), "columns": len(df.columns)})

    if "combined" in files and ("posts" in files or "comments" in files):
        errors.append("Upload either one combined CSV or split posts/comments CSVs, not both.")

    if "combined" not in files and "posts" not in files:
        errors.append("A combined CSV or posts CSV is required.")

    if "combined" in files:
        df = files["combined"][2]
        post_ids = df["post_id"].replace("", pd.NA).dropna()
        comment_ids = df["comment_id"].replace("", pd.NA).dropna() if "comment_id" in df else pd.Series(dtype=str)
        repeated_post_rows = max(len(post_ids) - post_ids.nunique(), 0)
        duplicate_comment_ids = int(comment_ids.duplicated().sum()) if not comment_ids.empty else 0
        summary.append({
            "file": "derived",
            "kind": "distinct posts",
            "rows": int(post_ids.nunique()),
            "columns": "",
        })
        summary.append({
            "file": "derived",
            "kind": "distinct comments",
            "rows": int(comment_ids.nunique()) if not comment_ids.empty else 0,
            "columns": "",
        })
        if repeated_post_rows:
            warnings.append(f"Combined CSV has {repeated_post_rows:,} repeated post rows from comment expansion; DB load deduplicates posts by post_id.")
        if duplicate_comment_ids:
            warnings.append(f"Combined CSV has {duplicate_comment_ids:,} duplicate comment IDs; DB load keeps distinct comment evidence IDs.")

    if "posts" in files:
        df = files["posts"][2]
        ids = df["id"].replace("", pd.NA).dropna()
        duplicate_posts = int(ids.duplicated().sum())
        summary.append({"file": "derived", "kind": "distinct posts", "rows": int(ids.nunique()), "columns": ""})
        if duplicate_posts:
            warnings.append(f"Posts CSV has {duplicate_posts:,} duplicate post IDs; DB load keeps one evidence row per post ID.")

    if "comments" in files:
        df = files["comments"][2]
        ids = df["comment_id"].replace("", pd.NA).dropna()
        duplicate_comments = int(ids.duplicated().sum())
        summary.append({"file": "derived", "kind": "distinct comments", "rows": int(ids.nunique()), "columns": ""})
        if duplicate_comments:
            warnings.append(f"Comments CSV has {duplicate_comments:,} duplicate comment IDs; DB load keeps one evidence row per comment ID.")

    return files, summary, errors, warnings


def read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def latest_run_summary(run: RunPaths) -> dict[str, Any] | None:
    if not run.runs_dir.exists():
        return None
    summaries = sorted(run.runs_dir.glob("*.json"))
    if not summaries:
        return None
    try:
        return json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def latest_run_id(run: RunPaths, posts: pd.DataFrame) -> str:
    summary = latest_run_summary(run)
    if summary and summary.get("run_id"):
        return str(summary["run_id"])
    if "run_id" in posts and not posts.empty:
        values = posts["run_id"].replace("", pd.NA).dropna()
        if not values.empty:
            return str(values.iloc[-1])
    return ""


def frame_for_run(df: pd.DataFrame, run_id_value: str) -> pd.DataFrame:
    if df.empty or not run_id_value or "run_id" not in df:
        return df
    return df[df["run_id"].astype(str) == run_id_value]


def date_range_text(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "-"
    dates = pd.to_datetime(df[column], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return "-"
    return f"{dates.min().date()} to {dates.max().date()}"


def subreddit_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df:
        return pd.DataFrame(columns=["subreddit", "posts"])
    return (
        df[column]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("subreddit")
        .reset_index(name="posts")
    )


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_csv_button(label: str, df: pd.DataFrame, file_name: str, key: str) -> None:
    st.download_button(
        label,
        data=csv_bytes(df) if not df.empty else b"",
        file_name=file_name,
        mime="text/csv",
        disabled=df.empty,
        key=key,
        use_container_width=True,
    )


def preview_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        "created_at",
        "subreddit",
        "title",
        "author",
        "score",
        "comment_count",
        "content_source",
        "content",
        "permalink",
    ]
    existing = [column for column in columns if column in df.columns]
    preview = df.loc[:, existing].copy()
    if "created_at" in preview:
        preview["created_at"] = pd.to_datetime(preview["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return preview.head(200)


def preview_combined(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        "post_created_at",
        "post_subreddit",
        "post_title",
        "post_score",
        "post_comment_count",
        "comment_rank",
        "comment_score",
        "comment_body",
        "post_permalink",
    ]
    existing = [column for column in columns if column in df.columns]
    preview = df.loc[:, existing].copy()
    if "post_created_at" in preview:
        preview["post_created_at"] = pd.to_datetime(preview["post_created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return preview.head(200)


def render_downloaded_data(run: RunPaths) -> None:
    st.subheader("Downloaded data")

    posts = read_csv_if_present(run.data_dir / "gm_posts.csv")
    comments = read_csv_if_present(run.data_dir / "gm_comments.csv")
    combined = read_csv_if_present(run.data_dir / "gm_posts_with_comments.csv")
    if posts.empty and comments.empty and combined.empty:
        st.info("No downloaded CSVs are stored for this run yet.")
        return

    run_id_value = latest_run_id(run, posts)
    latest_posts = frame_for_run(posts, run_id_value)
    latest_comments = frame_for_run(comments, run_id_value)
    latest_combined = frame_for_run(combined, run_id_value)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest run posts", fmt_int(len(latest_posts)))
    m2.metric("Latest run comments", fmt_int(len(latest_comments)))
    m3.metric("Latest combined rows", fmt_int(len(latest_combined)))
    m4.metric("Stored posts", fmt_int(len(posts)))
    m5.metric("Stored comments", fmt_int(len(comments)))

    latest_name = run_id_value or "latest"
    st.caption("Download CSVs")
    d1, d2, d3 = st.columns(3)
    with d1:
        download_csv_button(
            "Latest run: posts + comments",
            latest_combined,
            f"{run.tag}_{latest_name}_posts_with_comments.csv",
            f"{run.tag}_{latest_name}_latest_combined",
        )
    with d2:
        download_csv_button(
            "Latest run: posts only",
            latest_posts,
            f"{run.tag}_{latest_name}_posts.csv",
            f"{run.tag}_{latest_name}_latest_posts",
        )
    with d3:
        download_csv_button(
            "Latest run: comments only",
            latest_comments,
            f"{run.tag}_{latest_name}_comments.csv",
            f"{run.tag}_{latest_name}_latest_comments",
        )

    a1, a2, a3 = st.columns(3)
    with a1:
        download_csv_button(
            "All stored: posts + comments",
            combined,
            f"{run.tag}_all_posts_with_comments.csv",
            f"{run.tag}_all_combined",
        )
    with a2:
        download_csv_button(
            "All stored: posts only",
            posts,
            f"{run.tag}_all_posts.csv",
            f"{run.tag}_all_posts",
        )
    with a3:
        download_csv_button(
            "All stored: comments only",
            comments,
            f"{run.tag}_all_comments.csv",
            f"{run.tag}_all_comments",
        )

    summary = latest_run_summary(run)
    if summary:
        latest_rows = []
        for item in summary.get("subreddits", []):
            latest_rows.append({
                "subreddit": item.get("subreddit", ""),
                "status": item.get("status", ""),
                "inspected": item.get("inspected_posts", 0),
                "in_date_range": item.get("eligible_posts", item.get("new_posts", 0)),
                "new_posts": item.get("new_posts", 0),
                "comments": item.get("comments", 0),
                "combined_rows": item.get("combined_rows", 0),
            })
        st.caption(
            f"Latest run `{summary.get('run_id', run_id_value)}` | "
            f"created dates {date_range_text(latest_posts, 'created_at')}"
        )
        if latest_rows:
            st.dataframe(pd.DataFrame(latest_rows), width="stretch", hide_index=True)

    left, right = st.columns([1, 2])
    with left:
        st.caption("Stored posts by subreddit")
        st.dataframe(subreddit_counts(posts, "subreddit"), width="stretch", hide_index=True)
    with right:
        st.caption("Latest downloaded posts")
        st.dataframe(
            preview_posts(latest_posts if not latest_posts.empty else posts),
            width="stretch",
            hide_index=True,
            column_config={
                "permalink": st.column_config.LinkColumn("permalink"),
            },
        )

    if not latest_combined.empty or not combined.empty:
        with st.expander("Post/comment rows"):
            st.dataframe(
                preview_combined(latest_combined if not latest_combined.empty else combined),
                width="stretch",
                hide_index=True,
                column_config={
                    "post_permalink": st.column_config.LinkColumn("post_permalink"),
                },
            )


def data_page(run: RunPaths, settings: dict[str, Any]) -> None:
    st.title("Data")
    render_on_demand_download(settings)
    st.divider()

    if run.tag == "demo":
        st.info("The demo run is read-only. Choose or create a runtime tag to load new data.")

    status = run_status(run)
    st.markdown(
        status_badge("DB", status["db"]),
        unsafe_allow_html=True,
    )

    if run.db_path.exists():
        counts = fetch_df(
            run,
            """
            SELECT source_type, COUNT(*) AS evidence_units
            FROM evidence_units
            GROUP BY source_type
            ORDER BY source_type
            """,
        )
        st.dataframe(counts, width="stretch", hide_index=True)

    render_downloaded_data(run)

    with st.expander("CSV upload"):
        with st.form("upload_data"):
            tag_value = st.text_input("Tag", value="gm_vehicle_on_demand" if run.tag == "demo" else run.tag)
            uploaded = st.file_uploader("CSV", type=["csv"], accept_multiple_files=True)
            reset = st.checkbox("Reset this run before loading", value=False)
            submitted = st.form_submit_button("Append, dedupe, and load")

    if not submitted:
        return

    files, summary, errors, warnings = validate_uploads(uploaded)
    if summary:
        st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    for warning in warnings:
        st.warning(warning)
    if errors:
        for error in errors:
            st.error(error)
        return

    target_settings = load_settings()
    target_run = RunPaths.resolve(tag_value.strip() or target_settings["active_tag"], settings=target_settings)
    try:
        merge_stats = merge_upload_frames(files, target_run.data_dir, reset=reset)
    except ValueError as exc:
        st.error(str(exc))
        return
    try:
        counts = build_db(target_run.data_dir, target_run.db_path, reset=reset)
    except SystemExit:
        st.error("DB load failed. Check the uploaded CSV schema.")
        return

    input_counts = inspect_input_counts(target_run.data_dir)
    target_settings["active_tag"] = target_run.tag
    save_settings(target_settings)
    st.session_state["active_tag"] = target_run.tag
    st.success(f"Loaded `{target_run.tag}` from {input_counts['mode']} input.")
    st.dataframe(pd.DataFrame(merge_stats), width="stretch", hide_index=True)
    st.dataframe(pd.DataFrame([counts]), width="stretch", hide_index=True)
    render_downloaded_data(target_run)


def answer_page(run: RunPaths, settings: dict[str, Any]) -> None:
    st.title("Q&A")
    if not run.db_path.exists():
        render_empty_run(run)
        return

    question = st.text_area("Question", height=90, placeholder="What are the top Silverado pain themes?")
    top_k = st.slider(
        "Retrieved evidence chunks",
        1,
        12,
        int(settings.get("default_ask_top_k", 5)),
        help="How many top-matching RAG chunks to retrieve from the FAISS index. This follows the Lab 3 retrieve(..., k) pattern.",
    )
    if not st.button("Ask", type="primary") or not question.strip():
        return

    with st.spinner("Thinking through the run..."):
        try:
            result = answer(
                question.strip(),
                tag=None if run.tag == "demo" else run.tag,
                db_path=run.db_path,
                index_path=run.index_path,
                top_k=top_k,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    st.caption(f"Intent: {result.intent}")
    for warning in result.warnings:
        st.warning(warning)

    if result.sql:
        st.subheader("SQL")
        st.code(result.sql, language="sql")
        if result.cols:
            st.subheader("Result")
            st.dataframe(pd.DataFrame(result.rows, columns=result.cols), width="stretch", hide_index=True)

    st.subheader("Answer")
    st.markdown(result.answer_text)

    if result.sources:
        st.subheader("Retrieved Evidence Sources")
        st.caption("These are the Reddit posts/chunks pulled into the answer context, not citations from the SQL table.")
        st.dataframe(pd.DataFrame(result.sources), width="stretch", hide_index=True)


def load_subreddit_file(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_subreddit_file(path_value: str, content: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_list_area(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def parse_subreddit_lines(value: str) -> list[str]:
    subreddits: list[str] = []
    seen: set[str] = set()
    for line in value.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.lower().startswith("r/"):
            text = text[2:]
        text = text.strip().strip("/")
        key = text.lower()
        if text and key not in seen:
            subreddits.append(text)
            seen.add(key)
    return subreddits


def default_subreddit_text(settings: dict[str, Any]) -> str:
    subreddit_lists = settings.get("subreddit_lists", {})
    path = subreddit_lists.get("gm", "config/gm_vehicle_subreddits.txt")
    return load_subreddit_file(path)


def local_day_start(value: date) -> str:
    return datetime.combine(value, time(0, 0, 0), APP_TZ).isoformat()


def local_day_end(value: date) -> str:
    return datetime.combine(value, time(23, 59, 59), APP_TZ).isoformat()


def write_runtime_subreddits(run: RunPaths, subreddits: list[str]) -> Path:
    path = run.root / "config" / "subreddits.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(subreddits) + "\n", encoding="utf-8")
    return path


def render_on_demand_download(settings: dict[str, Any]) -> None:
    st.subheader("On-demand Reddit download")

    today = datetime.now(APP_TZ).date()
    default_start = today - timedelta(days=30)

    with st.form("reddit_download"):
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            tag_value = st.text_input(
                "Run tag",
                value=str(st.session_state.get("active_tag", settings.get("active_tag", "gm_vehicle_on_demand"))),
            )
        with c2:
            start_date = st.date_input("Start date", value=default_start)
        with c3:
            end_date = st.date_input("End date", value=today)

        d1, d2, d3 = st.columns(3)
        with d1:
            comments_limit = st.number_input("Top comments per post", min_value=0, max_value=25, value=5, step=1)
        with d2:
            listing_limit = st.number_input("Posts inspected per subreddit", min_value=1, max_value=5000, value=1000, step=100)
        with d3:
            request_delay = st.number_input("Request delay seconds", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

        subreddits_text = st.text_area(
            "Subreddits",
            value=default_subreddit_text(settings),
            height=190,
        )

        a1, a2 = st.columns(2)
        with a1:
            max_new = st.number_input("Max new posts per subreddit", min_value=0, max_value=10000, value=0, step=100)
        with a2:
            dry_run = st.checkbox("Dry run", value=False)

        submitted = st.form_submit_button("Download Reddit data", type="primary")

    if not submitted:
        return

    tag = tag_value.strip() or settings.get("active_tag", "gm_vehicle_on_demand")
    if tag == "demo":
        st.error("The demo run is read-only.")
        return
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    subreddits = parse_subreddit_lines(subreddits_text)
    if not subreddits:
        st.error("Add at least one subreddit.")
        return

    target_settings = load_settings()
    target_run = RunPaths.resolve(tag, settings=target_settings)
    subreddits_file = write_runtime_subreddits(target_run, subreddits)
    since = local_day_start(start_date)
    until = local_day_end(end_date)

    command = [
        str(PYTHON_BIN),
        "collect_incremental.py",
        "--subreddits-file",
        str(subreddits_file),
        "--data-dir",
        str(target_run.data_dir),
        "--state-file",
        str(target_run.state_file),
        "--runs-dir",
        str(target_run.runs_dir),
        "--listing-limit",
        str(int(listing_limit)),
        "--comments-limit",
        str(int(comments_limit)),
        "--request-delay",
        str(float(request_delay)),
        "--progress-every",
        "100",
        "--since-date",
        since,
        "--until-date",
        until,
    ]
    if int(max_new) > 0:
        command.extend(["--max-new-per-subreddit", str(int(max_new))])
    if dry_run:
        command.append("--dry-run")

    settings_summary = {
        "run_tag": target_run.tag,
        "start": since,
        "end": until,
        "top_comments_per_post": int(comments_limit),
        "subreddits": ", ".join(subreddits),
        "posts_inspected_per_subreddit": int(listing_limit),
        "request_delay_seconds": float(request_delay),
        "state_file": str(target_run.state_file),
    }
    st.dataframe(pd.DataFrame([settings_summary]), width="stretch", hide_index=True)
    st.code(" ".join(command), language="bash")

    ok = run_pipeline_step(command)
    if not ok or dry_run:
        return

    try:
        counts = build_db(target_run.data_dir, target_run.db_path, reset=False)
    except SystemExit:
        st.error("DB load failed after download. Check the generated CSV files.")
        return

    target_settings["active_tag"] = target_run.tag
    save_settings(target_settings)
    st.session_state["active_tag"] = target_run.tag
    st.success(f"Downloaded and loaded `{target_run.tag}`.")
    st.dataframe(pd.DataFrame([counts]), width="stretch", hide_index=True)
    render_downloaded_data(target_run)


def settings_page(run: RunPaths, settings: dict[str, Any]) -> None:
    st.title("Settings")

    editable = json.loads(json.dumps(settings))
    provider_options = ["auto", "openai", "openrouter", "anthropic", "jetstream"]

    with st.form("settings_form"):
        c1, c2 = st.columns(2)
        with c1:
            editable["active_tag"] = st.text_input("Default run", value=editable.get("active_tag", run.tag))
            preset_names = list(MODEL_PRESETS)
            selected_preset = st.selectbox("Model preset", preset_names, index=0)
            preset = MODEL_PRESETS[selected_preset]
            if preset:
                editable["generation_provider"], editable["generation_model"] = preset
            editable["generation_provider"] = st.selectbox(
                "Generation provider",
                provider_options,
                index=provider_options.index(editable.get("generation_provider", "auto"))
                if editable.get("generation_provider", "auto") in provider_options
                else 0,
                help="Use the preset above for class-style model switching, or set this manually for custom routing.",
            )
            editable["generation_model"] = st.text_input(
                "Generation model",
                value=editable.get("generation_model", ""),
                help="Examples from the labs: llama-4-scout, gpt-oss-120b, google/gemma-4-31b-it.",
            )
            editable["embedding_model"] = st.text_input(
                "Embedding model",
                value=editable.get("embedding_model", ""),
                help="Lab 3 uses text-embedding-3-large for chunk embeddings.",
            )
            editable["temperature"] = st.slider("Temperature", 0.0, 1.5, float(editable.get("temperature", 0.3)), 0.05)
            editable["max_tokens"] = st.number_input("Max tokens", min_value=128, max_value=8000, value=int(editable.get("max_tokens", 1024)), step=128)
        with c2:
            editable["default_ask_top_k"] = st.number_input("Ask top K", min_value=1, max_value=50, value=int(editable.get("default_ask_top_k", 5)))
            default_limit = editable.get("default_classify_limit")
            limit_text = "" if default_limit is None else str(default_limit)
            limit_text = st.text_input("Classify limit", value=limit_text, placeholder="blank for no limit")
            editable["default_classify_limit"] = int(limit_text) if limit_text.strip() else None
            editable["classification_workers"] = st.number_input("Classification workers", min_value=1, max_value=32, value=int(editable.get("classification_workers", 8)))
            editable["confidence_default"] = st.slider("Default confidence", 0.0, 1.0, float(editable.get("confidence_default", 0.5)), 0.05)

        st.subheader("Classification rules")
        prompts = editable.setdefault("prompts", {})
        old_classifier_prompt = settings["prompts"]["classifier"]
        old_taxonomy = settings["taxonomy"]
        prompts["classifier"] = st.text_area("Classifier prompt", value=prompts.get("classifier", ""), height=240)
        pain_text = st.text_area("Pain themes", value="\n".join(editable["taxonomy"]["pain"]), height=170)
        delight_text = st.text_area("Delight themes", value="\n".join(editable["taxonomy"]["delight"]), height=140)
        editable["taxonomy"]["pain"] = parse_list_area(pain_text)
        editable["taxonomy"]["delight"] = parse_list_area(delight_text)

        st.subheader("Answering prompts")
        prompts["router"] = st.text_area("Router prompt", value=prompts.get("router", ""), height=120)
        prompts["sql"] = st.text_area("SQL prompt", value=prompts.get("sql", ""), height=140)
        prompts["answer"] = st.text_area("Answer prompt", value=prompts.get("answer", ""), height=140)

        st.subheader("Subreddits")
        subreddit_lists = editable.setdefault("subreddit_lists", {"gm": "config/gm_vehicle_subreddits.txt"})
        gm_list_path = subreddit_lists.get("gm", "config/gm_vehicle_subreddits.txt")
        subreddit_lists["gm"] = st.text_input("GM subreddit list path", value=gm_list_path)
        gm_subreddits = st.text_area("GM subreddit list", value=load_subreddit_file(gm_list_path), height=160)

        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        classifier_changed = (
            prompts["classifier"] != old_classifier_prompt
            or editable["taxonomy"] != old_taxonomy
        )
        if classifier_changed:
            editable["relabel_required"] = True
        write_subreddit_file(editable["subreddit_lists"]["gm"], gm_subreddits)
        save_settings(editable)
        st.session_state["active_tag"] = editable["active_tag"]
        st.success("Settings saved.")
        if classifier_changed:
            st.warning("Classifier rules changed. Existing labels should be regenerated before analysis.")

    if settings.get("relabel_required"):
        st.warning("Re-label required")
        c1, c2, c3 = st.columns([1, 1, 2])
        source_type = c1.selectbox("Scope", ["post", "comment", "all"], index=0)
        limit = c2.text_input("Limit", value=str(settings.get("default_classify_limit") or ""))
        workers = c3.number_input("Workers", min_value=1, max_value=32, value=int(settings.get("classification_workers", 8)))
        if st.button("Clear selected labels and classify", type="primary", disabled=not run.db_path.exists() or run.tag == "demo"):
            limit_value = int(limit) if limit.strip() else None
            clear_source_type = None if source_type == "all" else source_type
            clear_labels(run, clear_source_type)
            ok = run_pipeline_step(
                [
                    str(PYTHON_BIN),
                    "classify_evidence.py",
                    "--tag",
                    run.tag,
                    "--workers",
                    str(int(workers)),
                    "--jsonl-progress",
                    *(["--source-type", clear_source_type] if clear_source_type else []),
                    *(["--limit", str(limit_value)] if limit_value else []),
                ],
                parse_json_progress=True,
            )
            if ok:
                refreshed = load_settings()
                refreshed["relabel_required"] = False
                save_settings(refreshed)
                st.success("Labels regenerated.")


def clear_labels(run: RunPaths, source_type: str | None) -> None:
    con = open_db(run, read_only=False)
    try:
        if source_type:
            con.execute(
                """
                DELETE FROM labels
                WHERE evidence_id IN (
                  SELECT evidence_id FROM evidence_units WHERE source_type = ?
                )
                """,
                [source_type],
            )
        else:
            con.execute("DELETE FROM labels")
    finally:
        con.close()


def estimate_pending(run: RunPaths, limit: int | None, source_type: str | None, workers: int, settings: dict[str, Any]) -> dict[str, Any] | None:
    if not run.db_path.exists():
        return None
    con = open_db(run, read_only=True)
    try:
        rows = fetch_unlabeled(con, limit=limit, source_type=source_type)
    finally:
        con.close()
    return estimate_classification(len(rows), settings, workers) | {"pending": len(rows)}


def run_pipeline_step(command: list[str], parse_json_progress: bool = False) -> bool:
    log_box = st.empty()
    progress = st.progress(0.0) if parse_json_progress else None
    lines: list[str] = []

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        lines.append(line)
        if parse_json_progress:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if event and event.get("total"):
                total = max(float(event.get("total", 0)), 1.0)
                completed = float(event.get("completed", 0))
                progress.progress(min(completed / total, 1.0))
        log_box.code("\n".join(lines[-240:]), language="text")

    return_code = process.wait()
    if return_code == 0:
        if progress:
            progress.progress(1.0)
        st.success("Step completed.")
        return True
    st.error(f"Step failed with exit code {return_code}.")
    return False


def pipeline_page(run: RunPaths, settings: dict[str, Any]) -> None:
    st.title("Pipeline")

    status = run_status(run)
    st.markdown(
        status_badge("1 Build DB", status["db"])
        + status_badge("2 Classify", status["classify"])
        + status_badge("3 RAG Index", status["index"])
        + status_badge("4 Report", status["report"]),
        unsafe_allow_html=True,
    )

    if run.tag == "demo":
        st.info("The demo run is read-only. Pipeline actions run on runtime tags.")
        return

    st.caption(f"Run `{run.tag}`")
    st.code(f"./run_gm_vehicle_on_demand.sh {run.tag}", language="bash")

    c1, c2, c3 = st.columns(3)
    source_type_choice = c1.selectbox("Classify source", ["all", "post", "comment"], index=1)
    source_type = None if source_type_choice == "all" else source_type_choice
    limit_text = c2.text_input("Limit", value=str(settings.get("default_classify_limit") or ""))
    limit = int(limit_text) if limit_text.strip() else None
    workers = c3.number_input("Workers", min_value=1, max_value=32, value=int(settings.get("classification_workers", 8)))

    estimate = estimate_pending(run, limit=limit, source_type=source_type, workers=int(workers), settings=settings)
    if estimate:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Pending calls", fmt_int(estimate["pending"]))
        e2.metric("Input tokens", fmt_int(estimate["input_tokens"]))
        e3.metric("Output tokens", fmt_int(estimate["output_tokens"]))
        e4.metric("Estimate", f"${estimate['estimated_cost_usd']:.2f} / {estimate['estimated_minutes']:.1f} min")

    step = st.radio("Step", ["Build DB", "Classify", "Build RAG Index", "Generate Report", "Run all"], horizontal=True)

    if st.button("Run selected", type="primary"):
        commands = {
            "Build DB": [[str(PYTHON_BIN), "build_analytics_db.py", "--tag", run.tag, "--data-dir", str(run.data_dir)]],
            "Classify": [[
                str(PYTHON_BIN),
                "classify_evidence.py",
                "--tag",
                run.tag,
                "--workers",
                str(int(workers)),
                "--jsonl-progress",
                *(["--source-type", source_type] if source_type else []),
                *(["--limit", str(limit)] if limit else []),
            ]],
            "Build RAG Index": [[str(PYTHON_BIN), "build_rag_index.py", "--tag", run.tag, "--data-dir", str(run.data_dir), "--jsonl-progress"]],
            "Generate Report": [[str(PYTHON_BIN), "report.py", "--tag", run.tag]],
        }
        selected_commands = (
            commands["Build DB"]
            + commands["Classify"]
            + commands["Build RAG Index"]
            + commands["Generate Report"]
            if step == "Run all"
            else commands[step]
        )
        for command in selected_commands:
            st.code(" ".join(command), language="bash")
            ok = run_pipeline_step(command, parse_json_progress="--jsonl-progress" in command)
            if not ok:
                break


PAGES = ["Dashboard", "Data", "Q&A", "Settings", "Pipeline"]


def render_sidebar(settings: dict[str, Any]) -> tuple[str, str]:
    tags = available_tags(settings)
    session_tag = st.session_state.get("active_tag", settings.get("active_tag", tags[0]))
    if session_tag not in tags:
        tags.insert(0, session_tag)
    st.sidebar.title("redditgm")
    selected_tag = st.sidebar.selectbox("Run", tags, index=tags.index(session_tag))
    st.session_state["active_tag"] = selected_tag
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    return selected_tag, page


def main() -> None:
    settings = load_settings()
    selected_tag, page = render_sidebar(settings)
    run = resolve_app_run(selected_tag, settings)

    status = run_status(run)
    st.sidebar.markdown(
        status_badge("DB", status["db"])
        + status_badge("Labels", status["classify"])
        + status_badge("Index", status["index"]),
        unsafe_allow_html=True,
    )

    if page == "Dashboard":
        dashboard_page(run, settings)
    elif page == "Data":
        data_page(run, settings)
    elif page == "Q&A":
        answer_page(run, settings)
    elif page == "Settings":
        settings_page(run, settings)
    elif page == "Pipeline":
        pipeline_page(run, settings)


if __name__ == "__main__":
    main()
