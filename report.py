"""
report.py — Generate a fixed analytics report from the DuckDB labels.

Outputs:
  reports/<tag>/analytics/report.md     — markdown summary
  reports/<tag>/analytics/dashboard.html — interactive dark HTML dashboard (Chart.js)
  reports/<tag>/analytics/*.png          — matplotlib charts (Lab 2 style)

Run:
  .venv311/bin/python report.py [--tag gm_vehicle_on_demand]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
DB_PATH = Path("analytics/redditgm.duckdb")

# Color palette — pain=red, delight=green, neutral=slate
COLORS = {
    "pain": "#ff4d6d",
    "delight": "#4ade80",
    "positive": "#4ade80",
    "negative": "#ff4d6d",
    "neutral": "#94a3b8",
    "mixed": "#f59e0b",
    "EV": "#6c63ff",
    "ICE": "#f59e0b",
    "PHEV": "#22d3ee",
    "unknown": "#64748b",
}
CHART_BG = "#0f1117"
CARD_BG = "#1a1d2e"
TEXT_COLOR = "#e2e8f0"


def _setup_dark_axes(ax, title: str = "") -> None:
    """Apply dark theme styling to a matplotlib axes (Lab 2 chart style)."""
    fig = ax.figure
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CARD_BG)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12, fontweight="bold")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def save_chart(fig, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CHART_BG)
    plt.close(fig)
    return out


def chart_sentiment_donut(con, out_dir: Path) -> Path:
    rows = con.execute("""
        SELECT sentiment, COUNT(*) AS cnt
        FROM labels
        WHERE sentiment IS NOT NULL
        GROUP BY sentiment
        ORDER BY cnt DESC
    """).fetchall()

    if not rows:
        return None

    labels = [r[0] for r in rows]
    sizes = [r[1] for r in rows]
    colors = [COLORS.get(l, "#64748b") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    _setup_dark_axes(ax, "Overall Sentiment")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75, wedgeprops={"width": 0.55},
    )
    for t in texts + autotexts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(9)
    return save_chart(fig, out_dir, "sentiment_donut.png")


def chart_pain_themes(con, out_dir: Path) -> Path:
    rows = con.execute("""
        SELECT pain_theme, COUNT(*) AS cnt
        FROM labels
        WHERE is_pain_point = TRUE AND pain_theme IS NOT NULL
        GROUP BY pain_theme
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    if not rows:
        return None

    themes = [r[0] for r in rows]
    counts = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    _setup_dark_axes(ax, "Top Pain-Point Themes")
    bars = ax.barh(themes[::-1], counts[::-1], color=COLORS["pain"], alpha=0.85)
    ax.bar_label(bars, fmt="%d", color=TEXT_COLOR, padding=3, fontsize=8)
    ax.set_xlabel("Count", color=TEXT_COLOR)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return save_chart(fig, out_dir, "pain_themes.png")


def chart_brand_sentiment(con, out_dir: Path) -> Path:
    rows = con.execute("""
        SELECT brand, sentiment, COUNT(*) AS cnt
        FROM labels
        WHERE brand NOT IN ('unknown', 'GM') AND sentiment IS NOT NULL
        GROUP BY brand, sentiment
        ORDER BY brand, sentiment
    """).fetchall()

    if not rows:
        return None

    from collections import defaultdict
    brand_data: dict[str, dict[str, int]] = defaultdict(dict)
    for brand, sentiment, cnt in rows:
        brand_data[brand][sentiment] = cnt

    brands = sorted(brand_data.keys())
    sentiments = ["positive", "negative", "neutral", "mixed"]

    fig, ax = plt.subplots(figsize=(8, 4))
    _setup_dark_axes(ax, "Sentiment by Brand")

    x = range(len(brands))
    width = 0.2
    for i, sentiment in enumerate(sentiments):
        vals = [brand_data[b].get(sentiment, 0) for b in brands]
        offset = (i - 1.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width,
                      label=sentiment, color=COLORS.get(sentiment, "#64748b"), alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(brands, color=TEXT_COLOR)
    ax.legend(facecolor=CARD_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Count", color=TEXT_COLOR)
    return save_chart(fig, out_dir, "brand_sentiment.png")


def chart_ev_vs_ice(con, out_dir: Path) -> Path:
    rows = con.execute("""
        SELECT powertrain,
               SUM(CASE WHEN is_pain_point THEN 1 ELSE 0 END) AS pain,
               SUM(CASE WHEN is_delight THEN 1 ELSE 0 END)    AS delight
        FROM labels
        WHERE powertrain IN ('EV', 'ICE', 'PHEV')
        GROUP BY powertrain
        ORDER BY powertrain
    """).fetchall()

    if not rows:
        return None

    powertrains = [r[0] for r in rows]
    pains = [r[1] for r in rows]
    delights = [r[2] for r in rows]
    x = range(len(powertrains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    _setup_dark_axes(ax, "Pain Points vs Delight — EV vs ICE")
    ax.bar([xi - width / 2 for xi in x], pains, width, label="Pain", color=COLORS["pain"], alpha=0.85)
    ax.bar([xi + width / 2 for xi in x], delights, width, label="Delight", color=COLORS["delight"], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(powertrains, color=TEXT_COLOR)
    ax.legend(facecolor=CARD_BG, edgecolor="#334155", labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Count", color=TEXT_COLOR)
    return save_chart(fig, out_dir, "ev_vs_ice.png")


def query_summary(con) -> dict:
    """Pull headline numbers for the report."""
    total = con.execute("SELECT COUNT(*) FROM evidence_units").fetchone()[0]
    labeled = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    pain_pct_row = con.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN is_pain_point THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM labels
    """).fetchone()
    delight_pct_row = con.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN is_delight THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM labels
    """).fetchone()

    top_pain = con.execute("""
        SELECT pain_theme, COUNT(*) AS cnt
        FROM labels WHERE is_pain_point = TRUE AND pain_theme IS NOT NULL
        GROUP BY pain_theme ORDER BY cnt DESC LIMIT 3
    """).fetchall()

    top_delight = con.execute("""
        SELECT delight_theme, COUNT(*) AS cnt
        FROM labels WHERE is_delight = TRUE AND delight_theme IS NOT NULL
        GROUP BY delight_theme ORDER BY cnt DESC LIMIT 3
    """).fetchall()

    return {
        "total": total,
        "labeled": labeled,
        "pain_pct": pain_pct_row[0] if pain_pct_row else 0,
        "delight_pct": delight_pct_row[0] if delight_pct_row else 0,
        "top_pain": top_pain,
        "top_delight": top_delight,
    }


def write_markdown(summary: dict, out_dir: Path) -> Path:
    pain_list = "\n".join(f"  {i+1}. {t} ({c})" for i, (t, c) in enumerate(summary["top_pain"]))
    delight_list = "\n".join(f"  {i+1}. {t} ({c})" for i, (t, c) in enumerate(summary["top_delight"]))
    md = f"""# GM Reddit Analytics Report

**Evidence Units:** {summary["total"]:,}
**Labeled:** {summary["labeled"]:,}
**Pain-Point Rate:** {summary["pain_pct"]}%
**Delight Rate:** {summary["delight_pct"]}%

## Top Pain-Point Themes
{pain_list or "  (no data yet)"}

## Top Delight Themes
{delight_list or "  (no data yet)"}

## Charts
- sentiment_donut.png — Overall sentiment breakdown
- pain_themes.png — Top pain-point categories
- brand_sentiment.png — Sentiment by GM brand
- ev_vs_ice.png — EV vs ICE pain & delight comparison
"""
    out = out_dir / "report.md"
    out.write_text(md, encoding="utf-8")
    return out


def write_html_dashboard(summary: dict, out_dir: Path) -> Path:
    """Interactive dark HTML dashboard with Chart.js (animated, engaging, clear)."""
    # Pull data for JS charts
    import duckdb as _duck

    # We can't pass `con` here so we re-open it — this function is called after charting
    db_path = DB_PATH  # module-level fallback; overridden in main

    pain_themes_data = []
    brand_sentiment_data = {}
    sentiment_data = []
    ev_ice_data = []

    try:
        _con = _duck.connect(str(db_path), read_only=True)

        sentiment_data = _con.execute("""
            SELECT sentiment, COUNT(*) FROM labels WHERE sentiment IS NOT NULL
            GROUP BY sentiment ORDER BY COUNT(*) DESC
        """).fetchall()

        pain_themes_data = _con.execute("""
            SELECT pain_theme, COUNT(*) FROM labels
            WHERE is_pain_point = TRUE AND pain_theme IS NOT NULL
            GROUP BY pain_theme ORDER BY COUNT(*) DESC LIMIT 8
        """).fetchall()

        brand_rows = _con.execute("""
            SELECT brand, sentiment, COUNT(*) FROM labels
            WHERE brand NOT IN ('unknown', 'GM') AND sentiment IS NOT NULL
            GROUP BY brand, sentiment ORDER BY brand
        """).fetchall()

        ev_ice_data = _con.execute("""
            SELECT powertrain,
                SUM(CASE WHEN is_pain_point THEN 1 ELSE 0 END),
                SUM(CASE WHEN is_delight THEN 1 ELSE 0 END)
            FROM labels WHERE powertrain IN ('EV', 'ICE', 'PHEV') GROUP BY powertrain
        """).fetchall()

        _con.close()
    except Exception:
        pass

    # Serialize to JS
    sentiment_js = json.dumps({r[0]: r[1] for r in sentiment_data})
    pain_js = json.dumps({"labels": [r[0] for r in pain_themes_data], "data": [r[1] for r in pain_themes_data]})

    brands = sorted(set(r[0] for r in brand_rows)) if brand_rows else []
    brand_dict: dict = {b: {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0} for b in brands}
    for b, s, c in brand_rows:
        if b in brand_dict and s in brand_dict[b]:
            brand_dict[b][s] = c
    brand_js = json.dumps({
        "brands": brands,
        "positive": [brand_dict[b]["positive"] for b in brands],
        "negative": [brand_dict[b]["negative"] for b in brands],
        "neutral": [brand_dict[b]["neutral"] for b in brands],
    })

    ev_js = json.dumps({
        "labels": [r[0] for r in ev_ice_data],
        "pain": [r[1] for r in ev_ice_data],
        "delight": [r[2] for r in ev_ice_data],
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GM Reddit Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d2e;
    --border: #2d3748;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #6c63ff;
    --pain: #ff4d6d;
    --delight: #4ade80;
    --warning: #f59e0b;
    --ev: #6c63ff;
    --ice: #f59e0b;
    --phev: #22d3ee;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: baseline;
    gap: 12px;
  }}
  header h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }}
  header span {{ color: var(--muted); font-size: 13px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 24px 32px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .stat-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 0 32px 0;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
  }}
  .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
  .stat .value {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
  .stat .value.pain {{ color: var(--pain); }}
  .stat .value.delight {{ color: var(--delight); }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .card h2 {{
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 16px;
  }}
  .chart-wrap {{ position: relative; height: 260px; }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    padding: 24px;
  }}
</style>
</head>
<body>
<header>
  <h1>GM Reddit Analytics</h1>
  <span>{summary["labeled"]:,} posts labeled &nbsp;·&nbsp; {summary["total"]:,} total evidence units</span>
</header>

<div class="stat-row" style="padding-top:24px">
  <div class="stat">
    <div class="label">Evidence Units</div>
    <div class="value">{summary["total"]:,}</div>
  </div>
  <div class="stat">
    <div class="label">Labeled</div>
    <div class="value">{summary["labeled"]:,}</div>
  </div>
  <div class="stat">
    <div class="label">Pain-Point Rate</div>
    <div class="value pain">{summary["pain_pct"]}%</div>
  </div>
  <div class="stat">
    <div class="label">Delight Rate</div>
    <div class="value delight">{summary["delight_pct"]}%</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Overall Sentiment</h2>
    <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div>
  </div>
  <div class="card" style="grid-column: span 2">
    <h2>Top Pain-Point Themes</h2>
    <div class="chart-wrap"><canvas id="painChart"></canvas></div>
  </div>
  <div class="card" style="grid-column: span 2">
    <h2>Sentiment by Brand</h2>
    <div class="chart-wrap"><canvas id="brandChart"></canvas></div>
  </div>
  <div class="card">
    <h2>EV vs ICE — Pain &amp; Delight</h2>
    <div class="chart-wrap"><canvas id="evIceChart"></canvas></div>
  </div>
</div>

<footer>Generated by redditgm · {__import__('datetime').date.today()}</footer>

<script>
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#2d3748';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Inter', sans-serif";
Chart.defaults.font.size = 12;
const anim = {{ duration: 700, easing: 'easeOutQuart' }};

// 1 — Sentiment donut
(function() {{
  const d = {sentiment_js};
  const labels = Object.keys(d);
  const data = Object.values(d);
  const colors = {{ positive:'#4ade80', negative:'#ff4d6d', neutral:'#94a3b8', mixed:'#f59e0b' }};
  new Chart(document.getElementById('sentimentChart'), {{
    type: 'doughnut',
    data: {{
      labels,
      datasets: [{{ data, backgroundColor: labels.map(l => colors[l] || '#64748b'),
                    borderWidth: 2, borderColor: '#1a1d2e', hoverOffset: 6 }}]
    }},
    options: {{
      animation: anim,
      cutout: '60%',
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ padding: 16, boxWidth: 12 }} }}
      }}
    }}
  }});
}})();

// 2 — Pain themes horizontal bar
(function() {{
  const d = {pain_js};
  new Chart(document.getElementById('painChart'), {{
    type: 'bar',
    data: {{
      labels: d.labels,
      datasets: [{{ label: 'Count', data: d.data,
                    backgroundColor: '#ff4d6d', borderRadius: 4, barThickness: 18 }}]
    }},
    options: {{
      animation: anim,
      indexAxis: 'y',
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#2d3748' }}, ticks: {{ precision: 0 }} }},
        y: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}})();

// 3 — Brand sentiment grouped bar
(function() {{
  const d = {brand_js};
  new Chart(document.getElementById('brandChart'), {{
    type: 'bar',
    data: {{
      labels: d.brands,
      datasets: [
        {{ label: 'Positive', data: d.positive, backgroundColor: '#4ade80', borderRadius: 4 }},
        {{ label: 'Negative', data: d.negative, backgroundColor: '#ff4d6d', borderRadius: 4 }},
        {{ label: 'Neutral',  data: d.neutral,  backgroundColor: '#94a3b8', borderRadius: 4 }},
      ]
    }},
    options: {{
      animation: anim,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        x: {{ grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#2d3748' }}, ticks: {{ precision: 0 }} }}
      }}
    }}
  }});
}})();

// 4 — EV vs ICE grouped bar
(function() {{
  const d = {ev_js};
  new Chart(document.getElementById('evIceChart'), {{
    type: 'bar',
    data: {{
      labels: d.labels,
      datasets: [
        {{ label: 'Pain Points', data: d.pain, backgroundColor: '#ff4d6d', borderRadius: 4 }},
        {{ label: 'Delight',     data: d.delight, backgroundColor: '#4ade80', borderRadius: 4 }},
      ]
    }},
    options: {{
      animation: anim,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        x: {{ grid: {{ display: false }} }},
        y: {{ grid: {{ color: '#2d3748' }}, ticks: {{ precision: 0 }} }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""

    out = out_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    global DB_PATH

    p = argparse.ArgumentParser(description="Generate GM Reddit analytics report.")
    p.add_argument("--db-path", default=str(DB_PATH))
    p.add_argument("--tag", default="gm_vehicle_on_demand")
    args = p.parse_args()

    db_path = Path(args.db_path)
    DB_PATH = db_path  # update module global so dashboard can re-open it

    out_dir = Path("reports") / args.tag / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        console.print(f"[red]✗[/] Database not found at {db_path}. Run build_analytics_db.py first.")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics — Report[/]\n"
        f"DB: [dim]{db_path}[/]\n"
        f"Out: [dim]{out_dir}[/]",
        border_style="cyan"
    ))

    con = duckdb.connect(str(db_path), read_only=True)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Querying summary stats...", total=5)
        summary = query_summary(con)
        progress.advance(task)

        progress.update(task, description="Rendering sentiment chart...")
        chart_sentiment_donut(con, out_dir)
        progress.advance(task)

        progress.update(task, description="Rendering pain themes chart...")
        chart_pain_themes(con, out_dir)
        progress.advance(task)

        progress.update(task, description="Rendering brand sentiment chart...")
        chart_brand_sentiment(con, out_dir)
        progress.advance(task)

        progress.update(task, description="Rendering EV vs ICE chart...")
        chart_ev_vs_ice(con, out_dir)
        progress.advance(task)

    con.close()

    md_path = write_markdown(summary, out_dir)
    html_path = write_html_dashboard(summary, out_dir)

    console.print(f"\n[green]✓[/] Report: [bold]{md_path}[/]")
    console.print(f"[green]✓[/] Dashboard: [bold]{html_path}[/]")
    console.print(f"[green]✓[/] Charts saved to [bold]{out_dir}[/]")
    console.print("\n[dim]Open dashboard.html in any browser for interactive charts.[/]")


if __name__ == "__main__":
    main()
