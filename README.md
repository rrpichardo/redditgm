# redditgm

Incremental Reddit collector plus GM-focused analytics/RAG pipeline.

It fetches newest posts from GM vehicle subreddits, skips posts already seen in
prior runs, fetches top comments for new posts, and appends rows to CSVs. The
analytics layer then loads those CSVs into DuckDB, classifies evidence, builds a
RAG index, and generates reports.

Counts are not fixed in code or docs. Post/comment totals are derived from the
active CSV or DuckDB run each time the pipeline loads data.

The default settings, pipeline, and verification path are GM-only.

## Setup

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add the keys you need for classification,
Ask/RAG, and embeddings.

## Collect GM Data

GM vehicle on-demand run:

```bash
./run_gm_vehicle_on_demand.sh
```

This uses `config/gm_vehicle_subreddits.txt`, inspects the newest 100 posts per
subreddit, collects the top 5 comments per new post, and writes stable additive
outputs under:

- `runtime/gm_vehicle_on_demand/data`
- `runtime/gm_vehicle_on_demand/state/seen_posts.json`
- `runtime/gm_vehicle_on_demand/runs`

To use a separate run tag or pass collector flags:

```bash
./run_gm_vehicle_on_demand.sh gm_vehicle_pilot --dry-run
```

## Build Analytics

Run the full GM analytics pipeline:

```bash
./build_pipeline.sh --tag gm_vehicle_on_demand
```

The pipeline runs:

1. Build DuckDB from CSVs
2. Classify unlabeled evidence units
3. Build the FAISS RAG index
4. Generate the report/dashboard

For a lower-cost sample:

```bash
./build_pipeline.sh --tag gm_vehicle_on_demand --source-type post --limit 100
```

The DB loader accepts either split files:

- `gm_posts.csv`
- `gm_comments.csv`

or the combined collector output:

- `gm_posts_with_comments.csv`

For the combined file, evidence counts are derived from distinct `post_id` and
distinct `comment_id`, so repeated post fields do not inflate post totals.

## Ask Questions

```bash
.venv311/bin/python ask.py --tag gm_vehicle_on_demand "What are people saying about EV range?"
```

`ask.py` exposes `answer(question, ...)` for an app or notebook. It returns the
intent, generated SQL, rows/columns, answer text, source links, and warnings.

## Generate Report

```bash
.venv311/bin/python report.py --tag gm_vehicle_on_demand
```

Outputs are written to:

- `runtime/gm_vehicle_on_demand/reports/analytics/report.md`
- `runtime/gm_vehicle_on_demand/reports/analytics/dashboard.html`
- `runtime/gm_vehicle_on_demand/reports/analytics/*.png`

## Activity Screen

Before choosing subreddits for a proposal, check actual activity instead of
relying only on member counts:

```bash
.venv311/bin/python subreddit_activity.py \
  --subreddits-file config/gm_vehicle_subreddits.txt \
  --out runtime/gm_vehicle_subreddit_activity.csv
```

The activity screen reports subscriber count, newest-post sample size, recent
posting volume, and an estimated average daily post rate. The estimate is for
sizing and source-quality documentation; it is not an official Reddit traffic
metric.

## Duplicate Rule

The collector is additive. It treats `post_id` as the durable unit. Once a post
ID is seen, future runs skip the post and do not refetch comments for it.

If `runtime/<tag>/state/seen_posts.json` is missing, the script seeds state from
the existing `runtime/<tag>/data/gm_posts.csv` before collecting, so reruns still
avoid duplicates when the master CSV exists.

Every collected row includes a `run_id`. To compare historical discussion
against the newest GM run, export a comparison bundle:

```bash
.venv311/bin/python export_run_comparison.py \
  --data-dir runtime/gm_vehicle_on_demand/data \
  --runs-dir runtime/gm_vehicle_on_demand/runs \
  --out-dir runtime/gm_vehicle_on_demand/reports/comparison
```

The exporter creates `historical/`, `new/`, and `all/` CSV folders plus
`comparison_manifest.json`. Use `new/` for newly collected information since the
last run, `historical/` for prior context, and the manifest for row counts, run
IDs, subreddit counts, and date ranges.
