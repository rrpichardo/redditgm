# redditgm

Incremental Reddit collector for GM, auto, and workflow-pain subreddits.

It fetches the newest posts from each subreddit, skips posts already seen in prior runs, fetches top comments for only the new posts, and appends rows to master CSVs.

## Setup

```bash
cd /Users/ricopichardo/Claude/redditgm
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
```

## Run

Default small manual run:

```bash
./run_daily.sh
```

Expanded manual run:

```bash
./run_weekly.sh
```

GM vehicle on-demand run:

```bash
./run_gm_vehicle_on_demand.sh
```

This uses `config/gm_vehicle_subreddits.txt`, inspects the newest 100 posts per subreddit, collects the top 5 comments per new post, and writes to stable additive outputs under `data/gm_vehicle_on_demand`, `state/gm_vehicle_on_demand_seen_posts.json`, and `runs/gm_vehicle_on_demand`.

To use a separate run tag or pass collector flags:

```bash
./run_gm_vehicle_on_demand.sh gm_vehicle_pilot --dry-run
```

Competitor on-demand benchmark run:

```bash
./run_competitor_on_demand.sh
```

This uses `config/competitor_subreddits.txt` to monitor Ford, Stellantis, Honda, Hyundai/Kia, and Toyota comparison communities. Its default outputs are stable and additive under `data/competitor_on_demand`, `state/competitor_on_demand_seen_posts.json`, and `runs/competitor_on_demand`.

## Activity Screen

Before choosing subreddits for a proposal, check actual activity instead of relying only on member counts:

```bash
.venv311/bin/python subreddit_activity.py \
  --subreddits-file config/gm_vehicle_subreddits.txt \
  --out reports/gm_vehicle_subreddit_activity.csv
```

The activity screen reports subscriber count, newest-post sample size, recent posting volume, and an estimated average daily post rate. The estimate is for sizing and source-quality documentation; it is not an official Reddit traffic metric.

## Outputs

- `data/gm_posts_with_comments.csv`: main combined file, one row per post/comment pair.
- `data/gm_posts.csv`: one row per collected post.
- `data/gm_comments.csv`: one row per collected top-level comment.
- `state/seen_posts.json`: post IDs already collected, used to avoid duplicates.
- `runs/run_manifest.jsonl`: per-subreddit run summaries.

## Duplicate Rule

The collector is meant to be additive. It treats `post_id` as the durable unit. Once a post ID is seen, future runs skip the post and do not refetch comments for it.

If `state/seen_posts.json` is missing, the script seeds state from `data/gm_posts.csv` before collecting, so reruns still avoid duplicates when the master CSV exists.

For the GM and competitor on-demand runs, use the default stable run tags unless you intentionally want a separate dataset. That keeps the state file consistent across runs, so each new run focuses on newly observed posts since the previous run rather than rebuilding the dataset from scratch. If a run returns `0 new posts`, that means the newest listing did not include unseen post IDs.

Every collected row includes a `run_id`. To compare historical discussion against the newest GM run, export a comparison bundle:

```bash
.venv311/bin/python export_run_comparison.py \
  --data-dir data/gm_vehicle_on_demand \
  --runs-dir runs/gm_vehicle_on_demand \
  --out-dir reports/gm_vehicle_comparison
```

The exporter creates `historical/`, `new/`, and `all/` CSV folders plus `comparison_manifest.json`. Use `new/` for newly collected information since the last run, `historical/` for prior context, and the manifest for row counts, run IDs, subreddit counts, and date ranges.

## Future Feature: Ask/RAG

The current project is a collector and comparison-data builder. It does not yet include a chat, Ask, or RAG interface.

A future Ask/RAG layer could sit on top of the additive CSVs and comparison exports so a user can ask questions like:

- What new GM pain points appeared since the last run?
- Are EV concerns showing up differently from ICE concerns?
- Which themes are unique to GM versus competitor communities?
- Which subreddit or vehicle line is driving a new issue?

That future feature should use `reports/*/new/` for newly collected evidence, `reports/*/historical/` for prior context, and `comparison_manifest.json` to cite run IDs, date ranges, and subreddit coverage. Any answer should trace claims back to specific posts/comments rather than inventing conclusions from summary text alone.
