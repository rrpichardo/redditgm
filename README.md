# redditgm

Incremental Reddit collector for GM/workflow-pain subreddits.

It fetches the newest posts from each subreddit, skips posts already seen in prior runs, fetches top comments for only the new posts, and appends rows to master CSVs.

## Setup

```bash
cd /Users/ricopichardo/Claude/redditgm
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
```

## Run

Daily-style run:

```bash
./run_daily.sh
```

Weekly-style run:

```bash
./run_weekly.sh
```

## Outputs

- `data/gm_posts_with_comments.csv`: main combined file, one row per post/comment pair.
- `data/gm_posts.csv`: one row per collected post.
- `data/gm_comments.csv`: one row per collected top-level comment.
- `state/seen_posts.json`: post IDs already collected, used to avoid duplicates.
- `runs/run_manifest.jsonl`: per-subreddit run summaries.

## Duplicate Rule

The collector treats `post_id` as the durable unit. Once a post ID is seen, future runs skip the post and do not refetch comments for it.

If `state/seen_posts.json` is missing, the script seeds state from `data/gm_posts.csv` before collecting, so reruns still avoid duplicates when the master CSV exists.
