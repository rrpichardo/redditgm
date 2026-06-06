#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_tag="${RUN_TAG:-gm_vehicle_on_demand}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_tag="$1"
  shift
fi

.venv311/bin/python collect_incremental.py \
  --subreddits-file config/gm_vehicle_subreddits.txt \
  --data-dir "data/${run_tag}" \
  --state-file "state/${run_tag}_seen_posts.json" \
  --runs-dir "runs/${run_tag}" \
  --listing-limit 100 \
  --comments-limit 5 \
  --request-delay 0.8 \
  --progress-every 25 \
  "$@"
