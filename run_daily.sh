#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
.venv311/bin/python collect_incremental.py \
  --listing-limit 100 \
  --comments-limit 5 \
  --request-delay 0.8 \
  --progress-every 25
