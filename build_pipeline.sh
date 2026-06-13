#!/usr/bin/env bash
# build_pipeline.sh - Run the full analytics pipeline in order.
# Usage: ./build_pipeline.sh [--tag gm_vehicle_on_demand] [--data-dir runtime/gm_vehicle_on_demand/data] [--limit N]
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-gm_vehicle_on_demand}"
DATA_DIR="${DATA_DIR:-}"
LIMIT_ARG=""
SOURCE_TYPE_ARG=""
WORKERS_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)      TAG="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --limit)    LIMIT_ARG="--limit $2"; shift 2 ;;
    --source-type) SOURCE_TYPE_ARG="--source-type $2"; shift 2 ;;
    --workers)  WORKERS_ARG="--workers $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="runtime/${TAG}/data"
fi

PYTHON=".venv311/bin/python"

echo ""
echo "══════════════════════════════════════════════════"
echo "  GM Reddit Analytics — Build Pipeline"
echo "  Tag:  $TAG"
echo "  Data: $DATA_DIR"
echo "══════════════════════════════════════════════════"
echo ""

# Step 1: Load CSVs into DuckDB
echo "▶ Step 1/4 — Build analytics database"
$PYTHON build_analytics_db.py --tag "$TAG" --data-dir "$DATA_DIR"
echo ""

# Step 2: Label evidence units via LLM
echo "▶ Step 2/4 — Classify evidence"
$PYTHON classify_evidence.py --tag "$TAG" $LIMIT_ARG $SOURCE_TYPE_ARG $WORKERS_ARG
echo ""

# Step 3: Build FAISS index for RAG
echo "▶ Step 3/4 — Build RAG index"
$PYTHON build_rag_index.py --tag "$TAG" --data-dir "$DATA_DIR"
echo ""

# Step 4: Generate report
echo "▶ Step 4/4 — Generate report"
$PYTHON report.py --tag "$TAG"
echo ""

echo "══════════════════════════════════════════════════"
echo "  Pipeline complete. Next steps:"
echo "  • Ask questions:   python ask.py --tag \"$TAG\" \"your question\""
echo "  • Report:          runtime/$TAG/reports/analytics/dashboard.html"
echo "══════════════════════════════════════════════════"
