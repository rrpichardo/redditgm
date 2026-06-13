#!/usr/bin/env bash
# build_pipeline.sh — Run the full analytics pipeline in order.
# Usage: ./build_pipeline.sh [--data-dir data/gm_vehicle_on_demand] [--limit N]
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-data/gm_vehicle_on_demand}"
LIMIT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --limit)    LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PYTHON=".venv311/bin/python"

echo ""
echo "══════════════════════════════════════════════════"
echo "  GM Reddit Analytics — Build Pipeline"
echo "  Data: $DATA_DIR"
echo "══════════════════════════════════════════════════"
echo ""

# Step 1: Load CSVs into DuckDB
echo "▶ Step 1/3 — Build analytics database"
$PYTHON build_analytics_db.py --data-dir "$DATA_DIR"
echo ""

# Step 2: Label evidence units via LLM
echo "▶ Step 2/3 — Classify evidence"
$PYTHON classify_evidence.py $LIMIT_ARG
echo ""

# Step 3: Build FAISS index for RAG
echo "▶ Step 3/3 — Build RAG index"
$PYTHON build_rag_index.py --data-dir "$DATA_DIR"
echo ""

echo "══════════════════════════════════════════════════"
echo "  Pipeline complete. Next steps:"
echo "  • Ask questions:   python ask.py \"your question\""
echo "  • Generate report: python report.py"
echo "══════════════════════════════════════════════════"
