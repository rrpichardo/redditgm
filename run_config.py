"""run_config.py — Canonical path layout for a named analysis run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    tag: str  # unique name for this run, e.g. "gm_vehicle_on_demand"

    @property
    def data_dir(self) -> Path:
        # raw Reddit data collected for this run
        return Path("data") / self.tag

    @property
    def db_path(self) -> Path:
        # DuckDB analytics database for this run
        return Path("runtime") / self.tag / "redditgm.duckdb"

    @property
    def index_path(self) -> Path:
        # FAISS/ChromaDB RAG index directory for this run
        return Path("runtime") / self.tag / "rag"

    @property
    def report_dir(self) -> Path:
        # output reports directory for this run
        return Path("runtime") / self.tag / "reports"
