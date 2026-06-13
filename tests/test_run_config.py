"""Tests for run_config.py — Run path conventions."""
from __future__ import annotations

from pathlib import Path
from run_config import Run


def test_run_paths_for_tag():
    run = Run("gm")
    assert run.db_path == Path("runtime/gm/redditgm.duckdb")
    assert run.index_path == Path("runtime/gm/rag")
    assert run.report_dir == Path("runtime/gm/reports")
    assert run.data_dir == Path("data/gm")


def test_run_paths_for_different_tag():
    run = Run("my_custom_run")
    assert run.db_path == Path("runtime/my_custom_run/redditgm.duckdb")
    assert run.index_path == Path("runtime/my_custom_run/rag")


def test_run_legacy_data_dir_accepts_custom():
    """data_dir is derivable but scripts can override it."""
    run = Run("gm_vehicle_on_demand")
    assert run.data_dir == Path("data/gm_vehicle_on_demand")
