"""Tests for run_config.py — RunPaths path conventions."""
from __future__ import annotations

from pathlib import Path

from run_config import RunPaths


def test_run_paths_for_tag():
    run = RunPaths.resolve("gm")
    assert run.db_path == Path("runtime/gm/analytics.duckdb")
    assert run.index_path == Path("runtime/gm/rag_index")
    assert run.report_dir == Path("runtime/gm/reports/analytics")
    assert run.data_dir == Path("runtime/gm/data")


def test_run_paths_for_different_tag():
    run = RunPaths.resolve("my_custom_run")
    assert run.db_path == Path("runtime/my_custom_run/analytics.duckdb")
    assert run.index_path == Path("runtime/my_custom_run/rag_index")


def test_run_custom_path_override():
    """RunPaths.resolve() accepts per-call path overrides."""
    run = RunPaths.resolve("gm", data_dir="custom/data")
    assert run.data_dir == Path("custom/data")
    # Other paths still derived from the tag
    assert run.db_path == Path("runtime/gm/analytics.duckdb")
