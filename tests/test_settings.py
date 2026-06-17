"""Tests for settings.py — defaults, round-trip, partial JSON merge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from settings import load_settings, save_settings


def test_defaults_load_without_json(tmp_path):
    """When no settings.json exists, load_settings() returns defaults."""
    cfg = load_settings(tmp_path / "settings.json")
    assert cfg["generation_model"] == "openai/gpt-4o-mini"
    assert cfg["embedding_model"] == "text-embedding-3-large"
    assert "transmission" in cfg["taxonomy"]["pain"]
    assert "performance" in cfg["taxonomy"]["delight"]
    assert cfg["default_ask_top_k"] == 5
    assert cfg["classification_workers"] == 8
    assert cfg["relabel_required"] is False


def test_save_load_roundtrip(tmp_path):
    """save_settings + load_settings round-trips values."""
    path = tmp_path / "settings.json"
    cfg = load_settings(path)
    cfg["generation_model"] = "openai/gpt-4o"
    cfg["default_ask_top_k"] = 10
    cfg["relabel_required"] = True
    save_settings(cfg, path)

    reloaded = load_settings(path)
    assert reloaded["generation_model"] == "openai/gpt-4o"
    assert reloaded["default_ask_top_k"] == 10
    assert reloaded["relabel_required"] is True


def test_partial_json_merges_with_defaults(tmp_path):
    """JSON with only some keys fills missing keys from defaults (deep merge)."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"generation_model": "openai/gpt-4o"}), encoding="utf-8")

    cfg = load_settings(path)
    assert cfg["generation_model"] == "openai/gpt-4o"
    # Unspecified keys fall back to defaults
    assert cfg["embedding_model"] == "text-embedding-3-large"
    assert cfg["classification_workers"] == 8
    assert len(cfg["taxonomy"]["pain"]) > 0
