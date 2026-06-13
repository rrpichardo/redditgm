"""Tests for settings.py — defaults, round-trip, partial JSON merge."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_defaults_load_without_json(monkeypatch, tmp_path):
    """When no settings.json exists, get_settings() returns defaults."""
    import importlib, sys
    # Patch SETTINGS_PATH before import
    monkeypatch.chdir(tmp_path)
    if "settings" in sys.modules:
        del sys.modules["settings"]
    import settings as s_mod
    s_mod.SETTINGS_PATH = tmp_path / "runtime" / "settings.json"
    s_mod._cache = None

    cfg = s_mod.get_settings()
    assert cfg.generation_model == "openai/gpt-4o-mini"
    assert cfg.embedding_model == "text-embedding-3-large"
    assert "transmission" in cfg.pain_themes
    assert "performance" in cfg.delight_themes
    assert cfg.ask_top_k == 5
    assert cfg.relabel_required is False


def test_save_load_roundtrip(tmp_path):
    """save_settings + get_settings round-trips all fields."""
    import sys
    if "settings" in sys.modules:
        del sys.modules["settings"]
    import settings as s_mod
    s_mod.SETTINGS_PATH = tmp_path / "runtime" / "settings.json"
    s_mod._cache = None

    cfg = s_mod.get_settings()
    cfg.generation_model = "gpt-4o"
    cfg.ask_top_k = 10
    cfg.relabel_required = True
    s_mod.save_settings(cfg)

    s_mod._cache = None  # force re-read
    reloaded = s_mod.get_settings()
    assert reloaded.generation_model == "gpt-4o"
    assert reloaded.ask_top_k == 10
    assert reloaded.relabel_required is True


def test_partial_json_merges_with_defaults(tmp_path):
    """JSON with only some keys fills missing keys from defaults."""
    import sys
    if "settings" in sys.modules:
        del sys.modules["settings"]
    import settings as s_mod
    settings_path = tmp_path / "runtime" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"generation_model": "gpt-4o", "ask_top_k": 7}))
    s_mod.SETTINGS_PATH = settings_path
    s_mod._cache = None

    cfg = s_mod.get_settings()
    assert cfg.generation_model == "gpt-4o"
    assert cfg.ask_top_k == 7
    assert cfg.embedding_model == "text-embedding-3-large"  # default preserved
    assert len(cfg.pain_themes) > 0  # default list preserved
