"""Central runtime settings for redditgm analytics.

Defaults live in code so the project works on first run. User/app edits are
persisted to runtime/settings.json and merged over these defaults.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path("runtime/settings.json")

DEFAULT_PAIN_THEMES = [
    "transmission",
    "reliability",
    "dealer_service",
    "pricing",
    "infotainment",
    "battery_range",
    "charging",
    "build_quality",
    "recall",
    "warranty",
    "performance",
    "comfort",
    "other",
]

DEFAULT_DELIGHT_THEMES = [
    "performance",
    "comfort",
    "value",
    "technology",
    "design",
    "safety",
    "dealer_service",
    "reliability",
    "other",
]

DEFAULT_ROUTER_PROMPT = """You classify the intent of a question about Reddit vehicle data.

Output ONE of these three tokens only:
  COUNT       - the answer is a number (how many, what share, top N, compare counts)
  QUALITATIVE - the answer is narrative (what are people saying, summarize themes)
  BOTH        - needs a number AND narrative examples

Output only the token, nothing else."""

DEFAULT_SQL_PROMPT = """You write DuckDB SQL queries for a Reddit vehicle analytics database.

{schema}

Output only the SQL query. No markdown. No explanation."""

DEFAULT_ANSWER_PROMPT = """You synthesize analytics results for a business audience.
Present findings clearly. Lead with the number, then interpret it, then cite evidence.
Be concise and honest. Never invent numbers."""

DEFAULT_CLASSIFIER_PROMPT = """You are a vehicle-brand sentiment analyst for GM (General Motors).

For each Reddit post/comment, output a single JSON object with exactly these fields:
{{
  "brand": "Chevy|GMC|Cadillac|Buick|GM|unknown",
  "model": "Silverado|Equinox|Tahoe|Sierra|Blazer|Escalade|Corvette|Camaro|<model>|unknown",
  "powertrain": "EV|ICE|PHEV|unknown",
  "is_pain_point": true|false,
  "pain_theme": "{pain_theme_choices}|null",
  "is_delight": true|false,
  "delight_theme": "{delight_theme_choices}|null",
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.0-1.0
}}

Rules:
- pain_theme must be one of the listed pain themes or null when is_pain_point is false
- delight_theme must be one of the listed delight themes or null when is_delight is false
- powertrain: EV if mentions electric/EV/battery/Bolt/Lyriq/Blazer EV; PHEV if plug-in hybrid; ICE otherwise
- confidence: certainty this is actually about a GM vehicle (0.0 = not sure, 1.0 = certain)
- Output ONLY the JSON object. No explanation and no markdown fences."""

DEFAULT_SETTINGS: dict[str, Any] = {
    "active_tag": "gm_vehicle_on_demand",
    "runtime_dir": "runtime",
    "generation_provider": "auto",
    "generation_model": "openai/gpt-4o-mini",
    "embedding_model": "text-embedding-3-large",
    "temperature": 0.3,
    "max_tokens": 1024,
    "default_ask_top_k": 5,
    "default_classify_limit": None,
    "classification_workers": 8,
    "confidence_default": 0.5,
    "classification_estimates": {
        "prompt_tokens_per_item": 700,
        "completion_tokens_per_item": 120,
        "usd_per_1m_input_tokens": 0.15,
        "usd_per_1m_output_tokens": 0.60,
        "seconds_per_item_serial": 2.0,
    },
    "subreddit_lists": {
        "gm": "config/gm_vehicle_subreddits.txt",
    },
    "taxonomy": {
        "pain": DEFAULT_PAIN_THEMES,
        "delight": DEFAULT_DELIGHT_THEMES,
    },
    "prompts": {
        "router": DEFAULT_ROUTER_PROMPT,
        "sql": DEFAULT_SQL_PROMPT,
        "answer": DEFAULT_ANSWER_PROMPT,
        "classifier": DEFAULT_CLASSIFIER_PROMPT,
    },
    "providers": {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "jetstream": {
            "base_url_env": "JETSTREAM_BASE_URL",
            "api_key_env": "JETSTREAM_API_KEY",
        },
    },
    "relabel_required": False,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    """Load settings from disk, merged over defaults."""
    settings_path = Path(path)
    if not settings_path.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    try:
        saved = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid settings JSON at {settings_path}: {exc}") from exc
    if not isinstance(saved, dict):
        raise ValueError(f"Settings file must contain a JSON object: {settings_path}")
    return _deep_merge(DEFAULT_SETTINGS, saved)


def save_settings(settings: dict[str, Any], path: Path | str = SETTINGS_PATH) -> Path:
    """Persist settings to disk."""
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return settings_path


def classifier_prompt(settings: dict[str, Any] | None = None) -> str:
    """Render the classifier prompt from editable settings and taxonomy."""
    loaded = settings or load_settings()
    pain = loaded["taxonomy"]["pain"]
    delight = loaded["taxonomy"]["delight"]
    template = loaded["prompts"]["classifier"]
    return template.format(
        pain_theme_choices="|".join(pain),
        delight_theme_choices="|".join(delight),
    )
