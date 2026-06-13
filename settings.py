"""
settings.py — Central editable settings for redditgm analytics.
All modules read from here. Edits persist to runtime/settings.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# default path; tests override this directly on the module
SETTINGS_PATH = Path("runtime/settings.json")  # intentionally module-level so tests can override it

# canonical theme enumerations used by the classifier prompt
PAIN_THEMES_DEFAULT: list[str] = [
    "transmission", "reliability", "dealer_service", "pricing",
    "infotainment", "battery_range", "charging", "build_quality",
    "recall", "warranty", "performance", "comfort", "other",
]
DELIGHT_THEMES_DEFAULT: list[str] = [
    "performance", "comfort", "value", "technology",
    "design", "safety", "dealer_service", "reliability", "other",
]

# ── System prompts ────────────────────────────────────────────────────────────

_ROUTER_PROMPT = """You classify the intent of a question about Reddit vehicle data.

Output ONE of these three tokens only:
  COUNT       — the answer is a number (how many, what share, top N, compare counts)
  QUALITATIVE — the answer is narrative (what are people saying, summarize themes)
  BOTH        — needs a number AND narrative examples

Output only the token, nothing else."""

_SCHEMA_DESCRIPTION = """Tables in the DuckDB database:

evidence_units (
    evidence_id  VARCHAR,   -- PK; format 'post_<id>' or 'comment_<id>'
    source_type  VARCHAR,   -- 'post' or 'comment'
    run_id       VARCHAR,
    subreddit    VARCHAR,   -- e.g. 'Silverado', 'GMC', 'Chevy'
    post_id      VARCHAR,
    comment_id   VARCHAR,
    author       VARCHAR,   -- '[deleted]' means removed
    created_at   TIMESTAMP,
    title        VARCHAR,
    text         VARCHAR,
    permalink    VARCHAR,
    score        INTEGER
)

labels (
    evidence_id    VARCHAR,  -- FK → evidence_units
    brand          VARCHAR,  -- 'Chevy', 'GMC', 'Cadillac', 'Buick', 'GM', 'unknown'
    model          VARCHAR,  -- 'Silverado', 'Tahoe', 'Sierra', etc.
    powertrain     VARCHAR,  -- 'EV', 'ICE', 'PHEV', 'unknown'
    is_pain_point  BOOLEAN,
    pain_theme     VARCHAR,  -- see THEME_ENUM below
    is_delight     BOOLEAN,
    delight_theme  VARCHAR,
    sentiment      VARCHAR,  -- 'positive', 'negative', 'neutral', 'mixed'
    confidence     FLOAT
)

THEME_ENUM: transmission | reliability | dealer_service | pricing | infotainment |
            battery_range | charging | build_quality | recall | warranty |
            performance | comfort | other

Rules you MUST follow when writing SQL:
- Always JOIN labels l ON e.evidence_id = l.evidence_id
- To count unique people, use COUNT(DISTINCT e.author)
- Always filter out deleted authors: WHERE e.author != '[deleted]'
- Use only columns listed above — do not invent column names
- Output only the SQL query, no explanation, no markdown fences"""

_SQL_PROMPT = f"""You write DuckDB SQL queries for a Reddit vehicle analytics database.

{_SCHEMA_DESCRIPTION}

Output only the SQL query. No markdown. No explanation."""

_ANSWER_PROMPT = """You synthesize analytics results for a business audience.
Present findings clearly. Lead with the number, then interpret it, then cite evidence.
Be concise and honest — never invent numbers."""


def _classifier_prompt_from_taxonomy(pain_themes: list[str], delight_themes: list[str]) -> str:
    """Build the classifier system prompt from the active theme taxonomy."""
    pain_enum = "|".join(pain_themes) + "|null"
    delight_enum = "|".join(delight_themes) + "|null"
    return f"""You are a vehicle-brand sentiment analyst for GM (General Motors).

For each Reddit post/comment, output a single JSON object with exactly these fields:
{{
  "brand": "Chevy|GMC|Cadillac|Buick|GM|unknown",
  "model": "Silverado|Equinox|Tahoe|Sierra|Blazer|Escalade|Corvette|Camaro|<model>|unknown",
  "powertrain": "EV|ICE|PHEV|unknown",
  "is_pain_point": true|false,
  "pain_theme": "{pain_enum}",
  "is_delight": true|false,
  "delight_theme": "{delight_enum}",
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.0-1.0
}}

Rules:
- pain_theme must be one of the listed values or null (if is_pain_point is false)
- delight_theme must be one of the listed values or null (if is_delight is false)
- powertrain: EV if mentions electric/EV/battery/Bolt/Lyriq/Blazer EV; PHEV if plug-in hybrid; ICE otherwise
- confidence: your certainty this is actually about a GM vehicle (0.0 = not sure, 1.0 = certain)
- Output ONLY the JSON object — no explanation, no markdown fences"""


# ── Settings dataclass ────────────────────────────────────────────────────────

@dataclass
class Settings:
    # LLM configuration
    generation_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    max_tokens: int = 1024

    # taxonomy lists used by the classifier
    pain_themes: list[str] = field(default_factory=lambda: list(PAIN_THEMES_DEFAULT))
    delight_themes: list[str] = field(default_factory=lambda: list(DELIGHT_THEMES_DEFAULT))

    # pipeline control
    classify_limit: Optional[int] = None   # None = classify everything
    ask_top_k: int = 5                     # RAG top-k chunks to retrieve
    confidence_default: float = 0.5
    active_tag: str = "gm_vehicle_on_demand"

    gm_subreddit_file: str = "config/gm_vehicle_subreddits.txt"

    # system prompts — stored in settings so they can be swapped without code changes
    router_prompt: str = field(default_factory=lambda: _ROUTER_PROMPT)
    sql_prompt: str = field(default_factory=lambda: _SQL_PROMPT)
    answer_prompt: str = field(default_factory=lambda: _ANSWER_PROMPT)
    classifier_prompt: str = field(
        default_factory=lambda: _classifier_prompt_from_taxonomy(
            PAIN_THEMES_DEFAULT, DELIGHT_THEMES_DEFAULT
        )
    )

    relabel_required: bool = False  # set True to force re-labeling on next run


# ── Module-level cache + public API ──────────────────────────────────────────

_cache: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the current settings, loading from disk on first call."""
    global _cache
    if _cache is not None:
        return _cache
    _cache = _load()
    return _cache


def save_settings(s: Settings) -> None:
    """Persist settings to disk and update the in-memory cache."""
    global _cache
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(s), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _cache = s


def invalidate_cache() -> None:
    """Force re-read from disk on the next get_settings() call."""
    global _cache
    _cache = None


def _load() -> Settings:
    """Read JSON from disk and merge known keys into a fresh Settings object."""
    if not SETTINGS_PATH.exists():
        return Settings()  # all defaults
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    # only pass keys the dataclass knows about — unknown keys are silently dropped
    known = {k: raw[k] for k in Settings.__dataclass_fields__ if k in raw}
    return Settings(**known)
