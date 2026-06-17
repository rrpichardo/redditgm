"""
rag_core.py — shared library for GM Reddit analytics.
Ported from: Lab 3 Rating Agent (chunk/embed/FAISS/retrieve/context),
             Lab 2 (model-selection switch, chat helper).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from settings import load_settings

load_dotenv()

# ==== KEYS: paste in .env (see .env.example) ====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY", "")
JETSTREAM_BASE_URL = os.getenv("JETSTREAM_BASE_URL", "")

# ==== MODEL SWITCH seed (#-style, like the class labs; runtime/settings.json wins) ====
GENERATION_MODEL = load_settings()["generation_model"]
# GENERATION_MODEL = "llama-4-scout"                      # Lab 2/OpenRouter
# GENERATION_MODEL = "gpt-oss-120b"                       # Lab 2/3/OpenRouter
# GENERATION_MODEL = "google/gemma-4-31b-it"              # Lab 2/OpenRouter
# GENERATION_MODEL = "gpt-4o-mini"                        # OpenAI direct
# GENERATION_MODEL = "claude-3-5-haiku-20241022"          # Anthropic / Claude
# GENERATION_MODEL = "llama-4-scout"                      # Jetstream when provider=jetstream

EMBEDDING_MODEL = load_settings()["embedding_model"]  # OpenAI; 3072 dims default

# Clients are instantiated lazily on first use so importing this module
# without API keys (e.g. in tests) doesn't raise an error.
_gen_clients: dict[tuple[str, str, str], OpenAI] = {}
_embed_clients: dict[tuple[str, str], OpenAI] = {}

# Tokenizer for the embedding model
_enc = tiktoken.get_encoding("cl100k_base")


def current_generation_model() -> str:
    return load_settings()["generation_model"]


def current_embedding_model() -> str:
    return load_settings()["embedding_model"]


def _provider_for_model(model: str, settings: dict[str, Any]) -> str:
    provider = settings.get("generation_provider", "auto")
    if provider and provider != "auto":
        return provider
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("jetstream/"):
        return "jetstream"
    if model in {"llama-4-scout", "gpt-oss-120b", "gemma-4-31b-it"}:
        return "openrouter"
    if model.startswith(("openai/", "meta-llama/", "anthropic/", "google/", "mistral/")):
        return "openrouter"
    return "openai"


def _provider_model_name(model: str, provider: str) -> str:
    if provider == "jetstream" and model.startswith("jetstream/"):
        return model.removeprefix("jetstream/")
    if provider == "openai" and model.startswith("openai/"):
        return model.removeprefix("openai/")
    if provider == "anthropic" and model.startswith("anthropic/"):
        return model.removeprefix("anthropic/")
    return model


def _provider_base_url(provider: str, settings: dict[str, Any]) -> str:
    provider_settings = settings["providers"].get(provider, {})
    env_name = provider_settings.get("base_url_env")
    if env_name and os.getenv(env_name):
        return os.getenv(env_name, "")
    if "base_url" in provider_settings:
        return provider_settings["base_url"]
    return ""


def _provider_api_key(provider: str, settings: dict[str, Any]) -> str:
    provider_settings = settings["providers"].get(provider, {})
    env_name = provider_settings.get("api_key_env", "")
    return os.getenv(env_name, "")


def _get_gen_client(provider: str, settings: dict[str, Any]) -> OpenAI:
    base_url = _provider_base_url(provider, settings)
    api_key = _provider_api_key(provider, settings)
    if not base_url:
        raise ValueError(f"No base URL configured for generation provider '{provider}'")
    cache_key = (provider, base_url, api_key)
    if cache_key not in _gen_clients:
        _gen_clients[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _gen_clients[cache_key]


def _get_embed_client() -> OpenAI:
    settings = load_settings()
    api_key = os.getenv(settings["providers"]["openai"]["api_key_env"], "")
    base_url = settings["providers"]["openai"]["base_url"]
    cache_key = (base_url, api_key)
    if cache_key not in _embed_clients:
        _embed_clients[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _embed_clients[cache_key]


def _anthropic_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    settings: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Anthropic Messages API without adding another dependency."""
    import httpx

    base_url = _provider_base_url("anthropic", settings).rstrip("/")
    api_key = _provider_api_key("anthropic", settings)
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models")

    system_messages = [m["content"] for m in messages if m.get("role") == "system"]
    user_messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in messages
        if m.get("role") != "system"
    ]
    if not user_messages:
        user_messages = [{"role": "user", "content": ""}]

    payload: dict[str, Any] = {
        "model": _provider_model_name(model, "anthropic"),
        "messages": user_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_messages:
        payload["system"] = "\n\n".join(system_messages)

    response = httpx.post(
        f"{base_url}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return "".join(
        part.get("text", "")
        for part in data.get("content", [])
        if part.get("type") == "text"
    )


def count_tokens(text: str) -> int:
    """Return the number of tokens in text (cl100k_base encoding)."""
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Chunking — Ported from Lab 3 Rating Agent (cell 25)
# ---------------------------------------------------------------------------

def chunk_csv_to_posts(data_dir: str | Path) -> list[dict[str, Any]]:
    """
    Read gm_posts.csv + gm_comments.csv from data_dir separately and combine
    into one chunk per post (title + body + top comments joined as text).

    IMPORTANT: Reads the two SEPARATE CSVs, NOT the combined file.
    The combined CSV repeats a post once per comment, which inflates COUNT(*) 5×+.
    """
    import csv

    data_dir = Path(data_dir)
    posts_path = data_dir / "gm_posts.csv"
    comments_path = data_dir / "gm_comments.csv"

    combined_path = data_dir / "gm_posts_with_comments.csv"

    if not posts_path.exists() and not combined_path.exists():
        raise FileNotFoundError(
            f"Expected gm_posts.csv or gm_posts_with_comments.csv in {data_dir}"
        )

    if not posts_path.exists():
        posts_by_id: dict[str, dict[str, Any]] = {}
        comments_by_post: dict[str, list[str]] = {}
        with combined_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                post_id = row.get("post_id", "")
                if not post_id:
                    continue
                posts_by_id.setdefault(post_id, row)
                body = (row.get("comment_body") or "").strip()
                comment_id = row.get("comment_id", "")
                if comment_id and body:
                    comments_by_post.setdefault(post_id, []).append(body)

        chunks = []
        for post_id, row in posts_by_id.items():
            title = (row.get("post_title") or "").strip()
            selftext = (row.get("post_selftext") or row.get("post_content") or "").strip()
            parts = [f"Title: {title}"]
            if selftext:
                parts.append(f"Post: {selftext}")
            for i, comment in enumerate(comments_by_post.get(post_id, [])[:5], 1):
                parts.append(f"Comment {i}: {comment}")
            chunks.append({
                "post_id": post_id,
                "subreddit": row.get("post_subreddit", ""),
                "author": row.get("post_author", ""),
                "created_at": row.get("post_created_at", ""),
                "title": title,
                "text": "\n".join(parts),
                "permalink": row.get("post_permalink", ""),
                "score": row.get("post_score", ""),
            })
        return chunks

    # Load all comments keyed by post_id
    comments_by_post: dict[str, list[str]] = {}
    if comments_path.exists():
        with comments_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pid = row.get("post_id", "")
                body = (row.get("body") or "").strip()
                if pid and body:
                    comments_by_post.setdefault(pid, []).append(body)

    chunks = []
    with posts_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            post_id = row.get("id", "")
            title = (row.get("title") or "").strip()
            selftext = (row.get("selftext") or "").strip()
            post_comments = comments_by_post.get(post_id, [])

            # Build the text chunk: title + body + top comments (Lab 3 pattern)
            parts = [f"Title: {title}"]
            if selftext:
                parts.append(f"Post: {selftext}")
            for i, c in enumerate(post_comments[:5], 1):
                parts.append(f"Comment {i}: {c}")
            text = "\n".join(parts)

            chunks.append({
                "post_id": post_id,
                "subreddit": row.get("subreddit", ""),
                "author": row.get("author", ""),
                "created_at": row.get("created_at", ""),
                "title": title,
                "text": text,
                "permalink": row.get("permalink", ""),
                "score": row.get("score", ""),
            })

    return chunks


# ---------------------------------------------------------------------------
# Embeddings — Ported from Lab 3 (embed_texts)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """Embed text blocks in batches, following the Lab 3 rating-agent pattern."""
    client = _get_embed_client()
    embedding_model = current_embedding_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=embedding_model, input=batch)
        batch_emb = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_emb)
    return np.array(all_embeddings, dtype="float32")


# ---------------------------------------------------------------------------
# FAISS index — Ported from Lab 3
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP (inner-product / cosine sim after L2-norm)."""
    import faiss

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks: list[dict], path: str | Path) -> None:
    """Persist FAISS index + chunk metadata to disk."""
    import faiss

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    with (path / "chunks.pkl").open("wb") as fh:
        pickle.dump(chunks, fh)


def load_index(path: str | Path):
    """Load FAISS index + chunk metadata from disk."""
    import faiss

    path = Path(path)
    index = faiss.read_index(str(path / "index.faiss"))
    with (path / "chunks.pkl").open("rb") as fh:
        chunks = pickle.load(fh)
    return index, chunks


# ---------------------------------------------------------------------------
# Retrieval + context formatting — Ported from Lab 3
# ---------------------------------------------------------------------------

def retrieve(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Embed the query and return the top-k most similar chunks."""
    import faiss

    if not chunks or top_k <= 0:
        return []
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    _scores, indices = index.search(q_emb, min(top_k, len(chunks)))
    return [chunks[int(i)] for i in indices[0] if 0 <= int(i) < len(chunks)]


def make_context(retrieved: list[dict]) -> str:
    """
    Format retrieved chunks into a context block for LLM prompts.
    Each chunk has a header: [r/<sub> | <post_id> | <permalink>]
    Ported from Lab 3 Rating Agent (make_context helper).
    """
    parts = []
    for chunk in retrieved:
        sub = chunk.get("subreddit", "unknown")
        pid = chunk.get("post_id", "")
        link = chunk.get("permalink", "")
        header = f"[r/{sub} | {pid} | {link}]"
        parts.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat helper — Ported from Lab 2 (model-selection switch)
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call the generation model and return the response text."""
    settings = load_settings()
    chosen_model = model or settings["generation_model"]
    provider = _provider_for_model(chosen_model, settings)
    chosen_temperature = settings["temperature"] if temperature is None else temperature
    chosen_max_tokens = settings["max_tokens"] if max_tokens is None else max_tokens

    if provider == "anthropic":
        return _anthropic_chat(
            messages=messages,
            model=chosen_model,
            settings=settings,
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
        )

    client = _get_gen_client(provider, settings)
    response = client.chat.completions.create(
        model=_provider_model_name(chosen_model, provider),
        messages=messages,
        temperature=chosen_temperature,
        max_tokens=chosen_max_tokens,
    )
    return response.choices[0].message.content or ""
