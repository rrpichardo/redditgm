"""
rag_core.py — shared library for GM Reddit analytics.
Ported from: Lab 3 Rating Agent (chunk/embed/FAISS/retrieve/context),
             Lab 2 (model-selection switch, chat helper).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==== KEYS: paste in .env (see .env.example) ====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY", "")

# Clients are instantiated lazily on first use so importing this module
# without API keys (e.g. in tests) doesn't raise an error.

# Per-model client cache — rebuilt automatically when model string changes
_gen_client_cache: dict[str, OpenAI] = {}
_embed_client: OpenAI | None = None

# Tokenizer for the embedding model
_enc = tiktoken.get_encoding("cl100k_base")


def _build_gen_client(model: str) -> OpenAI:
    """Dispatch to the right provider based on model id format."""
    # OpenRouter: slash-namespaced (openai/gpt-4o-mini, anthropic/claude-3-5-haiku, etc.)
    _openrouter_prefixes = ("openai/", "meta-llama/", "anthropic/", "google/", "mistral/", "cohere/")
    if any(model.startswith(p) for p in _openrouter_prefixes):
        return OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # Bare Claude IDs (claude-3-5-haiku-20241022) — route through OpenRouter with anthropic/ prefix
    if model.startswith("claude-"):
        return OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # Jetstream: any model when JETSTREAM_BASE_URL env var is configured
    jetstream_url = os.getenv("JETSTREAM_BASE_URL", "")
    if jetstream_url and JETSTREAM_API_KEY:
        return OpenAI(api_key=JETSTREAM_API_KEY, base_url=jetstream_url)

    # Default: OpenAI direct
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_gen_client(model: str) -> OpenAI:
    """Return a cached client for this model, building one if needed."""
    if model not in _gen_client_cache:
        _gen_client_cache[model] = _build_gen_client(model)
    return _gen_client_cache[model]


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=OPENAI_API_KEY)
    return _embed_client


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

    if not posts_path.exists():
        raise FileNotFoundError(f"gm_posts.csv not found in {data_dir}")

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

def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed a list of strings using the configured embedding model."""
    from settings import get_settings
    # Read model from settings so it can be changed without touching this file
    model = get_settings().embedding_model
    client = _get_embed_client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
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

    # Early exit: FAISS search on an empty index crashes; nothing to return anyway
    if not chunks:
        return []

    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    _scores, indices = index.search(q_emb, top_k)
    # FAISS fills extra slots with -1 when top_k > index size — guard against that
    # (without `0 <=`, Python's negative indexing silently wraps to the last chunk)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


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
    from settings import get_settings
    s = get_settings()

    # Caller can override any param; fall back to settings values
    chosen_model = model or s.generation_model
    t = temperature if temperature is not None else s.temperature
    mt = max_tokens if max_tokens is not None else s.max_tokens

    # Bare Claude IDs need the anthropic/ prefix for OpenRouter routing
    routed_model = chosen_model
    if chosen_model.startswith("claude-") and "/" not in chosen_model:
        routed_model = f"anthropic/{chosen_model}"

    response = _get_gen_client(chosen_model).chat.completions.create(
        model=routed_model,
        messages=messages,
        temperature=t,
        max_tokens=mt,
    )
    return response.choices[0].message.content or ""
