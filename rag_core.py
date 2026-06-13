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

# ==== MODEL SWITCH (#-style, like the labs) ====
GENERATION_MODEL = "openai/gpt-4o-mini"                   # OpenRouter (default)
# GENERATION_MODEL = "gpt-4o-mini"                        # OpenAI direct
# GENERATION_MODEL = "claude-3-5-haiku-20241022"          # Anthropic / Claude
# GENERATION_MODEL = "meta-llama/llama-3.1-70b-instruct"  # Jetstream / open-source

EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI; 3072 dims, best accuracy

# Clients are instantiated lazily on first use so importing this module
# without API keys (e.g. in tests) doesn't raise an error.
_gen_client: OpenAI | None = None
_embed_client: OpenAI | None = None

# Tokenizer for the embedding model
_enc = tiktoken.get_encoding("cl100k_base")


def _get_gen_client() -> OpenAI:
    global _gen_client
    if _gen_client is None:
        is_openrouter = GENERATION_MODEL.startswith(
            ("openai/", "meta-llama/", "anthropic/", "google/", "mistral/")
        )
        _gen_client = OpenAI(
            api_key=OPENROUTER_API_KEY if is_openrouter else OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1" if is_openrouter else "https://api.openai.com/v1",
        )
    return _gen_client


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
    """Embed a list of strings using OpenAI text-embedding-3-large."""
    client = _get_embed_client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
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

    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    _scores, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


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
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call the generation model and return the response text."""
    chosen_model = model or GENERATION_MODEL
    response = _get_gen_client().chat.completions.create(
        model=chosen_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
