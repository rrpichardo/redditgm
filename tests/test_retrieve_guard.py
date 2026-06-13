"""
tests/test_retrieve_guard.py — Unit tests for rag_core.retrieve() FAISS negative-index guard.

FAISS fills extra slots with -1 when top_k > number of indexed vectors.
Without a guard, chunks[-1] silently returns the last chunk (Python wraps negatives),
which is a silent correctness bug — not a crash, but wrong results.
"""
from __future__ import annotations

import numpy as np
import pytest


def make_dummy_index(n_vectors: int, dim: int = 8):
    """Build a tiny FAISS IndexFlatIP with n_vectors entries, normalized for cosine sim."""
    import faiss

    vecs = np.random.randn(n_vectors, dim).astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def test_retrieve_when_top_k_exceeds_index_size(monkeypatch):
    """
    Requesting more results than the index holds should return only the
    real matches (3), not crash or silently include wrong chunks via chunks[-1].
    """
    import faiss
    from rag_core import retrieve

    index = make_dummy_index(n_vectors=3, dim=8)
    chunks = [
        {"post_id": f"p{i}", "text": f"text {i}", "subreddit": "x", "permalink": ""}
        for i in range(3)
    ]

    # Mock embed_texts to return a normalized vector without hitting the API
    def fake_embed(texts, batch_size=100):
        vecs = np.random.randn(len(texts), 8).astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    monkeypatch.setattr("rag_core.embed_texts", fake_embed)

    # top_k=10 with only 3 vectors — FAISS fills the extra 7 slots with -1
    results = retrieve("test query about cars", index, chunks, top_k=10)

    # Must return exactly 3 real results, no crash, no phantom entries
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for r in results:
        assert r in chunks, f"Result {r} not in original chunks"


def test_retrieve_returns_empty_for_empty_index(monkeypatch):
    """
    Empty chunks list returns [] immediately — FAISS search on empty index would crash
    without the early-exit guard.
    """
    import faiss
    from rag_core import retrieve

    dim = 8
    index = faiss.IndexFlatIP(dim)  # empty index — no vectors added
    chunks: list = []

    # fake_embed won't be called because the early-exit fires first,
    # but set it up anyway to avoid any accidental API call
    def fake_embed(texts, batch_size=100):
        return np.random.randn(len(texts), dim).astype("float32")

    monkeypatch.setattr("rag_core.embed_texts", fake_embed)

    results = retrieve("any question", index, chunks, top_k=5)
    assert results == [], f"Expected [], got {results}"
