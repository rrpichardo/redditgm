"""
build_rag_index.py — Embed posts into a FAISS index for RAG retrieval.

Reads gm_posts.csv + gm_comments.csv from the run's data_dir, chunks each post
into a single text blob, embeds with OpenAI text-embedding-3-large, saves to
the run's index_path.

Run:
  .venv311/bin/python build_rag_index.py --tag gm_vehicle_on_demand
  .venv311/bin/python build_rag_index.py --tag gm --jsonl-progress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rag_core import chunk_csv_to_posts, embed_texts, build_faiss_index, save_index
from run_config import Run

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from Reddit CSVs.")
    p.add_argument("--tag", default="gm_vehicle_on_demand",
                   help="Run tag — determines data source and output paths")
    p.add_argument("--limit", type=int, default=None, help="Cap posts (for testing)")
    p.add_argument("--jsonl-progress", action="store_true",
                   help="Emit one JSON line per batch (for Streamlit pipeline page)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = Run(args.tag)              # canonical paths derived from tag
    data_dir = run.data_dir
    index_path = run.index_path

    if not args.jsonl_progress:
        console.print(Panel.fit(
            "[bold cyan]GM Reddit Analytics — Build RAG Index[/]\n"
            f"Source: [dim]{data_dir}[/]\n"
            f"Output: [dim]{index_path}[/]",
            border_style="cyan"
        ))

    # Step 1: chunk posts (always quiet — this is fast)
    if not args.jsonl_progress:
        with console.status("[cyan]Chunking posts...[/]"):
            chunks = chunk_csv_to_posts(data_dir)
            if args.limit:
                chunks = chunks[: args.limit]
        console.print(f"[green]✓[/] {len(chunks)} posts chunked")
    else:
        chunks = chunk_csv_to_posts(data_dir)
        if args.limit:
            chunks = chunks[: args.limit]

    if not chunks:
        msg = "No posts found. Run the scraper first."
        if args.jsonl_progress:
            print(json.dumps({"error": msg}), flush=True)
        else:
            console.print(f"[red]✗[/] {msg}")
        sys.exit(1)

    # Step 2: embed in batches
    import numpy as np

    texts = [c["text"] for c in chunks]
    batch_size = 100
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)
    all_embeddings = []

    if args.jsonl_progress:
        # Machine-readable mode: one JSON line per batch so Streamlit can render a progress bar
        for batch_num, batch in enumerate(batches, 1):
            emb = embed_texts(batch)
            all_embeddings.append(emb)
            posts_done = min(batch_num * batch_size, len(chunks))
            print(json.dumps({
                "step": "embed",
                "batch": batch_num,
                "total_batches": total_batches,
                "posts_done": posts_done,
                "total_posts": len(chunks),
            }), flush=True)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding batches...", total=total_batches)
            for batch in batches:
                emb = embed_texts(batch)
                all_embeddings.append(emb)
                progress.advance(task)

    embeddings = np.vstack(all_embeddings)

    if not args.jsonl_progress:
        console.print(f"[green]✓[/] Embedded {len(chunks)} posts ({embeddings.shape[1]} dims)")

    # Step 3: build and save index
    if not args.jsonl_progress:
        with console.status("[cyan]Building FAISS index...[/]"):
            index = build_faiss_index(embeddings)
            save_index(index, chunks, index_path)
        console.print(f"[green]✓[/] Index saved to [bold]{index_path}[/]")
        console.print("[dim]Next: run ask.py to query the index[/]")
    else:
        index = build_faiss_index(embeddings)
        save_index(index, chunks, index_path)
        # Final line signals completion to the caller
        print(json.dumps({
            "step": "done",
            "index_path": str(index_path),
            "chunks": len(chunks),
        }), flush=True)


if __name__ == "__main__":
    main()
