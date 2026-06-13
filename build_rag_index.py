"""
build_rag_index.py — Embed posts into a FAISS index for RAG retrieval.

Reads gm_posts.csv + gm_comments.csv, chunks each post into a single text blob,
embeds with OpenAI text-embedding-3-large, saves to state/rag/<tag>/.

Run:
  .venv311/bin/python build_rag_index.py --data-dir data/gm_vehicle_on_demand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rag_core import chunk_csv_to_posts, embed_texts, build_faiss_index, save_index

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from Reddit CSVs.")
    p.add_argument("--data-dir", default="data/gm_vehicle_on_demand")
    p.add_argument("--tag", default="gm_vehicle_on_demand",
                   help="Subdirectory name under state/rag/ for this index")
    p.add_argument("--limit", type=int, default=None, help="Cap posts (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    index_path = Path("state") / "rag" / args.tag

    console.print(Panel.fit(
        "[bold cyan]GM Reddit Analytics — Build RAG Index[/]\n"
        f"Source: [dim]{data_dir}[/]\n"
        f"Output: [dim]{index_path}[/]",
        border_style="cyan"
    ))

    # Step 1: chunk posts
    with console.status("[cyan]Chunking posts...[/]"):
        chunks = chunk_csv_to_posts(data_dir)
        if args.limit:
            chunks = chunks[: args.limit]
    console.print(f"[green]✓[/] {len(chunks)} posts chunked")

    if not chunks:
        console.print("[red]✗[/] No posts found. Run run_gm_vehicle_on_demand.sh first.")
        sys.exit(1)

    # Step 2: embed in batches with progress bar
    texts = [c["text"] for c in chunks]
    batch_size = 100
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    import numpy as np
    all_embeddings = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding batches...", total=len(batches))
        for batch in batches:
            emb = embed_texts(batch)
            all_embeddings.append(emb)
            progress.advance(task)

    embeddings = np.vstack(all_embeddings)
    console.print(f"[green]✓[/] Embedded {len(chunks)} posts ({embeddings.shape[1]} dims)")

    # Step 3: build and save index
    with console.status("[cyan]Building FAISS index...[/]"):
        index = build_faiss_index(embeddings)
        save_index(index, chunks, index_path)

    console.print(f"[green]✓[/] Index saved to [bold]{index_path}[/]")
    console.print("[dim]Next: run ask.py to query the index[/]")


if __name__ == "__main__":
    main()
