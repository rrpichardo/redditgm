"""
build_rag_index.py - Embed posts into a FAISS index for RAG retrieval.

Reads split CSVs or the combined gm_posts_with_comments.csv, chunks each
distinct post into a single text blob, embeds it, and saves under the active
runtime run.

Run:
  .venv311/bin/python build_rag_index.py --tag gm_vehicle_on_demand
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from rag_core import build_faiss_index, chunk_csv_to_posts, embed_texts, save_index
from run_config import RunPaths
from settings import load_settings

console = Console()


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Build FAISS index from Reddit CSVs.")
    p.add_argument("--tag", default=settings["active_tag"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--index-path", default=None)
    p.add_argument("--limit", type=int, default=None, help="Cap posts for testing")
    p.add_argument("--jsonl-progress", action="store_true", help="Emit machine-readable progress")
    return p.parse_args()


def emit_progress(enabled: bool, **payload) -> None:
    if enabled:
        print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    run = RunPaths.resolve(args.tag, data_dir=args.data_dir, index_path=args.index_path)

    if not args.jsonl_progress:
        console.print(Panel.fit(
            "[bold cyan]GM Reddit Analytics - Build RAG Index[/]\n"
            f"Tag: [dim]{run.tag}[/]\n"
            f"Source: [dim]{run.data_dir}[/]\n"
            f"Output: [dim]{run.index_path}[/]",
            border_style="cyan",
        ))

    with console.status("[cyan]Chunking posts...[/]", spinner="dots") if not args.jsonl_progress else nullcontext():
        chunks = chunk_csv_to_posts(run.data_dir)
        if args.limit:
            chunks = chunks[: args.limit]

    emit_progress(args.jsonl_progress, step="chunk", completed=len(chunks), total=len(chunks))
    if not args.jsonl_progress:
        console.print(f"[green]+[/] {len(chunks)} posts chunked")

    if not chunks:
        message = "No posts found. Run collection or upload a CSV first."
        if args.jsonl_progress:
            emit_progress(args.jsonl_progress, step="error", message=message)
        else:
            console.print(f"[red]x[/] {message}")
        sys.exit(1)

    texts = [c["text"] for c in chunks]
    batch_size = 50
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    import numpy as np

    all_embeddings = []

    if args.jsonl_progress:
        for index, batch in enumerate(batches, start=1):
            all_embeddings.append(embed_texts(batch))
            emit_progress(args.jsonl_progress, step="embed", completed=index, total=len(batches))
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding batches...", total=len(batches))
            for batch in batches:
                all_embeddings.append(embed_texts(batch))
                progress.advance(task)

    embeddings = np.vstack(all_embeddings)
    if not args.jsonl_progress:
        console.print(f"[green]+[/] Embedded {len(chunks)} posts ({embeddings.shape[1]} dims)")

    if not args.jsonl_progress:
        with console.status("[cyan]Building FAISS index...[/]"):
            index = build_faiss_index(embeddings)
            save_index(index, chunks, run.index_path)
    else:
        index = build_faiss_index(embeddings)
        save_index(index, chunks, run.index_path)
        emit_progress(args.jsonl_progress, step="save", completed=len(chunks), total=len(chunks))

    if not args.jsonl_progress:
        console.print(f"[green]+[/] Index saved to [bold]{run.index_path}[/]")
        console.print("[dim]Next: run ask.py to query the index[/]")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
