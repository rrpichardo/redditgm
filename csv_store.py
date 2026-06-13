"""Cumulative CSV storage helpers for redditgm runtime uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_FILES = {
    "combined": "gm_posts_with_comments.csv",
    "posts": "gm_posts.csv",
    "comments": "gm_comments.csv",
}


def existing_storage_mode(data_dir: Path) -> str | None:
    """Return the CSV storage mode already present in a runtime data folder."""
    has_combined = (data_dir / CANONICAL_FILES["combined"]).exists()
    has_split = (data_dir / CANONICAL_FILES["posts"]).exists() or (
        data_dir / CANONICAL_FILES["comments"]
    ).exists()
    if has_combined and has_split:
        return "mixed"
    if has_combined:
        return "combined"
    if has_split:
        return "split"
    return None


def _ordered_columns(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for frame in [existing, incoming]:
        if frame is None:
            continue
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    return columns


def _dedupe_key(frame: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "combined":
        comments = frame.get("comment_id", pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str)
        posts = frame.get("post_id", pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str)
        keys = []
        for idx, post_id, comment_id in zip(frame.index, posts, comments):
            if comment_id.strip():
                keys.append(f"comment:{comment_id.strip()}")
            elif post_id.strip():
                keys.append(f"post:{post_id.strip()}")
            else:
                keys.append(f"row:{idx}")
        return pd.Series(keys, index=frame.index)

    id_column = "id" if kind == "posts" else "comment_id"
    ids = frame.get(id_column, pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str)
    keys = [
        f"{kind}:{value.strip()}" if value.strip() else f"row:{idx}"
        for idx, value in zip(frame.index, ids)
    ]
    return pd.Series(keys, index=frame.index)


def _clear_conflicting_files(data_dir: Path, incoming_kinds: set[str]) -> None:
    """Keep one storage shape per tag: combined OR split."""
    if "combined" in incoming_kinds:
        for kind in ["posts", "comments"]:
            (data_dir / CANONICAL_FILES[kind]).unlink(missing_ok=True)
    else:
        (data_dir / CANONICAL_FILES["combined"]).unlink(missing_ok=True)


def merge_upload_frames(
    files: dict[str, tuple[str, bytes, pd.DataFrame]],
    data_dir: Path,
    *,
    reset: bool = False,
) -> list[dict[str, Any]]:
    """Append uploaded CSV frames into canonical runtime CSVs with ID dedupe.

    Overlapping exports are expected. For example, loading a one-day CSV and
    later loading a three-day CSV should grow the stored data only by new
    post/comment IDs.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    incoming_kinds = set(files)
    incoming_mode = "combined" if "combined" in incoming_kinds else "split"
    current_mode = existing_storage_mode(data_dir)

    if current_mode == "mixed":
        raise ValueError(
            f"{data_dir} contains both combined and split CSVs. Reset this run before loading more data."
        )
    if current_mode and current_mode != incoming_mode and not reset:
        raise ValueError(
            f"{data_dir} already uses {current_mode} CSV storage. Reset this run to switch to {incoming_mode} uploads."
        )

    if reset:
        _clear_conflicting_files(data_dir, incoming_kinds)

    stats: list[dict[str, Any]] = []
    for kind, (name, _raw_bytes, incoming) in files.items():
        target = data_dir / CANONICAL_FILES[kind]
        existing: pd.DataFrame | None = None
        if target.exists() and not reset:
            existing = pd.read_csv(target, dtype=str, keep_default_na=False)
            merged = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            merged = incoming.copy()

        columns = _ordered_columns(existing, incoming)
        if columns:
            merged = merged.reindex(columns=columns, fill_value="")

        before_dedupe = len(merged)
        deduped = (
            merged.assign(__dedupe_key=_dedupe_key(merged, kind))
            .drop_duplicates("__dedupe_key", keep="last")
            .drop(columns=["__dedupe_key"])
        )
        deduped.to_csv(target, index=False)

        stats.append({
            "file": name,
            "kind": kind,
            "existing_rows": 0 if existing is None else len(existing),
            "incoming_rows": len(incoming),
            "stored_rows": len(deduped),
            "duplicate_rows_removed": before_dedupe - len(deduped),
            "path": str(target),
        })

    return stats
