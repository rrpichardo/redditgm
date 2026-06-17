"""Run/tag path resolution for redditgm.

The tag is the stable dataset identity. Runtime artifacts live under
runtime/<tag>/ by default so app and pipeline runs do not dirty git.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from settings import load_settings


@dataclass(frozen=True)
class RunPaths:
    tag: str
    root: Path
    data_dir: Path
    db_path: Path
    index_path: Path
    report_dir: Path
    state_file: Path
    runs_dir: Path

    @classmethod
    def resolve(
        cls,
        tag: str | None = None,
        *,
        data_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        index_path: str | Path | None = None,
        report_dir: str | Path | None = None,
        settings: dict | None = None,
    ) -> "RunPaths":
        loaded = settings or load_settings()
        resolved_tag = tag or loaded["active_tag"]
        root = Path(loaded["runtime_dir"]) / resolved_tag
        return cls(
            tag=resolved_tag,
            root=root,
            data_dir=Path(data_dir) if data_dir else root / "data",
            db_path=Path(db_path) if db_path else root / "analytics.duckdb",
            index_path=Path(index_path) if index_path else root / "rag_index",
            report_dir=Path(report_dir) if report_dir else root / "reports" / "analytics",
            state_file=root / "state" / "seen_posts.json",
            runs_dir=root / "runs",
        )
