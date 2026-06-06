#!/usr/bin/env python3
"""Screen subreddit size and recent posting activity."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import redditwarp.SYNC

from collect_incremental import read_subreddits


FIELDS = [
    "subreddit",
    "subscriber_count",
    "sample_posts",
    "window_days",
    "posts_in_window",
    "sample_span_days",
    "avg_daily_posts_est",
    "avg_method",
    "latest_post_at",
    "oldest_sample_at",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate subreddit members and average daily posts from the newest listing."
    )
    parser.add_argument(
        "--subreddits-file",
        default="config/subreddits.txt",
        help="Text file with one subreddit per line.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Newest posts to sample per subreddit.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Recent activity window for average daily post estimates.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.8,
        help="Seconds to wait between subreddit requests.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional CSV output path. Defaults to stdout.",
    )
    return parser.parse_args()


def iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat()


def estimate_average(
    *,
    dates: list[datetime],
    sample_limit: int,
    window_days: int,
    now: datetime,
) -> tuple[int, float | None, float, str]:
    cutoff = now - timedelta(days=window_days)
    posts_in_window = sum(1 for value in dates if value >= cutoff)
    span_days = None
    avg = posts_in_window / window_days
    method = f"posts_last_{window_days}d/window_days"

    if len(dates) >= 2:
        newest = max(dates)
        oldest = min(dates)
        span_days = max((newest - oldest).total_seconds() / 86400, 1 / 24)
        if sample_limit >= 30 and len(dates) >= sample_limit and span_days < window_days:
            avg = len(dates) / span_days
            method = f"latest_{len(dates)}_posts/sample_span_days"

    return posts_in_window, span_days, avg, method


def subscriber_count_from_posts(posts: list[Any]) -> int | None:
    if not posts:
        return None
    subreddit = getattr(posts[0], "subreddit", None)
    if subreddit is None:
        return None
    return getattr(subreddit, "subscriber_count", None)


def screen_subreddit(
    client: redditwarp.SYNC.Client,
    subreddit: str,
    *,
    sample_limit: int,
    window_days: int,
    now: datetime,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subreddit": f"r/{subreddit}",
        "subscriber_count": "",
        "sample_posts": 0,
        "window_days": window_days,
        "posts_in_window": "",
        "sample_span_days": "",
        "avg_daily_posts_est": "",
        "avg_method": "",
        "latest_post_at": "",
        "oldest_sample_at": "",
        "status": "failed",
        "error": "",
    }

    try:
        posts = list(client.p.subreddit.pull.new(subreddit, sample_limit))
        dates = [post.created_at for post in posts if getattr(post, "created_at", None)]
        posts_in_window, span_days, avg, method = estimate_average(
            dates=dates,
            sample_limit=sample_limit,
            window_days=window_days,
            now=now,
        )

        row.update(
            {
                "subscriber_count": subscriber_count_from_posts(posts) or "",
                "sample_posts": len(posts),
                "posts_in_window": posts_in_window,
                "sample_span_days": round(span_days, 1) if span_days is not None else "",
                "avg_daily_posts_est": round(avg, 1),
                "avg_method": method,
                "latest_post_at": iso(max(dates) if dates else None),
                "oldest_sample_at": iso(min(dates) if dates else None),
                "status": "ok",
            }
        )
    except Exception as error:  # Keep one bad/private subreddit from aborting the screen.
        row["error"] = f"{type(error).__name__}: {error}"

    return row


def write_rows(rows: list[dict[str, Any]], out_path: str) -> None:
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be at least 1")
    if args.window_days < 1:
        raise ValueError("--window-days must be at least 1")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be 0 or greater")

    subreddits = read_subreddits(Path(args.subreddits_file))
    client = redditwarp.SYNC.Client()
    now = datetime.now(timezone.utc)
    rows = []

    for index, subreddit in enumerate(subreddits, start=1):
        rows.append(
            screen_subreddit(
                client,
                subreddit,
                sample_limit=args.sample_limit,
                window_days=args.window_days,
                now=now,
            )
        )
        if index < len(subreddits) and args.request_delay:
            time.sleep(args.request_delay)

    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
