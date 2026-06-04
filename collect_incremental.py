#!/usr/bin/env python3
"""Incrementally collect Reddit posts and top comments into append-only CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

import redditwarp.SYNC


POST_CSV_FIELDS = [
    "run_id",
    "downloaded_at",
    "source_tool",
    "category",
    "time_filter",
    "id",
    "subreddit",
    "title",
    "author",
    "score",
    "comment_count",
    "created_at",
    "post_type",
    "content_source",
    "selftext",
    "outbound_url",
    "domain",
    "permalink",
    "content",
]

COMMENT_CSV_FIELDS = [
    "run_id",
    "downloaded_at",
    "post_id",
    "post_subreddit",
    "post_title",
    "post_permalink",
    "comment_rank",
    "comment_id",
    "author",
    "score",
    "body",
    "reply_count",
    "comment_permalink",
]

COMBINED_CSV_FIELDS = [
    "run_id",
    "downloaded_at",
    "source_tool",
    "category",
    "time_filter",
    "post_id",
    "post_subreddit",
    "post_title",
    "post_author",
    "post_score",
    "post_comment_count",
    "post_created_at",
    "post_type",
    "post_content_source",
    "post_content",
    "post_selftext",
    "post_outbound_url",
    "post_domain",
    "post_permalink",
    "comment_rank",
    "comment_id",
    "comment_author",
    "comment_score",
    "comment_body",
    "comment_reply_count",
    "comment_permalink",
]

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally collect new Reddit posts plus top comments."
    )
    parser.add_argument(
        "--subreddits-file",
        default="config/subreddits.txt",
        help="Text file with one subreddit per line.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--state-file", default="state/seen_posts.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--listing-limit",
        type=int,
        default=100,
        help="Newest posts to inspect per subreddit each run.",
    )
    parser.add_argument(
        "--comments-limit",
        type=int,
        default=5,
        help="Top-level top comments to collect per new post.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.8,
        help="Seconds between per-post comment requests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N processed new posts per subreddit.",
    )
    parser.add_argument(
        "--max-new-per-subreddit",
        type=int,
        default=0,
        help="Optional cap on new posts collected per subreddit. 0 means no cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report unseen posts without writing CSV or state.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def clean_subreddit(value: str) -> str:
    text = value.strip()
    if not text or text.startswith("#"):
        return ""
    if text.lower().startswith("r/"):
        text = text[2:]
    return text.strip().strip("/")


def read_subreddits(path: Path) -> list[str]:
    subreddits = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        subreddit = clean_subreddit(line)
        if subreddit and subreddit.lower() not in seen:
            subreddits.append(subreddit)
            seen.add(subreddit.lower())
    if not subreddits:
        raise ValueError(f"No subreddits found in {path}")
    return subreddits


def empty_state() -> dict[str, Any]:
    return {"version": 1, "updated_at": "", "subreddits": {}}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now().isoformat()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def state_entry(state: dict[str, Any], subreddit: str) -> dict[str, Any]:
    key = subreddit.lower()
    subreddits = state.setdefault("subreddits", {})
    entry = subreddits.setdefault(key, {"name": subreddit, "seen_post_ids": [], "last_run_at": ""})
    entry["name"] = subreddit
    entry.setdefault("seen_post_ids", [])
    return entry


def seed_state_from_posts_csv(path: Path, state: dict[str, Any]) -> int:
    if not path.exists():
        return 0

    added = 0
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            subreddit = row.get("subreddit", "")
            post_id = row.get("id", "")
            if not subreddit or not post_id:
                continue
            entry = state_entry(state, subreddit)
            seen = set(entry["seen_post_ids"])
            if post_id not in seen:
                entry["seen_post_ids"].append(post_id)
                added += 1
    return added


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def with_retries(label: str, fn: Callable[[], T], attempts: int = 3) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = 10 * attempt
            print(
                f"Warning: {label} failed on attempt {attempt}: {error}; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} failed unexpectedly")


def submission_post_type(submission: Any) -> str:
    class_name = type(submission).__name__
    if class_name == "LinkPost":
        return "link"
    if class_name == "TextPost":
        return "text"
    if class_name == "GalleryPost":
        return "gallery"
    return "unknown"


def permalink(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith("/"):
        return f"https://www.reddit.com{text}"
    return text


def comment_permalink(post_permalink: str, comment_id: Any) -> str:
    comment_id_text = "" if comment_id is None else str(comment_id)
    if not post_permalink or not comment_id_text:
        return ""
    return f"{post_permalink.rstrip('/')}/{comment_id_text}/"


def build_post_from_submission(submission: Any) -> dict[str, Any]:
    selftext = getattr(submission, "body", None) or ""
    outbound_url = getattr(submission, "link", None) or ""
    gallery_link = getattr(submission, "gallery_link", None)
    if gallery_link and not outbound_url:
        outbound_url = str(gallery_link)

    post_permalink = permalink(getattr(submission, "permalink", ""))
    content = selftext or outbound_url or post_permalink
    domain = urlparse(outbound_url).netloc if outbound_url else ""

    return {
        "id": getattr(submission, "id36", ""),
        "title": getattr(submission, "title", ""),
        "author": getattr(submission, "author_display_name", None) or "[deleted]",
        "score": getattr(submission, "score", ""),
        "subreddit": getattr(getattr(submission, "subreddit", None), "name", ""),
        "url": post_permalink,
        "created_at": submission.created_at.astimezone().isoformat(),
        "comment_count": getattr(submission, "comment_count", ""),
        "post_type": submission_post_type(submission),
        "content": content,
        "_detail_selftext": selftext,
        "_detail_outbound_url": outbound_url,
        "_detail_domain": domain,
    }


def pull_new_posts(
    client: redditwarp.SYNC.Client,
    subreddit: str,
    listing_limit: int,
) -> list[dict[str, Any]]:
    submissions = with_retries(
        f"listing r/{subreddit}",
        lambda: list(client.p.subreddit.pull.new(subreddit, listing_limit)),
    )
    return [build_post_from_submission(submission) for submission in submissions]


def post_type_value(post: dict[str, Any]) -> str:
    return str(post.get("post_type") or "")


def post_text(post: dict[str, Any]) -> str:
    selftext = str(post.get("_detail_selftext") or "").strip()
    if selftext:
        return selftext
    if post_type_value(post) == "text":
        return str(post.get("content") or "").strip()
    return ""


def post_outbound_url(post: dict[str, Any]) -> str:
    outbound_url = str(post.get("_detail_outbound_url") or "").strip()
    post_permalink = permalink(post.get("url", ""))
    if not outbound_url or outbound_url.rstrip("/") == post_permalink.rstrip("/"):
        return ""
    return outbound_url


def post_content(post: dict[str, Any]) -> tuple[str, str]:
    selftext = post_text(post)
    if selftext:
        return "selftext", selftext

    outbound_url = post_outbound_url(post)
    if outbound_url:
        return "outbound_url", outbound_url

    fallback = str(post.get("content") or "").strip()
    if fallback:
        return "content", fallback
    return "", ""


def build_comment(comment: Any) -> dict[str, Any]:
    return {
        "id": getattr(comment, "id36", ""),
        "author": getattr(comment, "author_display_name", None) or "[deleted]",
        "body": getattr(comment, "body", ""),
        "score": getattr(comment, "score", ""),
    }


def pull_top_comments(
    client: redditwarp.SYNC.Client,
    post_id: str,
    comments_limit: int,
) -> list[dict[str, Any]]:
    if comments_limit <= 0:
        return []

    def fetch_tree() -> Any:
        return client.p.comment_tree.fetch(post_id, sort="top", limit=comments_limit)

    tree = with_retries(f"comments {post_id}", fetch_tree)
    comments = []
    for node in tree.children[:comments_limit]:
        comment = build_comment(node.value)
        comment["reply_count"] = len(node.children)
        comments.append(comment)
    return comments


def post_csv_row(
    post: dict[str, Any],
    *,
    current_run_id: str,
    downloaded_at: str,
) -> dict[str, Any]:
    content_source, content = post_content(post)
    return {
        "run_id": current_run_id,
        "downloaded_at": downloaded_at,
        "source_tool": "redditwarp",
        "category": "new",
        "time_filter": "",
        "id": post.get("id", ""),
        "subreddit": post.get("subreddit", ""),
        "title": post.get("title", ""),
        "author": post.get("author", ""),
        "score": post.get("score", ""),
        "comment_count": post.get("comment_count", ""),
        "created_at": post.get("created_at", ""),
        "post_type": post_type_value(post),
        "content_source": content_source,
        "selftext": post_text(post),
        "outbound_url": post_outbound_url(post),
        "domain": post.get("_detail_domain", ""),
        "permalink": permalink(post.get("url", "")),
        "content": content,
    }


def comment_csv_rows(
    post: dict[str, Any],
    comments: list[dict[str, Any]],
    *,
    current_run_id: str,
    downloaded_at: str,
) -> list[dict[str, Any]]:
    post_permalink = permalink(post.get("url", ""))
    rows = []
    for index, comment in enumerate(comments, start=1):
        rows.append(
            {
                "run_id": current_run_id,
                "downloaded_at": downloaded_at,
                "post_id": post.get("id", ""),
                "post_subreddit": post.get("subreddit", ""),
                "post_title": post.get("title", ""),
                "post_permalink": post_permalink,
                "comment_rank": index,
                "comment_id": comment.get("id", ""),
                "author": comment.get("author", ""),
                "score": comment.get("score", ""),
                "body": comment.get("body", ""),
                "reply_count": comment.get("reply_count", ""),
                "comment_permalink": comment_permalink(post_permalink, comment.get("id", "")),
            }
        )
    return rows


def combined_csv_rows(
    post: dict[str, Any],
    comments: list[dict[str, Any]],
    *,
    current_run_id: str,
    downloaded_at: str,
) -> list[dict[str, Any]]:
    content_source, content = post_content(post)
    post_permalink = permalink(post.get("url", ""))
    base_row = {
        "run_id": current_run_id,
        "downloaded_at": downloaded_at,
        "source_tool": "redditwarp",
        "category": "new",
        "time_filter": "",
        "post_id": post.get("id", ""),
        "post_subreddit": post.get("subreddit", ""),
        "post_title": post.get("title", ""),
        "post_author": post.get("author", ""),
        "post_score": post.get("score", ""),
        "post_comment_count": post.get("comment_count", ""),
        "post_created_at": post.get("created_at", ""),
        "post_type": post_type_value(post),
        "post_content_source": content_source,
        "post_content": content,
        "post_selftext": post_text(post),
        "post_outbound_url": post_outbound_url(post),
        "post_domain": post.get("_detail_domain", ""),
        "post_permalink": post_permalink,
    }

    if not comments:
        return [
            {
                **base_row,
                "comment_rank": "",
                "comment_id": "",
                "comment_author": "",
                "comment_score": "",
                "comment_body": "",
                "comment_reply_count": "",
                "comment_permalink": "",
            }
        ]

    rows = []
    for index, comment in enumerate(comments, start=1):
        rows.append(
            {
                **base_row,
                "comment_rank": index,
                "comment_id": comment.get("id", ""),
                "comment_author": comment.get("author", ""),
                "comment_score": comment.get("score", ""),
                "comment_body": comment.get("body", ""),
                "comment_reply_count": comment.get("reply_count", ""),
                "comment_permalink": comment_permalink(post_permalink, comment.get("id", "")),
            }
        )
    return rows


def collect_subreddit(
    *,
    client: redditwarp.SYNC.Client,
    subreddit: str,
    args: argparse.Namespace,
    state: dict[str, Any],
    current_run_id: str,
    downloaded_at: str,
    data_dir: Path,
) -> dict[str, Any]:
    entry = state_entry(state, subreddit)
    seen_post_ids = set(entry["seen_post_ids"])

    posts = pull_new_posts(client, subreddit, args.listing_limit)
    new_posts = [post for post in posts if post.get("id") and post["id"] not in seen_post_ids]
    if args.max_new_per_subreddit > 0:
        new_posts = new_posts[: args.max_new_per_subreddit]

    print(
        f"r/{subreddit}: inspected {len(posts)} posts, {len(new_posts)} new",
        flush=True,
    )

    if args.dry_run:
        return {
            "subreddit": subreddit,
            "status": "dry_run",
            "inspected_posts": len(posts),
            "new_posts": len(new_posts),
            "comments": 0,
            "combined_rows": 0,
        }

    post_rows = []
    comment_rows = []
    combined_rows = []

    for index, post in enumerate(new_posts, start=1):
        post_id = str(post["id"])
        comments = pull_top_comments(client, post_id, args.comments_limit)
        post_rows.append(
            post_csv_row(
                post,
                current_run_id=current_run_id,
                downloaded_at=downloaded_at,
            )
        )
        comment_rows.extend(
            comment_csv_rows(
                post,
                comments,
                current_run_id=current_run_id,
                downloaded_at=downloaded_at,
            )
        )
        combined_rows.extend(
            combined_csv_rows(
                post,
                comments,
                current_run_id=current_run_id,
                downloaded_at=downloaded_at,
            )
        )
        seen_post_ids.add(post_id)

        if index % args.progress_every == 0 or index == len(new_posts):
            print(f"r/{subreddit}: collected {index}/{len(new_posts)} new posts", flush=True)
        if index < len(new_posts) and args.request_delay:
            time.sleep(args.request_delay)

    append_csv(data_dir / "gm_posts.csv", post_rows, POST_CSV_FIELDS)
    append_csv(data_dir / "gm_comments.csv", comment_rows, COMMENT_CSV_FIELDS)
    append_csv(data_dir / "gm_posts_with_comments.csv", combined_rows, COMBINED_CSV_FIELDS)

    entry["seen_post_ids"] = sorted(seen_post_ids)
    entry["last_run_at"] = downloaded_at

    return {
        "subreddit": subreddit,
        "status": "completed",
        "inspected_posts": len(posts),
        "new_posts": len(new_posts),
        "comments": len(comment_rows),
        "combined_rows": len(combined_rows),
    }


def main() -> None:
    args = parse_args()
    if args.listing_limit < 1:
        raise ValueError("--listing-limit must be at least 1")
    if args.comments_limit < 0:
        raise ValueError("--comments-limit must be 0 or greater")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be 0 or greater")

    subreddits_file = Path(args.subreddits_file)
    data_dir = Path(args.data_dir)
    state_file = Path(args.state_file)
    runs_dir = Path(args.runs_dir)
    current_run_id = run_id()
    downloaded_at = utc_now().isoformat()

    state = load_state(state_file)
    seeded = seed_state_from_posts_csv(data_dir / "gm_posts.csv", state)
    if seeded and not args.dry_run:
        save_state(state_file, state)

    subreddits = read_subreddits(subreddits_file)
    client = redditwarp.SYNC.Client()
    manifest_path = runs_dir / "run_manifest.jsonl"

    run_summary = {
        "run_id": current_run_id,
        "started_at": downloaded_at,
        "subreddits": [],
        "dry_run": args.dry_run,
        "listing_limit": args.listing_limit,
        "comments_limit": args.comments_limit,
    }

    for subreddit in subreddits:
        started = time.monotonic()
        try:
            summary = collect_subreddit(
                client=client,
                subreddit=subreddit,
                args=args,
                state=state,
                current_run_id=current_run_id,
                downloaded_at=downloaded_at,
                data_dir=data_dir,
            )
        except Exception as error:
            summary = {
                "subreddit": subreddit,
                "status": "failed",
                "error": str(error),
                "inspected_posts": 0,
                "new_posts": 0,
                "comments": 0,
                "combined_rows": 0,
            }
            print(f"Error: r/{subreddit} failed: {error}", file=sys.stderr, flush=True)
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        run_summary["subreddits"].append(summary)
        append_jsonl(manifest_path, {"run_id": current_run_id, **summary})
        if not args.dry_run:
            save_state(state_file, state)

    run_summary["completed_at"] = utc_now().isoformat()
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{current_run_id}.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_new = sum(item["new_posts"] for item in run_summary["subreddits"])
    total_rows = sum(item["combined_rows"] for item in run_summary["subreddits"])
    print(f"Run {current_run_id} complete: {total_new} new posts, {total_rows} combined rows")


if __name__ == "__main__":
    main()
