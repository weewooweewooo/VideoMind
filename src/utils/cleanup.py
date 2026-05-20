"""Cleanup utilities for VideoMind cached data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def clean_video(
    video_name: str,
    targets: list[str],
    pairs_dir: str | Path = "data/pairs",
) -> dict[str, Any]:
    """Delete specific cached data files for a given video.

    Args:
        video_name: Name of the video to clean (stems match video_name)
        targets: List of target types to clean: "pairs", "redis"
        pairs_dir: Directory containing pair JSON files

    Returns:
        Dictionary with keys: "deleted" (list of deleted items), "bytes_freed" (total bytes)
    """
    deleted: list[str] = []
    bytes_freed = 0

    if "pairs" in targets:
        pairs_path = Path(pairs_dir)
        for pair_file in pairs_path.glob(f"{video_name}*.json"):
            bytes_freed += pair_file.stat().st_size
            pair_file.unlink()
            deleted.append(f"pairs/{pair_file.name}")

    if "redis" in targets:
        from src.retrieval.store import VideoMindStore

        deleted_count = VideoMindStore().delete_video(video_name)
        deleted.append(f"redis/{video_name} ({deleted_count} docs)")

    return {
        "deleted": deleted,
        "bytes_freed": bytes_freed,
    }


def clean_all(
    targets: list[str],
    videos_dir: str | Path = "data/videos",
    pairs_dir: str | Path = "data/pairs",
) -> dict[str, Any]:
    """Clean selected cached data for all videos.

    Args:
        targets: List of target types to clean: "pairs", "redis"
        videos_dir: Unused, kept for backward compatibility
        pairs_dir: Directory containing pair JSON files

    Returns:
        Dictionary with keys: "deleted" (list of deleted items), "bytes_freed" (total bytes)
    """
    _ = videos_dir
    all_deleted: list[str] = []
    total_bytes_freed = 0

    if "pairs" in targets:
        pairs_path = Path(pairs_dir)
        if pairs_path.exists():
            for pair_file in pairs_path.glob("*.json"):
                total_bytes_freed += pair_file.stat().st_size
                pair_file.unlink()
                all_deleted.append(f"pairs/{pair_file.name}")

    if "redis" in targets:
        from src.retrieval.store import VideoMindStore

        store = VideoMindStore()
        for video_name in store.list_videos():
            deleted_count = store.delete_video(video_name)
            all_deleted.append(f"redis/{video_name} ({deleted_count} docs)")

    return {
        "deleted": all_deleted,
        "bytes_freed": total_bytes_freed,
    }
