"""Cleanup utilities for VideoMind cached data."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def get_directory_size(path: Path) -> int:
    """Calculate total size in bytes of a directory and its contents."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def clean_video(
    video_name: str,
    targets: list[str],
    frames_root: str | Path = "data/frames",
    transcripts_dir: str | Path = "data/transcripts",
    pairs_dir: str | Path = "data/pairs",
    chroma_dir: str | Path = "data/chroma",
) -> dict[str, Any]:
    """Delete specific cached data files for a given video.

    Args:
        video_name: Name of the video to clean (stems match video_name)
        targets: List of target types to clean: "frames", "transcripts", "pairs", "chroma"
        frames_root: Root directory containing frame subdirectories
        transcripts_dir: Directory containing transcript JSON files
        pairs_dir: Directory containing pair JSON files
        chroma_dir: Root ChromaDB directory

    Returns:
        Dictionary with keys: "deleted" (list of deleted items), "bytes_freed" (total bytes)
    """
    deleted: list[str] = []
    bytes_freed = 0

    if "frames" in targets:
        frames_path = Path(frames_root) / video_name
        if frames_path.exists():
            bytes_freed += get_directory_size(frames_path)
            shutil.rmtree(frames_path)
            deleted.append(f"frames/{video_name}")

    if "transcripts" in targets:
        transcript_path = Path(transcripts_dir) / f"{video_name}.json"
        if transcript_path.exists():
            bytes_freed += transcript_path.stat().st_size
            transcript_path.unlink()
            deleted.append(f"transcripts/{video_name}.json")

    if "pairs" in targets:
        pairs_path = Path(pairs_dir)
        for pair_file in pairs_path.glob(f"{video_name}*.json"):
            bytes_freed += pair_file.stat().st_size
            pair_file.unlink()
            deleted.append(f"pairs/{pair_file.name}")

    if "chroma" in targets:
        deleted.append(f"chroma/{video_name}")

    return {
        "deleted": deleted,
        "bytes_freed": bytes_freed,
    }


def clean_all(
    targets: list[str],
    videos_dir: str | Path = "data/videos",
    frames_root: str | Path = "data/frames",
    transcripts_dir: str | Path = "data/transcripts",
    pairs_dir: str | Path = "data/pairs",
    chroma_dir: str | Path = "data/chroma",
) -> dict[str, Any]:
    """Clean selected cached data for all videos.

    Args:
        targets: List of target types to clean: "frames", "transcripts", "pairs", "chroma"
        videos_dir: Directory containing video files to determine video names
        frames_root: Root directory containing frame subdirectories
        transcripts_dir: Directory containing transcript JSON files
        pairs_dir: Directory containing pair JSON files
        chroma_dir: Root ChromaDB directory

    Returns:
        Dictionary with keys: "deleted" (list of deleted items), "bytes_freed" (total bytes)
    """
    video_dir = Path(videos_dir)
    if not video_dir.exists():
        return {"deleted": [], "bytes_freed": 0}

    all_deleted: list[str] = []
    total_bytes_freed = 0

    video_files = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )

    for video_file in video_files:
        video_name = video_file.stem
        result = clean_video(
            video_name,
            targets,
            frames_root=frames_root,
            transcripts_dir=transcripts_dir,
            pairs_dir=pairs_dir,
            chroma_dir=chroma_dir,
        )
        all_deleted.extend(result["deleted"])
        total_bytes_freed += result["bytes_freed"]

    return {
        "deleted": all_deleted,
        "bytes_freed": total_bytes_freed,
    }
