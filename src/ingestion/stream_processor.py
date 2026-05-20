"""Streaming and download pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.archive_search import sanitize_title, search_archive_org
from src.ingestion.archive_utils import resolve_direct_url
from src.ingestion.sector_analyzer import (
    analyze_sectors_with_llm,
    convert_llm_sectors_to_dict,
    display_sectors,
)


def filter_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter videos by identifier and title presence.

    Only filters out videos with invalid identifiers or titles.
    All videos with valid metadata are kept regardless of duration.

    Args:
        videos: List of video metadata dictionaries

    Returns:
        Filtered list of videos
    """
    filtered = []
    for video in videos:
        title = video.get("title", "")
        identifier = video.get("identifier", "")

        if not title or not identifier:
            continue

        filtered.append(video)

    return filtered


def display_videos(videos: list[dict[str, Any]]) -> tuple[bool, list[int]]:
    """Display available videos and get user selection.

    Args:
        videos: List of video metadata dictionaries

    Returns:
        Tuple of (all selected flag, selected_indices: list[int])
    """
    import sys

    print(f"\n{'='*70}")
    print(f"Found {len(videos)} lecture videos\n")
    print(f"{'='*70}\n")

    for i, video in enumerate(videos, 1):
        title = video.get("title", "")[:70]
        duration = video.get("duration_formatted", "Unknown")
        identifier = video.get("identifier", "")
        print(f"{i:3d}. [{duration}] {title}")
        print(f"      {identifier}")

    print(f"\n{'='*70}")

    if not sys.stdin.isatty():
        print("Non-interactive mode: downloading all videos")
        return True, list(range(len(videos)))

    while True:
        user_input = (
            input("Download all? (y/n) or select numbers (e.g. 1,3,5): ")
            .strip()
            .lower()
        )

        if user_input == "y":
            return True, list(range(len(videos)))

        elif user_input == "n":
            return False, []

        else:
            try:
                selected = [int(x.strip()) - 1 for x in user_input.split(",")]
                if all(0 <= idx < len(videos) for idx in selected):
                    return False, selected
                print(
                    f"Invalid selections. Please enter numbers between 1-{len(videos)}"
                )
            except ValueError:
                print(
                    "Invalid input. Please enter 'y', 'n', or comma-separated numbers"
                )


def process_video_from_url(url: str, video_name: str | None = None) -> bool:
    """Stream and process a video from URL without downloading it to disk.

    Extracts frames and transcribes directly from URL.
    If streaming fails, falls back to temporary file.

    Args:
        url: Video URL
        video_name: Optional video name (extracted from URL if not provided)

    Returns:
        True if successful, False otherwise
    """
    try:
        from src.ingestion.extractor import extract_frames_from_url
        from src.ingestion.transcriber import transcribe_url
        from src.dataset.builder import build_video_pairs

        if video_name is None:
            video_name = Path(url.split("?")[0]).stem or "video"

        resolved_url = resolve_direct_url(url)
        if resolved_url != url:
            url = resolved_url
            print(f"Resolved direct URL: {url}")

        print(f"\nProcessing video: {video_name}")
        print(f"Source: {url[:80]}...")
        print("=" * 70)

        print("\n1. Extracting frames from URL...")
        extract_frames_from_url(url, video_name=video_name)

        print("\n2. Transcribing audio from URL...")
        transcribe_url(url, video_name=video_name)

        frames_dir = Path("data/frames") / video_name
        if frames_dir.exists():
            print("\n3. Building training pairs...")
            build_video_pairs(frames_dir)

        print("\n✓ Video processed successfully (no local file saved)")
        return True

    except Exception as exc:
        print(f"\n✗ Processing failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def discover_and_download(
    topic: str, limit: int = 5, output_dir: str = "data/videos"
) -> None:
    """Main discovery workflow: search, filter, select, and process videos via streaming.

    Streams videos directly without downloading them to disk.
    Extracts frames and transcribes on the fly.
    Uses LLM for intelligent categorization when available.

    Args:
        topic: Search topic/keyword
        limit: Maximum videos to process
        output_dir: Unused (kept for backward compatibility)
    """
    print(f"\nDiscovering content for: \"{topic}\"")
    print("=" * 70)

    videos = search_archive_org(topic, limit=50)
    if not videos:
        print("No videos found")
        return

    sectors = None
    sector_descriptions = {}

    llm_result = analyze_sectors_with_llm(videos, topic)
    if llm_result:
        sectors, sector_descriptions = convert_llm_sectors_to_dict(llm_result, videos)

    if sectors:
        sector_idx = display_sectors(sectors, sector_descriptions)
        selected_sector = list(sectors.keys())[sector_idx]
        filtered_videos = filter_videos(sectors[selected_sector])
        found_label = selected_sector
    else:
        filtered_videos = filter_videos(videos)
        found_label = topic

    if not filtered_videos:
        print(f"No lecture videos found in {found_label}")
        return

    print(f"\n✓ Found {len(filtered_videos)} lecture videos for \"{found_label}\"")

    _, selected_indices = display_videos(filtered_videos)

    if not selected_indices:
        print("No videos selected")
        return

    selected_videos = [filtered_videos[i] for i in selected_indices]

    print(f"\n{'='*70}")
    print(f"Processing {len(selected_videos)} video(s) via streaming")
    print("(No permanent files saved to disk)")
    print(f"{'='*70}")

    success_count = 0
    failed_count = 0

    for i, video in enumerate(selected_videos, 1):
        title = video["title"][:60]
        print(f"\n[{i}/{len(selected_videos)}] {title}")
        print("-" * 70)

        url = video["url"]
        video_name = video.get("video_name") or sanitize_title(
            video.get("title", f"video_{i}")
        )

        if process_video_from_url(url, video_name):
            success_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*70}")
    print("Streaming Processing Summary")
    print(f"{'='*70}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print("\nFrames and transcripts saved to data/frames/ and data/transcripts/")
    print("Training pairs saved to data/pairs/")
    print("No video files saved (streamed directly)")
    print(f"{'='*70}\n")
