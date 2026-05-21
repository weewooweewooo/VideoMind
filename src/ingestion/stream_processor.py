"""Streaming and download pipeline orchestration."""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
import time
from typing import Any

from src.ingestion.archive_search import sanitize_title, search_archive_org
from src.ingestion.archive_utils import resolve_direct_url
from src.ingestion.sector_analyzer import (
    analyze_sectors_with_llm,
    convert_llm_sectors_to_dict,
    display_sectors,
)

logger = logging.getLogger(__name__)


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

    _print_video_list(videos)

    if not sys.stdin.isatty():
        print("Non-interactive mode: downloading all videos")
        return True, list(range(len(videos)))

    return _prompt_video_selection(len(videos))


def _print_video_list(videos: list[dict[str, Any]]) -> None:
    """Print discovered video choices."""
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


def _prompt_video_selection(video_count: int) -> tuple[bool, list[int]]:
    """Prompt for a video selection and return selected indices."""
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
                if all(0 <= idx < video_count for idx in selected):
                    return False, selected
                print(f"Invalid selections. Please enter numbers between 1-{video_count}")
            except ValueError:
                print(
                    "Invalid input. Please enter 'y', 'n', or comma-separated numbers"
                )


def _resolve_video_source(url: str) -> str:
    """Resolve an input URL to the direct source used by ingestion."""
    resolved_url = resolve_direct_url(url)
    if resolved_url != url:
        logger.debug("Resolved direct URL: %s", resolved_url)
    return resolved_url


def _extract_transcript_and_frames(
    url: str, video_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Extract frames and transcript concurrently."""
    from src.ingestion.extractor import extract_frames_to_memory
    from src.ingestion.transcriber import transcribe_to_memory

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        frames_future = executor.submit(extract_frames_to_memory, url)
        transcript_future = executor.submit(transcribe_to_memory, url, video_name)
        frames = frames_future.result()
        transcript = transcript_future.result()
    return frames, transcript, time.time() - start


def _index_processed_video(
    frames: list[dict[str, Any]],
    transcript: dict[str, Any],
    video_name: str,
) -> int:
    """Embed extracted frames and index them in Redis."""
    from src.retrieval.embedder import embed_frames_from_memory
    from src.retrieval.store import VideoMindStore

    embeddings = embed_frames_from_memory(frames)
    return VideoMindStore().index_frames_from_memory(embeddings, transcript, video_name)


def _handle_processing_error(exc: Exception) -> bool:
    """Print processing errors for the interactive CLI."""
    print(f"\nProcessing failed: {exc}")
    import traceback

    traceback.print_exc()
    return False


def process_video_from_url(url: str, video_name: str | None = None) -> bool:
    """Stream, transcribe, embed, and index a video without writing files.

    Args:
        url: Video URL
        video_name: Optional video name (extracted from URL if not provided)

    Returns:
        True if successful, False otherwise
    """
    try:
        if video_name is None:
            video_name = Path(url.split("?")[0]).stem or "video"

        url = _resolve_video_source(url)
        print(f"\nProcessing video: {video_name}")
        print("=" * 70)

        frames, transcript, elapsed = _extract_transcript_and_frames(url, video_name)
        print(f"Ingestion complete in {elapsed:.1f}s")

        indexed = _index_processed_video(frames, transcript, video_name)
        print(f"\nVideo processed successfully ({indexed} frames indexed)")
        return True

    except Exception as exc:
        return _handle_processing_error(exc)


def _select_videos_for_topic(
    videos: list[dict[str, Any]], topic: str
) -> tuple[str, list[dict[str, Any]]]:
    """Select the topic sector and return filtered videos."""
    llm_result = analyze_sectors_with_llm(videos, topic)
    if llm_result:
        sectors, descriptions = convert_llm_sectors_to_dict(llm_result, videos)
        if sectors:
            sector_idx = display_sectors(sectors, descriptions)
            selected_sector = list(sectors.keys())[sector_idx]
            return selected_sector, filter_videos(sectors[selected_sector])
    return topic, filter_videos(videos)


def _process_selected_videos(selected_videos: list[dict[str, Any]]) -> tuple[int, int]:
    """Process selected videos and return success/failure counts."""
    success_count = 0
    failed_count = 0
    for i, video in enumerate(selected_videos, 1):
        title = video["title"][:60]
        print(f"\n[{i}/{len(selected_videos)}] {title}")
        print("-" * 70)

        video_name = video.get("video_name") or sanitize_title(
            video.get("title", f"video_{i}")
        )
        if process_video_from_url(video["url"], video_name):
            success_count += 1
        else:
            failed_count += 1
    return success_count, failed_count


def _print_processing_summary(success_count: int, failed_count: int) -> None:
    """Print the streaming processing summary."""
    print(f"\n{'='*70}")
    print("Streaming Processing Summary")
    print(f"{'='*70}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print("\nNo video, frame, or transcript files saved (processed in memory)")
    print(f"{'='*70}\n")


def _print_discovery_header(topic: str) -> None:
    """Print discovery mode header."""
    print(f"\nDiscovering content for: \"{topic}\"")
    print("=" * 70)


def _print_selected_processing_header(selected_count: int) -> None:
    """Print selected video processing header."""
    print(f"\n{'='*70}")
    print(f"Processing {selected_count} video(s) via streaming")
    print("(No permanent files saved to disk)")
    print(f"{'='*70}")


def _selected_video_items(
    filtered_videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prompt for selected videos and return the chosen items."""
    _, selected_indices = display_videos(filtered_videos)
    return [filtered_videos[i] for i in selected_indices]


def discover_and_download(topic: str, limit: int = 5, output_dir: str = "data/videos") -> None:
    """Search, filter, select, and process lecture videos via streaming."""
    _ = output_dir
    _print_discovery_header(topic)

    videos = search_archive_org(topic, limit=50)
    if not videos:
        print("No videos found")
        return

    found_label, filtered_videos = _select_videos_for_topic(videos, topic)
    if not filtered_videos:
        print(f"No lecture videos found in {found_label}")
        return

    print(f"\nFound {len(filtered_videos)} lecture videos for \"{found_label}\"")
    selected_videos = _selected_video_items(filtered_videos)
    if not selected_videos:
        print("No videos selected")
        return

    _print_selected_processing_header(len(selected_videos))
    success_count, failed_count = _process_selected_videos(selected_videos)
    _print_processing_summary(success_count, failed_count)
