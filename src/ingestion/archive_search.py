"""Archive.org lecture video search and metadata helpers."""

from __future__ import annotations

import re
from typing import Any

import requests

from src.ingestion.archive_utils import fetch_archive_metadata


def sanitize_title(title: str) -> str:
    """Convert a video title into a stable filesystem-safe video name."""
    normalized = title.lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_-]+", "", normalized)
    normalized = re.sub(r"_+-+_+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_-")
    return (normalized or "video")[:60]


def format_duration(seconds: int) -> str:
    """Format seconds to HH:MM:SS format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fetch_video_duration(identifier: str) -> str:
    """Fetch video duration from archive.org metadata API with caching.

    Only fetches metadata for individual videos to avoid excessive API calls.
    Caches results to prevent duplicate requests for same identifier.

    Args:
        identifier: Archive.org item identifier

    Returns:
        Duration formatted as HH:MM:SS or "Unknown" if not found
    """
    try:
        data = fetch_archive_metadata(identifier)

        metadata = data.get("metadata", {})
        runtime = metadata.get("runtime")

        if runtime:
            try:
                duration_seconds = int(runtime)
                result = format_duration(duration_seconds)
                return result
            except (ValueError, TypeError):
                pass

        files = data.get("files", [])
        for file_info in files:
            if file_info.get("name", "").endswith(".mp4"):
                file_length = file_info.get("length")
                if file_length:
                    try:
                        duration_seconds = int(file_length)
                        result = format_duration(duration_seconds)
                        return result
                    except (ValueError, TypeError):
                        pass

        return "Unknown"

    except Exception:
        return "Unknown"


def fetch_video_metadata(item: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch relevant video metadata from archive.org result.

    Args:
        item: Archive.org search result item

    Returns:
        Extracted video metadata or None if invalid
    """
    try:
        identifier = item.get("identifier", "")
        title = item.get("title", "")

        if not identifier or not title:
            return None

        description = item.get("description", "")
        url = f"https://archive.org/details/{identifier}"

        duration_formatted = fetch_video_duration(identifier)

        return {
            "identifier": identifier,
            "title": title,
            "video_name": sanitize_title(title),
            "description": description if description else "",
            "duration_formatted": duration_formatted,
            "url": url,
        }

    except Exception as exc:
        return None


def filter_lecture_videos(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter archive.org search results to lecture videos with metadata."""
    lecture_keywords = [
        "lecture",
        "course",
        "class",
        "tutorial",
        "lesson",
        "seminar",
        "mit",
        "stanford",
        "cs",
    ]

    videos: list[dict[str, Any]] = []
    for doc in docs:
        title = doc.get("title", "").lower()

        has_lecture_keyword = any(kw in title for kw in lecture_keywords)
        if not has_lecture_keyword:
            continue

        video_info = fetch_video_metadata(doc)
        if video_info:
            videos.append(video_info)

    return videos


def search_archive_org(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Search archive.org for lecture videos.

    Args:
        query: Search query (topic/keyword)
        limit: Maximum number of results to retrieve

    Returns:
        List of video metadata dictionaries
    """
    try:
        search_query = f"{query} lecture mediatype:movies"
        search_url = "https://archive.org/advancedsearch.php"
        params = {
            "q": search_query,
            "fl": "identifier,title,description,runtime",
            "rows": limit,
            "output": "json",
        }

        print(f"Searching archive.org for: {query}")
        print(f"Query: {search_query}")
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        docs = data.get("response", {}).get("docs", [])

        print(f"Raw results from archive.org: {len(docs)}")

        videos = filter_lecture_videos(docs)

        print(f"Filtered to {len(videos)} lecture videos\n")
        return videos

    except Exception as exc:
        print(f"Error searching archive.org: {exc}")
        import traceback

        traceback.print_exc()
        return []
