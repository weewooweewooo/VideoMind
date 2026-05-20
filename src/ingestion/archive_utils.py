"""Shared archive.org metadata and URL resolution helpers."""

from __future__ import annotations

from functools import lru_cache
import re
from urllib.parse import urlparse

import requests


@lru_cache(maxsize=256)
def fetch_archive_metadata(identifier: str) -> dict:
    """Fetch archive.org metadata for an item identifier with process-local caching.

    Args:
        identifier: Archive.org item identifier

    Returns:
        Metadata response dictionary from archive.org
    """
    url = f"https://archive.org/metadata/{identifier}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def resolve_direct_url(url: str) -> str:
    """Resolve archive.org details URLs to direct MP4 download URLs.

    Args:
        url: Original video URL

    Returns:
        Direct MP4 URL for archive.org details pages, otherwise the original URL

    Raises:
        ValueError: If an archive.org item has no MP4 file
        requests.RequestException: If the metadata request fails
    """
    if "archive.org/details/" not in url:
        return url

    identifier = url.split("/details/")[1].split("/")[0]
    metadata = fetch_archive_metadata(identifier)
    files = metadata.get("files", [])

    mp4_files = [
        file_info
        for file_info in files
        if file_info.get("name", "").endswith(".mp4")
    ]
    if not mp4_files:
        raise ValueError(f"No mp4 file found for {identifier}")

    identifier_words = {
        word
        for word in re.split(r"[^a-z0-9]+", identifier.lower())
        if word
    }

    def name_score(file_info: dict) -> int:
        filename = file_info.get("name", "").lower()
        return sum(1 for word in identifier_words if word in filename)

    scored_files = [(name_score(file_info), file_info) for file_info in mp4_files]
    best_score = max(score for score, _ in scored_files)
    if best_score > 0:
        best_file = max(scored_files, key=lambda item: item[0])[1]
    else:
        best_file = max(mp4_files, key=lambda f: int(f.get("size", 0)))

    filename = best_file["name"]
    parsed = urlparse(url)
    base_url = f"{parsed.scheme or 'https'}://archive.org/download/{identifier}"
    return f"{base_url}/{filename}"
