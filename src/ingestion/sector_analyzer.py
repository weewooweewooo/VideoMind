"""Sector categorization for discovered lecture videos."""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any

logger = logging.getLogger(__name__)


def is_ollama_available(host: str = "localhost", port: int = 11434) -> bool:
    """Check whether the local Ollama server is reachable."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def analyze_sectors_with_llm(
    videos: list[dict[str, Any]], topic: str
) -> dict[str, Any] | None:
    """Use Ollama LLaMA 3.2 3B to intelligently categorize videos into sectors.

    Args:
        videos: List of video metadata dictionaries
        topic: Search topic/keyword for context

    Returns:
        Dictionary with sectors or None if LLM analysis fails
    """
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        logger.warning("langchain_ollama not installed; skipping sector selection")
        return None

    titles_list = "\n".join(
        [
            f"{i}. {v['title']} - {v.get('description', '')[:100]}"
            for i, v in enumerate(videos)
        ]
    )

    prompt = f"""You are an academic content classifier.

I have {len(videos)} lecture videos about "{topic}" from archive.org.
Analyze their titles and descriptions, then group them into 4-7 
meaningful academic sectors.

Videos:
{titles_list}

Return ONLY valid JSON, no explanation, no markdown, no backticks:
{{
  "sectors": [
    {{
      "name": "Sector Name",
      "description": "One line description",
      "video_indices": [0, 2, 5]
    }}
  ]
}}

Rules:
- Every video must belong to exactly one sector
- Sector names should be specific and academic
- 4-7 sectors maximum
- Return ONLY the JSON object"""

    try:
        if not is_ollama_available():
            logger.warning("Ollama server not available at http://localhost:11434")
            logger.warning("LLM categorization failed; skipping sector selection")
            return None

        llm = OllamaLLM(model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)

        logger.info("Analyzing content with local AI (LLaMA 3.2)")
        response = llm.invoke(prompt)

        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean)
        return result
    except Exception as exc:
        logger.error("LLM analysis failed: %s", exc)
        logger.warning("LLM categorization failed; skipping sector selection")
        return None


def convert_llm_sectors_to_dict(
    llm_result: dict[str, Any], videos: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Convert LLM sector result to dict format compatible with display_sectors.

    Args:
        llm_result: Result from analyze_sectors_with_llm
        videos: Original videos list

    Returns:
        Tuple of (sectors_dict, descriptions_dict)
    """
    sectors: dict[str, list[dict[str, Any]]] = {}
    sectors_meta: dict[str, str] = {}

    try:
        for sector_info in llm_result.get("sectors", []):
            name = sector_info.get("name", "Unknown Sector")
            description = sector_info.get("description", "")
            indices = sector_info.get("video_indices", [])

            sectors[name] = [videos[i] for i in indices if 0 <= i < len(videos)]
            sectors_meta[name] = description

        return sectors, sectors_meta
    except Exception:
        return {}, {}


def display_sectors(
    sectors: dict[str, list[dict[str, Any]]],
    sector_descriptions: dict[str, str] | None = None,
) -> int:
    """Display available sectors and get user selection.

    Args:
        sectors: Dictionary of sectors and their videos
        sector_descriptions: Optional descriptions for each sector

    Returns:
        Selected sector index (0-based)
    """
    import sys

    sector_names = list(sectors.keys())
    if sector_descriptions is None:
        sector_descriptions = {}

    print(f"\n{'='*70}")
    print("Found these sectors:")
    print(f"{'='*70}\n")

    for i, sector in enumerate(sector_names, 1):
        count = len(sectors[sector])
        description = sector_descriptions.get(sector, "")
        print(f"{i:2d}. {sector:<45} ({count} videos)")
        if description:
            print(f"    {description}")

    print(f"\n{'='*70}")

    if not sys.stdin.isatty():
        print("Non-interactive mode: selecting first sector automatically")
        return 0

    while True:
        try:
            choice = input("Select a sector (1-{}): ".format(len(sector_names)))
            idx = int(choice) - 1
            if 0 <= idx < len(sector_names):
                return idx
            print(f"Invalid selection. Please enter 1-{len(sector_names)}")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1-{len(sector_names)}")
