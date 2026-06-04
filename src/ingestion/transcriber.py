"""In-memory transcription helpers using faster-whisper."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from faster_whisper import WhisperModel

from src.ingestion.archive_utils import archive_identifier, fetch_archive_metadata
from src.utils.model_utils import resolve_device

logger = logging.getLogger(__name__)


def parse_srt_time(time_str: str) -> float:
    """Convert an SRT or VTT timestamp to seconds.

    Args:
        time_str: Timestamp in HH:MM:SS,mmm or HH:MM:SS.mmm format

    Returns:
        Timestamp in seconds
    """
    time_part = time_str.strip().replace(",", ".")
    hours_text, minutes_text, seconds_text = time_part.split(":")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + float(seconds_text)
    )


def get_archive_transcript(identifier: str) -> dict | None:
    """Fetch and parse an existing archive.org SRT or VTT transcript.

    Args:
        identifier: Archive.org item identifier

    Returns:
        Whisper-compatible transcript dictionary, or None if none exists
    """
    try:
        data = fetch_archive_metadata(identifier)
        files = data.get("files", [])

        transcript_extensions = [".srt", ".vtt"]
        transcript_file = None
        for file_info in files:
            name = file_info.get("name", "")
            if any(name.endswith(ext) for ext in transcript_extensions):
                transcript_file = name
                break

        if not transcript_file:
            return None

        file_url = f"https://archive.org/download/{identifier}/{transcript_file}"
        transcript_response = requests.get(file_url, timeout=10)
        transcript_response.raise_for_status()
        content = transcript_response.text

        if transcript_file.endswith(".vtt"):
            lines = content.strip().splitlines()
            if lines and lines[0].strip() == "WEBVTT":
                content = "\n".join(lines[1:])

        segments = []
        blocks = content.strip().split("\n\n")
        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                times = lines[1].split(" --> ")
                start = parse_srt_time(times[0])
                end = parse_srt_time(times[1])
                text = " ".join(lines[2:])
                segments.append(
                    {"id": i, "start": start, "end": end, "text": text, "words": []}
                )

        return {
            "video": identifier,
            "duration": segments[-1]["end"] if segments else 0,
            "language": "en",
            "segments": segments,
        }

    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        logger.warning("Could not fetch archive.org transcript for %s: %s", identifier, exc)
        return None


def _create_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    """Create a Whisper model from a model size name."""
    return WhisperModel(model_size, device=device, compute_type="int8")


def transcribe_to_memory(
    video_path_or_url: str,
    video_name: str,
    model_size: str = os.environ.get("WHISPER_MODEL", "medium"),
    device: str = os.environ.get("DEVICE", "auto"),
    compute_type: str = "int8",
) -> dict[str, Any]:
    """Transcribe audio from a path or URL into a Whisper-compatible dict."""
    identifier = archive_identifier(video_path_or_url)
    if identifier:
        existing = get_archive_transcript(identifier)
        if existing:
            existing["video"] = video_name
            return existing

    start_time = time.time()
    model = _create_whisper_model(model_size, resolve_device(device), compute_type)
    segments_gen, info = model.transcribe(
        video_path_or_url,
        beam_size=int(os.environ.get("WHISPER_BEAM_SIZE", "5")),
        word_timestamps=True,
        language="en",
    )
    raw_segments = list(segments_gen)
    segments = [
        {
            "id": i,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [
                {
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                }
                for word in (seg.words or [])
            ],
        }
        for i, seg in enumerate(raw_segments)
        if seg.text.strip()
    ]
    elapsed = time.time() - start_time
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    logger.info(
        "Transcribed %.0fs audio in %.1fs (%s segments)",
        duration,
        elapsed,
        len(segments),
    )
    return {
        "video": video_name,
        "duration": duration,
        "language": getattr(info, "language", "en") or "en",
        "segments": segments,
    }
