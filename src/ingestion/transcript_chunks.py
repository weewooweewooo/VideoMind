"""Dependency-free validation and chunking for timestamped transcripts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_CHUNK_WORDS = 70


def _segment_value(segment: Any, field: str) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(field)
    return getattr(segment, field, None)


def normalize_transcript_segments(
    segments: Iterable[Any],
) -> list[dict[str, str | float]]:
    """Validate timestamped segments and return their minimal JSON-safe shape."""
    normalized: list[dict[str, str | float]] = []
    previous_start: float | None = None
    previous_end: float | None = None

    for index, segment in enumerate(segments):
        text = " ".join(str(_segment_value(segment, "text") or "").split())
        if not text:
            continue

        raw_start = _segment_value(segment, "start")
        raw_end = _segment_value(segment, "end")
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Transcript segment {index} has non-numeric timestamps"
            ) from exc

        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Transcript segment {index} has non-finite timestamps")
        if start < 0:
            raise ValueError(f"Transcript segment {index} starts before zero")
        if end <= start:
            raise ValueError(
                f"Transcript segment {index} must end after it starts"
            )
        if previous_start is not None and start < previous_start:
            raise ValueError(f"Transcript segment {index} is out of timestamp order")
        if previous_end is not None and end < previous_end:
            raise ValueError(f"Transcript segment {index} ends out of timestamp order")

        normalized.append({"start": start, "end": end, "text": text})
        previous_start = start
        previous_end = end

    if not normalized:
        raise ValueError("Transcript contains no usable segments")
    return normalized


def chunk_transcript_segments(
    segments: Iterable[Any],
    max_words: int = DEFAULT_CHUNK_WORDS,
) -> list[dict[str, str | float]]:
    """Merge adjacent transcript segments into ordered timestamped chunks."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    normalized = normalize_transcript_segments(segments)
    chunks: list[dict[str, str | float]] = []
    current: list[dict[str, str | float]] = []
    current_word_count = 0

    def flush() -> None:
        nonlocal current, current_word_count
        if not current:
            return
        chunks.append(
            {
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": " ".join(str(segment["text"]) for segment in current),
            }
        )
        current = []
        current_word_count = 0

    for segment in normalized:
        words = str(segment["text"]).split()
        segment_parts = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": " ".join(words[offset : offset + max_words]),
            }
            for offset in range(0, len(words), max_words)
        ]
        for segment_part in segment_parts:
            part_word_count = len(str(segment_part["text"]).split())
            if current and current_word_count + part_word_count > max_words:
                flush()
            current.append(segment_part)
            current_word_count += part_word_count

    flush()
    return chunks
