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
    overlap_words: int = 0,
) -> list[dict[str, str | float]]:
    """Merge adjacent transcript segments into ordered timestamped chunks.

    Zero overlap preserves the original word-limited behavior. Positive overlap
    reuses complete trailing segments so every timestamp remains tied to real
    transcript segment boundaries.
    """
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise ValueError("overlap_words must be an integer")
    if overlap_words < 0:
        raise ValueError("overlap_words must not be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be less than max_words")

    normalized = normalize_transcript_segments(segments)

    if overlap_words:
        return _chunk_with_segment_overlap(
            normalized,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    return _chunk_without_overlap(normalized, max_words=max_words)


def _format_chunk(
    segments: list[dict[str, str | float]],
) -> dict[str, str | float]:
    return {
        "start": float(segments[0]["start"]),
        "end": float(segments[-1]["end"]),
        "text": " ".join(str(segment["text"]) for segment in segments),
    }


def _chunk_without_overlap(
    normalized: list[dict[str, str | float]],
    *,
    max_words: int,
) -> list[dict[str, str | float]]:
    """Preserve the original non-overlapping chunk construction exactly."""
    chunks: list[dict[str, str | float]] = []
    current: list[dict[str, str | float]] = []
    current_word_count = 0

    def flush() -> None:
        nonlocal current, current_word_count
        if not current:
            return
        chunks.append(_format_chunk(current))
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


def _chunk_with_segment_overlap(
    normalized: list[dict[str, str | float]],
    *,
    max_words: int,
    overlap_words: int,
) -> list[dict[str, str | float]]:
    """Build overlapping chunks by reusing only complete transcript segments."""
    word_counts = [len(str(segment["text"]).split()) for segment in normalized]
    chunks: list[dict[str, str | float]] = []
    start_index = 0

    while start_index < len(normalized):
        end_index = start_index
        chunk_word_count = 0

        while end_index < len(normalized):
            segment_word_count = word_counts[end_index]
            if (
                end_index > start_index
                and chunk_word_count + segment_word_count > max_words
            ):
                break
            chunk_word_count += segment_word_count
            end_index += 1
            if chunk_word_count >= max_words:
                break

        chunks.append(_format_chunk(normalized[start_index:end_index]))
        if end_index >= len(normalized):
            break

        next_start = end_index
        retained_words = 0
        while next_start - 1 > start_index and retained_words < overlap_words:
            next_start -= 1
            retained_words += word_counts[next_start]

        start_index = max(start_index + 1, next_start)

    return chunks
