"""Local-video transcription, segment normalization, and transcript caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pysbd

from src.config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)


def _normalize_segments(segments: Iterable[Any]) -> list[dict[str, str | float]]:
    """Normalize Whisper objects or cached dictionaries into timestamped text."""
    normalized: list[dict[str, str | float]] = []
    previous_start = previous_end = None
    for index, segment in enumerate(segments):
        if isinstance(segment, Mapping):
            raw_start = segment.get("start")
            raw_end = segment.get("end")
            raw_text = segment.get("text")
        else:
            raw_start = getattr(segment, "start", None)
            raw_end = getattr(segment, "end", None)
            raw_text = getattr(segment, "text", None)

        text = " ".join(str(raw_text or "").split())
        if not text:
            continue
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid transcript segment at index {index}") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"Invalid transcript segment at index {index}")
        if (
            previous_start is not None
            and previous_end is not None
            and (start < previous_start or end < previous_end)
        ):
            raise ValueError(f"Transcript segments are out of order at index {index}")

        normalized.append({"start": start, "end": end, "text": text})
        previous_start, previous_end = start, end
    if not normalized:
        raise ValueError("Transcript contains no usable segments")
    return normalized


def _transcribe_video(video_path: Path) -> dict[str, Any]:
    """Transcribe one local video with fixed Faster-Whisper defaults."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for transcription; "
            "install the CPU dependencies before running this command"
        ) from exc

    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        num_workers=4,
        cpu_threads=8,
    )
    raw_segments, info = model.transcribe(
        str(video_path),
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=False,
    )
    segments = _normalize_segments(raw_segments)
    duration = float(getattr(info, "duration", 0) or segments[-1]["end"])
    language = getattr(info, "language", None)
    return {
        "language": language if isinstance(language, str) and language else None,
        "duration": duration,
        "segments": segments,
    }


def _cache_directory() -> Path:
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
            / "VideoMind"
            / "cache"
        )
    return Path.home() / ".cache" / "videomind"


def _cache_path_for(video_path: Path) -> Path:
    """Return the stable cache file for one resolved local video path."""
    resolved_path = os.path.normcase(str(video_path.resolve()))
    cache_key = hashlib.sha256(resolved_path.encode("utf-8")).hexdigest()
    return _cache_directory() / f"{cache_key}.json"


def _cache_profile() -> dict[str, str | int]:
    return {
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "beam_size": WHISPER_BEAM_SIZE,
    }


def _load_cache(
    cache_path: Path,
    source: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a compatible normalized transcript or treat the entry as a miss."""
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, Mapping):
        return None
    if data.get("source") != source or data.get("profile") != profile:
        return None
    language = data.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        return None
    duration_value = data.get("duration")
    if isinstance(duration_value, bool):
        return None
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return None
    try:
        duration = float(duration_value)
        segments = _normalize_segments(raw_segments)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return {"language": language, "duration": duration, "segments": segments}


def _save_cache(
    cache_path: Path,
    source: Mapping[str, Any],
    profile: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> None:
    """Atomically write one inspectable transcript cache entry."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source": dict(source),
        "profile": dict(profile),
        "language": transcript.get("language"),
        "duration": transcript["duration"],
        "segments": transcript["segments"],
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(record, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, cache_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def prepare_transcript(
    transcript: Mapping[str, Any],
) -> list[dict[str, str | float]]:
    """Return validated timestamped segments without file or ASR access."""
    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Transcript must contain a segment list")
    return _normalize_segments(raw_segments)


def _reconstruct_sentences(
    segments: list[dict[str, str | float]],
) -> list[dict[str, str | float | list[int]]]:
    """Reconstruct exact pySBD sentence slices with source timestamps."""
    transcript_parts: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        if index:
            transcript_parts.append(" ")
            cursor += 1
        text = str(segment["text"])
        char_start = cursor
        transcript_parts.append(text)
        cursor += len(text)
        source_spans.append((char_start, cursor))

    transcript = "".join(transcript_parts)
    detected = pysbd.Segmenter(
        language="en",
        clean=False,
        char_span=True,
    ).segment(transcript)

    sentences: list[dict[str, str | float | list[int]]] = []
    source_cursor = 0
    for span in detected:
        if span.sent != transcript[span.start : span.end]:
            raise RuntimeError("pySBD rewrote canonical transcript text")
        start = span.start
        end = span.end
        while start < end and transcript[start].isspace():
            start += 1
        while end > start and transcript[end - 1].isspace():
            end -= 1
        if start >= end:
            continue

        while (
            source_cursor < len(source_spans)
            and source_spans[source_cursor][1] <= start
        ):
            source_cursor += 1
        if source_cursor == len(source_spans):
            raise RuntimeError("pySBD sentence has no source segment")
        contributor_end = source_cursor
        while (
            contributor_end < len(source_spans)
            and source_spans[contributor_end][0] < end
        ):
            contributor_end += 1
        source_indices = list(range(source_cursor, contributor_end))
        if not source_indices:
            raise RuntimeError("pySBD sentence has no source segment")

        sentences.append(
            {
                "start": segments[source_indices[0]]["start"],
                "end": segments[source_indices[-1]]["end"],
                "text": transcript[start:end],
                "source_segments": source_indices,
            }
        )
    if not sentences:
        raise RuntimeError("pySBD produced no sentences")
    return sentences


def ingest_video(
    video_path: str | Path,
) -> list[dict[str, str | float | list[int]]]:
    """Validate, cache or transcribe, and return timestamped sentences."""
    if not isinstance(video_path, (str, Path)) or not str(video_path).strip():
        raise ValueError("A local video path is required")
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    stat = path.stat()
    source = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    profile = _cache_profile()
    cache_path = _cache_path_for(path)
    transcript = _load_cache(cache_path, source, profile)
    if transcript is None:
        transcript = _transcribe_video(path)
        _save_cache(cache_path, source, profile, transcript)
    segments = prepare_transcript(transcript)
    return _reconstruct_sentences(segments)
