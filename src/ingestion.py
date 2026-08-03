"""Single-video transcription, chunking, and transcript caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.config import (
    CACHE_SCHEMA_VERSION,
    TRANSCRIPT_CHUNK_WORDS,
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
            raise ValueError(
                f"Transcript segment {index} has non-numeric timestamps"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Transcript segment {index} has non-finite timestamps")
        if start < 0:
            raise ValueError(f"Transcript segment {index} starts before zero")
        if end <= start:
            raise ValueError(f"Transcript segment {index} must end after it starts")
        if previous_start is not None and start < previous_start:
            raise ValueError(f"Transcript segment {index} is out of timestamp order")
        if previous_end is not None and end < previous_end:
            raise ValueError(f"Transcript segment {index} ends out of timestamp order")

        normalized.append({"start": start, "end": end, "text": text})
        previous_start, previous_end = start, end
    if not normalized:
        raise ValueError("Transcript contains no usable segments")
    return normalized


def _build_chunks(segments: list[dict[str, str | float]]) -> list[dict[str, Any]]:
    """Build fixed deterministic chunks without reusing transcript segments."""
    chunks: list[dict[str, str | float]] = []
    current: list[dict[str, str | float]] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        chunks.append(
            {
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": " ".join(str(segment["text"]) for segment in current),
            }
        )
        current = []
        current_words = 0

    for segment in segments:
        segment_words = len(str(segment["text"]).split())
        if current and current_words + segment_words > TRANSCRIPT_CHUNK_WORDS:
            flush()
        current.append(segment)
        current_words += segment_words
    flush()
    for chunk, next_chunk in zip(chunks, chunks[1:]):
        if chunk["end"] > next_chunk["start"]:
            if next_chunk["start"] <= chunk["start"]:
                raise ValueError("Transcript cannot form non-overlapping chunks")
            chunk["end"] = float(next_chunk["start"])
    return chunks


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
        WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE,
        num_workers=4, cpu_threads=8,
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


def _cache_path_for(video_path: Path) -> Path:
    """Return a local cache identity, not a content-integrity hash."""
    stat = video_path.stat()
    identity = (
        f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{CACHE_SCHEMA_VERSION}|{WHISPER_MODEL}"
    )
    cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    if os.name == "nt":
        cache_dir = (
            Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
            / "VideoMind" / "cache"
        )
    else:
        cache_dir = Path.home() / ".cache" / "videomind"
    return cache_dir / f"{cache_key}.json"


def _load_cache(cache_path: Path) -> dict[str, Any] | None:
    """Return a compatible normalized transcript or treat the entry as a miss."""
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Cache entry must be an object")
        if data.get("schema_version") != CACHE_SCHEMA_VERSION or (
            data.get("model") != WHISPER_MODEL
        ):
            raise ValueError("Cache identity mismatch")
        transcript = data.get("transcript")
        if not isinstance(transcript, Mapping):
            raise ValueError("Cached transcript must be an object")
        language = transcript.get("language")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("Cached language is invalid")
        duration_value = transcript.get("duration")
        if isinstance(duration_value, bool):
            raise ValueError("Cached duration is invalid")
        duration = float(duration_value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Cached duration is invalid")
        segments = transcript.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("Cached segments must be a nonempty list")
        return {"language": language, "duration": duration,
                "segments": _normalize_segments(segments)}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _save_cache(cache_path: Path, transcript: Mapping[str, Any]) -> None:
    """Atomically write one inspectable transcript cache entry."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": WHISPER_MODEL,
        "transcript": transcript,
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
            json.dump(entry, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, cache_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def ingest_video(video_path: str | Path) -> dict[str, Any]:
    """Validate, cache or transcribe, chunk, and return one prepared transcript."""
    if not isinstance(video_path, (str, Path)) or not str(video_path).strip():
        raise ValueError("A local video path is required")
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cache_path = _cache_path_for(path)
    transcript = _load_cache(cache_path)
    if transcript is None:
        transcript = _transcribe_video(path)
        _save_cache(cache_path, transcript)

    transcript["video"] = str(path)
    chunks = _build_chunks(transcript["segments"])
    return {
        **transcript,
        "segment_count": len(transcript["segments"]),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
