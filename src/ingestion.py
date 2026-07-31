"""VideoMind video ingestion, transcription, chunking, and transcript caching."""

from __future__ import annotations

# Standard-library imports

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Constants and lightweight data contracts

DEFAULT_CHUNK_WORDS = 70
CACHE_SCHEMA_VERSION = 1
CACHE_DIRECTORY_ENVIRONMENT_VARIABLE = "VIDEOMIND_CACHE_DIR"
_HASH_BLOCK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _TranscriptCacheIdentity:
    """Deterministic identity for one video and transcription configuration."""

    cache_key: str
    video_sha256: str
    transcription: dict[str, Any]


# Segment normalization and validation


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


# Transcript chunk construction


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


def resolve_chunk_words(
    chunk_words: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve the transcript chunk word limit."""
    environment = os.environ if environ is None else environ
    raw_value: int | str = (
        chunk_words
        if chunk_words is not None
        else environment.get("TRANSCRIPT_CHUNK_WORDS", str(DEFAULT_CHUNK_WORDS))
    )
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("transcript chunk word limit must be an integer") from exc
    if resolved <= 0:
        raise ValueError("transcript chunk word limit must be greater than zero")
    return resolved


def resolve_chunk_overlap_words(
    overlap_words: int = 0,
    *,
    chunk_words: int,
) -> int:
    """Validate optional transcript chunk overlap against the word limit."""
    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise ValueError("transcript chunk overlap must be an integer")
    if overlap_words < 0:
        raise ValueError("transcript chunk overlap must not be negative")
    if overlap_words >= chunk_words:
        raise ValueError(
            "transcript chunk overlap must be less than the chunk word limit"
        )
    return overlap_words


def build_transcript_output(
    transcript: Mapping[str, Any],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = 0,
) -> dict[str, Any]:
    """Build transcript-only JSON with validated segments and chunks."""
    segments = normalize_transcript_segments(transcript.get("segments", []))
    chunks = chunk_transcript_segments(
        segments,
        max_words=chunk_words,
        overlap_words=chunk_overlap_words,
    )
    return {
        "video": str(transcript.get("video", "")),
        "language": transcript.get("language"),
        "duration": float(transcript.get("duration", segments[-1]["end"])),
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "segments": segments,
        "chunks": chunks,
    }


# Whisper configuration resolution


def _resolve_whisper_model(
    model_size: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve CLI value, then WHISPER_MODEL, then the CPU-friendly base default."""
    environment = os.environ if environ is None else environ
    resolved = model_size or environment.get("WHISPER_MODEL") or "base"
    resolved = resolved.strip()
    if not resolved:
        raise ValueError("Whisper model name or path must not be empty")
    return resolved


def resolve_whisper_device(
    device: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a Faster-Whisper device without importing optional dependencies."""
    environment = os.environ if environ is None else environ
    resolved = (device or environment.get("DEVICE") or "cpu").strip().lower()
    if resolved not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto")
    return resolved


def _resolve_compute_type(
    compute_type: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the Faster-Whisper compute type, defaulting to CPU int8."""
    environment = os.environ if environ is None else environ
    resolved = (
        compute_type or environment.get("WHISPER_COMPUTE_TYPE") or "int8"
    ).strip()
    if not resolved:
        raise ValueError("compute type must not be empty")
    return resolved


def _resolve_beam_size(
    beam_size: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve the Faster-Whisper beam size."""
    environment = os.environ if environ is None else environ
    raw_value: int | str = (
        beam_size
        if beam_size is not None
        else environment.get("WHISPER_BEAM_SIZE", "5")
    )
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Whisper beam size must be an integer") from exc
    if resolved <= 0:
        raise ValueError("Whisper beam size must be greater than zero")
    return resolved


# Lazy Faster-Whisper loading and transcription


def _load_whisper_model_class() -> Any:
    """Import Faster-Whisper only when transcription is actually requested."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for transcription; "
            "install the CPU dependencies before running this command"
        ) from exc
    return WhisperModel


def _create_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> Any:
    """Create a Whisper model from the caller-selected model name or path."""
    whisper_model = _load_whisper_model_class()
    return whisper_model(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=4,
        cpu_threads=8,
    )


def _transcribe_to_memory(
    video_path_or_url: str,
    video_name: str | None = None,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio into validated timestamped segment dictionaries."""
    video_identifier = (
        video_name or Path(video_path_or_url.split("?")[0]).stem or "video"
    )
    start_time = time.time()
    resolved_model = _resolve_whisper_model(model_size)
    resolved_device = resolve_whisper_device(device)
    resolved_compute_type = _resolve_compute_type(compute_type)
    model = _create_whisper_model(
        resolved_model,
        resolved_device,
        resolved_compute_type,
    )
    transcribe_options: dict[str, Any] = {
        "beam_size": _resolve_beam_size(beam_size),
        "word_timestamps": False,
    }
    if language:
        transcribe_options["language"] = language
    segments_gen, info = model.transcribe(video_path_or_url, **transcribe_options)
    segments = normalize_transcript_segments(list(segments_gen))
    elapsed = time.time() - start_time
    duration = float(getattr(info, "duration", 0.0) or segments[-1]["end"])
    logger.info(
        "Transcribed %.0fs audio in %.1fs (%s segments) with %s on %s/%s",
        duration,
        elapsed,
        len(segments),
        resolved_model,
        resolved_device,
        resolved_compute_type,
    )
    return {
        "video": video_identifier,
        "duration": duration,
        "language": getattr(info, "language", None) or language,
        "segments": segments,
    }


# Transcript cache identity and validation


def _validated_video_path(video_path: str | Path) -> Path:
    if not isinstance(video_path, (str, Path)) or not str(video_path).strip():
        raise ValueError("A local video path is required")
    path = Path(video_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def video_content_sha256(video_path: Path) -> str:
    """Hash local video contents for library deduplication and cache identity."""
    digest = hashlib.sha256()
    try:
        with video_path.open("rb") as video_file:
            for block in iter(lambda: video_file.read(_HASH_BLOCK_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise OSError(
            f"Unable to hash video for transcript cache: {video_path}"
        ) from exc
    return digest.hexdigest()


def _model_identifier(model: str) -> str:
    configured = model.strip()
    if not configured:
        raise ValueError("Whisper model name or path must not be empty")
    possible_path = Path(configured).expanduser()
    if possible_path.exists():
        return str(possible_path.resolve())
    return configured


def _build_cache_identity(
    video_path: Path,
    *,
    model: str,
    language: str | None,
    beam_size: int,
    device: str,
    compute_type: str,
) -> _TranscriptCacheIdentity:
    """Hash video contents and canonical transcription settings."""
    video_sha256 = video_content_sha256(video_path)
    transcription = {
        "model": _model_identifier(model),
        "language": language,
        "beam_size": beam_size,
        "device": device,
        "compute_type": compute_type,
    }
    identity_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "video_sha256": video_sha256,
        **transcription,
    }
    canonical_payload = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    cache_key = hashlib.sha256(canonical_payload).hexdigest()
    return _TranscriptCacheIdentity(
        cache_key=cache_key,
        video_sha256=video_sha256,
        transcription=transcription,
    )


def _cache_entry_path(
    cache_dir: Path,
    identity: _TranscriptCacheIdentity,
) -> Path:
    return cache_dir / f"{identity.cache_key}.json"


def _validated_cached_transcript(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Cached transcript must be an object")

    video = value.get("video")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("Cached transcript has no video identifier")

    raw_duration = value.get("duration")
    if isinstance(raw_duration, bool):
        raise ValueError("Cached transcript duration is invalid")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cached transcript duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Cached transcript duration is invalid")

    language = value.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("Cached transcript language must be a string or null")

    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Cached transcript segments must be a nonempty list")
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise ValueError(
                f"Cached transcript segment {index} must be an object"
            )
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Cached transcript segment {index} has empty text")

    segments = normalize_transcript_segments(raw_segments)
    if len(segments) != len(raw_segments):
        raise ValueError("Cached transcript contains unusable segments")

    return {
        "video": video.strip(),
        "duration": duration,
        "language": language,
        "segments": segments,
    }


def _validated_cache_entry(
    value: Any,
    identity: _TranscriptCacheIdentity,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Transcript cache entry must be an object")
    metadata = value.get("cache")
    if not isinstance(metadata, Mapping):
        raise ValueError("Transcript cache metadata must be an object")
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported transcript cache schema")
    if metadata.get("cache_key") != identity.cache_key:
        raise ValueError("Transcript cache key mismatch")
    if metadata.get("video_sha256") != identity.video_sha256:
        raise ValueError("Transcript cache video hash mismatch")
    if metadata.get("transcription") != identity.transcription:
        raise ValueError("Transcript cache configuration mismatch")
    created_at = metadata.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("Transcript cache creation time is missing")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError("Transcript cache creation time is invalid") from exc
    source_video = metadata.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError("Transcript cache source video is missing")
    return _validated_cached_transcript(value.get("transcript"))


# Atomic cache loading and storage


def _resolve_cache_directory(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve CLI, environment, then platform-default cache location."""
    environment = os.environ if environ is None else environ
    if explicit is not None:
        configured = str(explicit).strip()
        if not configured:
            raise ValueError("Cache directory must not be empty")
        cache_directory = Path(configured)
    else:
        environment_value = environment.get(
            CACHE_DIRECTORY_ENVIRONMENT_VARIABLE,
            "",
        ).strip()
        if environment_value:
            cache_directory = Path(environment_value)
        elif os.name == "nt":
            local_app_data = environment.get("LOCALAPPDATA", "").strip()
            platform_root = (
                Path(local_app_data)
                if local_app_data
                else Path.home() / "AppData" / "Local"
            )
            cache_directory = platform_root / "VideoMind" / "cache"
        else:
            cache_directory = Path.home() / ".cache" / "videomind"
    return cache_directory.expanduser().resolve()


def _validate_cache_directory(cache_dir: Path) -> None:
    """Reject an explicitly unusable cache path before transcription."""
    if cache_dir.exists():
        if not cache_dir.is_dir():
            raise NotADirectoryError(
                f"Transcript cache path is not a directory: {cache_dir}"
            )
        try:
            with os.scandir(cache_dir):
                pass
        except OSError as exc:
            raise OSError(
                f"Transcript cache directory is unreadable: {cache_dir}"
            ) from exc
        return

    existing_parent = cache_dir.parent
    while (
        not existing_parent.exists()
        and existing_parent != existing_parent.parent
    ):
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise NotADirectoryError(
            f"Transcript cache parent is not a directory: {existing_parent}"
        )


def _prepare_cache_directory_for_write(cache_dir: Path) -> None:
    """Create and verify a writable cache directory before transcription."""
    probe_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, probe_name = tempfile.mkstemp(
            dir=cache_dir,
            prefix=".videomind-write-probe.",
            suffix=".tmp",
        )
        os.close(descriptor)
        probe_path = Path(probe_name)
        probe_path.unlink()
    except OSError as exc:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OSError(
            f"Transcript cache directory is not writable: {cache_dir}"
        ) from exc


def _load_cached_transcript(
    cache_dir: Path,
    identity: _TranscriptCacheIdentity,
) -> dict[str, Any] | None:
    """Return a valid cached transcript, or ``None`` for an invalid entry."""
    path = _cache_entry_path(cache_dir, identity)
    if not path.exists() or not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return _validated_cache_entry(loaded, identity)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _store_cached_transcript(
    cache_dir: Path,
    identity: _TranscriptCacheIdentity,
    transcript: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    replace: bool = False,
) -> Path:
    """Atomically store one validated transcript cache entry."""
    validated_transcript = _validated_cached_transcript(transcript)
    destination = _cache_entry_path(cache_dir, identity)
    if destination.exists() and not replace:
        if _load_cached_transcript(cache_dir, identity) is not None:
            return destination

    cache_dir.mkdir(parents=True, exist_ok=True)
    source_video = metadata.get("source_video")
    entry = {
        "cache": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": identity.cache_key,
            "video_sha256": identity.video_sha256,
            "source_video": str(source_video or validated_transcript["video"]),
            "transcription": identity.transcription,
        },
        "transcript": validated_transcript,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=f".{identity.cache_key}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(entry, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


# Public ingest_video() operation


def ingest_video(
    video_path: str | Path,
    *,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int | None = None,
    language: str | None = None,
    chunk_words: int | None = None,
    chunk_overlap_words: int = 0,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ingest one local video into a transcript and cache-status pair."""
    path = _validated_video_path(video_path)
    if refresh_cache and not use_cache:
        raise ValueError("refresh_cache requires use_cache")

    resolved_model = _resolve_whisper_model(model)
    resolved_device = resolve_whisper_device(device)
    resolved_compute_type = _resolve_compute_type(compute_type)
    resolved_beam_size = _resolve_beam_size(beam_size)
    resolved_language = language.strip() if language else None
    resolved_chunk_words = resolve_chunk_words(chunk_words)
    resolved_chunk_overlap = resolve_chunk_overlap_words(
        chunk_overlap_words,
        chunk_words=resolved_chunk_words,
    )

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    transcript: dict[str, Any] | None = None
    cache_metadata: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
    }
    identity = None
    resolved_cache_dir = None
    if use_cache:
        resolved_cache_dir = _resolve_cache_directory(cache_dir)
        _validate_cache_directory(resolved_cache_dir)
        identity = _build_cache_identity(
            path,
            model=resolved_model,
            language=resolved_language,
            beam_size=resolved_beam_size,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
        destination = _cache_entry_path(resolved_cache_dir, identity)
        cache_status = "refreshed" if refresh_cache else "miss"
        cache_metadata = {
            "enabled": True,
            "status": cache_status,
            "path": str(destination),
        }
        if not refresh_cache:
            transcript = _load_cached_transcript(resolved_cache_dir, identity)
            if transcript is not None:
                transcript["video"] = str(path)
                cache_metadata["status"] = "hit"
                report("VideoMind transcript cache hit.")

    if transcript is None:
        if use_cache:
            if resolved_cache_dir is None:
                raise RuntimeError("Transcript cache initialization failed")
            _prepare_cache_directory_for_write(resolved_cache_dir)
            if refresh_cache:
                report(
                    "VideoMind transcript cache refresh requested; "
                    "transcribing video."
                )
            else:
                report("VideoMind transcript cache miss; transcribing video.")
        else:
            report("VideoMind transcript cache disabled; transcribing video.")

        transcript = _transcribe_to_memory(
            str(path),
            video_name=str(path),
            model_size=resolved_model,
            device=resolved_device,
            compute_type=resolved_compute_type,
            beam_size=resolved_beam_size,
            language=resolved_language,
        )
        if use_cache:
            try:
                if resolved_cache_dir is None or identity is None:
                    raise RuntimeError("Transcript cache initialization failed")
                _store_cached_transcript(
                    resolved_cache_dir,
                    identity,
                    transcript,
                    {"source_video": str(path.resolve())},
                    replace=refresh_cache,
                )
                if refresh_cache:
                    report("VideoMind transcript cache refreshed.")
            except OSError as exc:
                report(f"VideoMind transcript cache write warning: {exc}")

    output = build_transcript_output(
        transcript,
        chunk_words=resolved_chunk_words,
        chunk_overlap_words=resolved_chunk_overlap,
    )
    return output, cache_metadata


# Optional lower-level CLI


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the dependency-free transcript-only CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one local video on CPU and emit timestamped transcript JSON."
        )
    )
    parser.add_argument("video", help="Path to a local video file.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Faster-Whisper model name or local path "
            '(default: WHISPER_MODEL or "base").'
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help='Inference device (default: DEVICE or "cpu").',
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help='Faster-Whisper compute type (default: WHISPER_COMPUTE_TYPE or "int8").',
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Transcription beam size (default: WHISPER_BEAM_SIZE or 5).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code; omit to use Faster-Whisper detection.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help=(
            "Maximum target words per chunk "
            f"(default: TRANSCRIPT_CHUNK_WORDS or {DEFAULT_CHUNK_WORDS})."
        ),
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=0,
        help=(
            "Approximate overlap using complete trailing transcript segments "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file; stdout is used when omitted.",
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    """Run the transcript-only command and return a process exit code."""
    args = _build_argument_parser().parse_args(argv)
    try:
        video_path = Path(args.video).expanduser()
        result, _ = ingest_video(
            video_path,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
            chunk_words=args.chunk_words,
            chunk_overlap_words=args.chunk_overlap_words,
            use_cache=False,
        )
        result["video"] = str(video_path.resolve())
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output is None:
            print(json_output)
        else:
            args.output.write_text(f"{json_output}\n", encoding="utf-8")
        return 0
    except KeyboardInterrupt:
        print("Transcription interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
