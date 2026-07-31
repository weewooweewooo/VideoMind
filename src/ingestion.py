"""VideoMind video ingestion, transcription, chunking, and transcript caching."""

from __future__ import annotations

# Standard-library imports

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


# Constants

_DEFAULT_CHUNK_WORDS = 70
_CACHE_SCHEMA_VERSION = 1
_HASH_BLOCK_SIZE = 1024 * 1024


# Segment normalization and validation


def _segment_value(segment: Any, field: str) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(field)
    return getattr(segment, field, None)


def _normalize_transcript_segments(
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


def _chunk_transcript_segments(
    segments: list[dict[str, str | float]],
    max_words: int = _DEFAULT_CHUNK_WORDS,
    overlap_words: int = 0,
) -> list[dict[str, str | float]]:
    """Merge normalized transcript segments into ordered timestamped chunks.

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

    if overlap_words:
        return _chunk_with_segment_overlap(
            segments,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    return _chunk_without_overlap(segments, max_words=max_words)


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


def _build_transcript_output(
    transcript: Mapping[str, Any],
    chunk_words: int = _DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = 0,
) -> dict[str, Any]:
    """Build transcript-only JSON from normalized segments."""
    segments = transcript["segments"]
    chunks = _chunk_transcript_segments(
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


# Settings resolution


def _resolve_settings(
    *,
    model: str | None,
    device: str | None,
    compute_type: str | None,
    beam_size: int | None,
    language: str | None,
    chunk_words: int | None,
    chunk_overlap_words: int,
    cache_dir: str | Path | None,
    use_cache: bool,
) -> dict[str, Any]:
    """Resolve and validate all ingestion settings once."""
    environment = os.environ

    resolved_model = (model or environment.get("WHISPER_MODEL") or "base").strip()
    if not resolved_model:
        raise ValueError("Whisper model name or path must not be empty")

    resolved_device = (device or environment.get("DEVICE") or "cpu").strip().lower()
    if resolved_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto")

    resolved_compute_type = (
        compute_type or environment.get("WHISPER_COMPUTE_TYPE") or "int8"
    ).strip()
    if not resolved_compute_type:
        raise ValueError("compute type must not be empty")

    raw_beam_size: int | str = (
        beam_size
        if beam_size is not None
        else environment.get("WHISPER_BEAM_SIZE", "5")
    )
    try:
        resolved_beam_size = int(raw_beam_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Whisper beam size must be an integer") from exc
    if resolved_beam_size <= 0:
        raise ValueError("Whisper beam size must be greater than zero")

    raw_chunk_words: int | str = (
        chunk_words
        if chunk_words is not None
        else environment.get("TRANSCRIPT_CHUNK_WORDS", str(_DEFAULT_CHUNK_WORDS))
    )
    try:
        resolved_chunk_words = int(raw_chunk_words)
    except (TypeError, ValueError) as exc:
        raise ValueError("transcript chunk word limit must be an integer") from exc
    if resolved_chunk_words <= 0:
        raise ValueError("transcript chunk word limit must be greater than zero")
    if (
        isinstance(chunk_overlap_words, bool)
        or not isinstance(chunk_overlap_words, int)
    ):
        raise ValueError("transcript chunk overlap must be an integer")
    if chunk_overlap_words < 0:
        raise ValueError("transcript chunk overlap must not be negative")
    if chunk_overlap_words >= resolved_chunk_words:
        raise ValueError(
            "transcript chunk overlap must be less than the chunk word limit"
        )

    resolved_cache_dir: Path | None = None
    if use_cache:
        if cache_dir is not None:
            configured_cache_dir = str(cache_dir).strip()
            if not configured_cache_dir:
                raise ValueError("Cache directory must not be empty")
            resolved_cache_dir = Path(configured_cache_dir)
        else:
            environment_cache_dir = environment.get(
                "VIDEOMIND_CACHE_DIR",
                "",
            ).strip()
            if environment_cache_dir:
                resolved_cache_dir = Path(environment_cache_dir)
            elif os.name == "nt":
                local_app_data = environment.get("LOCALAPPDATA", "").strip()
                platform_root = (
                    Path(local_app_data)
                    if local_app_data
                    else Path.home() / "AppData" / "Local"
                )
                resolved_cache_dir = platform_root / "VideoMind" / "cache"
            else:
                resolved_cache_dir = Path.home() / ".cache" / "videomind"
        resolved_cache_dir = resolved_cache_dir.expanduser().resolve()

    return {
        "model": resolved_model,
        "device": resolved_device,
        "compute_type": resolved_compute_type,
        "beam_size": resolved_beam_size,
        "language": language.strip() if language else None,
        "chunk_words": resolved_chunk_words,
        "chunk_overlap_words": chunk_overlap_words,
        "cache_dir": resolved_cache_dir,
    }


# Lazy Faster-Whisper transcription


def _transcribe_video(
    video_path: Path,
    *,
    model: str,
    device: str,
    compute_type: str,
    beam_size: int,
    language: str | None,
) -> dict[str, Any]:
    """Transcribe one local video into normalized timestamped segments."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for transcription; "
            "install the CPU dependencies before running this command"
        ) from exc

    whisper_model = WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        num_workers=4,
        cpu_threads=8,
    )
    transcribe_options: dict[str, Any] = {
        "beam_size": beam_size,
        "word_timestamps": False,
    }
    if language:
        transcribe_options["language"] = language
    raw_segments, info = whisper_model.transcribe(
        str(video_path),
        **transcribe_options,
    )
    segments = _normalize_transcript_segments(raw_segments)
    duration = float(getattr(info, "duration", 0.0) or segments[-1]["end"])
    return {
        "video": str(video_path),
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


def _video_content_sha256(video_path: Path) -> str:
    """Hash local video contents for transcript cache identity."""
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


def _build_cache_identity(
    video_path: Path,
    *,
    model: str,
    language: str | None,
    beam_size: int,
    device: str,
    compute_type: str,
) -> dict[str, Any]:
    """Hash video contents and canonical transcription settings."""
    video_sha256 = _video_content_sha256(video_path)
    possible_model_path = Path(model).expanduser()
    model_identifier = (
        str(possible_model_path.resolve())
        if possible_model_path.exists()
        else model
    )
    transcription = {
        "model": model_identifier,
        "language": language,
        "beam_size": beam_size,
        "device": device,
        "compute_type": compute_type,
    }
    identity_payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
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
    return {
        "cache_key": cache_key,
        "video_sha256": video_sha256,
        "transcription": transcription,
    }


def _cache_entry_path(
    cache_dir: Path,
    identity: Mapping[str, Any],
) -> Path:
    return cache_dir / f"{identity['cache_key']}.json"


def _validated_cache_entry(
    value: Any,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Transcript cache entry must be an object")
    metadata = value.get("cache")
    if not isinstance(metadata, Mapping):
        raise ValueError("Transcript cache metadata must be an object")
    if metadata.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported transcript cache schema")
    if metadata.get("cache_key") != identity["cache_key"]:
        raise ValueError("Transcript cache key mismatch")
    if metadata.get("video_sha256") != identity["video_sha256"]:
        raise ValueError("Transcript cache video hash mismatch")
    if metadata.get("transcription") != identity["transcription"]:
        raise ValueError("Transcript cache configuration mismatch")

    transcript = value.get("transcript")
    if not isinstance(transcript, Mapping):
        raise ValueError("Cached transcript must be an object")
    video = transcript.get("video")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("Cached transcript has no video identifier")
    raw_duration = transcript.get("duration")
    if isinstance(raw_duration, bool):
        raise ValueError("Cached transcript duration is invalid")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cached transcript duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Cached transcript duration is invalid")
    language = transcript.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("Cached transcript language must be a string or null")
    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Cached transcript segments must be a nonempty list")
    if any(
        not isinstance(segment, Mapping)
        or not isinstance(segment.get("text"), str)
        or not segment["text"].strip()
        for segment in raw_segments
    ):
        raise ValueError("Cached transcript contains unusable segments")
    segments = _normalize_transcript_segments(raw_segments)
    return {
        "video": video.strip(),
        "duration": duration,
        "language": language,
        "segments": segments,
    }


# Atomic cache loading and storage


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
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=".videomind-write-probe.",
            suffix=".tmp",
        ):
            pass
    except OSError as exc:
        raise OSError(
            f"Transcript cache directory is not writable: {cache_dir}"
        ) from exc


def _load_cached_transcript(
    cache_dir: Path,
    identity: Mapping[str, Any],
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
    identity: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> Path:
    """Atomically store one normalized transcript cache entry."""
    destination = _cache_entry_path(cache_dir, identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "cache": {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "cache_key": identity["cache_key"],
            "video_sha256": identity["video_sha256"],
            "transcription": identity["transcription"],
        },
        "transcript": transcript,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=f".{identity['cache_key']}.",
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

    settings = _resolve_settings(
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    transcript: dict[str, Any] | None = None
    cache_metadata: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
    }
    identity: dict[str, Any] | None = None
    resolved_cache_dir = settings["cache_dir"]
    if use_cache:
        _validate_cache_directory(resolved_cache_dir)
        identity = _build_cache_identity(
            path,
            model=settings["model"],
            language=settings["language"],
            beam_size=settings["beam_size"],
            device=settings["device"],
            compute_type=settings["compute_type"],
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

        transcript = _transcribe_video(
            path,
            model=settings["model"],
            device=settings["device"],
            compute_type=settings["compute_type"],
            beam_size=settings["beam_size"],
            language=settings["language"],
        )
        if use_cache:
            try:
                _store_cached_transcript(
                    resolved_cache_dir,
                    identity,
                    transcript,
                )
                if refresh_cache:
                    report("VideoMind transcript cache refreshed.")
            except OSError as exc:
                report(f"VideoMind transcript cache write warning: {exc}")

    output = _build_transcript_output(
        transcript,
        chunk_words=settings["chunk_words"],
        chunk_overlap_words=settings["chunk_overlap_words"],
    )
    return output, cache_metadata
