"""Persistent standard-library cache for timestamped video transcripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion.transcript_chunks import normalize_transcript_segments


CACHE_SCHEMA_VERSION = 1
CACHE_DIRECTORY_ENVIRONMENT_VARIABLE = "VIDEOMIND_CACHE_DIR"
_HASH_BLOCK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class TranscriptCacheIdentity:
    """Deterministic identity for one video and transcription configuration."""

    cache_key: str
    video_sha256: str
    transcription: dict[str, Any]


def resolve_cache_directory(
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


def validate_cache_directory(cache_dir: Path) -> None:
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


def prepare_cache_directory_for_write(cache_dir: Path) -> None:
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


def _hash_video(video_path: Path) -> str:
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


def build_cache_identity(
    video_path: Path,
    *,
    model: str,
    language: str | None,
    beam_size: int,
    device: str,
    compute_type: str,
) -> TranscriptCacheIdentity:
    """Hash video contents and canonical transcription settings."""
    video_sha256 = _hash_video(video_path)
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
    return TranscriptCacheIdentity(
        cache_key=cache_key,
        video_sha256=video_sha256,
        transcription=transcription,
    )


def build_cache_key(
    video_path: Path,
    *,
    model: str,
    language: str | None,
    beam_size: int,
    device: str,
    compute_type: str,
) -> str:
    """Return only the deterministic cache key for public callers."""
    return build_cache_identity(
        video_path,
        model=model,
        language=language,
        beam_size=beam_size,
        device=device,
        compute_type=compute_type,
    ).cache_key


def cache_entry_path(
    cache_dir: Path,
    identity: TranscriptCacheIdentity,
) -> Path:
    """Return the JSON entry path for one cache identity."""
    return cache_dir / f"{identity.cache_key}.json"


def _validated_cached_transcript(
    value: Any,
) -> dict[str, Any]:
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
    identity: TranscriptCacheIdentity,
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


def load_cached_transcript(
    cache_dir: Path,
    identity: TranscriptCacheIdentity,
) -> dict[str, Any] | None:
    """Return a valid cached transcript, or ``None`` for an invalid entry."""
    path = cache_entry_path(cache_dir, identity)
    if not path.exists() or not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return _validated_cache_entry(loaded, identity)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def store_cached_transcript(
    cache_dir: Path,
    identity: TranscriptCacheIdentity,
    transcript: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    replace: bool = False,
) -> Path:
    """Atomically store one validated transcript cache entry."""
    validated_transcript = _validated_cached_transcript(transcript)
    destination = cache_entry_path(cache_dir, identity)
    if destination.exists() and not replace:
        if load_cached_transcript(cache_dir, identity) is not None:
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
