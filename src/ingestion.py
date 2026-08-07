"""YouTube transcript orchestration, chunking, and caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src import youtube
from src.config import (
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
            raise ValueError(f"Invalid transcript segment at index {index}") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"Invalid transcript segment at index {index}")
        if previous_start is not None and previous_end is not None and (
            start < previous_start or end < previous_end
        ):
            raise ValueError(f"Transcript segments are out of order at index {index}")

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
    return chunks


def _transcribe_video(video_path: Path) -> dict[str, Any]:
    """Transcribe temporary YouTube audio with fixed Faster-Whisper defaults."""
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


def _cache_directory() -> Path:
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
            / "VideoMind" / "cache"
        )
    return Path.home() / ".cache" / "videomind"


def _cache_path_for(identity: youtube.YouTubeIdentity) -> Path:
    """Return the policy-level cache path derivable without network access."""
    cache_key = hashlib.sha256(
        (
            f"youtube:{identity.video_id}:caption-first:"
            f"{youtube.CAPTION_PROFILE_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    return _cache_directory() / f"{cache_key}.json"


def _cache_profile() -> dict[str, Any]:
    """Describe every setting that can materially change acquisition."""
    return {
        "acquisition": "youtube_caption_first",
        "preferred_language": "en",
        "caption_priority": ["manual", "automatic", "whisper"],
        "source_types": [
            "youtube_manual_caption",
            "youtube_auto_caption",
            "whisper",
        ],
        "caption_formats": list(youtube.CAPTION_FORMATS),
        "caption_parser_version": youtube.CAPTION_PROFILE_VERSION,
        "whisper": {
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "beam_size": WHISPER_BEAM_SIZE,
        },
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
    if data.get("source") != source:
        return None
    if data.get("profile") != profile:
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
    transcript = {"language": language, "duration": duration, "segments": segments}
    source_metadata = data.get("transcript_source")
    if isinstance(source_metadata, Mapping):
        transcript["transcript_source"] = dict(source_metadata)
    return transcript


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
    if isinstance(transcript.get("transcript_source"), Mapping):
        record["transcript_source"] = dict(transcript["transcript_source"])
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


def _acquire_transcript(
    identity: youtube.YouTubeIdentity,
    metadata: youtube.YouTubeMetadata,
) -> dict[str, Any]:
    try:
        duration = float(metadata.duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("YouTube video metadata has no valid duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("YouTube video metadata has no valid duration")

    caption_errors: list[str] = []
    for track in metadata.caption_tracks:
        try:
            payload = youtube.download_caption(track)
            segments = _normalize_segments(
                youtube.parse_captions(payload, track.extension)
            )
            youtube.validate_captions(segments, duration, track.language)
        except (KeyError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            caption_errors.append(str(exc))
            continue
        source_type = (
            "youtube_auto_caption"
            if track.is_generated
            else "youtube_manual_caption"
        )
        return {
            "language": track.language,
            "duration": duration,
            "segments": segments,
            "transcript_source": {
                "source_type": source_type,
                "source_language": track.language,
                "source_video_id": identity.video_id,
                "canonical_source": identity.canonical_url,
                "generated_or_manual": (
                    "generated" if track.is_generated else "manual"
                ),
                "caption_format": track.extension,
                "parser_version": youtube.CAPTION_PROFILE_VERSION,
            },
        }

    try:
        with youtube.acquire_audio(identity) as audio_path:
            transcript = _transcribe_video(audio_path)
    except Exception as exc:
        detail = caption_errors[-1] if caption_errors else "no usable English captions"
        raise RuntimeError(
            f"YouTube caption acquisition failed ({detail}); Whisper audio fallback "
            f"also failed: {exc}"
        ) from exc
    transcript["transcript_source"] = {
        "source_type": "whisper",
        "source_language": transcript.get("language"),
        "source_video_id": identity.video_id,
        "canonical_source": identity.canonical_url,
        "generated_or_manual": None,
    }
    return transcript


def _prepare_transcript(transcript: dict[str, Any], video: str) -> dict[str, Any]:
    transcript["video"] = video
    chunks = _build_chunks(transcript["segments"])
    return {
        **transcript,
        "segment_count": len(transcript["segments"]),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _ingest_youtube_video(identity: youtube.YouTubeIdentity) -> dict[str, Any]:
    source = {
        "video_id": identity.video_id,
        "canonical_url": identity.canonical_url,
    }
    profile = _cache_profile()
    cache_path = _cache_path_for(identity)
    transcript = _load_cache(cache_path, source, profile)
    if transcript is not None and not isinstance(
        transcript.get("transcript_source"), Mapping
    ):
        transcript = None
    if transcript is None:
        metadata = youtube.load_metadata(identity)
        if metadata.video_id != identity.video_id:
            raise RuntimeError("YouTube metadata did not match the requested video ID")
        transcript = _acquire_transcript(identity, metadata)
        _save_cache(cache_path, source, profile, transcript)
    return _prepare_transcript(transcript, identity.canonical_url)


def ingest_video(youtube_url: str) -> dict[str, Any]:
    """Acquire one YouTube transcript and prepare common chunks."""
    if not isinstance(youtube_url, str) or not youtube_url.strip():
        raise ValueError("Input must be a supported YouTube URL")
    return _ingest_youtube_video(youtube.resolve_identity(youtube_url.strip()))
