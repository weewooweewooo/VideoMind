"""YouTube identity, caption acquisition/parsing, and temporary audio."""

from __future__ import annotations

import html
import json
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.config import (
    YOUTUBE_CANONICAL_HOST,
    YOUTUBE_CANONICAL_SCHEME,
    YOUTUBE_ROOT_HOST,
    YOUTUBE_SHORT_HOST,
    YOUTUBE_SUPPORTED_SCHEMES,
    YOUTUBE_VIDEO_ID_PATTERN,
    YOUTUBE_VIDEO_ID_QUERY_KEY,
    YOUTUBE_WATCH_HOSTS,
    YOUTUBE_WATCH_PATH,
)


CAPTION_PROFILE_VERSION = 2
CAPTION_FORMATS = ("json3", "vtt", "srt")
_CAPTION_TAG_PATTERN = re.compile(r"<[^>]*>")
_CAPTION_BRACE_PATTERN = re.compile(r"\{\\[^}]*\}")
_CAPTION_WORD_KEY_PATTERN = re.compile(r"[\w]+(?:'[\w]+)?", re.UNICODE)
_CUE_TIMING_PATTERN = re.compile(
    r"(?P<start>(?:\d{1,3}:)?\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,3}:)?\d{1,2}:\d{2}[.,]\d{3})"
)


@dataclass(frozen=True, slots=True)
class YouTubeIdentity:
    video_id: str
    canonical_url: str


@dataclass(frozen=True, slots=True)
class YouTubeCaptionTrack:
    language: str
    is_generated: bool
    extension: str
    url: str


@dataclass(frozen=True, slots=True)
class YouTubeMetadata:
    video_id: Any
    duration: Any
    caption_tracks: tuple[YouTubeCaptionTrack, ...]


def resolve_identity(value: str) -> YouTubeIdentity:
    """Return a validated video ID and canonical URL."""
    if "://" not in value:
        raise ValueError("Input must be a supported YouTube URL")
    parsed = urlparse(value)
    if parsed.scheme.lower() not in YOUTUBE_SUPPORTED_SCHEMES:
        raise ValueError("Unsupported video URL scheme; use https://")
    host = (parsed.hostname or "").lower().rstrip(".")
    video_id: str | None = None
    if host == YOUTUBE_SHORT_HOST:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif (
        host in YOUTUBE_WATCH_HOSTS
        and parsed.path.rstrip("/") == YOUTUBE_WATCH_PATH
    ):
        video_id = (
            parse_qs(parsed.query).get(YOUTUBE_VIDEO_ID_QUERY_KEY) or [None]
        )[0]
    elif host.endswith(YOUTUBE_ROOT_HOST) or host == YOUTUBE_SHORT_HOST:
        raise ValueError(
            "Unsupported YouTube URL; use youtube.com/watch?v=... or youtu.be/..."
        )
    else:
        raise ValueError(f"Unsupported video URL host: {host or 'missing host'}")
    if not isinstance(video_id, str) or not YOUTUBE_VIDEO_ID_PATTERN.fullmatch(
        video_id
    ):
        raise ValueError("Invalid YouTube URL: a valid video ID is required")
    canonical_url = (
        f"{YOUTUBE_CANONICAL_SCHEME}://{YOUTUBE_CANONICAL_HOST}{YOUTUBE_WATCH_PATH}"
        f"?{YOUTUBE_VIDEO_ID_QUERY_KEY}={video_id}"
    )
    return YouTubeIdentity(video_id, canonical_url)


def _clean_caption_text(value: Any) -> str:
    """Remove caption markup while preserving the caption's spoken wording."""
    text = html.unescape(str(value or ""))
    text = _CAPTION_TAG_PATTERN.sub("", text)
    text = _CAPTION_BRACE_PATTERN.sub("", text)
    return " ".join(text.replace("\u200b", "").split())


def _deduplicate_caption_segments(
    segments: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    """Strip exact rolling word overlap from timestamp-ordered caption cues."""
    deduplicated: list[dict[str, str | float]] = []
    emitted_keys: list[str] = []
    for raw_segment in sorted(segments, key=lambda item: float(item["start"])):
        text = _clean_caption_text(raw_segment.get("text"))
        if not text:
            continue
        start = float(raw_segment["start"])
        end = float(raw_segment["end"])
        current_words = text.split()
        current_keys = [
            "".join(_CAPTION_WORD_KEY_PATTERN.findall(word.casefold()))
            for word in current_words
        ]
        if deduplicated:
            previous = deduplicated[-1]
            overlap = 0
            if start <= float(previous["end"]) + 0.25:
                for size in range(
                    min(len(emitted_keys), len(current_keys)), 0, -1
                ):
                    if emitted_keys[-size:] == current_keys[:size]:
                        overlap = size
                        break
            if overlap == len(current_words):
                previous["end"] = max(float(previous["end"]), end)
                continue
            if overlap:
                text = " ".join(current_words[overlap:])
                current_keys = current_keys[overlap:]
            end = max(end, float(previous["end"]))
        deduplicated.append({"start": start, "end": end, "text": text})
        emitted_keys.extend(current_keys)
    return deduplicated


def _parse_json3_captions(payload: bytes) -> list[dict[str, str | float]]:
    """Convert YouTube JSON3 events into common timestamped segments."""
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Caption track is not valid JSON3") from exc
    events = data.get("events") if isinstance(data, Mapping) else None
    if not isinstance(events, list):
        raise ValueError("Caption JSON3 contains no events")
    segments: list[dict[str, str | float]] = []
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("segs"), list):
            continue
        try:
            start = float(event["tStartMs"]) / 1000.0
            duration = float(event.get("dDurationMs") or 0) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        text = "".join(
            str(segment.get("utf8") or "")
            for segment in event["segs"]
            if isinstance(segment, Mapping)
        )
        if duration <= 0:
            duration = 0.001
        segments.append({"start": start, "end": start + duration, "text": text})
    return _deduplicate_caption_segments(segments)


def _caption_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError("Invalid caption timestamp")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_text_captions(payload: bytes) -> list[dict[str, str | float]]:
    """Convert WebVTT or SRT cues into common timestamped segments."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Caption track is not valid UTF-8") from exc
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[dict[str, str | float]] = []
    index = 0
    while index < len(lines):
        match = _CUE_TIMING_PATTERN.search(lines[index])
        if not match:
            index += 1
            continue
        start = _caption_timestamp(match.group("start"))
        end = _caption_timestamp(match.group("end"))
        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index])
            index += 1
        segments.append({"start": start, "end": end, "text": " ".join(cue_lines)})
    return _deduplicate_caption_segments(segments)


def parse_captions(
    payload: bytes, extension: str
) -> list[dict[str, str | float]]:
    if extension == "json3":
        return _parse_json3_captions(payload)
    if extension in {"vtt", "srt"}:
        return _parse_text_captions(payload)
    raise ValueError(f"Unsupported caption format: {extension}")


def validate_captions(
    segments: list[dict[str, str | float]], duration: float, language: str
) -> None:
    """Reject empty, malformed, implausibly short, or mistimed caption tracks."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Caption track has no language metadata")
    minimum_cues = 1 if duration <= 15 else 3
    if len(segments) < minimum_cues:
        raise ValueError("Caption track contains too few usable cues")
    word_count = sum(len(str(segment["text"]).split()) for segment in segments)
    if duration > 30 and word_count < 10:
        raise ValueError("Caption track contains too little usable text")
    first_start = float(segments[0]["start"])
    last_end = float(segments[-1]["end"])
    if duration > 60 and (last_end - first_start) / duration < 0.25:
        raise ValueError("Caption track does not cover enough of the video timeline")
    if last_end > duration + max(30.0, duration * 0.02):
        raise ValueError("Caption timing extends implausibly beyond the video")


def _language_rank(language: str, generated: bool) -> tuple[int, str] | None:
    """Rank original English variants without choosing translated tracks."""
    normalized = language.strip().lower().replace("_", "-")
    if normalized == "en-orig":
        return (0 if generated else 4, normalized)
    preferred = {"en": 1, "en-us": 2, "en-gb": 3}
    if normalized in preferred:
        return preferred[normalized], normalized
    if normalized.startswith("en-"):
        return 4, normalized
    return None


def _caption_candidates(info: Mapping[str, Any]) -> tuple[YouTubeCaptionTrack, ...]:
    """Return manual English candidates before original automatic English."""
    candidates: list[YouTubeCaptionTrack] = []
    for generated, key in ((False, "subtitles"), (True, "automatic_captions")):
        tracks = info.get(key)
        if not isinstance(tracks, Mapping):
            continue
        ranked_languages = sorted(
            (
                (rank, language, formats)
                for language, formats in tracks.items()
                if isinstance(language, str)
                and isinstance(formats, list)
                and (rank := _language_rank(language, generated)) is not None
            ),
            key=lambda item: item[0],
        )
        for _, language, formats in ranked_languages:
            for extension in CAPTION_FORMATS:
                track = next(
                    (
                        item
                        for item in formats
                        if isinstance(item, Mapping)
                        and item.get("ext") == extension
                        and isinstance(item.get("url"), str)
                    ),
                    None,
                )
                if track is not None:
                    candidates.append(
                        YouTubeCaptionTrack(
                            language=language,
                            is_generated=generated,
                            extension=extension,
                            url=str(track["url"]),
                        )
                    )
    return tuple(candidates)


def _downloader(options: Mapping[str, Any] | None = None) -> Any:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube input; install the runtime dependencies"
        ) from exc
    defaults = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
    }
    defaults.update(options or {})
    return YoutubeDL(defaults)


def load_metadata(identity: YouTubeIdentity) -> YouTubeMetadata:
    try:
        with _downloader({"skip_download": True}) as downloader:
            info = downloader.extract_info(identity.canonical_url, download=False)
    except Exception as exc:
        raise RuntimeError(f"Could not access YouTube video: {exc}") from exc
    if not isinstance(info, Mapping):
        raise RuntimeError("YouTube returned invalid video metadata")
    return YouTubeMetadata(
        video_id=info.get("id"),
        duration=info.get("duration"),
        caption_tracks=_caption_candidates(info),
    )


def download_caption(track: YouTubeCaptionTrack) -> bytes:
    try:
        with _downloader() as downloader:
            with downloader.urlopen(track.url) as response:
                return response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download YouTube captions: {exc}") from exc


def _download_audio(canonical_url: str, directory: Path) -> Path:
    output_template = str(directory / "audio.%(ext)s")
    options = {
        "format": "bestaudio",
        "outtmpl": output_template,
        "skip_download": False,
    }
    try:
        with _downloader(options) as downloader:
            downloader.extract_info(canonical_url, download=True)
    except Exception as exc:
        raise RuntimeError(f"Could not download YouTube audio: {exc}") from exc
    files = [path for path in directory.iterdir() if path.is_file()]
    if len(files) != 1:
        raise RuntimeError("YouTube audio download did not produce one usable file")
    return files[0]


@contextmanager
def acquire_audio(identity: YouTubeIdentity) -> Iterator[Path]:
    """Yield downloaded audio and remove its temporary directory afterward."""
    with tempfile.TemporaryDirectory(prefix="videomind-youtube-") as temp_name:
        yield _download_audio(identity.canonical_url, Path(temp_name))
