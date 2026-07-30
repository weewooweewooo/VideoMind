"""Transcript-only CPU entry point with lazy Faster-Whisper loading."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion.transcript_chunks import (
    DEFAULT_CHUNK_WORDS,
    chunk_transcript_segments,
    normalize_transcript_segments,
)

logger = logging.getLogger(__name__)


def resolve_whisper_model(
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
    """Resolve a Faster-Whisper device without importing PyTorch or OpenCLIP."""
    environment = os.environ if environ is None else environ
    resolved = (device or environment.get("DEVICE") or "cpu").strip().lower()
    if resolved not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto")
    return resolved


def resolve_compute_type(
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


def resolve_beam_size(
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


def transcribe_to_memory(
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
    resolved_model = resolve_whisper_model(model_size)
    resolved_device = resolve_whisper_device(device)
    resolved_compute_type = resolve_compute_type(compute_type)
    model = _create_whisper_model(
        resolved_model,
        resolved_device,
        resolved_compute_type,
    )
    transcribe_options: dict[str, Any] = {
        "beam_size": resolve_beam_size(beam_size),
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


def build_transcript_output(
    transcript: Mapping[str, Any],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> dict[str, Any]:
    """Build transcript-only JSON with validated segments and chunks."""
    segments = normalize_transcript_segments(transcript.get("segments", []))
    chunks = chunk_transcript_segments(segments, max_words=chunk_words)
    return {
        "video": str(transcript.get("video", "")),
        "language": transcript.get("language"),
        "duration": float(transcript.get("duration", segments[-1]["end"])),
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "segments": segments,
        "chunks": chunks,
    }


def build_argument_parser() -> argparse.ArgumentParser:
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
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file; stdout is used when omitted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the transcript-only command and return a process exit code."""
    args = build_argument_parser().parse_args(argv)
    try:
        video_path = Path(args.video).expanduser()
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        transcript = transcribe_to_memory(
            str(video_path),
            video_name=str(video_path.resolve()),
            model_size=resolve_whisper_model(args.model),
            device=resolve_whisper_device(args.device),
            compute_type=resolve_compute_type(args.compute_type),
            beam_size=resolve_beam_size(args.beam_size),
            language=args.language,
        )
        result = build_transcript_output(
            transcript,
            chunk_words=resolve_chunk_words(args.chunk_words),
        )
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
    raise SystemExit(main())
