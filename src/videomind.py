"""Unified CPU orchestration for transcript-only VideoMind retrieval."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion.transcriber import (
    build_transcript_output,
    resolve_chunk_words,
    transcribe_to_memory,
)
from src.ingestion.transcript_chunks import normalize_transcript_segments
from src.retrieval.local_retriever import (
    LocalTfidfRetriever,
    TranscriptDocument,
    format_search_output,
    validate_transcript_document,
)


def _validated_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    return question.strip()


def _validated_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _validated_min_score(min_score: float) -> float:
    try:
        score = float(min_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_score must be a finite number") from exc
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError("min_score must be between zero and one")
    return score


def _validated_video_path(video_path: str | Path) -> Path:
    if not isinstance(video_path, (str, Path)) or not str(video_path).strip():
        raise ValueError("A local video path is required")
    path = Path(video_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def _validated_transcript_mapping(
    transcript: Mapping[str, Any],
) -> tuple[TranscriptDocument, int]:
    if not isinstance(transcript, Mapping):
        raise ValueError("Transcript JSON must contain an object")
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Transcript JSON segments must be a list")
    normalized_segments = normalize_transcript_segments(segments)
    document = validate_transcript_document(transcript)
    return document, len(normalized_segments)


def query_transcript(
    transcript: Mapping[str, Any],
    question: str,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
    include_zero_scores: bool = False,
) -> dict[str, Any]:
    """Search one in-memory transcript using the existing local retriever."""
    normalized_question = _validated_question(question)
    resolved_top_k = _validated_top_k(top_k)
    resolved_min_score = _validated_min_score(min_score)
    document, segment_count = _validated_transcript_mapping(transcript)
    retriever = LocalTfidfRetriever(document)
    results = retriever.search(
        normalized_question,
        top_k=resolved_top_k,
        min_score=resolved_min_score,
        include_zero_scores=include_zero_scores,
    )
    retrieval_output = format_search_output(
        retriever,
        normalized_question,
        results,
    )
    return {
        "video": retrieval_output["video"],
        "query": retrieval_output["query"],
        "language": transcript.get("language"),
        "segment_count": segment_count,
        "chunk_count": retrieval_output["chunk_count"],
        "result_count": retrieval_output["result_count"],
        "results": retrieval_output["results"],
    }


def _transcribe_video(
    video_path: str | Path,
    *,
    model: str | None,
    device: str,
    compute_type: str,
    chunk_words: int | None,
) -> dict[str, Any]:
    path = _validated_video_path(video_path)
    resolved_chunk_words = resolve_chunk_words(chunk_words)
    transcript = transcribe_to_memory(
        str(path),
        video_name=str(path),
        model_size=model,
        device=device,
        compute_type=compute_type,
    )
    return build_transcript_output(
        transcript,
        chunk_words=resolved_chunk_words,
    )


def _query_video_with_transcript(
    video_path: str | Path,
    question: str,
    *,
    model: str | None,
    device: str,
    compute_type: str,
    chunk_words: int | None,
    top_k: int,
    min_score: float,
    include_zero_scores: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_question = _validated_question(question)
    _validated_top_k(top_k)
    _validated_min_score(min_score)
    transcript = _transcribe_video(
        video_path,
        model=model,
        device=device,
        compute_type=compute_type,
        chunk_words=chunk_words,
    )
    result = query_transcript(
        transcript,
        normalized_question,
        top_k=top_k,
        min_score=min_score,
        include_zero_scores=include_zero_scores,
    )
    return result, transcript


def query_video(
    video_path: str,
    question: str,
    *,
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    chunk_words: int | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
    include_zero_scores: bool = False,
) -> dict[str, Any]:
    """Transcribe one local video and return ranked timestamped chunks."""
    result, _ = _query_video_with_transcript(
        video_path,
        question,
        model=model,
        device=device,
        compute_type=compute_type,
        chunk_words=chunk_words,
        top_k=top_k,
        min_score=min_score,
        include_zero_scores=include_zero_scores,
    )
    return result


def load_transcript_input(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 or UTF-8-with-BOM transcript JSON object."""
    transcript_path = Path(path).expanduser()
    if not transcript_path.exists() or not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")
    try:
        loaded = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid transcript JSON: {transcript_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Transcript JSON must contain an object")
    return loaded


def _validate_destination(path: Path | None, label: str) -> None:
    if path is None:
        return
    destination = path.expanduser()
    if destination.exists():
        raise FileExistsError(f"{label} already exists: {destination}")
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise FileNotFoundError(
            f"{label} parent directory not found: {destination.parent}"
        )


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    pretty: bool,
) -> None:
    serialized = json.dumps(
        value,
        indent=2 if pretty else None,
        ensure_ascii=False,
    )
    with path.expanduser().open("x", encoding="utf-8") as output_file:
        output_file.write(f"{serialized}\n")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the unified dependency-light command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a local video and search it, or search an existing "
            "transcript JSON."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        help=(
            "Local video path, or the question when --transcript-input is used."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Lexical question when a local video path is supplied.",
    )
    parser.add_argument(
        "--transcript-input",
        type=Path,
        default=None,
        help="Search an existing transcript JSON without invoking Faster-Whisper.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='Faster-Whisper model name or local path (default: "base").',
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help='Faster-Whisper device for video mode (default: "cpu").',
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help='Faster-Whisper compute type for video mode (default: "int8").',
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help="Maximum target words per transcript chunk.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum retrieval results (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum cosine score from 0 to 1 (default: 0).",
    )
    parser.add_argument(
        "--include-zero-scores",
        action="store_true",
        help="Explicitly include unrelated zero-score chunks.",
    )
    parser.add_argument(
        "--save-transcript",
        type=Path,
        default=None,
        help="Save the complete generated transcript; video mode only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the unified JSON result instead of printing it.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the unified JSON result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified CLI and return a process exit code."""
    args = build_argument_parser().parse_args(argv)
    try:
        _validate_destination(args.output, "Output file")
        _validate_destination(args.save_transcript, "Transcript output")
        if args.chunk_words is not None:
            resolve_chunk_words(args.chunk_words)
        if (
            args.output is not None
            and args.save_transcript is not None
            and args.output.expanduser().resolve()
            == args.save_transcript.expanduser().resolve()
        ):
            raise ValueError("--output and --save-transcript must use different paths")

        if args.transcript_input is not None:
            if args.question is not None:
                raise ValueError(
                    "Do not supply a video path together with --transcript-input"
                )
            if args.source is None:
                raise ValueError(
                    "A question is required when --transcript-input is used"
                )
            if args.save_transcript is not None:
                raise ValueError("--save-transcript is only valid in video mode")
            transcript = load_transcript_input(args.transcript_input)
            result = query_transcript(
                transcript,
                args.source,
                top_k=args.top_k,
                min_score=args.min_score,
                include_zero_scores=args.include_zero_scores,
            )
        else:
            if args.source is None:
                raise ValueError(
                    "Supply a local video path or use --transcript-input"
                )
            if args.question is None:
                raise ValueError("A question is required after the video path")
            result, transcript = _query_video_with_transcript(
                args.source,
                args.question,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_words=args.chunk_words,
                top_k=args.top_k,
                min_score=args.min_score,
                include_zero_scores=args.include_zero_scores,
            )
            if args.save_transcript is not None:
                _write_json(args.save_transcript, transcript, pretty=True)

        json_output = json.dumps(
            result,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        if args.output is None:
            print(json_output)
        else:
            _write_json(args.output, result, pretty=args.pretty)
        return 0
    except KeyboardInterrupt:
        print("VideoMind interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"VideoMind failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
