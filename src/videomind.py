"""Single-video VideoMind command-line application."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion import _normalize_transcript_segments, ingest_video
from src.retrieval import build_retriever


def _validated_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    return question.strip()


def _format_evidence(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep algorithm-specific ranking details out of the product result."""
    return [
        {
            "rank": int(result["rank"]),
            "chunk_id": int(result["chunk_id"]),
            "start": float(result["start"]),
            "end": float(result["end"]),
            "text": str(result["text"]),
            "score": round(float(result["score"]), 6),
        }
        for result in results
    ]


class _VideoMindSession:
    """Build one transcript index and reuse it for independent questions."""

    def __init__(self, transcript: Mapping[str, Any]) -> None:
        if not isinstance(transcript, Mapping):
            raise ValueError("Transcript JSON must contain an object")
        segments = transcript.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Transcript JSON segments must be a list")

        self.transcript = transcript
        self.segment_count = len(_normalize_transcript_segments(segments))
        self.retriever = build_retriever(transcript)

    def query(self, question: str) -> dict[str, Any]:
        """Search the reusable index and return transcript evidence."""
        normalized_question = _validated_question(question)
        results = _format_evidence(self.retriever.search(normalized_question))
        return {
            "video": self.retriever.document.video,
            "query": normalized_question,
            "language": self.transcript.get("language"),
            "segment_count": self.segment_count,
            "chunk_count": self.retriever.chunk_count,
            "result_count": len(results),
            "results": results,
        }


def _load_transcript_input(path: str | Path) -> dict[str, Any]:
    """Load a diagnostic transcript JSON object."""
    transcript_path = Path(path).expanduser()
    if not transcript_path.exists() or not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")
    try:
        loaded = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid transcript JSON: {transcript_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Transcript JSON must contain an object")
    return loaded


def _validate_destination(path: Path | None) -> None:
    if path is None:
        return
    destination = path.expanduser()
    if destination.exists():
        raise FileExistsError(f"Transcript output already exists: {destination}")
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Transcript output parent not found: {destination.parent}"
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False)
    with path.expanduser().open("x", encoding="utf-8") as output_file:
        output_file.write(f"{serialized}\n")


def _print_json(value: Mapping[str, Any], *, pretty: bool) -> None:
    print(
        json.dumps(
            value,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )
    )


def _query_and_print(
    session: _VideoMindSession,
    question: str,
    *,
    pretty: bool,
    cache_metadata: Mapping[str, Any] | None,
) -> None:
    try:
        result = session.query(question)
    except Exception as exc:
        print(f"VideoMind query failed: {exc}", file=sys.stderr)
        return
    if cache_metadata is not None:
        result["transcript_cache"] = cache_metadata
    _print_json(result, pretty=pretty)


def _run_interactive_session(
    session: _VideoMindSession,
    *,
    initial_question: str | None,
    pretty: bool,
    cache_metadata: Mapping[str, Any] | None,
) -> int:
    """Read independent questions while reusing one prepared transcript index."""
    try:
        if initial_question is not None:
            _query_and_print(
                session,
                initial_question,
                pretty=pretty,
                cache_metadata=cache_metadata,
            )

        print(
            "VideoMind ready. Enter a question, or use :help for commands.",
            file=sys.stderr,
        )
        while True:
            print("videomind> ", end="", file=sys.stderr, flush=True)
            question_input = sys.stdin.readline()
            if question_input == "":
                print("", file=sys.stderr)
                return 0

            question = question_input.strip()
            if not question:
                continue
            if question == ":help":
                print("Commands: :help, :quit, :exit", file=sys.stderr)
                continue
            if question in {":quit", ":exit"}:
                return 0

            _query_and_print(
                session,
                question,
                pretty=pretty,
                cache_metadata=cache_metadata,
            )
    except KeyboardInterrupt:
        print("\nVideoMind interactive session ended", file=sys.stderr)
        return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one local video and retrieve transcript evidence "
            "for one or more questions."
        )
    )
    parser.add_argument("source", nargs="?", help="Path to one local video.")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question about the selected video.",
    )
    parser.add_argument(
        "--transcript-input",
        type=Path,
        default=None,
        help=(
            "Diagnostic: use a saved transcript JSON without retranscribing; "
            "the first positional value is then the question."
        ),
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
        help='Faster-Whisper device (default: "cpu").',
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help='Faster-Whisper compute type (default: "int8").',
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Faster-Whisper beam size (default: 5).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional transcription language; omit for automatic detection.",
    )
    parser.add_argument(
        "--save-transcript",
        type=Path,
        default=None,
        help="Save the prepared transcript JSON for diagnostics.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Transcript cache directory (default: VIDEOMIND_CACHE_DIR or "
            "the platform user cache)."
        ),
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable transcript cache reads and writes.",
    )
    cache_mode.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Retranscribe and replace the matching cache entry.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Reuse the prepared transcript for questions read from stdin.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print evidence JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single-video CLI and return a process exit code."""
    args = _build_argument_parser().parse_args(argv)
    try:
        _validate_destination(args.save_transcript)

        cache_metadata: Mapping[str, Any] | None
        if args.transcript_input is not None:
            if args.question is not None:
                raise ValueError(
                    "Supply only one optional question with --transcript-input"
                )
            if args.source is None and not args.interactive:
                raise ValueError(
                    "A question is required when --transcript-input is used"
                )
            if args.save_transcript is not None:
                raise ValueError(
                    "--save-transcript is only valid when a video is selected"
                )
            if args.cache_dir is not None or args.no_cache or args.refresh_cache:
                raise ValueError(
                    "Transcript cache options require a selected video"
                )
            transcript = _load_transcript_input(args.transcript_input)
            initial_question = args.source
            cache_metadata = None
        else:
            if args.source is None:
                raise ValueError("Supply one local video path")
            if args.question is None and not args.interactive:
                raise ValueError("A question is required after the video path")
            transcript, cache_metadata = ingest_video(
                args.source,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                language=args.language,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                status_callback=lambda message: print(message, file=sys.stderr),
            )
            initial_question = args.question
            if args.save_transcript is not None:
                _write_json(args.save_transcript, transcript)

        session = _VideoMindSession(transcript)
        if args.interactive:
            return _run_interactive_session(
                session,
                initial_question=initial_question,
                pretty=args.pretty,
                cache_metadata=cache_metadata,
            )

        result = session.query(initial_question)
        if cache_metadata is not None:
            result["transcript_cache"] = cache_metadata
        _print_json(result, pretty=args.pretty)
        return 0
    except KeyboardInterrupt:
        print("VideoMind interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"VideoMind failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
