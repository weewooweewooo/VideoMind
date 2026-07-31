"""Unified CPU orchestration for transcript-only VideoMind retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion import (
    build_transcript_output,
    ingest_video,
    normalize_transcript_segments,
    resolve_chunk_overlap_words,
    resolve_chunk_words,
)
from src.library import ingest_video_library
from src.retrieval import (
    TranscriptDocument,
    build_retriever,
    format_search_output,
    resolve_retrieval_options,
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


class VideoMindSession:
    """Build one transcript index and reuse it for independent questions."""

    def __init__(
        self,
        transcript: Mapping[str, Any],
        *,
        retriever: str = "tfidf",
        embedding_model: str | None = None,
        device: str = "cpu",
        min_score: float | None = None,
        semantic_min_score: float | None = None,
        chunk_words: int | None = None,
        chunk_overlap_words: int = 0,
    ) -> None:
        resolved_chunk_words = resolve_chunk_words(chunk_words, environ={})
        resolved_chunk_overlap = resolve_chunk_overlap_words(
            chunk_overlap_words,
            chunk_words=resolved_chunk_words,
        )
        session_transcript = transcript
        if chunk_words is not None or resolved_chunk_overlap:
            session_transcript = build_transcript_output(
                transcript,
                chunk_words=resolved_chunk_words,
                chunk_overlap_words=resolved_chunk_overlap,
            )

        document, segment_count = _validated_transcript_mapping(
            session_transcript
        )
        retrieval_backend = build_retriever(
            document,
            backend=retriever,
            embedding_model=embedding_model,
            device=device,
            min_score=min_score,
            semantic_min_score=semantic_min_score,
        )

        self.transcript = session_transcript
        self.document = document
        self.segment_count = segment_count
        self.backend = retrieval_backend.backend
        self.retriever = retrieval_backend

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the session's reusable index."""
        return self.retriever.chunk_count

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        include_zero_scores: bool = False,
    ) -> dict[str, Any]:
        """Search the reusable index and return the existing unified schema."""
        normalized_question = _validated_question(question)
        results = self.retriever.search(
            normalized_question,
            top_k=top_k,
            min_score=min_score,
            include_zero_scores=include_zero_scores,
        )

        retrieval_output = format_search_output(
            self.retriever,
            normalized_question,
            results,
        )
        output = {
            "video": retrieval_output["video"],
            "query": retrieval_output["query"],
            "language": self.transcript.get("language"),
            "segment_count": self.segment_count,
            "chunk_count": retrieval_output["chunk_count"],
            "retriever": self.backend,
            "result_count": retrieval_output["result_count"],
            "results": retrieval_output["results"],
        }
        if self.retriever.embedding_model is not None:
            output["embedding_model"] = self.retriever.embedding_model
        if self.retriever.rrf_k is not None:
            output["rrf_k"] = self.retriever.rrf_k
        return output


def query_transcript(
    transcript: Mapping[str, Any],
    question: str,
    *,
    retriever: str = "tfidf",
    embedding_model: str | None = None,
    device: str = "cpu",
    top_k: int = 5,
    min_score: float | None = None,
    semantic_min_score: float | None = None,
    include_zero_scores: bool = False,
    chunk_words: int | None = None,
    chunk_overlap_words: int = 0,
) -> dict[str, Any]:
    """Build one session, search it once, and return the unified schema."""
    session = VideoMindSession(
        transcript,
        retriever=retriever,
        embedding_model=embedding_model,
        device=device,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    return session.query(
        question,
        top_k=top_k,
        include_zero_scores=include_zero_scores,
    )


def _query_video_with_transcript(
    video_path: str | Path,
    question: str,
    *,
    model: str | None,
    device: str,
    compute_type: str,
    beam_size: int | None,
    language: str | None,
    chunk_words: int | None,
    chunk_overlap_words: int,
    retriever: str,
    embedding_model: str | None,
    top_k: int,
    min_score: float | None,
    semantic_min_score: float | None,
    include_zero_scores: bool,
    cache_dir: str | Path | None,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_question = _validated_question(question)
    _validated_top_k(top_k)
    resolve_retrieval_options(
        retriever,
        embedding_model,
        min_score,
        semantic_min_score,
    )
    transcript, cache_metadata = ingest_video(
        video_path,
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    result = query_transcript(
        transcript,
        normalized_question,
        retriever=retriever,
        embedding_model=embedding_model,
        device=device,
        top_k=top_k,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
        include_zero_scores=include_zero_scores,
    )
    result["transcript_cache"] = cache_metadata
    return result, transcript


def query_video(
    video_path: str,
    question: str,
    *,
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int | None = None,
    language: str | None = None,
    chunk_words: int | None = None,
    chunk_overlap_words: int = 0,
    retriever: str = "tfidf",
    embedding_model: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    semantic_min_score: float | None = None,
    include_zero_scores: bool = False,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Transcribe one local video and return ranked timestamped chunks."""
    result, _ = _query_video_with_transcript(
        video_path,
        question,
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
        retriever=retriever,
        embedding_model=embedding_model,
        top_k=top_k,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
        include_zero_scores=include_zero_scores,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
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


def _print_json(value: Mapping[str, Any], *, pretty: bool) -> None:
    print(
        json.dumps(
            value,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )
    )


def _query_and_print(
    session: Any,
    question: str,
    *,
    top_k: int,
    include_zero_scores: bool,
    pretty: bool,
    result_metadata: Mapping[str, Any] | None = None,
) -> None:
    try:
        result = session.query(
            question,
            top_k=top_k,
            include_zero_scores=include_zero_scores,
        )
    except Exception as exc:
        print(f"VideoMind query failed: {exc}", file=sys.stderr)
        return
    if result_metadata is not None:
        result.update(result_metadata)
    _print_json(result, pretty=pretty)


def _run_interactive_session(
    session: Any,
    *,
    initial_question: str | None,
    top_k: int,
    include_zero_scores: bool,
    pretty: bool,
    result_metadata: Mapping[str, Any] | None = None,
    ready_message: str | None = None,
) -> int:
    """Read independent questions from stdin and reuse one indexed session."""
    try:
        if initial_question is not None:
            _query_and_print(
                session,
                initial_question,
                top_k=top_k,
                include_zero_scores=include_zero_scores,
                pretty=pretty,
                result_metadata=result_metadata,
            )

        print(
            ready_message
            or "VideoMind ready. Enter a question, or use :help for commands.",
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
                top_k=top_k,
                include_zero_scores=include_zero_scores,
                pretty=pretty,
                result_metadata=result_metadata,
            )
    except KeyboardInterrupt:
        print("\nVideoMind interactive session ended", file=sys.stderr)
        return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the unified dependency-light command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Search a local video, an existing transcript, or a small local "
            "video/transcript library."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        help=(
            "Local video path, or the question with a transcript/library mode."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Search question when a local video path is supplied.",
    )
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--transcript-input",
        type=Path,
        default=None,
        help="Search an existing transcript JSON without invoking Faster-Whisper.",
    )
    input_mode.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Search supported videos in one non-recursive local directory.",
    )
    input_mode.add_argument(
        "--transcript-library",
        type=Path,
        default=None,
        help="Search transcript JSON files in one non-recursive local directory.",
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
        "--beam-size",
        type=int,
        default=None,
        help="Faster-Whisper beam size for video mode (default: 5).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional transcription language; omit for automatic detection.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help="Maximum target words per transcript chunk.",
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
        "--retriever",
        choices=("tfidf", "semantic", "hybrid"),
        default="tfidf",
        help="Retrieval backend (default: tfidf).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Semantic model name or local path; valid only with "
            "--retriever semantic or hybrid."
        ),
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
        default=None,
        help=(
            "Minimum backend-specific cosine score from 0 to 1 "
            "(default: 0 for TF-IDF, 0.4 for semantic/hybrid)."
        ),
    )
    parser.add_argument(
        "--semantic-min-score",
        type=float,
        default=None,
        help=(
            "Semantic threshold for semantic/hybrid retrieval; takes the "
            "place of --min-score when supplied."
        ),
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
        help="Retranscribe and atomically replace the matching cache entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the unified JSON result instead of printing it.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Reuse one retrieval index for independent questions read from stdin.",
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
        if args.interactive and args.output is not None:
            raise ValueError(
                "--output is not supported with --interactive; redirect stdout instead"
            )
        _validate_destination(args.output, "Output file")
        _validate_destination(args.save_transcript, "Transcript output")
        validation_chunk_words = resolve_chunk_words(
            args.chunk_words,
            environ=(
                {}
                if (
                    args.transcript_input is not None
                    or args.transcript_library is not None
                )
                else None
            ),
        )
        resolve_chunk_overlap_words(
            args.chunk_overlap_words,
            chunk_words=validation_chunk_words,
        )
        if (
            args.output is not None
            and args.save_transcript is not None
            and args.output.expanduser().resolve()
            == args.save_transcript.expanduser().resolve()
        ):
            raise ValueError("--output and --save-transcript must use different paths")

        _validated_top_k(args.top_k)
        selected_backend, _ = resolve_retrieval_options(
            args.retriever,
            args.embedding_model,
            args.min_score,
            args.semantic_min_score,
        )
        if selected_backend == "hybrid" and args.include_zero_scores:
            raise ValueError(
                "--include-zero-scores is not supported with hybrid retrieval"
            )

        if args.transcript_input is not None:
            if args.question is not None:
                raise ValueError(
                    "Do not supply a video path together with --transcript-input"
                )
            if args.source is None and not args.interactive:
                raise ValueError(
                    "A question is required when --transcript-input is used"
                )
            if args.save_transcript is not None:
                raise ValueError("--save-transcript is only valid in video mode")
            if (
                args.cache_dir is not None
                or args.no_cache
                or args.refresh_cache
            ):
                raise ValueError(
                    "Transcript cache options are only valid in video mode"
                )
            transcript = load_transcript_input(args.transcript_input)
            session = VideoMindSession(
                transcript,
                retriever=args.retriever,
                embedding_model=args.embedding_model,
                device=args.device,
                min_score=args.min_score,
                semantic_min_score=args.semantic_min_score,
                chunk_words=args.chunk_words,
                chunk_overlap_words=args.chunk_overlap_words,
            )
            initial_question = args.source
            result_metadata = None
            ready_message = None
        elif args.library is not None or args.transcript_library is not None:
            library_option = (
                "--library"
                if args.library is not None
                else "--transcript-library"
            )
            if args.question is not None:
                raise ValueError(
                    f"Supply only an optional question with {library_option}"
                )
            if args.source is None and not args.interactive:
                raise ValueError(
                    f"A question is required when {library_option} is used"
                )
            if args.save_transcript is not None:
                raise ValueError(
                    "--save-transcript is not supported with library modes"
                )

            if args.library is not None:
                session, cache_summary = ingest_video_library(
                    args.library,
                    model=args.model,
                    device=args.device,
                    compute_type=args.compute_type,
                    beam_size=args.beam_size,
                    language=args.language,
                    chunk_words=args.chunk_words,
                    chunk_overlap_words=args.chunk_overlap_words,
                    retriever=args.retriever,
                    embedding_model=args.embedding_model,
                    min_score=args.min_score,
                    semantic_min_score=args.semantic_min_score,
                    cache_dir=args.cache_dir,
                    use_cache=not args.no_cache,
                    refresh_cache=args.refresh_cache,
                    status_callback=lambda message: print(
                        message,
                        file=sys.stderr,
                    ),
                )
                result_metadata = {"transcript_cache": cache_summary}
            else:
                if (
                    args.cache_dir is not None
                    or args.no_cache
                    or args.refresh_cache
                ):
                    raise ValueError(
                        "Transcript cache options are only valid in video modes"
                    )
                from src.library import VideoLibrary, load_transcript_library

                documents, source_labels = load_transcript_library(
                    args.transcript_library
                )
                session = VideoLibrary(
                    documents,
                    retriever=args.retriever,
                    embedding_model=args.embedding_model,
                    device=args.device,
                    min_score=args.min_score,
                    semantic_min_score=args.semantic_min_score,
                    chunk_words=args.chunk_words,
                    chunk_overlap_words=args.chunk_overlap_words,
                    source_labels=source_labels,
                    warning_callback=lambda message: print(
                        message,
                        file=sys.stderr,
                    ),
                )
                result_metadata = None
            initial_question = args.source
            ready_message = (
                "VideoMind library ready: "
                f"{session.video_count} videos, {session.chunk_count} chunks."
            )
        else:
            if args.source is None:
                raise ValueError(
                    "Supply a local video path or select a transcript/library mode"
                )
            if args.question is None and not args.interactive:
                raise ValueError("A question is required after the video path")
            transcript, cache_metadata = ingest_video(
                args.source,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                language=args.language,
                chunk_words=args.chunk_words,
                chunk_overlap_words=args.chunk_overlap_words,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                status_callback=lambda message: print(message, file=sys.stderr),
            )
            session = VideoMindSession(
                transcript,
                retriever=args.retriever,
                embedding_model=args.embedding_model,
                device=args.device,
                min_score=args.min_score,
                semantic_min_score=args.semantic_min_score,
            )
            initial_question = args.question
            result_metadata = {"transcript_cache": cache_metadata}
            ready_message = None
            if args.save_transcript is not None:
                _write_json(args.save_transcript, transcript, pretty=True)

        if args.interactive:
            return _run_interactive_session(
                session,
                initial_question=initial_question,
                top_k=args.top_k,
                include_zero_scores=args.include_zero_scores,
                pretty=args.pretty,
                result_metadata=result_metadata,
                ready_message=ready_message,
            )

        result = session.query(
            initial_question,
            top_k=args.top_k,
            include_zero_scores=args.include_zero_scores,
        )
        if result_metadata is not None:
            result.update(result_metadata)
        if args.output is None:
            _print_json(result, pretty=args.pretty)
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
