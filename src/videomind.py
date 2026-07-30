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
    resolve_chunk_overlap_words,
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


def _resolve_retrieval_options(
    retriever: str,
    embedding_model: str | None,
    min_score: float | None,
    semantic_min_score: float | None,
) -> tuple[str, float]:
    if not isinstance(retriever, str):
        raise ValueError("retriever must be one of: tfidf, semantic, hybrid")
    backend = retriever.strip().lower()
    if backend not in {"tfidf", "semantic", "hybrid"}:
        raise ValueError("retriever must be one of: tfidf, semantic, hybrid")
    if backend == "tfidf":
        if embedding_model is not None:
            raise ValueError(
                "embedding_model is only valid with semantic or hybrid retrieval"
            )
        if semantic_min_score is not None:
            raise ValueError(
                "semantic_min_score is only valid with semantic or hybrid retrieval"
            )
        threshold = 0.0 if min_score is None else min_score
    else:
        from src.retrieval.semantic_retriever import DEFAULT_SEMANTIC_MIN_SCORE

        if min_score is not None and semantic_min_score is not None:
            raise ValueError(
                "Use either min_score or semantic_min_score, not both"
            )
        configured_threshold = (
            semantic_min_score if semantic_min_score is not None else min_score
        )
        threshold = (
            DEFAULT_SEMANTIC_MIN_SCORE
            if configured_threshold is None
            else configured_threshold
        )
    return backend, _validated_min_score(threshold)


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
        backend, default_min_score = _resolve_retrieval_options(
            retriever,
            embedding_model,
            min_score,
            semantic_min_score,
        )

        if backend == "tfidf":
            retrieval_backend = LocalTfidfRetriever(document)
        elif backend == "semantic":
            from src.retrieval.semantic_retriever import SemanticRetriever

            retrieval_backend = SemanticRetriever(
                document,
                model_name=embedding_model,
                device=device,
            )
        else:
            from src.retrieval.hybrid_retriever import HybridRetriever

            retrieval_backend = HybridRetriever(
                document,
                model_name=embedding_model,
                device=device,
            )

        self.transcript = session_transcript
        self.document = document
        self.segment_count = segment_count
        self.backend = backend
        self.default_min_score = default_min_score
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
        resolved_top_k = _validated_top_k(top_k)
        resolved_min_score = (
            self.default_min_score
            if min_score is None
            else _validated_min_score(min_score)
        )

        if self.backend == "tfidf":
            results = self.retriever.search(
                normalized_question,
                top_k=resolved_top_k,
                min_score=resolved_min_score,
                include_zero_scores=include_zero_scores,
            )
        elif self.backend == "semantic":
            results = self.retriever.search(
                normalized_question,
                top_k=resolved_top_k,
                min_score=resolved_min_score,
                include_zero_scores=include_zero_scores,
            )
        else:
            if include_zero_scores:
                raise ValueError(
                    "include_zero_scores is not supported with hybrid retrieval"
                )
            results = self.retriever.search(
                normalized_question,
                top_k=resolved_top_k,
                semantic_min_score=resolved_min_score,
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
        if self.backend in {"semantic", "hybrid"}:
            output["embedding_model"] = self.retriever.model_name
        if self.backend == "hybrid":
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


def _transcribe_video(
    video_path: str | Path,
    *,
    model: str | None,
    device: str,
    compute_type: str,
    chunk_words: int | None,
    chunk_overlap_words: int,
) -> dict[str, Any]:
    path = _validated_video_path(video_path)
    resolved_chunk_words = resolve_chunk_words(chunk_words)
    resolved_chunk_overlap = resolve_chunk_overlap_words(
        chunk_overlap_words,
        chunk_words=resolved_chunk_words,
    )
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
        chunk_overlap_words=resolved_chunk_overlap,
    )


def _query_video_with_transcript(
    video_path: str | Path,
    question: str,
    *,
    model: str | None,
    device: str,
    compute_type: str,
    chunk_words: int | None,
    chunk_overlap_words: int,
    retriever: str,
    embedding_model: str | None,
    top_k: int,
    min_score: float | None,
    semantic_min_score: float | None,
    include_zero_scores: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_question = _validated_question(question)
    _validated_top_k(top_k)
    _resolve_retrieval_options(
        retriever,
        embedding_model,
        min_score,
        semantic_min_score,
    )
    transcript = _transcribe_video(
        video_path,
        model=model,
        device=device,
        compute_type=compute_type,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
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
    return result, transcript


def query_video(
    video_path: str,
    question: str,
    *,
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    chunk_words: int | None = None,
    chunk_overlap_words: int = 0,
    retriever: str = "tfidf",
    embedding_model: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    semantic_min_score: float | None = None,
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
        chunk_overlap_words=chunk_overlap_words,
        retriever=retriever,
        embedding_model=embedding_model,
        top_k=top_k,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
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


def _print_json(value: Mapping[str, Any], *, pretty: bool) -> None:
    print(
        json.dumps(
            value,
            indent=2 if pretty else None,
            ensure_ascii=False,
        )
    )


def _query_and_print(
    session: VideoMindSession,
    question: str,
    *,
    top_k: int,
    include_zero_scores: bool,
    pretty: bool,
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
    _print_json(result, pretty=pretty)


def _run_interactive_session(
    session: VideoMindSession,
    *,
    initial_question: str | None,
    top_k: int,
    include_zero_scores: bool,
    pretty: bool,
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
                top_k=top_k,
                include_zero_scores=include_zero_scores,
                pretty=pretty,
            )
    except KeyboardInterrupt:
        print("\nVideoMind interactive session ended", file=sys.stderr)
        return 0


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
        help="Search question when a local video path is supplied.",
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
        "--output",
        type=Path,
        default=None,
        help="Write the unified JSON result instead of printing it.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Reuse one transcript index for independent questions read from stdin.",
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
            environ={} if args.transcript_input is not None else None,
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
        selected_backend, _ = _resolve_retrieval_options(
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
        else:
            if args.source is None:
                raise ValueError(
                    "Supply a local video path or use --transcript-input"
                )
            if args.question is None and not args.interactive:
                raise ValueError("A question is required after the video path")
            transcript = _transcribe_video(
                args.source,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_words=args.chunk_words,
                chunk_overlap_words=args.chunk_overlap_words,
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
            if args.save_transcript is not None:
                _write_json(args.save_transcript, transcript, pretty=True)

        if args.interactive:
            return _run_interactive_session(
                session,
                initial_question=initial_question,
                top_k=args.top_k,
                include_zero_scores=args.include_zero_scores,
                pretty=args.pretty,
            )

        result = session.query(
            initial_question,
            top_k=args.top_k,
            include_zero_scores=args.include_zero_scores,
        )
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
