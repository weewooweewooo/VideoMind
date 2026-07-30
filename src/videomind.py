"""Unified CPU orchestration for transcript-only VideoMind retrieval."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion.transcript_cache import (
    TranscriptCacheIdentity,
    build_cache_identity,
    cache_entry_path,
    load_cached_transcript,
    prepare_cache_directory_for_write,
    resolve_cache_directory,
    store_cached_transcript,
    validate_cache_directory,
)
from src.ingestion.transcriber import (
    build_transcript_output,
    resolve_beam_size,
    resolve_chunk_overlap_words,
    resolve_chunk_words,
    resolve_compute_type,
    resolve_whisper_device,
    resolve_whisper_model,
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
    beam_size: int | None,
    language: str | None,
    chunk_words: int | None,
    chunk_overlap_words: int,
    cache_dir: str | Path | None,
    use_cache: bool,
    refresh_cache: bool,
    cache_identity: TranscriptCacheIdentity | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _validated_video_path(video_path)
    if refresh_cache and not use_cache:
        raise ValueError("refresh_cache requires use_cache")

    resolved_model = resolve_whisper_model(model)
    resolved_device = resolve_whisper_device(device)
    resolved_compute_type = resolve_compute_type(compute_type)
    resolved_beam_size = resolve_beam_size(beam_size)
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
        resolved_cache_dir = resolve_cache_directory(cache_dir)
        validate_cache_directory(resolved_cache_dir)
        identity = cache_identity
        if identity is None:
            identity = build_cache_identity(
                path,
                model=resolved_model,
                language=resolved_language,
                beam_size=resolved_beam_size,
                device=resolved_device,
                compute_type=resolved_compute_type,
            )
        destination = cache_entry_path(resolved_cache_dir, identity)
        cache_status = "refreshed" if refresh_cache else "miss"
        cache_metadata = {
            "enabled": True,
            "status": cache_status,
            "path": str(destination),
        }
        if not refresh_cache:
            transcript = load_cached_transcript(resolved_cache_dir, identity)
            if transcript is not None:
                transcript["video"] = str(path)
                cache_metadata["status"] = "hit"
                report("VideoMind transcript cache hit.")

    if transcript is None:
        if use_cache:
            if resolved_cache_dir is None:
                raise RuntimeError("Transcript cache initialization failed")
            prepare_cache_directory_for_write(resolved_cache_dir)
            if refresh_cache:
                report(
                    "VideoMind transcript cache refresh requested; "
                    "transcribing video."
                )
            else:
                report("VideoMind transcript cache miss; transcribing video.")
        else:
            report("VideoMind transcript cache disabled; transcribing video.")

        transcript = transcribe_to_memory(
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
                store_cached_transcript(
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
    _resolve_retrieval_options(
        retriever,
        embedding_model,
        min_score,
        semantic_min_score,
    )
    transcript, cache_metadata = _transcribe_video(
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


def _initialize_video_library(
    directory: str | Path,
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
    min_score: float | None,
    semantic_min_score: float | None,
    cache_dir: str | Path | None,
    use_cache: bool,
    refresh_cache: bool,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load distinct videos once and build one combined retrieval index."""
    from src.library import VideoLibrary, discover_video_files

    resolved_model = resolve_whisper_model(model)
    resolved_device = resolve_whisper_device(device)
    resolved_compute_type = resolve_compute_type(compute_type)
    resolved_beam_size = resolve_beam_size(beam_size)
    resolved_language = language.strip() if language else None
    paths = discover_video_files(directory)
    transcripts = []
    source_labels = []
    seen_video_hashes: dict[str, Path] = {}
    cache_summary = {
        "enabled": use_cache,
        "hits": 0,
        "misses": 0,
        "refreshed": 0,
        "disabled": 0,
    }

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    for path in paths:
        identity = build_cache_identity(
            path,
            model=resolved_model,
            language=resolved_language,
            beam_size=resolved_beam_size,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
        duplicate_of = seen_video_hashes.get(identity.video_sha256)
        if duplicate_of is not None:
            report(
                "VideoMind duplicate video content skipped: "
                f"{path.name}; using {duplicate_of.name}."
            )
            continue
        seen_video_hashes[identity.video_sha256] = path

        transcript, cache_metadata = _transcribe_video(
            path,
            model=resolved_model,
            device=resolved_device,
            compute_type=resolved_compute_type,
            beam_size=resolved_beam_size,
            language=resolved_language,
            chunk_words=chunk_words,
            chunk_overlap_words=chunk_overlap_words,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_identity=identity,
            status_callback=(
                lambda message, video_name=path.name: report(
                    f"[{video_name}] {message}"
                )
            ),
        )
        transcript["video"] = path.name
        transcripts.append(transcript)
        source_labels.append(str(path))
        status = str(cache_metadata["status"])
        summary_field = {
            "hit": "hits",
            "miss": "misses",
            "refreshed": "refreshed",
            "disabled": "disabled",
        }[status]
        cache_summary[summary_field] += 1

    library = VideoLibrary(
        transcripts,
        retriever=retriever,
        embedding_model=embedding_model,
        device=resolved_device,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
        source_labels=source_labels,
        warning_callback=report,
    )
    return library, cache_summary


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
                session, cache_summary = _initialize_video_library(
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
            transcript, cache_metadata = _transcribe_video(
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
