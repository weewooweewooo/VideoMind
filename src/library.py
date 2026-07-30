"""Small combined in-memory library for cross-video transcript retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.ingestion.transcriber import (
    build_transcript_output,
    resolve_chunk_overlap_words,
    resolve_chunk_words,
)
from src.ingestion.transcript_chunks import normalize_transcript_segments
from src.retrieval.local_retriever import (
    LocalTfidfRetriever,
    TranscriptChunk,
    TranscriptDocument,
    format_search_output,
    validate_transcript_document,
)


SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4a", ".mov", ".mkv", ".webm"}
)


def _validated_directory(path: str | Path, label: str) -> Path:
    directory = Path(path).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {directory}")
    return directory.resolve()


def discover_video_files(directory: str | Path) -> list[Path]:
    """Return supported top-level video files in deterministic order."""
    library_directory = _validated_directory(directory, "Video library")
    try:
        videos = [
            path.resolve()
            for path in library_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
    except OSError as exc:
        raise OSError(
            f"Unable to read video library directory: {library_directory}"
        ) from exc
    videos.sort(key=lambda path: (path.name.casefold(), str(path)))
    if not videos:
        raise ValueError(
            f"Video library contains no supported media files: {library_directory}"
        )
    return videos


def load_transcript_library(
    directory: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every top-level JSON transcript or fail with the selected path."""
    library_directory = _validated_directory(directory, "Transcript library")
    try:
        paths = sorted(
            (
                path.resolve()
                for path in library_directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".json"
            ),
            key=lambda path: (path.name.casefold(), str(path)),
        )
    except OSError as exc:
        raise OSError(
            f"Unable to read transcript library directory: {library_directory}"
        ) from exc
    if not paths:
        raise ValueError(
            f"Transcript library contains no JSON files: {library_directory}"
        )

    transcripts = []
    labels = []
    for path in paths:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid transcript JSON: {path}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid transcript JSON object: {path}")
        transcripts.append(loaded)
        labels.append(str(path))
    return transcripts, labels


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


def _transcript_content_identity(transcript: Mapping[str, Any]) -> str:
    segments = normalize_transcript_segments(transcript.get("segments", []))
    canonical = json.dumps(
        {"segments": segments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class VideoLibrary:
    """Build one retrieval index over validated chunks from multiple videos."""

    def __init__(
        self,
        transcript_documents: Sequence[Mapping[str, Any]],
        *,
        retriever: str = "tfidf",
        embedding_model: str | None = None,
        device: str = "cpu",
        min_score: float | None = None,
        semantic_min_score: float | None = None,
        chunk_words: int | None = None,
        chunk_overlap_words: int = 0,
        source_labels: Sequence[str] | None = None,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not transcript_documents:
            raise ValueError("Video library contains no transcript documents")
        if (
            source_labels is not None
            and len(source_labels) != len(transcript_documents)
        ):
            raise ValueError("source_labels must match transcript_documents")

        resolved_chunk_words = resolve_chunk_words(chunk_words, environ={})
        resolved_chunk_overlap = resolve_chunk_overlap_words(
            chunk_overlap_words,
            chunk_words=resolved_chunk_words,
        )
        backend, default_min_score = _resolve_retrieval_options(
            retriever,
            embedding_model,
            min_score,
            semantic_min_score,
        )

        global_chunks = []
        chunk_sources = []
        accepted_videos = []
        seen_transcripts: dict[str, str] = {}
        for document_index, transcript in enumerate(transcript_documents):
            label = (
                source_labels[document_index]
                if source_labels is not None
                else f"transcript {document_index}"
            )
            try:
                if not isinstance(transcript, Mapping):
                    raise ValueError("Transcript JSON must contain an object")
                transcript_identity = _transcript_content_identity(transcript)
                first_label = seen_transcripts.get(transcript_identity)
                if first_label is not None:
                    if warning_callback is not None:
                        warning_callback(
                            "VideoMind duplicate transcript content skipped: "
                            f"{label}; using {first_label}."
                        )
                    continue

                search_transcript = transcript
                if chunk_words is not None or resolved_chunk_overlap:
                    search_transcript = build_transcript_output(
                        transcript,
                        chunk_words=resolved_chunk_words,
                        chunk_overlap_words=resolved_chunk_overlap,
                    )
                document = validate_transcript_document(search_transcript)
            except Exception as exc:
                raise ValueError(f"Invalid library transcript {label}: {exc}") from exc

            seen_transcripts[transcript_identity] = label
            accepted_videos.append(document.video)
            for source_chunk in document.chunks:
                library_chunk_id = len(global_chunks)
                global_chunks.append(
                    TranscriptChunk(
                        chunk_id=library_chunk_id,
                        start=source_chunk.start,
                        end=source_chunk.end,
                        text=source_chunk.text,
                    )
                )
                chunk_sources.append(
                    {
                        "video": document.video,
                        "source_chunk_id": source_chunk.chunk_id,
                    }
                )

        if not global_chunks:
            raise ValueError("Video library contains no usable transcript chunks")

        combined_document = TranscriptDocument(
            video="VideoMind library",
            chunks=tuple(global_chunks),
        )
        if backend == "tfidf":
            retrieval_backend = LocalTfidfRetriever(combined_document)
        elif backend == "semantic":
            from src.retrieval.semantic_retriever import SemanticRetriever

            retrieval_backend = SemanticRetriever(
                combined_document,
                model_name=embedding_model,
                device=device,
            )
        else:
            from src.retrieval.hybrid_retriever import HybridRetriever

            retrieval_backend = HybridRetriever(
                combined_document,
                model_name=embedding_model,
                device=device,
            )

        self.backend = backend
        self.default_min_score = default_min_score
        self.retriever = retrieval_backend
        self.video_count = len(accepted_videos)
        self.chunk_count = len(global_chunks)
        self.videos = tuple(accepted_videos)
        self._chunk_sources = tuple(chunk_sources)

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        include_zero_scores: bool = False,
    ) -> dict[str, Any]:
        """Search the combined index and return cross-video evidence."""
        normalized_question = _validated_question(question)
        resolved_top_k = _validated_top_k(top_k)
        resolved_min_score = (
            self.default_min_score
            if min_score is None
            else _validated_min_score(min_score)
        )

        if self.backend in {"tfidf", "semantic"}:
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

        formatted = format_search_output(
            self.retriever,
            normalized_question,
            results,
        )
        library_results = []
        for result in formatted["results"]:
            source = self._chunk_sources[int(result["chunk_id"])]
            library_result = {
                "rank": result["rank"],
                "chunk_id": result["chunk_id"],
                "source_chunk_id": source["source_chunk_id"],
                "video": source["video"],
                "start": result["start"],
                "end": result["end"],
                "text": result["text"],
                "score": result["score"],
            }
            for field in (
                "tfidf_rank",
                "semantic_rank",
                "tfidf_score",
                "semantic_score",
            ):
                if field in result:
                    library_result[field] = result[field]
            library_results.append(library_result)

        output = {
            "query": normalized_question,
            "retriever": self.backend,
            "video_count": self.video_count,
            "chunk_count": self.chunk_count,
            "result_count": len(library_results),
            "results": library_results,
        }
        if self.backend in {"semantic", "hybrid"}:
            output["embedding_model"] = self.retriever.model_name
        if self.backend == "hybrid":
            output["rrf_k"] = self.retriever.rrf_k
        return output
