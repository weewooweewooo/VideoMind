"""VideoMind transcript retrieval."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion import normalize_transcript_segments


# Shared constants and transcript contracts

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SEMANTIC_MIN_SCORE = 0.4
RRF_K = 60
MINIMUM_CANDIDATE_POOL = 20
RETRIEVAL_BACKENDS = ("tfidf", "semantic", "hybrid")

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SEMANTIC_MIN_SCORE",
    "HybridRetriever",
    "LocalTfidfRetriever",
    "MINIMUM_CANDIDATE_POOL",
    "RETRIEVAL_BACKENDS",
    "RRF_K",
    "SemanticRetriever",
    "TOKEN_PATTERN",
    "TranscriptChunk",
    "TranscriptDocument",
    "VideoMindRetriever",
    "build_retriever",
    "format_search_output",
    "load_transcript_json",
    "resolve_embedding_model",
    "resolve_retrieval_options",
    "tokenize",
    "validate_transcript_document",
]


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """One validated timestamped chunk from the transcript JSON."""

    chunk_id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptDocument:
    """Validated transcript metadata and chunks."""

    video: str
    chunks: tuple[TranscriptChunk, ...]


# Transcript validation and result formatting


def validate_transcript_document(
    document: Mapping[str, Any],
) -> TranscriptDocument:
    """Validate the transcript JSON contract without altering timestamps or text."""
    if not isinstance(document, Mapping):
        raise ValueError("Transcript JSON must contain an object")

    raw_video = document.get("video")
    if not isinstance(raw_video, str) or not raw_video.strip():
        raise ValueError("Transcript JSON must contain a nonempty video field")

    raw_chunks = document.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Transcript JSON chunks must be a list")
    if not raw_chunks:
        raise ValueError("Transcript JSON contains no chunks")

    for index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"Transcript chunk {index} must be an object")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Transcript chunk {index} has empty text")

    normalized_chunks = normalize_transcript_segments(raw_chunks)
    chunks = tuple(
        TranscriptChunk(
            chunk_id=index,
            start=float(normalized["start"]),
            end=float(normalized["end"]),
            text=str(raw_chunks[index]["text"]),
        )
        for index, normalized in enumerate(normalized_chunks)
    )
    return TranscriptDocument(video=raw_video.strip(), chunks=chunks)


def load_transcript_json(path: str | Path) -> TranscriptDocument:
    """Load and validate one transcript JSON file."""
    transcript_path = Path(path)
    if not transcript_path.exists() or not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_path}")
    try:
        document = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON: {transcript_path}") from exc
    return validate_transcript_document(document)


def format_search_output(
    retriever: Any,
    query: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Format compatible retrieval results for stable JSON output."""
    formatted_results = []
    for result in results:
        formatted_result = {
            "rank": int(result["rank"]),
            "chunk_id": int(result["chunk_id"]),
            "start": float(result["start"]),
            "end": float(result["end"]),
            "text": str(result["text"]),
            "score": round(float(result["score"]), 6),
        }
        for field in ("tfidf_rank", "semantic_rank"):
            if field in result:
                value = result[field]
                formatted_result[field] = None if value is None else int(value)
        for field in ("tfidf_score", "semantic_score"):
            if field in result:
                value = result[field]
                formatted_result[field] = (
                    None if value is None else round(float(value), 6)
                )
        formatted_results.append(formatted_result)
    return {
        "query": query,
        "video": retriever.document.video,
        "chunk_count": retriever.chunk_count,
        "result_count": len(formatted_results),
        "results": formatted_results,
    }


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


def _validate_search_options(top_k: int, min_score: float) -> tuple[int, float]:
    return _validated_top_k(top_k), _validated_min_score(min_score)


# TF-IDF tokenisation, vectors, and retriever


def tokenize(text: str) -> list[str]:
    """Lowercase text and return ASCII word/number tokens without punctuation."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _normalized_tfidf_vector(
    tokens: Sequence[str],
    inverse_document_frequency: Mapping[str, float],
) -> dict[str, float]:
    """Build a unit-normalized TF-IDF vector over the index vocabulary."""
    if not tokens:
        return {}

    counts = Counter(tokens)
    token_count = len(tokens)
    weighted = {
        term: (count / token_count) * inverse_document_frequency[term]
        for term, count in counts.items()
        if term in inverse_document_frequency
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in weighted.items()}


def _tfidf_cosine_similarity(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    """Return the dot product of two unit-normalized sparse vectors."""
    if len(first) > len(second):
        first, second = second, first
    score = sum(value * second.get(term, 0.0) for term, value in first.items())
    return min(1.0, max(0.0, score))


class LocalTfidfRetriever:
    """Build one in-process TF-IDF index and reuse it across queries."""

    def __init__(self, document: TranscriptDocument) -> None:
        if not document.chunks:
            raise ValueError("Transcript document contains no chunks")

        tokenized_chunks: list[list[str]] = []
        for chunk in document.chunks:
            tokens = tokenize(chunk.text)
            if not tokens:
                raise ValueError(
                    f"Transcript chunk {chunk.chunk_id} contains no usable tokens"
                )
            tokenized_chunks.append(tokens)

        document_count = len(tokenized_chunks)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_chunks:
            document_frequency.update(set(tokens))

        self.document = document
        self.inverse_document_frequency = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._document_vectors = tuple(
            _normalized_tfidf_vector(tokens, self.inverse_document_frequency)
            for tokens in tokenized_chunks
        )

    @classmethod
    def from_json(cls, path: str | Path) -> LocalTfidfRetriever:
        """Load transcript JSON and build a reusable local index."""
        return cls(load_transcript_json(path))

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed transcript chunks."""
        return len(self.document.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        include_zero_scores: bool = False,
    ) -> list[dict[str, Any]]:
        """Rank transcript chunks by cosine similarity to a lexical query.

        Zero-score chunks are omitted unless ``include_zero_scores`` is true.
        Ties are resolved by the original zero-based chunk position.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Query contains no usable tokens")
        resolved_top_k, threshold = _validate_search_options(top_k, min_score)

        query_vector = _normalized_tfidf_vector(
            query_tokens,
            self.inverse_document_frequency,
        )
        if not query_vector and not include_zero_scores:
            return []

        candidates: list[tuple[float, TranscriptChunk]] = []
        for chunk, document_vector in zip(
            self.document.chunks,
            self._document_vectors,
        ):
            score = _tfidf_cosine_similarity(query_vector, document_vector)
            if score < threshold:
                continue
            if score == 0 and not include_zero_scores:
                continue
            candidates.append((score, chunk))

        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "score": score,
            }
            for rank, (score, chunk) in enumerate(
                candidates[:resolved_top_k],
                start=1,
            )
        ]


# Lazy semantic model loading and semantic retriever


def resolve_embedding_model(
    model_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve explicit value, environment value, then the lightweight default."""
    environment = os.environ if environ is None else environ
    resolved = (
        model_name
        or environment.get("VIDEOMIND_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    ).strip()
    if not resolved:
        raise ValueError("Embedding model name or path must not be empty")
    return resolved


def _load_text_embedding_class() -> Any:
    """Import FastEmbed only when semantic retrieval is selected."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "Semantic retrieval requires the optional semantic dependency. "
            "Install it using: python -m pip install -r "
            "requirements-semantic.txt"
        ) from exc
    return TextEmbedding


def _embedding_constructor_options(
    configured_model: str,
) -> tuple[str, dict[str, str]]:
    model_path = Path(configured_model).expanduser()
    if not model_path.exists():
        return configured_model, {}
    if not model_path.is_dir():
        raise ValueError(f"Embedding model path is not a directory: {model_path}")
    return DEFAULT_EMBEDDING_MODEL, {
        "specific_model_path": str(model_path.resolve())
    }


def _normalize_semantic_vector(values: Iterable[Any]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("Embedding model returned an empty vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding model returned an invalid vector")
    return tuple(value / norm for value in vector)


def _semantic_cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding vectors have inconsistent dimensions")
    score = sum(left * right for left, right in zip(first, second))
    return min(1.0, max(-1.0, score))


class SemanticRetriever:
    """Build one in-process semantic index and reuse it across queries."""

    def __init__(
        self,
        document: TranscriptDocument,
        *,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not document.chunks:
            raise ValueError("Transcript document contains no chunks")
        if device.strip().lower() != "cpu":
            raise ValueError("Semantic retrieval currently supports only CPU")

        self.document = document
        self.model_name = resolve_embedding_model(model_name)
        text_embedding = _load_text_embedding_class()
        fastembed_model_name, model_options = _embedding_constructor_options(
            self.model_name
        )

        model_started = time.perf_counter()
        self._model = text_embedding(
            model_name=fastembed_model_name,
            providers=["CPUExecutionProvider"],
            **model_options,
        )
        self.model_load_seconds = time.perf_counter() - model_started

        index_started = time.perf_counter()
        chunk_vectors = self._model.passage_embed(
            [chunk.text for chunk in document.chunks]
        )
        self._document_vectors = tuple(
            _normalize_semantic_vector(vector) for vector in chunk_vectors
        )
        self.index_build_seconds = time.perf_counter() - index_started
        if len(self._document_vectors) != len(document.chunks):
            raise ValueError(
                "Embedding model returned an unexpected number of chunk vectors"
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> SemanticRetriever:
        """Load transcript JSON and build a reusable semantic index."""
        return cls(
            load_transcript_json(path),
            model_name=model_name,
            device=device,
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed transcript chunks."""
        return len(self.document.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
        include_zero_scores: bool = False,
    ) -> list[dict[str, Any]]:
        """Rank transcript chunks by normalized semantic cosine similarity.

        ``min_score`` is a backend-specific cosine threshold from zero to one.
        Negative scores are always excluded. Zero scores are excluded unless
        ``include_zero_scores`` is explicitly enabled.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")
        resolved_top_k, threshold = _validate_search_options(top_k, min_score)

        query_vectors = tuple(self._model.query_embed(query.strip()))
        if len(query_vectors) != 1:
            raise ValueError("Embedding model returned an invalid query vector")
        query_vector = _normalize_semantic_vector(query_vectors[0])

        candidates = []
        for chunk, document_vector in zip(
            self.document.chunks,
            self._document_vectors,
        ):
            score = _semantic_cosine_similarity(query_vector, document_vector)
            if score < threshold:
                continue
            if score == 0 and not include_zero_scores:
                continue
            candidates.append((score, chunk))

        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "score": score,
            }
            for rank, (score, chunk) in enumerate(
                candidates[:resolved_top_k],
                start=1,
            )
        ]


# Hybrid candidate admission and reciprocal-rank fusion


class HybridRetriever:
    """Build one lexical index and one reusable semantic index over shared chunks."""

    def __init__(
        self,
        document: TranscriptDocument,
        *,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not document.chunks:
            raise ValueError("Transcript document contains no chunks")

        self.document = document
        self.rrf_k = RRF_K
        started = time.perf_counter()

        tfidf_started = time.perf_counter()
        self.tfidf_retriever = LocalTfidfRetriever(document)
        self.tfidf_index_seconds = time.perf_counter() - tfidf_started

        self.semantic_retriever = SemanticRetriever(
            document,
            model_name=model_name,
            device=device,
        )
        self.model_name = self.semantic_retriever.model_name
        self.construction_seconds = time.perf_counter() - started

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> HybridRetriever:
        """Load transcript JSON and build both reusable component indexes."""
        return cls(
            load_transcript_json(path),
            model_name=model_name,
            device=device,
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks shared by both component indexes."""
        return len(self.document.chunks)

    @property
    def semantic_model_load_seconds(self) -> float:
        return self.semantic_retriever.model_load_seconds

    @property
    def semantic_index_build_seconds(self) -> float:
        return self.semantic_retriever.index_build_seconds

    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        """Fuse component ranks after lexical-or-confident-semantic admission."""
        resolved_top_k, threshold = _validate_search_options(
            top_k,
            semantic_min_score,
        )
        candidate_pool_size = min(
            self.chunk_count,
            max(resolved_top_k * 4, MINIMUM_CANDIDATE_POOL),
        )

        if tokenize(query):
            tfidf_results = self.tfidf_retriever.search(
                query,
                top_k=candidate_pool_size,
                min_score=0.0,
            )
        else:
            tfidf_results = []
        semantic_results = self.semantic_retriever.search(
            query,
            top_k=candidate_pool_size,
            min_score=0.0,
        )

        tfidf_by_chunk = {
            int(result["chunk_id"]): result for result in tfidf_results
        }
        semantic_by_chunk = {
            int(result["chunk_id"]): result for result in semantic_results
        }
        admitted_chunk_ids = set(tfidf_by_chunk)
        admitted_chunk_ids.update(
            chunk_id
            for chunk_id, result in semantic_by_chunk.items()
            if float(result["score"]) >= threshold
        )

        candidates = []
        for chunk_id in admitted_chunk_ids:
            tfidf_result = tfidf_by_chunk.get(chunk_id)
            semantic_result = semantic_by_chunk.get(chunk_id)
            tfidf_rank = (
                int(tfidf_result["rank"]) if tfidf_result is not None else None
            )
            semantic_rank = (
                int(semantic_result["rank"]) if semantic_result is not None else None
            )
            fusion_score = 0.0
            if tfidf_rank is not None:
                fusion_score += 1.0 / (RRF_K + tfidf_rank)
            if semantic_rank is not None:
                fusion_score += 1.0 / (RRF_K + semantic_rank)

            chunk = self.document.chunks[chunk_id]
            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "score": fusion_score,
                    "tfidf_rank": tfidf_rank,
                    "semantic_rank": semantic_rank,
                    "tfidf_score": (
                        float(tfidf_result["score"])
                        if tfidf_result is not None
                        else None
                    ),
                    "semantic_score": (
                        float(semantic_result["score"])
                        if semantic_result is not None
                        else None
                    ),
                }
            )

        candidates.sort(key=lambda item: (-float(item["score"]), item["chunk_id"]))
        return [
            {**candidate, "rank": rank}
            for rank, candidate in enumerate(
                candidates[:resolved_top_k],
                start=1,
            )
        ]


# Unified retrieval facade


def _validated_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    return question.strip()


def resolve_retrieval_options(
    backend: str,
    embedding_model: str | None,
    min_score: float | None,
    semantic_min_score: float | None,
) -> tuple[str, float]:
    """Validate backend options and return its name and default score."""
    if not isinstance(backend, str):
        raise ValueError("retriever must be one of: tfidf, semantic, hybrid")
    resolved_backend = backend.strip().lower()
    if resolved_backend not in RETRIEVAL_BACKENDS:
        raise ValueError("retriever must be one of: tfidf, semantic, hybrid")

    if resolved_backend == "tfidf":
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

    return resolved_backend, _validated_min_score(threshold)


class VideoMindRetriever:
    """One reusable VideoMind retriever with a backend-neutral search method."""

    def __init__(
        self,
        document: TranscriptDocument,
        *,
        backend: str = "tfidf",
        embedding_model: str | None = None,
        device: str = "cpu",
        min_score: float | None = None,
        semantic_min_score: float | None = None,
    ) -> None:
        resolved_backend, default_min_score = resolve_retrieval_options(
            backend,
            embedding_model,
            min_score,
            semantic_min_score,
        )

        if resolved_backend == "tfidf":
            retriever: Any = LocalTfidfRetriever(document)
        elif resolved_backend == "semantic":
            retriever = SemanticRetriever(
                document,
                model_name=embedding_model,
                device=device,
            )
        else:
            retriever = HybridRetriever(
                document,
                model_name=embedding_model,
                device=device,
            )

        self._backend = resolved_backend
        self._default_min_score = default_min_score
        self._retriever = retriever

    @property
    def backend(self) -> str:
        """Return the selected backend name."""
        return self._backend

    @property
    def document(self) -> TranscriptDocument:
        """Return the shared validated transcript document."""
        return self._retriever.document

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed transcript chunks."""
        return self._retriever.chunk_count

    @property
    def embedding_model(self) -> str | None:
        """Return semantic model metadata when the selected backend uses it."""
        return getattr(self._retriever, "model_name", None)

    @property
    def model_name(self) -> str | None:
        """Preserve the existing semantic model metadata attribute."""
        return self.embedding_model

    @property
    def rrf_k(self) -> int | None:
        """Return the hybrid RRF constant when applicable."""
        return getattr(self._retriever, "rrf_k", None)

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        include_zero_scores: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the selected backend while preserving its score semantics."""
        normalized_question = _validated_question(question)
        configured_min_score = (
            self._default_min_score
            if min_score is None
            else min_score
        )
        resolved_top_k, resolved_min_score = _validate_search_options(
            top_k,
            configured_min_score,
        )

        if self._backend in {"tfidf", "semantic"}:
            return self._retriever.search(
                normalized_question,
                top_k=resolved_top_k,
                min_score=resolved_min_score,
                include_zero_scores=include_zero_scores,
            )

        if include_zero_scores:
            raise ValueError(
                "include_zero_scores is not supported with hybrid retrieval"
            )
        return self._retriever.search(
            normalized_question,
            top_k=resolved_top_k,
            semantic_min_score=resolved_min_score,
        )


def build_retriever(
    document: TranscriptDocument,
    *,
    backend: str = "tfidf",
    embedding_model: str | None = None,
    device: str = "cpu",
    min_score: float | None = None,
    semantic_min_score: float | None = None,
) -> VideoMindRetriever:
    """Build one reusable retriever over an existing transcript document."""
    return VideoMindRetriever(
        document,
        backend=backend,
        embedding_model=embedding_model,
        device=device,
        min_score=min_score,
        semantic_min_score=semantic_min_score,
    )
