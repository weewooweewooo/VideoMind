"""Direct construction and querying for VideoMind retrieval backends."""

from __future__ import annotations

import math
from typing import Any

from src.retrieval.local_retriever import (
    LocalTfidfRetriever,
    TranscriptDocument,
)


RETRIEVAL_BACKENDS = ("tfidf", "semantic", "hybrid")

__all__ = [
    "RETRIEVAL_BACKENDS",
    "VideoMindRetriever",
    "build_retriever",
    "resolve_retrieval_options",
]


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
            from src.retrieval.semantic_retriever import SemanticRetriever

            retriever = SemanticRetriever(
                document,
                model_name=embedding_model,
                device=device,
            )
        else:
            from src.retrieval.hybrid_retriever import HybridRetriever

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
        resolved_top_k = _validated_top_k(top_k)
        resolved_min_score = (
            self._default_min_score
            if min_score is None
            else _validated_min_score(min_score)
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
