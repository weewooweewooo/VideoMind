"""VideoMind BM25 transcript retrieval."""

from __future__ import annotations

# Standard-library imports

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.ingestion import _normalize_transcript_segments


# Tokenization and constants

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_DEFAULT_TOP_K = 5
# Generic function words plus "work", a non-discriminative process verb.
_STOPWORDS = frozenset(
    (
        "a an and are as at be been being but by can could did do does doing "
        "for from had has have having he her hers him his how i if in into is "
        "it its itself may me might my of on or our ours she should so that "
        "the their theirs them themselves then there these they this those "
        "through to was we were what when where which while who why will with "
        "work would you your yours"
    ).split()
)


# Internal transcript contracts and validation


@dataclass(frozen=True, slots=True)
class _TranscriptChunk:
    """One validated timestamped chunk from the transcript JSON."""

    chunk_id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class _TranscriptDocument:
    """Validated transcript metadata and chunks."""

    video: str
    chunks: tuple[_TranscriptChunk, ...]


def _validate_transcript_document(
    document: Mapping[str, Any],
) -> _TranscriptDocument:
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

    normalized_chunks = _normalize_transcript_segments(raw_chunks)
    chunks = tuple(
        _TranscriptChunk(
            chunk_id=index,
            start=float(normalized["start"]),
            end=float(normalized["end"]),
            text=str(raw_chunks[index]["text"]),
        )
        for index, normalized in enumerate(normalized_chunks)
    )
    return _TranscriptDocument(video=raw_video.strip(), chunks=chunks)


def _validated_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    return question.strip()


# Shared lexical tokenization


def _tokenize(text: str) -> list[str]:
    """Return deterministic lowercase word tokens without generic stopwords."""
    return [
        token
        for match in _TOKEN_PATTERN.finditer(text)
        if (token := match.group(0).lower()) not in _STOPWORDS
    ]


# Reusable BM25 retriever


class _Bm25Retriever:
    """Build one in-process BM25 index and reuse it across questions."""

    def __init__(self, document: _TranscriptDocument) -> None:
        if not document.chunks:
            raise ValueError("Transcript document contains no chunks")

        self._tokenized_chunks: list[list[str]] = []
        for chunk in document.chunks:
            tokens = _tokenize(chunk.text)
            if not tokens:
                raise ValueError(
                    f"Transcript chunk {chunk.chunk_id} contains no usable tokens"
                )
            self._tokenized_chunks.append(tokens)

        self.document = document
        self._vocabulary = frozenset(
            token
            for tokens in self._tokenized_chunks
            for token in tokens
        )

        # Keep lightweight module imports independent of optional inference
        # dependencies and rank-bm25's transitive NumPy dependency.
        from rank_bm25 import BM25Okapi

        self._index = BM25Okapi(self._tokenized_chunks)

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed transcript chunks."""
        return len(self.document.chunks)

    def search(self, question: str) -> list[dict[str, Any]]:
        """Rank transcript evidence by raw BM25 score, where higher is stronger."""
        normalized_question = _validated_question(question)
        query_tokens = _tokenize(normalized_question)
        if not query_tokens:
            return []

        if self._vocabulary.isdisjoint(query_tokens):
            return []

        candidates: list[tuple[float, _TranscriptChunk]] = []
        for score, chunk in zip(
            self._index.get_scores(query_tokens),
            self.document.chunks,
        ):
            raw_score = float(score)
            if raw_score <= 0.0:
                continue
            candidates.append((raw_score, chunk))

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
                candidates[:_DEFAULT_TOP_K],
                start=1,
            )
        ]


# Public retrieval entry point


def build_retriever(
    transcript: Mapping[str, Any],
) -> _Bm25Retriever:
    """Build one reusable BM25 index for a prepared transcript."""
    return _Bm25Retriever(_validate_transcript_document(transcript))
