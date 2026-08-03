"""VideoMind BM25 transcript retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_DEFAULT_TOP_K = 5
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


@dataclass(frozen=True, slots=True)
class _TranscriptChunk:
    chunk_id: int
    start: float
    end: float
    text: str


def _validated_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")
    return question.strip()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for match in _TOKEN_PATTERN.finditer(text)
        if (token := match.group(0).lower()) not in _STOPWORDS
    ]


class _Bm25Retriever:
    """Build one in-process BM25 index and reuse it across questions."""

    def __init__(self, video: str, chunks: tuple[_TranscriptChunk, ...]) -> None:
        self.video = video
        self._chunks = chunks
        tokenized_chunks = [_tokenize(chunk.text) for chunk in chunks]
        for chunk, tokens in zip(chunks, tokenized_chunks):
            if not tokens:
                raise ValueError(
                    f"Transcript chunk {chunk.chunk_id} contains no usable tokens"
                )
        self._vocabulary = frozenset(
            token for tokens in tokenized_chunks for token in tokens
        )

        from rank_bm25 import BM25Okapi

        self._index = BM25Okapi(tokenized_chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, question: str) -> list[dict[str, Any]]:
        """Return up to five deterministic positive-score BM25 matches."""
        query_tokens = _tokenize(_validated_question(question))
        if not query_tokens or self._vocabulary.isdisjoint(query_tokens):
            return []

        candidates = [
            (float(score), chunk)
            for score, chunk in zip(self._index.get_scores(query_tokens), self._chunks)
            if float(score) > 0.0
        ]
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
                candidates[:_DEFAULT_TOP_K], start=1
            )
        ]


def build_retriever(transcript: Mapping[str, Any]) -> _Bm25Retriever:
    """Validate prepared chunks and build one reusable BM25 index."""
    if not isinstance(transcript, Mapping):
        raise ValueError("Transcript must be an object")
    video = transcript.get("video")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("Transcript must contain a nonempty video field")
    raw_chunks = transcript.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("Transcript must contain nonempty chunks")

    chunks: list[_TranscriptChunk] = []
    previous_start: float | None = None
    previous_end: float | None = None
    for chunk_id, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, Mapping):
            raise ValueError(f"Transcript chunk {chunk_id} must be an object")
        text = raw_chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Transcript chunk {chunk_id} has empty text")
        try:
            start = float(raw_chunk.get("start"))
            end = float(raw_chunk.get("end"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Transcript chunk {chunk_id} has invalid timestamps"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Transcript chunk {chunk_id} has invalid timestamps")
        if previous_start is not None and start < previous_start:
            raise ValueError(f"Transcript chunk {chunk_id} is out of timestamp order")
        if previous_end is not None and end < previous_end:
            raise ValueError(f"Transcript chunk {chunk_id} ends out of timestamp order")
        chunks.append(_TranscriptChunk(chunk_id, start, end, text.strip()))
        previous_start = start
        previous_end = end

    return _Bm25Retriever(video.strip(), tuple(chunks))
