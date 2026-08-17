"""VideoMind BM25S transcript retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import bm25s
import Stemmer

from src.config import DEFAULT_TOP_K


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


class _Bm25Retriever:
    """Build one in-process BM25S chunk index and reuse it across questions."""

    def __init__(self, video: str, chunks: tuple[_TranscriptChunk, ...]) -> None:
        self.video = video
        self._chunks = chunks
        self._stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in chunks],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens, show_progress=False)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self, question: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """Return up to top_k deterministic positive-score BM25 matches."""
        if top_k <= 0:
            return []
        query_tokens = bm25s.tokenize(
            _validated_question(question),
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        document_ids, scores = self._index.retrieve(
            query_tokens,
            k=len(self._chunks),
            show_progress=False,
        )
        candidates = [
            (float(score), self._chunks[int(document_id)])
            for document_id, score in zip(document_ids[0], scores[0])
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
                candidates[:top_k], start=1
            )
        ]


def build_retriever(
    transcript: Mapping[str, Any],
) -> _Bm25Retriever:
    """Validate prepared chunks and build one reusable BM25S retriever."""
    if not isinstance(transcript, Mapping):
        raise ValueError("Invalid transcript")
    video = transcript.get("video")
    raw_chunks = transcript.get("chunks")
    if (
        not isinstance(video, str)
        or not video.strip()
        or not isinstance(raw_chunks, list)
        or not raw_chunks
    ):
        raise ValueError("Invalid transcript")

    chunks: list[_TranscriptChunk] = []
    previous_start: float | None = None
    previous_end: float | None = None
    for chunk_id, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, Mapping):
            raise ValueError(f"Invalid transcript chunk at index {chunk_id}")
        text = raw_chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid transcript chunk at index {chunk_id}")
        try:
            start = float(raw_chunk.get("start"))
            end = float(raw_chunk.get("end"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid transcript chunk at index {chunk_id}") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid transcript chunk at index {chunk_id}")
        if previous_start is not None and previous_end is not None and (
            start < previous_start or end < previous_end
        ):
            raise ValueError(f"Transcript chunks are out of order at index {chunk_id}")
        chunks.append(_TranscriptChunk(chunk_id, start, end, text.strip()))
        previous_start = start
        previous_end = end

    return _Bm25Retriever(video.strip(), tuple(chunks))
