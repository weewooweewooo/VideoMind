"""VideoMind BM25 transcript retrieval."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.normalization import (
    _apply_compound_splits,
    _base_tokens,
    _discover_compound_splits,
    _split_sentences,
    _tokenize,
)

_DEFAULT_TOP_K = 5


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


def _score_bm25_sentences(
    query_tokens: list[str],
    tokenized_sentences: list[tuple[str, ...]],
    *,
    k1: float,
    b: float,
    epsilon: float,
) -> list[float]:
    """Score prepared eligible sentences with BM25Okapi's formula."""
    if not query_tokens or not tokenized_sentences:
        return [0.0] * len(tokenized_sentences)

    document_count = len(tokenized_sentences)
    average_length = sum(map(len, tokenized_sentences)) / document_count
    document_frequencies = Counter(
        term for tokens in tokenized_sentences for term in set(tokens)
    )
    raw_idf = {
        term: math.log(document_count - frequency + 0.5)
        - math.log(frequency + 0.5)
        for term, frequency in document_frequencies.items()
    }
    average_idf = sum(raw_idf.values()) / len(raw_idf)
    epsilon_idf = epsilon * average_idf
    idf = {
        term: epsilon_idf if value < 0.0 else value
        for term, value in raw_idf.items()
    }

    scores: list[float] = []
    for tokens in tokenized_sentences:
        frequencies = Counter(tokens)
        length_factor = k1 * (1 - b + b * len(tokens) / average_length)
        scores.append(
            sum(
                idf.get(term, 0.0)
                * (
                    frequencies.get(term, 0) * (k1 + 1)
                    / (frequencies.get(term, 0) + length_factor)
                )
                for term in query_tokens
            )
        )
    return scores


class _Bm25Retriever:
    """Build one in-process BM25 chunk index and reuse it across questions."""

    def __init__(self, video: str, chunks: tuple[_TranscriptChunk, ...]) -> None:
        self.video = video
        self._chunks = chunks
        base_chunk_tokens = [_base_tokens(chunk.text) for chunk in chunks]
        self._compound_splits = _discover_compound_splits(base_chunk_tokens)
        tokenized_chunks = [
            _apply_compound_splits(tokens, self._compound_splits)
            for tokens in base_chunk_tokens
        ]
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
        self._sentences = tuple(
            (chunk.chunk_id, sentence_index, sentence, tuple(tokens))
            for chunk in chunks
            for sentence_index, sentence in enumerate(_split_sentences(chunk.text))
            if (tokens := _tokenize(sentence, self._compound_splits))
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def sentence_count(self) -> int:
        return len(self._sentences)

    def search(self, question: str) -> list[dict[str, Any]]:
        """Return up to five deterministic positive-score BM25 matches."""
        query_tokens = _tokenize(
            _validated_question(question), self._compound_splits
        )
        return self._search_normalized(query_tokens)

    def _search_normalized(self, query_tokens: list[str]) -> list[dict[str, Any]]:
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

    def _select_best_sentence(self, question: str) -> dict[str, Any] | None:
        """Select the strongest exact sentence for focused retrieval."""
        query_tokens = _tokenize(
            _validated_question(question), self._compound_splits
        )
        chunk_results = self._search_normalized(query_tokens)
        if not chunk_results:
            return None

        parent_ranks = {
            int(result["chunk_id"]): int(result["rank"])
            for result in chunk_results
        }
        sentence_candidates = [
            candidate
            for candidate in self._sentences
            if candidate[0] in parent_ranks
        ]
        sentence_scores = _score_bm25_sentences(
            query_tokens,
            [candidate[3] for candidate in sentence_candidates],
            k1=self._index.k1,
            b=self._index.b,
            epsilon=self._index.epsilon,
        )
        scored_candidates = [
            (
                float(score),
                parent_ranks[chunk_id],
                sentence_position,
                chunk_id,
                sentence,
            )
            for score, (chunk_id, sentence_position, sentence, _) in zip(
                sentence_scores,
                sentence_candidates,
            )
            if float(score) > 0.0
        ]
        if not scored_candidates:
            return None

        score, parent_rank, sentence_position, chunk_id, sentence = min(
            scored_candidates,
            key=lambda candidate: (
                -candidate[0],
                candidate[1],
                candidate[2],
                candidate[3],
            ),
        )
        return {
            "text": sentence,
            "chunk_id": chunk_id,
            "sentence_score": score,
            "parent_rank": parent_rank,
            "sentence_index": sentence_position,
        }

    def search_focused(self, question: str) -> dict[str, Any] | None:
        """Return the strongest exact sentence within retrieved BM25 chunks."""
        return self._select_best_sentence(question)


def build_retriever(
    transcript: Mapping[str, Any],
) -> _Bm25Retriever:
    """Validate prepared chunks and build one reusable BM25 retriever."""
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
