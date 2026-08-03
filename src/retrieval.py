"""VideoMind BM25 transcript retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
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


def _light_stem(token: str) -> str:
    """Normalize a small set of safe English suffix patterns."""
    if (
        len(token) <= 3
        or token.isdigit()
        or token in _STOPWORDS
        or "'" in token
    ):
        return token

    replacements = (
        ("izations", "ize"),
        ("ization", "ize"),
        ("izing", "ize"),
        ("ized", "ize"),
        ("ating", "ate"),
        ("ated", "ate"),
        ("izes", "ize"),
        ("ates", "ate"),
    )
    for suffix, replacement in replacements:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + replacement
            return stem if len(stem) >= 4 else token

    for suffix in ("ations", "ation"):
        if token.endswith(suffix):
            root = token[: -len(suffix)]
            if len(root) < 4:
                return token
            if root.endswith(("form", "ment")):
                return root
            if root.endswith("vers"):
                return root + "e"
            return root + "ate"

    if token.endswith("s") and not token.endswith(
        ("es", "is", "ss", "us", "ys")
    ):
        stem = token[:-1]
        return stem if len(stem) >= 4 else token
    return token


def _base_tokens(text: str) -> list[str]:
    return [
        _light_stem(match.group(0).lower())
        for match in _TOKEN_PATTERN.finditer(text)
    ]


def _discover_compound_splits(
    tokenized_chunks: list[list[str]],
) -> dict[str, tuple[str, str]]:
    document_frequencies = Counter(
        token
        for tokens in tokenized_chunks
        for token in set(tokens)
        if token not in _STOPWORDS
    )
    splits: dict[str, tuple[str, str]] = {}
    for token in sorted(document_frequencies):
        if len(token) < 6 or token.isdigit():
            continue
        candidates: list[tuple[int, int, int, str, str]] = []
        for position in range(3, len(token) - 2):
            left, right = token[:position], token[position:]
            left_frequency = document_frequencies.get(left, 0)
            right_frequency = document_frequencies.get(right, 0)
            if left_frequency < 2 or right_frequency < 2:
                continue
            candidates.append(
                (
                    left_frequency + right_frequency,
                    -abs(len(left) - len(right)),
                    -position,
                    left,
                    right,
                )
            )
        if candidates:
            _, _, _, left, right = max(candidates)
            splits[token] = (left, right)
    return splits


def _apply_compound_splits(
    base_tokens: list[str],
    compound_splits: Mapping[str, tuple[str, str]],
) -> list[str]:
    tokens: list[str] = []
    for token in base_tokens:
        tokens.extend(compound_splits.get(token, (token,)))
    return [token for token in tokens if token not in _STOPWORDS]


def _tokenize(
    text: str,
    compound_splits: Mapping[str, tuple[str, str]],
) -> list[str]:
    return _apply_compound_splits(_base_tokens(text), compound_splits)


class _Bm25Retriever:
    """Build one in-process BM25 index and reuse it across questions."""

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

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, question: str) -> list[dict[str, Any]]:
        """Return up to five deterministic positive-score BM25 matches."""
        query_tokens = _tokenize(
            _validated_question(question), self._compound_splits
        )
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
