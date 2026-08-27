"""Rank timestamped video evidence for one sentence corpus."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any


BM25_CANDIDATES = 10
DENSE_CANDIDATES = 10
CONTEXT_RADIUS = 2
OVERLAP_THRESHOLD = 0.8
DEFAULT_TOP_K = 5
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class VideoRetriever:
    """Own the in-memory indexes and models for one video's sentences."""

    def __init__(self, sentences: Sequence[Mapping[str, Any]]) -> None:
        self._sentences = _validate_sentences(sentences)
        texts = [sentence["text"] for sentence in self._sentences]

        try:
            import bm25s
            import numpy as np
            import Stemmer
            from fastembed import TextEmbedding
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Video retrieval dependencies are unavailable; install the "
                "production requirements before constructing VideoRetriever"
            ) from exc

        self._bm25s = bm25s
        self._np = np
        self._stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(
            texts,
            lower=True,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._bm25 = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
        self._bm25.index(corpus_tokens, show_progress=False)

        threads = max(1, min(8, os.cpu_count() or 1))
        self._dense_model = TextEmbedding(
            model_name=DENSE_MODEL,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        dense_matrix = np.asarray(
            list(self._dense_model.passage_embed(texts, batch_size=64)),
            dtype=np.float32,
        )
        norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Dense model produced a zero corpus embedding")
        self._dense_matrix = dense_matrix / norms
        self._reranker = TextCrossEncoder(
            model_name=RERANKER_MODEL,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the highest-scoring timestamped evidence windows for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty retrieval query is required")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        bm25_ranking = self._rank_bm25(query)
        dense_ranking = self._rank_dense(query)
        candidates = self._candidate_pool(bm25_ranking, dense_ranking)
        windows = self._expand_and_deduplicate(candidates)
        ranked = self._rerank(query, windows)
        return [
            {
                "start": window["start"],
                "end": window["end"],
                "text": window["text"],
                "score": window["reranker_score"],
                "sentence_indices": window["included_sentence_indices"],
                "source_segments": window["source_segments"],
            }
            for window in ranked[:top_k]
        ]

    def _rank_bm25(self, query: str) -> list[tuple[int, float]]:
        query_tokens = self._bm25s.tokenize(
            [query],
            lower=True,
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        indices, scores = self._bm25.retrieve(
            query_tokens,
            k=len(self._sentences),
            show_progress=False,
        )
        ranking = _sort_scores(
            [int(index) for index in indices[0]],
            [float(score) for score in scores[0]],
        )
        return ranking[:BM25_CANDIDATES]

    def _rank_dense(self, query: str) -> list[tuple[int, float]]:
        vector = self._np.asarray(
            list(self._dense_model.query_embed([query]))[0],
            dtype=self._np.float32,
        )
        norm = self._np.linalg.norm(vector)
        if norm == 0:
            raise RuntimeError("Dense model produced a zero query embedding")
        scores = self._dense_matrix @ (vector / norm)
        ranking = _sort_scores(
            list(range(len(scores))),
            [float(score) for score in scores],
        )
        return ranking[:DENSE_CANDIDATES]

    def _candidate_pool(
        self,
        bm25_ranking: list[tuple[int, float]],
        dense_ranking: list[tuple[int, float]],
    ) -> list[dict[str, Any]]:
        bm25 = {
            index: (rank, score)
            for rank, (index, score) in enumerate(
                bm25_ranking[:BM25_CANDIDATES],
                start=1,
            )
        }
        dense = {
            index: (rank, score)
            for rank, (index, score) in enumerate(
                dense_ranking[:DENSE_CANDIDATES],
                start=1,
            )
        }
        candidates = []
        for index in bm25.keys() | dense.keys():
            lexical = bm25.get(index)
            semantic = dense.get(index)
            candidates.append(
                {
                    "sentence_index": index,
                    "bm25_rank": lexical[0] if lexical else None,
                    "bm25_score": lexical[1] if lexical else None,
                    "dense_rank": semantic[0] if semantic else None,
                    "dense_score": semantic[1] if semantic else None,
                }
            )
        return sorted(candidates, key=_candidate_strength)

    def _expand_and_deduplicate(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=_candidate_strength):
            anchor = candidate["sentence_index"]
            sentence_indices = list(
                range(
                    max(0, anchor - CONTEXT_RADIUS),
                    min(len(self._sentences), anchor + CONTEXT_RADIUS + 1),
                )
            )
            if any(
                _window_overlap(
                    sentence_indices,
                    window["included_sentence_indices"],
                )
                >= OVERLAP_THRESHOLD
                for window in windows
            ):
                continue
            selected = [self._sentences[index] for index in sentence_indices]
            windows.append(
                {
                    **candidate,
                    "included_sentence_indices": sentence_indices,
                    "start": selected[0]["start"],
                    "end": selected[-1]["end"],
                    "text": " ".join(sentence["text"] for sentence in selected),
                    "source_segments": list(
                        dict.fromkeys(
                            source_index
                            for sentence in selected
                            for source_index in sentence["source_segments"]
                        )
                    ),
                }
            )
        return windows

    def _rerank(
        self,
        query: str,
        windows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scores = [
            float(score)
            for score in self._reranker.rerank(
                query,
                [window["text"] for window in windows],
            )
        ]
        ranked = []
        for window, score in zip(windows, scores, strict=True):
            ranked.append({**window, "reranker_score": score})
        return sorted(
            ranked,
            key=lambda window: (
                -window["reranker_score"],
                window["sentence_index"],
            ),
        )


def _validate_sentences(
    sentences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
        raise ValueError("Sentences must be a non-empty ordered sequence")
    if not sentences:
        raise ValueError("Sentences must be a non-empty ordered sequence")

    validated = []
    previous_start = previous_end = None
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, Mapping):
            raise ValueError(f"Invalid sentence at index {index}")
        start = sentence.get("start")
        end = sentence.get("end")
        text = sentence.get("text")
        source_segments = sentence.get("source_segments")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) <= float(start)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(source_segments, list)
            or not source_segments
            or any(
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                for source_index in source_segments
            )
            or (previous_start is not None and float(start) < previous_start)
            or (previous_end is not None and float(end) < previous_end)
        ):
            raise ValueError(f"Invalid sentence at index {index}")
        validated.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "source_segments": list(source_segments),
            }
        )
        previous_start, previous_end = float(start), float(end)
    return validated


def _sort_scores(
    indices: list[int],
    scores: list[float],
) -> list[tuple[int, float]]:
    return sorted(
        zip(indices, scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )


def _candidate_strength(candidate: Mapping[str, Any]) -> tuple[int, int, int]:
    bm25_rank = candidate.get("bm25_rank") or 9999
    dense_rank = candidate.get("dense_rank") or 9999
    return (
        min(bm25_rank, dense_rank),
        bm25_rank + dense_rank,
        candidate["sentence_index"],
    )


def _window_overlap(first: list[int], second: list[int]) -> float:
    return len(set(first) & set(second)) / min(len(first), len(second))
