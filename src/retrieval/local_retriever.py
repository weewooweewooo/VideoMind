"""Dependency-free TF-IDF retrieval for timestamped transcript chunks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.transcript_chunks import normalize_transcript_segments


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase text and return ASCII word/number tokens without punctuation."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


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


def validate_transcript_document(document: Mapping[str, Any]) -> TranscriptDocument:
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


def _cosine_similarity(
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
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        try:
            threshold = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_score must be a finite number") from exc
        if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
            raise ValueError("min_score must be between zero and one")

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
            score = _cosine_similarity(query_vector, document_vector)
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
            for rank, (score, chunk) in enumerate(candidates[:top_k], start=1)
        ]


def format_search_output(
    retriever: LocalTfidfRetriever,
    query: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Format search results for stable JSON output."""
    formatted_results = [
        {
            "rank": int(result["rank"]),
            "chunk_id": int(result["chunk_id"]),
            "start": float(result["start"]),
            "end": float(result["end"]),
            "text": str(result["text"]),
            "score": round(float(result["score"]), 6),
        }
        for result in results
    ]
    return {
        "query": query,
        "video": retriever.document.video,
        "chunk_count": retriever.chunk_count,
        "result_count": len(formatted_results),
        "results": formatted_results,
    }


def search_transcript_json(
    transcript_json: str | Path,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    include_zero_scores: bool = False,
) -> dict[str, Any]:
    """Load a transcript, build its local index, and return formatted results."""
    retriever = LocalTfidfRetriever.from_json(transcript_json)
    results = retriever.search(
        query,
        top_k=top_k,
        min_score=min_score,
        include_zero_scores=include_zero_scores,
    )
    return format_search_output(retriever, query, results)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the dependency-free local retrieval CLI parser."""
    parser = argparse.ArgumentParser(
        description="Rank timestamped transcript chunks with local TF-IDF."
    )
    parser.add_argument(
        "transcript_json",
        help="Transcript JSON produced by transcriber.",
    )
    parser.add_argument("query", help="Lexical search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum results to return (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum cosine score from 0 to 1 (default: 0).",
    )
    parser.add_argument(
        "--include-zero-scores",
        action="store_true",
        help="Explicitly allow unrelated zero-score chunks in the result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file; stdout is used when omitted.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local retrieval command and return a process exit code."""
    args = build_argument_parser().parse_args(argv)
    try:
        result = search_transcript_json(
            args.transcript_json,
            args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            include_zero_scores=args.include_zero_scores,
        )
        json_output = json.dumps(
            result,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        if args.output is None:
            print(json_output)
        else:
            args.output.write_text(f"{json_output}\n", encoding="utf-8")
        return 0
    except KeyboardInterrupt:
        print("Local retrieval interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Local retrieval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
