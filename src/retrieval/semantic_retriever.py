"""Optional CPU semantic retrieval for timestamped transcript chunks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.retrieval.local_retriever import (
    TranscriptDocument,
    format_search_output,
    load_transcript_json,
)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SEMANTIC_MIN_SCORE = 0.4


def resolve_embedding_model(
    model_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve CLI value, environment value, then the lightweight default."""
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


def _normalize_vector(values: Iterable[Any]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("Embedding model returned an empty vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding model returned an invalid vector")
    return tuple(value / norm for value in vector)


def _cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding vectors have inconsistent dimensions")
    score = sum(left * right for left, right in zip(first, second))
    return min(1.0, max(-1.0, score))


def _validate_search_options(top_k: int, min_score: float) -> tuple[int, float]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    try:
        threshold = float(min_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_score must be a finite number") from exc
    if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("min_score must be between zero and one")
    return top_k, threshold


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
            _normalize_vector(vector) for vector in chunk_vectors
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
        query_vector = _normalize_vector(query_vectors[0])

        candidates = []
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
            for rank, (score, chunk) in enumerate(
                candidates[:resolved_top_k],
                start=1,
            )
        ]


def search_transcript_json(
    transcript_json: str | Path,
    query: str,
    *,
    model_name: str | None = None,
    device: str = "cpu",
    top_k: int = 5,
    min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
    include_zero_scores: bool = False,
) -> dict[str, Any]:
    """Build a semantic index and return formatted timestamped results."""
    retriever = SemanticRetriever.from_json(
        transcript_json,
        model_name=model_name,
        device=device,
    )
    results = retriever.search(
        query,
        top_k=top_k,
        min_score=min_score,
        include_zero_scores=include_zero_scores,
    )
    output = format_search_output(retriever, query, results)
    output["retriever"] = "semantic"
    output["embedding_model"] = retriever.model_name
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the optional semantic retrieval CLI parser."""
    parser = argparse.ArgumentParser(
        description="Rank timestamped transcript chunks with semantic embeddings."
    )
    parser.add_argument(
        "transcript_json",
        help="Transcript JSON produced by transcriber.",
    )
    parser.add_argument("query", help="Semantic search query.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "FastEmbed model name or local default-model directory "
            "(default: VIDEOMIND_EMBEDDING_MODEL or "
            f"{DEFAULT_EMBEDDING_MODEL})."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="Embedding device (default: cpu).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum results to return (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_SEMANTIC_MIN_SCORE,
        help=(
            "Minimum semantic cosine score from 0 to 1 "
            f"(default: {DEFAULT_SEMANTIC_MIN_SCORE})."
        ),
    )
    parser.add_argument(
        "--include-zero-scores",
        action="store_true",
        help="Allow zero-score chunks when --min-score is 0.",
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
    """Run the optional semantic retrieval command."""
    args = build_argument_parser().parse_args(argv)
    try:
        result = search_transcript_json(
            args.transcript_json,
            args.query,
            model_name=args.model,
            device=args.device,
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
        print("Semantic retrieval interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Semantic retrieval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
