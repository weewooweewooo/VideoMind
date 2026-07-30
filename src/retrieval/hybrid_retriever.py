"""Optional reciprocal-rank fusion over local TF-IDF and semantic retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.retrieval.local_retriever import (
    LocalTfidfRetriever,
    TranscriptDocument,
    format_search_output,
    load_transcript_json,
    tokenize,
)
from src.retrieval.semantic_retriever import (
    DEFAULT_SEMANTIC_MIN_SCORE,
    SemanticRetriever,
    _validate_search_options,
)


RRF_K = 60
MINIMUM_CANDIDATE_POOL = 20


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


def search_transcript_json(
    transcript_json: str | Path,
    query: str,
    *,
    model_name: str | None = None,
    device: str = "cpu",
    top_k: int = 5,
    semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
) -> dict[str, Any]:
    """Build both indexes and return formatted hybrid results."""
    retriever = HybridRetriever.from_json(
        transcript_json,
        model_name=model_name,
        device=device,
    )
    results = retriever.search(
        query,
        top_k=top_k,
        semantic_min_score=semantic_min_score,
    )
    output = format_search_output(retriever, query, results)
    output["retriever"] = "hybrid"
    output["embedding_model"] = retriever.model_name
    output["rrf_k"] = RRF_K
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the focused hybrid retrieval CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fuse local TF-IDF and semantic transcript retrieval."
    )
    parser.add_argument("transcript_json", help="Transcript JSON produced by transcriber.")
    parser.add_argument("query", help="Search query.")
    parser.add_argument(
        "--model",
        default=None,
        help="FastEmbed model name or local default-model directory.",
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
        "--semantic-min-score",
        type=float,
        default=DEFAULT_SEMANTIC_MIN_SCORE,
        help=(
            "Semantic candidate-admission threshold "
            f"(default: {DEFAULT_SEMANTIC_MIN_SCORE})."
        ),
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
    """Run the hybrid retrieval command."""
    args = build_argument_parser().parse_args(argv)
    try:
        result = search_transcript_json(
            args.transcript_json,
            args.query,
            model_name=args.model,
            device=args.device,
            top_k=args.top_k,
            semantic_min_score=args.semantic_min_score,
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
        print("Hybrid retrieval interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Hybrid retrieval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
