"""Evaluate the current VideoMind retrieval behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.ingestion import ingest_video
from src.retrieval import build_retriever


EVALUATION_PATH = Path(__file__).with_name("evaluation.json")
POSITIVE_TYPES = {"exact", "paraphrase"}


def _overlaps(result: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    overlap_seconds = min(
        float(result["end"]), float(expected["end"])
    ) - max(
        float(result["start"]), float(expected["start"])
    )
    return overlap_seconds > 1.0


def _load_evaluation() -> dict[str, Any]:
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if not isinstance(evaluation.get("video_path"), str):
        raise ValueError("evaluation.json must contain a video_path")

    queries = evaluation.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("evaluation.json must contain queries")

    for index, item in enumerate(queries):
        if not isinstance(item, dict) or item.get("type") not in (
            POSITIVE_TYPES | {"negative"}
        ):
            raise ValueError(f"Invalid query at index {index}")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Invalid query at index {index}")

        expected = item.get("expected")
        if item["type"] in POSITIVE_TYPES and not (
            isinstance(expected, dict)
            and isinstance(expected.get("start"), (int, float))
            and isinstance(expected.get("end"), (int, float))
            and expected["start"] < expected["end"]
        ):
            raise ValueError(f"Invalid expected window at index {index}")
        if item["type"] == "negative" and expected is not None:
            raise ValueError(f"Negative query {index} must expect no result")

    return evaluation


def _evaluate(
    transcript: Mapping[str, Any], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    retriever = build_retriever(transcript)
    rank_one = 0
    top_three = 0
    reciprocal_rank = 0.0
    negative_false_positives = 0
    rows: list[dict[str, Any]] = []

    for item in queries:
        results = retriever.search(item["query"], top_k=retriever.chunk_count)
        if item["type"] in POSITIVE_TYPES:
            match = next(
                (
                    result
                    for result in results
                    if _overlaps(result, item["expected"])
                ),
                None,
            )
            rank = int(match["rank"]) if match is not None else None
            rank_one += int(rank == 1)
            top_three += int(rank is not None and rank <= 3)
            reciprocal_rank += 1.0 / rank if rank is not None else 0.0
            rows.append({"item": item, "rank": rank})
        else:
            false_positive = bool(results)
            negative_false_positives += int(false_positive)
            rows.append({"item": item, "false_positive": false_positive})

    positive_count = sum(item["type"] in POSITIVE_TYPES for item in queries)
    negative_count = len(queries) - positive_count
    return {
        "rank_one": rank_one,
        "top_three": top_three,
        "mrr": reciprocal_rank / positive_count,
        "negative_false_positives": negative_false_positives,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "rows": rows,
    }


def _print_report(result: Mapping[str, Any]) -> None:
    print("VideoMind retrieval evaluation\n")
    print(f"Rank-1: {result['rank_one']}/{result['positive_count']}")
    print(f"Top-3: {result['top_three']}/{result['positive_count']}")
    print(f"MRR: {result['mrr']:.3f}")
    print(
        "Negative false positives: "
        f"{result['negative_false_positives']}/{result['negative_count']}"
    )

    print("\nPer-query results")
    for row in result["rows"]:
        item = row["item"]
        if item["type"] in POSITIVE_TYPES:
            rank = row["rank"]
            outcome = f"rank {rank}" if rank is not None else "not found"
        else:
            outcome = "false positive" if row["false_positive"] else "no result"
        print(f"- [{item['type']}] {outcome}: {item['query']}")


def main() -> int:
    evaluation = _load_evaluation()
    video_path = EVALUATION_PATH.parent / evaluation["video_path"]
    transcript = ingest_video(video_path)
    result = _evaluate(transcript, evaluation["queries"])
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
