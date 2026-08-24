"""Evaluate the current VideoMind retrieval behavior."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from src.ingestion import prepare_transcript
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

    fixture = evaluation.get("transcript_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("evaluation.json must contain a transcript_fixture")
    if not isinstance(fixture.get("path"), str) or not fixture["path"].strip():
        raise ValueError("Invalid transcript fixture path")
    fixture_sha256 = fixture.get("sha256")
    if not isinstance(fixture_sha256, str) or len(fixture_sha256) != 64:
        raise ValueError("Invalid transcript fixture SHA-256")
    for count_name in ("segment_count", "chunk_count"):
        count = fixture.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid transcript fixture {count_name}")

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


def _load_transcript_fixture(
    evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_definition = evaluation["transcript_fixture"]
    fixture_path = EVALUATION_PATH.parent / fixture_definition["path"]
    fixture_bytes = fixture_path.read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if fixture_sha256.lower() != fixture_definition["sha256"].lower():
        raise ValueError("Transcript fixture SHA-256 does not match evaluation.json")

    fixture = json.loads(fixture_bytes.decode("utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("Invalid transcript fixture")
    manifest = fixture.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Transcript fixture must contain a manifest")

    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("path") != evaluation["video_path"]:
        raise ValueError("Transcript fixture source does not match evaluation.json")
    source_size = source.get("size")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise ValueError("Invalid transcript fixture source size")
    source_sha256 = source.get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("Invalid transcript fixture source SHA-256")

    expected_profile = {
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "beam_size": WHISPER_BEAM_SIZE,
    }
    if manifest.get("profile") != expected_profile:
        raise ValueError("Transcript fixture profile is not the production profile")

    faster_whisper_version = manifest.get("faster_whisper_version")
    if (
        not isinstance(faster_whisper_version, str)
        or not faster_whisper_version.strip()
    ):
        raise ValueError("Invalid Faster-Whisper fixture provenance")
    transcription_seconds = manifest.get("transcription_seconds")
    if isinstance(transcription_seconds, bool):
        raise ValueError("Invalid fixture transcription time")
    try:
        transcription_seconds = float(transcription_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid fixture transcription time") from exc
    if not math.isfinite(transcription_seconds) or transcription_seconds <= 0:
        raise ValueError("Invalid fixture transcription time")

    language = fixture.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("Invalid transcript fixture language")
    duration_value = fixture.get("duration")
    if isinstance(duration_value, bool):
        raise ValueError("Invalid transcript fixture duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid transcript fixture duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid transcript fixture duration")
    if not isinstance(fixture.get("segments"), list) or not fixture["segments"]:
        raise ValueError("Transcript fixture contains no segments")

    transcript = prepare_transcript(
        {
            "language": language,
            "duration": duration,
            "segments": fixture["segments"],
        },
        source["path"],
    )
    if transcript["segment_count"] != fixture_definition["segment_count"]:
        raise ValueError(
            "Transcript fixture segment count does not match evaluation.json"
        )
    if transcript["chunk_count"] != fixture_definition["chunk_count"]:
        raise ValueError(
            "Transcript fixture chunk count does not match evaluation.json"
        )

    provenance = {
        "path": fixture_definition["path"],
        "sha256": fixture_sha256,
        "source": source,
        "profile": expected_profile,
        "faster_whisper_version": faster_whisper_version,
        "transcription_seconds": transcription_seconds,
        "segment_count": transcript["segment_count"],
        "chunk_count": transcript["chunk_count"],
    }
    return transcript, provenance


def _evaluate(
    transcript: Mapping[str, Any], queries: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, float]]:
    index_started = time.perf_counter()
    retriever = build_retriever(transcript)
    index_seconds = time.perf_counter() - index_started
    rank_one = 0
    top_three = 0
    reciprocal_rank = 0.0
    negative_false_positives = 0
    rows: list[dict[str, Any]] = []

    queries_started = time.perf_counter()
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
    query_seconds = time.perf_counter() - queries_started

    positive_count = sum(item["type"] in POSITIVE_TYPES for item in queries)
    negative_count = len(queries) - positive_count
    return (
        {
            "rank_one": rank_one,
            "top_three": top_three,
            "mrr": reciprocal_rank / positive_count,
            "negative_false_positives": negative_false_positives,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "rows": rows,
        },
        {
            "index_seconds": index_seconds,
            "query_seconds": query_seconds,
            "mean_query_seconds": query_seconds / len(queries),
        },
    )


def _print_report(
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
    runtime: Mapping[str, float],
) -> None:
    print("VideoMind retrieval evaluation\n")
    profile = provenance["profile"]
    source = provenance["source"]
    print("Fixture provenance")
    print(f"- Fixture: {provenance['path']}")
    print(f"- Fixture SHA-256: {provenance['sha256']}")
    print(f"- Source: {source['path']} ({source['size']} bytes)")
    print(f"- Source SHA-256: {source['sha256']}")
    print(
        "- Profile: "
        f"{profile['model']}, {profile['device']}, {profile['compute_type']}, "
        f"beam {profile['beam_size']}"
    )
    print(
        "- Faster-Whisper version at fixture generation: "
        f"{provenance['faster_whisper_version']}"
    )
    print(f"- Segments: {provenance['segment_count']}")
    print(f"- Chunks: {provenance['chunk_count']}")
    print(f"- Recorded ASR time: {provenance['transcription_seconds']:.3f}s\n")

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

    print("\nRetrieval runtime")
    print(f"- Index construction: {runtime['index_seconds']:.6f}s")
    print(f"- Queries total: {runtime['query_seconds']:.6f}s")
    print(f"- Query mean: {runtime['mean_query_seconds']:.6f}s")


def main() -> int:
    evaluation = _load_evaluation()
    transcript, provenance = _load_transcript_fixture(evaluation)
    result, runtime = _evaluate(transcript, evaluation["queries"])
    _print_report(result, provenance, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
