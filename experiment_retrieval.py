"""Compare sentence-level BM25, dense, and hybrid retrieval on a frozen fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import bm25s
import numpy as np
import psutil
import Stemmer
from fastembed import TextEmbedding


FIXTURE_PATH = Path("data/test2.pysbd.sentences.json")
EXPECTED_FIXTURE_HASH = (
    "3e34ce8693508cddc9827ecee365c08f4b27c49615275e7a16a2aa0e4852727f"
)
EXPECTED_SENTENCE_COUNT = 445
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60
TOP_K = 5
POSITIVE_CATEGORIES = ("exact", "semantic", "topic", "multi_evidence")
EXPECTED_CATEGORY_COUNTS = {
    "exact": 4,
    "semantic": 4,
    "topic": 3,
    "multi_evidence": 2,
    "negative": 3,
}


def parse_args() -> argparse.Namespace:
    default_root = Path(tempfile.gettempdir()) / "videomind-retrieval-experiment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_root,
        help="External directory containing the frozen evaluation and generated reports.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution_requirements(name: str) -> list[str]:
    try:
        requirements = importlib.metadata.requires(name) or []
    except importlib.metadata.PackageNotFoundError:
        return []
    return sorted(requirement for requirement in requirements if "extra ==" not in requirement)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def rss_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(FIXTURE_PATH) != EXPECTED_FIXTURE_HASH:
        raise AssertionError("Frozen pySBD fixture hash does not match")
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("Fixture must be a JSON object")
    sentences = data.get("sentences")
    if (
        data.get("method") != "pysbd"
        or data.get("version") != "0.3.4"
        or data.get("sentence_count") != EXPECTED_SENTENCE_COUNT
        or not isinstance(sentences, list)
        or len(sentences) != EXPECTED_SENTENCE_COUNT
    ):
        raise AssertionError("Fixture is not the frozen 445-sentence pySBD artifact")

    previous_start = previous_end = None
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict) or set(sentence) != {
            "start",
            "end",
            "text",
            "source_segments",
        }:
            raise AssertionError(f"Sentence {index} has an invalid shape")
        start = sentence["start"]
        end = sentence["end"]
        text = sentence["text"]
        source_segments = sentence["source_segments"]
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
            or not text
            or not isinstance(source_segments, list)
            or not source_segments
            or any(
                isinstance(source_index, bool) or not isinstance(source_index, int)
                for source_index in source_segments
            )
            or (previous_start is not None and float(start) < previous_start)
            or (previous_end is not None and float(end) < previous_end)
        ):
            raise AssertionError(f"Sentence {index} has invalid content")
        previous_start = float(start)
        previous_end = float(end)
    return data, sentences


def validate_evaluation(
    path: Path,
    sentences: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise AssertionError("Evaluation must be a JSON object")
    input_metadata = evaluation.get("input")
    if not isinstance(input_metadata, dict) or (
        input_metadata.get("sha256") != EXPECTED_FIXTURE_HASH
        or input_metadata.get("sentence_count") != EXPECTED_SENTENCE_COUNT
    ):
        raise AssertionError("Evaluation does not target the frozen fixture")
    policy = evaluation.get("ground_truth_policy")
    if not isinstance(policy, dict) or policy.get("frozen_before_retrieval") is not True:
        raise AssertionError("Ground truth was not declared frozen before retrieval")
    queries = evaluation.get("queries")
    if not isinstance(queries, list):
        raise AssertionError("Evaluation queries must be a list")
    counts = Counter(query.get("category") for query in queries if isinstance(query, dict))
    if counts != Counter(EXPECTED_CATEGORY_COUNTS):
        raise AssertionError(f"Unexpected query category counts: {dict(counts)}")
    query_ids: set[str] = set()
    for query in queries:
        if not isinstance(query, dict):
            raise AssertionError("Each query must be an object")
        query_id = query.get("id")
        question = query.get("query")
        category = query.get("category")
        truth = query.get("ground_truth")
        if (
            not isinstance(query_id, str)
            or query_id in query_ids
            or not isinstance(question, str)
            or not question.strip()
            or category not in EXPECTED_CATEGORY_COUNTS
            or not isinstance(truth, dict)
        ):
            raise AssertionError(f"Invalid query record: {query_id!r}")
        query_ids.add(query_id)
        answer = truth.get("answer")
        regions = truth.get("regions")
        if not isinstance(regions, list):
            raise AssertionError(f"Invalid ground-truth regions for {query_id}")
        if category == "negative":
            if answer != "NO_ANSWER" or regions:
                raise AssertionError(f"Negative query {query_id} must be NO_ANSWER")
            continue
        if answer != "EVIDENCE" or not regions:
            raise AssertionError(f"Positive query {query_id} requires evidence")
        for region in regions:
            if not isinstance(region, dict):
                raise AssertionError(f"Invalid region for {query_id}")
            indices = region.get("sentence_indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(sentences)
                    for index in indices
                )
                or indices != sorted(set(indices))
            ):
                raise AssertionError(f"Invalid sentence indices for {query_id}")
            selected = [sentences[index] for index in indices]
            expected_text = " ".join(str(sentence["text"]) for sentence in selected)
            if (
                region.get("text") != expected_text
                or float(region.get("start")) != float(selected[0]["start"])
                or float(region.get("end")) != float(selected[-1]["end"])
            ):
                raise AssertionError(f"Ground truth does not match fixture for {query_id}")

    chat_cases = evaluation.get("future_chat_routing_cases")
    if (
        not isinstance(chat_cases, list)
        or len(chat_cases) != 2
        or any(case.get("scored_as_retrieval") is not False for case in chat_cases)
    ):
        raise AssertionError("Future chat-routing cases must remain unscored")
    return evaluation


def ranked_items(
    indices: list[int],
    scores: list[float],
) -> list[tuple[int, float]]:
    return sorted(
        zip(indices, scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )


def run_bm25_query(
    retriever: bm25s.BM25,
    stemmer: Stemmer.Stemmer,
    query: str,
    sentence_count: int,
) -> list[tuple[int, float]]:
    query_tokens = bm25s.tokenize(
        [query],
        lower=True,
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    )
    indices, scores = retriever.retrieve(
        query_tokens,
        k=sentence_count,
        show_progress=False,
    )
    return ranked_items(
        [int(index) for index in indices[0]],
        [float(score) for score in scores[0]],
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise AssertionError("Dense model produced a zero embedding")
    return values / norms


def run_dense_query(
    model: TextEmbedding,
    matrix: np.ndarray,
    query: str,
) -> list[tuple[int, float]]:
    query_vector = np.asarray(list(model.query_embed([query]))[0], dtype=np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    scores = matrix @ query_vector
    return ranked_items(list(range(len(scores))), [float(score) for score in scores])


def reciprocal_rank_fusion(
    bm25_ranking: list[tuple[int, float]],
    dense_ranking: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    fused = Counter()
    for ranking in (bm25_ranking, dense_ranking):
        for rank, (sentence_index, _) in enumerate(ranking, start=1):
            fused[sentence_index] += 1 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def first_relevant_rank(
    ranking: list[tuple[int, float]],
    relevant_indices: set[int],
) -> int | None:
    for rank, (sentence_index, _) in enumerate(ranking, start=1):
        if sentence_index in relevant_indices:
            return rank
    return None


def coverage_at(
    ranking: list[tuple[int, float]],
    regions: list[dict[str, Any]],
    k: int,
) -> tuple[int, int]:
    top_indices = {sentence_index for sentence_index, _ in ranking[:k]}
    covered = sum(bool(top_indices & set(region["sentence_indices"])) for region in regions)
    return covered, len(regions)


def result_rows(
    ranking: list[tuple[int, float]],
    sentences: list[dict[str, Any]],
    k: int = TOP_K,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "score": score,
            "sentence_index": sentence_index,
            "start": sentences[sentence_index]["start"],
            "end": sentences[sentence_index]["end"],
            "text": sentences[sentence_index]["text"],
        }
        for rank, (sentence_index, score) in enumerate(ranking[:k], start=1)
    ]


def summarize_metrics(
    query_results: list[dict[str, Any]],
    retriever: str,
    category: str | None = None,
) -> dict[str, Any]:
    eligible = [
        result
        for result in query_results
        if result["category"] != "negative"
        and (category is None or result["category"] == category)
    ]
    ranks = [result["retrievers"][retriever]["first_relevant_rank"] for result in eligible]
    count = len(ranks)
    if not count:
        raise AssertionError("Metric group contains no positive queries")
    rank1_count = sum(rank == 1 for rank in ranks)
    recall3_count = sum(rank is not None and rank <= 3 for rank in ranks)
    recall5_count = sum(rank is not None and rank <= 5 for rank in ranks)
    reciprocal_rank_sum = sum(1 / rank for rank in ranks if rank is not None)
    return {
        "query_count": count,
        "rank1_count": rank1_count,
        "rank1": rank1_count / count,
        "recall_at_3_count": recall3_count,
        "recall_at_3": recall3_count / count,
        "recall_at_5_count": recall5_count,
        "recall_at_5": recall5_count / count,
        "mrr": reciprocal_rank_sum / count,
    }


def score_distributions(
    query_results: list[dict[str, Any]],
    retriever: str,
) -> dict[str, Any]:
    positive = [
        result["retrievers"][retriever]["top5"][0]["score"]
        for result in query_results
        if result["category"] != "negative"
    ]
    negative = [
        result["retrievers"][retriever]["top5"][0]["score"]
        for result in query_results
        if result["category"] == "negative"
    ]
    return {
        "positive_top_score": {
            "min": min(positive),
            "max": max(positive),
            "mean": sum(positive) / len(positive),
        },
        "negative_top_score": {
            "min": min(negative),
            "max": max(negative),
            "mean": sum(negative) / len(negative),
        },
        "strictly_separable_by_top_score": min(positive) > max(negative),
        "note": "Descriptive only; no rejection threshold was selected.",
    }


def choose_recommendation(metrics: dict[str, Any]) -> tuple[str, str]:
    overall = metrics["overall"]
    category = metrics["by_category"]
    bm25 = overall["bm25"]
    dense = overall["dense"]
    hybrid = overall["hybrid"]
    best_component_rank1 = max(bm25["rank1_count"], dense["rank1_count"])
    best_component_recall3 = max(
        bm25["recall_at_3_count"], dense["recall_at_3_count"]
    )
    if (
        hybrid["recall_at_5_count"]
        >= max(bm25["recall_at_5_count"], dense["recall_at_5_count"])
        and (
            hybrid["rank1_count"] >= best_component_rank1 + 2
            or hybrid["recall_at_3_count"] >= best_component_recall3 + 2
        )
    ):
        return (
            "HYBRID LEADS",
            "Hybrid produced a meaningful gain of at least two positive queries at "
            "Rank-1 or Recall@3 without losing Recall@5, enough to justify retaining "
            "the more complex candidate for further testing.",
        )
    semantic_bm25 = category["semantic"]["bm25"]
    semantic_dense = category["semantic"]["dense"]
    if (
        dense["rank1_count"] >= bm25["rank1_count"]
        and dense["recall_at_3_count"] >= bm25["recall_at_3_count"] + 1
        and semantic_dense["recall_at_3_count"]
        >= semantic_bm25["recall_at_3_count"] + 1
        and hybrid["recall_at_3_count"] < dense["recall_at_3_count"] + 2
    ):
        return (
            "DENSE LEADS",
            "Dense improved both overall and semantic Recall@3 without a Rank-1 "
            "regression, while hybrid did not add a large enough gain to justify "
            "its additional fusion complexity.",
        )
    best_rank1 = max(item["rank1_count"] for item in overall.values())
    best_recall3 = max(item["recall_at_3_count"] for item in overall.values())
    best_recall5 = max(item["recall_at_5_count"] for item in overall.values())
    best_mrr = max(item["mrr"] for item in overall.values())
    if (
        bm25["rank1_count"] >= best_rank1 - 1
        and bm25["recall_at_3_count"] >= best_recall3 - 1
        and bm25["recall_at_5_count"] >= best_recall5 - 1
        and bm25["mrr"] >= best_mrr - 0.05
    ):
        return (
            "BM25 LEADS",
            "BM25 stayed within one positive query of the best candidate at each "
            "cutoff and within 0.05 MRR, so the measured gain from added model and "
            "fusion complexity was not substantial.",
        )
    return (
        "INCONCLUSIVE",
        "Quality differences were mixed: no more complex candidate cleared the "
        "predeclared meaningful-gain rule, and BM25 was not close enough across all "
        "metrics to call the methods essentially tied.",
    )


def important_failures(query_results: list[dict[str, Any]]) -> dict[str, list[str]]:
    comparisons = {
        "bm25_fails_dense_succeeds": [],
        "dense_fails_bm25_succeeds": [],
        "hybrid_improves_both": [],
        "hybrid_makes_ranking_worse": [],
    }
    for result in query_results:
        if result["category"] == "negative":
            continue
        ranks = {
            name: result["retrievers"][name]["first_relevant_rank"]
            for name in ("bm25", "dense", "hybrid")
        }
        comparable = {name: rank if rank is not None else math.inf for name, rank in ranks.items()}
        if comparable["bm25"] > TOP_K and comparable["dense"] <= TOP_K:
            comparisons["bm25_fails_dense_succeeds"].append(result["id"])
        if comparable["dense"] > TOP_K and comparable["bm25"] <= TOP_K:
            comparisons["dense_fails_bm25_succeeds"].append(result["id"])
        if comparable["hybrid"] < min(comparable["bm25"], comparable["dense"]):
            comparisons["hybrid_improves_both"].append(result["id"])
        if comparable["hybrid"] > min(comparable["bm25"], comparable["dense"]):
            comparisons["hybrid_makes_ranking_worse"].append(result["id"])
    return comparisons


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def build_report(results: dict[str, Any]) -> str:
    metrics = results["metrics"]
    lines = [
        "# VideoMind sentence retrieval experiment",
        "",
        "The frozen input is 445 atomic pySBD sentences. This experiment does not "
        "change production retrieval, implement chat, or select a rejection threshold.",
        "",
        "## Evaluation set",
        "",
    ]
    for category, count in EXPECTED_CATEGORY_COUNTS.items():
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "- future chat-routing case (unscored): `Summarize the whole video.`",
            "- future chat-routing case (unscored): `Simplify that explanation.`",
            "",
            "## Overall retrieval results",
            "",
            "| Retriever | Rank-1 | Recall@3 | Recall@5 | MRR | Avg query latency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("bm25", "dense", "hybrid"):
        item = metrics["overall"][name]
        lines.append(
            f"| {name.upper()} | {item['rank1_count']}/{item['query_count']} "
            f"({format_metric(item['rank1'])}) | {item['recall_at_3_count']}/"
            f"{item['query_count']} ({format_metric(item['recall_at_3'])}) | "
            f"{item['recall_at_5_count']}/{item['query_count']} "
            f"({format_metric(item['recall_at_5'])}) | {format_metric(item['mrr'])} | "
            f"{results['latency_ms'][name]['mean']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            "Latencies include query preparation and ranking. Hybrid latency includes "
            "both component queries plus RRF; one-time dense model loading and passage "
            "embedding are reported separately below.",
            "",
            "## Results by category",
            "",
            "| Category | Retriever | Rank-1 | Recall@3 | Recall@5 | MRR |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for category in POSITIVE_CATEGORIES:
        for name in ("bm25", "dense", "hybrid"):
            item = metrics["by_category"][category][name]
            lines.append(
                f"| {category} | {name.upper()} | {item['rank1_count']}/"
                f"{item['query_count']} | {item['recall_at_3_count']}/"
                f"{item['query_count']} | {item['recall_at_5_count']}/"
                f"{item['query_count']} | {format_metric(item['mrr'])} |"
            )
    lines.extend(["", "### Multi-evidence coverage", ""])
    for result in results["queries"]:
        if result["category"] != "multi_evidence":
            continue
        lines.append(f"- `{result['id']}` — {result['query']}")
        for name in ("bm25", "dense", "hybrid"):
            coverage = result["retrievers"][name]["evidence_coverage"]
            lines.append(
                f"  - {name.upper()}: top 3 {coverage['at_3']['covered']}/"
                f"{coverage['at_3']['total']}; top 5 {coverage['at_5']['covered']}/"
                f"{coverage['at_5']['total']} regions"
            )
    lines.extend(
        [
            "",
            "## Negative-query analysis",
            "",
            "All retrievers rank a sentence for every query. These scores are diagnostics, "
            "not false-positive decisions; no rejection threshold was fitted.",
            "",
        ]
    )
    for result in results["queries"]:
        if result["category"] != "negative":
            continue
        lines.append(f"### {result['id']}: {result['query']}")
        lines.append("")
        for name in ("bm25", "dense", "hybrid"):
            top = result["retrievers"][name]["top5"][0]
            suffix = ""
            if name == "bm25":
                suffix = (
                    f"; positive lexical scores: {result['bm25_positive_score_count']}; "
                    f"zero lexical evidence: {result['bm25_zero_lexical_evidence']}"
                )
            lines.append(
                f"- {name.upper()}: score `{top['score']:.8f}`, sentence "
                f"{top['sentence_index']} [{top['start']:.2f}-{top['end']:.2f}] — "
                f"{top['text']}{suffix}"
            )
        lines.append("")
    lines.extend(
        [
            "### Top-score distributions",
            "",
            "| Retriever | Positive min-max | Negative min-max | Strictly separable |",
            "|---|---:|---:|---|",
        ]
    )
    for name in ("bm25", "dense", "hybrid"):
        item = metrics["score_distributions"][name]
        positive = item["positive_top_score"]
        negative = item["negative_top_score"]
        lines.append(
            f"| {name.upper()} | {positive['min']:.8f}-{positive['max']:.8f} | "
            f"{negative['min']:.8f}-{negative['max']:.8f} | "
            f"{item['strictly_separable_by_top_score']} |"
        )
    lines.extend(["", "## Important failures", ""])
    labels = {
        "bm25_fails_dense_succeeds": "BM25 fails top 5 but dense succeeds",
        "dense_fails_bm25_succeeds": "Dense fails top 5 but BM25 succeeds",
        "hybrid_improves_both": "Hybrid ranks evidence above both components",
        "hybrid_makes_ranking_worse": "Hybrid ranks evidence below the better component",
    }
    for key, label in labels.items():
        query_ids = results["important_failures"][key]
        lines.append(f"- {label}: {', '.join(query_ids) if query_ids else 'none'}")
    complexity = results["complexity"]
    lines.extend(
        [
            "",
            "## Complexity and cost",
            "",
            "| Retriever | Build/load time | Incremental RSS | Stored/index size | Dependencies |",
            "|---|---:|---:|---:|---|",
            f"| BM25 | {complexity['bm25']['build_seconds']:.3f} s | "
            f"{complexity['bm25']['rss_delta_mib']:.2f} MiB | in-memory sparse index | "
            "bm25s + PyStemmer |",
            f"| Dense | model {complexity['dense']['model_load_seconds']:.3f} s; "
            f"embed {complexity['dense']['embedding_seconds']:.3f} s | model "
            f"{complexity['dense']['model_rss_delta_mib']:.2f} MiB; matrix "
            f"{complexity['dense']['matrix_rss_delta_mib']:.2f} MiB | cache "
            f"{complexity['dense']['cache_size_mib']:.2f} MiB; matrix "
            f"{complexity['dense']['matrix_size_mib']:.2f} MiB | FastEmbed + ONNX Runtime + model |",
            f"| Hybrid | shared components {complexity['hybrid']['shared_build_seconds']:.3f} s | "
            "component memory | no additional persistent index | both stacks + fixed RRF |",
            "",
            f"Dense model: `{MODEL_NAME}`. RRF constant: `{RRF_K}`. The experiment "
            "used installed packages only and did not edit `requirements.txt`.",
            "",
            "## Recommendation",
            "",
            f"**{results['recommendation']['decision']}**",
            "",
            results["recommendation"]["rationale"],
            "",
            "This is an experiment-level recommendation only. It is not permission to "
            "modify production retrieval.",
            "",
            "## Full top-5 inspection",
            "",
        ]
    )
    for result in results["queries"]:
        lines.extend(
            [
                f"### {result['id']} ({result['category']})",
                "",
                f"Query: {result['query']}",
                "",
            ]
        )
        truth = result["ground_truth"]
        if truth["answer"] == "NO_ANSWER":
            lines.append("Expected evidence: `NO_ANSWER`")
        else:
            lines.append("Expected evidence:")
            for region_index, region in enumerate(truth["regions"], start=1):
                lines.append(
                    f"- Region {region_index}; sentences {region['sentence_indices']}; "
                    f"[{region['start']:.2f}-{region['end']:.2f}]: {region['text']}"
                )
        lines.append("")
        actual_ranks = []
        for name in ("bm25", "dense", "hybrid"):
            rank = result["retrievers"][name]["first_relevant_rank"]
            actual_ranks.append(f"{name.upper()}={rank if rank is not None else 'not found'}")
        lines.append(f"Actual first relevant rank: {'; '.join(actual_ranks)}")
        lines.append("")
        for name in ("bm25", "dense", "hybrid"):
            lines.extend(
                [
                    f"#### {name.upper()}",
                    "",
                    "| Rank | Score | Sentence | Timestamp | Text |",
                    "|---:|---:|---:|---:|---|",
                ]
            )
            for row in result["retrievers"][name]["top5"]:
                safe_text = str(row["text"]).replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {row['rank']} | {row['score']:.8f} | {row['sentence_index']} | "
                    f"{row['start']:.2f}-{row['end']:.2f} | {safe_text} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_root / "evaluation_queries.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Frozen evaluation not found: {evaluation_path}")
    manifest_path = output_root / "manifest.json"
    results_path = output_root / "retrieval_results.json"
    report_path = output_root / "retrieval_report.md"
    model_cache = output_root / "fastembed-cache"

    initial_status = git_output("status", "--short", "--untracked-files=all")
    initial_src_diff = git_output("diff", "--", "src")
    initial_requirements_hash = sha256_file(Path("requirements.txt"))
    fixture_metadata, sentences = load_fixture()
    evaluation = validate_evaluation(evaluation_path, sentences)
    texts = [str(sentence["text"]) for sentence in sentences]

    baseline_rss = rss_mib()
    stemmer = Stemmer.Stemmer("english")
    started = perf_counter()
    corpus_tokens = bm25s.tokenize(
        texts,
        lower=True,
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    )
    bm25_retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bm25_retriever.index(corpus_tokens, show_progress=False)
    bm25_build_seconds = perf_counter() - started
    bm25_rss = rss_mib()

    cache_bytes_before = directory_size(model_cache)
    started = perf_counter()
    dense_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(model_cache),
        threads=max(1, min(8, os.cpu_count() or 1)),
        providers=["CPUExecutionProvider"],
    )
    model_load_seconds = perf_counter() - started
    model_rss = rss_mib()
    started = perf_counter()
    dense_matrix = np.asarray(
        list(dense_model.passage_embed(texts, batch_size=64)),
        dtype=np.float32,
    )
    dense_matrix = normalize_rows(dense_matrix)
    embedding_seconds = perf_counter() - started
    dense_rss = rss_mib()
    cache_bytes_after = directory_size(model_cache)

    query_results: list[dict[str, Any]] = []
    latency_samples: dict[str, list[float]] = {
        "bm25": [],
        "dense": [],
        "hybrid": [],
    }
    first_rankings: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for query_record in evaluation["queries"]:
        query_id = query_record["id"]
        question = query_record["query"]
        truth = query_record["ground_truth"]
        relevant_indices = {
            sentence_index
            for region in truth["regions"]
            for sentence_index in region["sentence_indices"]
        }

        started = perf_counter()
        bm25_ranking = run_bm25_query(
            bm25_retriever,
            stemmer,
            question,
            len(sentences),
        )
        latency_samples["bm25"].append((perf_counter() - started) * 1000)

        started = perf_counter()
        dense_ranking = run_dense_query(dense_model, dense_matrix, question)
        latency_samples["dense"].append((perf_counter() - started) * 1000)

        started = perf_counter()
        hybrid_bm25 = run_bm25_query(
            bm25_retriever,
            stemmer,
            question,
            len(sentences),
        )
        hybrid_dense = run_dense_query(dense_model, dense_matrix, question)
        hybrid_ranking = reciprocal_rank_fusion(hybrid_bm25, hybrid_dense)
        latency_samples["hybrid"].append((perf_counter() - started) * 1000)

        rankings = {
            "bm25": bm25_ranking,
            "dense": dense_ranking,
            "hybrid": hybrid_ranking,
        }
        first_rankings[query_id] = rankings
        retriever_results: dict[str, Any] = {}
        for name, ranking in rankings.items():
            entry: dict[str, Any] = {
                "first_relevant_rank": (
                    first_relevant_rank(ranking, relevant_indices)
                    if relevant_indices
                    else None
                ),
                "top5": result_rows(ranking, sentences),
            }
            if query_record["category"] == "multi_evidence":
                covered3, total = coverage_at(ranking, truth["regions"], 3)
                covered5, _ = coverage_at(ranking, truth["regions"], 5)
                entry["evidence_coverage"] = {
                    "at_3": {"covered": covered3, "total": total},
                    "at_5": {"covered": covered5, "total": total},
                }
            retriever_results[name] = entry
        positive_bm25_scores = sum(score > 0 for _, score in bm25_ranking)
        query_results.append(
            {
                "id": query_id,
                "category": query_record["category"],
                "query": question,
                "ground_truth": truth,
                "bm25_positive_score_count": positive_bm25_scores,
                "bm25_zero_lexical_evidence": positive_bm25_scores == 0,
                "retrievers": retriever_results,
            }
        )

    deterministic = True
    deterministic_details: dict[str, dict[str, bool]] = {}
    for query_record in evaluation["queries"]:
        query_id = query_record["id"]
        question = query_record["query"]
        repeated_bm25 = run_bm25_query(
            bm25_retriever,
            stemmer,
            question,
            len(sentences),
        )
        repeated_dense = run_dense_query(dense_model, dense_matrix, question)
        repeated_hybrid = reciprocal_rank_fusion(repeated_bm25, repeated_dense)
        repeated = {
            "bm25": repeated_bm25,
            "dense": repeated_dense,
            "hybrid": repeated_hybrid,
        }
        deterministic_details[query_id] = {}
        for name in ("bm25", "dense", "hybrid"):
            identical = first_rankings[query_id][name] == repeated[name]
            deterministic_details[query_id][name] = identical
            deterministic = deterministic and identical
    if not deterministic:
        raise AssertionError("Repeated retrieval was not deterministic")

    metrics = {
        "overall": {
            name: summarize_metrics(query_results, name)
            for name in ("bm25", "dense", "hybrid")
        },
        "by_category": {
            category: {
                name: summarize_metrics(query_results, name, category)
                for name in ("bm25", "dense", "hybrid")
            }
            for category in POSITIVE_CATEGORIES
        },
        "score_distributions": {
            name: score_distributions(query_results, name)
            for name in ("bm25", "dense", "hybrid")
        },
    }
    decision, rationale = choose_recommendation(metrics)
    latency = {
        name: {
            "mean": sum(samples) / len(samples),
            "min": min(samples),
            "max": max(samples),
            "sample_count": len(samples),
        }
        for name, samples in latency_samples.items()
    }
    complexity = {
        "bm25": {
            "build_seconds": bm25_build_seconds,
            "rss_delta_mib": bm25_rss - baseline_rss,
            "configuration": {
                "library": "bm25s",
                "method": "lucene",
                "k1": 1.5,
                "b": 0.75,
                "lowercase": True,
                "stopwords": "en",
                "stemmer": "PyStemmer english",
            },
        },
        "dense": {
            "model": MODEL_NAME,
            "model_load_seconds": model_load_seconds,
            "embedding_seconds": embedding_seconds,
            "model_rss_delta_mib": model_rss - bm25_rss,
            "matrix_rss_delta_mib": dense_rss - model_rss,
            "matrix_shape": list(dense_matrix.shape),
            "matrix_size_mib": dense_matrix.nbytes / (1024 * 1024),
            "cache_path": str(model_cache),
            "cache_existed_before": cache_bytes_before > 0,
            "cache_bytes_before": cache_bytes_before,
            "cache_bytes_after": cache_bytes_after,
            "cache_size_mib": cache_bytes_after / (1024 * 1024),
            "similarity": "cosine",
            "provider": "CPUExecutionProvider",
        },
        "hybrid": {
            "method": "Reciprocal Rank Fusion",
            "rrf_k": RRF_K,
            "weights": "none",
            "shared_build_seconds": bm25_build_seconds
            + model_load_seconds
            + embedding_seconds,
            "additional_persistent_index": False,
        },
    }
    results = {
        "experiment": "VideoMind frozen sentence retrieval comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": evaluation["input"],
        "query_counts": EXPECTED_CATEGORY_COUNTS,
        "future_chat_routing_cases": evaluation["future_chat_routing_cases"],
        "metrics": metrics,
        "latency_ms": latency,
        "complexity": complexity,
        "determinism": {
            "passed": deterministic,
            "comparison": "full ranked indices and raw scores repeated exactly",
            "per_query": deterministic_details,
        },
        "important_failures": important_failures(query_results),
        "recommendation": {"decision": decision, "rationale": rationale},
        "queries": query_results,
    }
    write_json(results_path, results)
    report_path.write_text(build_report(results) + "\n", encoding="utf-8")

    if git_output("diff", "--", "src") != initial_src_diff:
        raise AssertionError("src/ changed during retrieval experiment")
    if sha256_file(Path("requirements.txt")) != initial_requirements_hash:
        raise AssertionError("requirements.txt changed during retrieval experiment")
    manifest = {
        "experiment": "VideoMind frozen sentence retrieval comparison",
        "generated_at_utc": results["generated_at_utc"],
        "repository": str(Path.cwd()),
        "head": git_output("rev-parse", "HEAD"),
        "initial_git_status_short": initial_status.splitlines(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "bm25s",
                "PyStemmer",
                "fastembed",
                "onnxruntime",
                "numpy",
                "psutil",
                "tokenizers",
                "huggingface-hub",
            )
        },
        "dependency_requirements": {
            "bm25s": distribution_requirements("bm25s"),
            "fastembed": distribution_requirements("fastembed"),
        },
        "fixture": {
            "path": str(FIXTURE_PATH),
            "sha256": sha256_file(FIXTURE_PATH),
            "method": fixture_metadata["method"],
            "version": fixture_metadata["version"],
            "sentence_count": len(sentences),
            "atomic_units": True,
            "windows_or_chunks_created": False,
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
            "ground_truth_validated_before_retrieval": True,
            "query_counts": EXPECTED_CATEGORY_COUNTS,
        },
        "configurations": {
            "bm25": complexity["bm25"]["configuration"],
            "dense": {
                "library": "fastembed",
                "model": MODEL_NAME,
                "similarity": "cosine",
                "provider": "CPUExecutionProvider",
            },
            "hybrid": complexity["hybrid"],
        },
        "timing_and_memory": complexity,
        "determinism_passed": deterministic,
        "scope": {
            "faster_whisper_invoked": False,
            "external_inference_api_used": False,
            "model_artifacts_downloaded": cache_bytes_before == 0
            and cache_bytes_after > 0,
            "model_download_source": "Hugging Face Hub via FastEmbed",
            "llm_used": False,
            "production_src_unchanged_by_harness": True,
            "requirements_unchanged_by_harness": True,
            "production_retriever_implemented": False,
        },
        "outputs": {
            "evaluation_queries": str(evaluation_path),
            "retrieval_results": {
                "path": str(results_path),
                "sha256": sha256_file(results_path),
            },
            "retrieval_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
        },
        "recommendation": results["recommendation"],
    }
    write_json(manifest_path, manifest)
    print(f"Fixture validation: PASS ({len(sentences)} pySBD sentences)")
    print("Ground-truth validation: PASS")
    print(f"Deterministic repeated retrieval: {'PASS' if deterministic else 'FAIL'}")
    for name in ("bm25", "dense", "hybrid"):
        item = metrics["overall"][name]
        print(
            f"{name.upper()}: Rank-1 {item['rank1_count']}/{item['query_count']}, "
            f"Recall@3 {item['recall_at_3_count']}/{item['query_count']}, "
            f"Recall@5 {item['recall_at_5_count']}/{item['query_count']}, "
            f"MRR {item['mrr']:.3f}, mean latency {latency[name]['mean']:.3f} ms"
        )
    print(f"Recommendation: {decision}")
    print(f"Reports: {output_root}")


if __name__ == "__main__":
    main()
