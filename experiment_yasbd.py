"""Freeze and compare yasbd and pySBD sentences from the test2 transcript."""

from __future__ import annotations

import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pysbd
from yasbd import BoundaryDetector


FIXTURE_PATH = Path("data/test2.small.transcript.json")
YASBD_OUTPUT_PATH = Path("data/test2.yasbd.sentences.json")
PYSBD_OUTPUT_PATH = Path("data/test2.pysbd.sentences.json")
SEPARATOR = " "


def load_segments(path: Path) -> list[dict[str, str | float]]:
    """Load the frozen fixture without normalizing its text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Fixture must contain a non-empty segment list")

    segments: list[dict[str, str | float]] = []
    previous_start = previous_end = None
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"Segment {index} is not an object")
        start = raw.get("start")
        end = raw.get("end")
        text = raw.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError(f"Segment {index} has an invalid shape")
        start = float(start)
        end = float(end)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or (previous_start is not None and start < previous_start)
            or (previous_end is not None and end < previous_end)
        ):
            raise ValueError(f"Segment {index} has invalid timestamps")
        segments.append({"start": start, "end": end, "text": text})
        previous_start, previous_end = start, end
    return segments


def build_canonical_transcript(
    segments: list[dict[str, str | float]],
) -> tuple[str, list[dict[str, Any]]]:
    """Join segment text once and retain its exact canonical offsets."""
    parts: list[str] = []
    mappings: list[dict[str, Any]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        if index:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
        text = str(segment["text"])
        char_start = cursor
        parts.append(text)
        cursor += len(text)
        mappings.append(
            {
                "index": index,
                "char_start": char_start,
                "char_end": cursor,
                "start": segment["start"],
                "end": segment["end"],
            }
        )
    return "".join(parts), mappings


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Move offsets over whitespace while leaving transcript text unchanged."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def map_sentence_spans(
    transcript: str,
    raw_spans: list[tuple[int, int]],
    mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map detector spans to every overlapping source segment."""
    sentences: list[dict[str, Any]] = []
    for raw_start, raw_end in raw_spans:
        start, end = trim_span(transcript, raw_start, raw_end)
        if start < end:
            contributors = [
                mapping
                for mapping in mappings
                if mapping["char_start"] < end and mapping["char_end"] > start
            ]
            if not contributors:
                raise AssertionError(f"Sentence span {start}:{end} has no source segment")
            sentences.append(
                {
                    "start": contributors[0]["start"],
                    "end": contributors[-1]["end"],
                    "text": transcript[start:end],
                    "char_start": start,
                    "char_end": end,
                    "source_indices": [item["index"] for item in contributors],
                }
            )
    return sentences


def yasbd_spans(transcript: str, detector: BoundaryDetector) -> list[tuple[int, int]]:
    boundaries = list(detector.detect(transcript))
    return list(zip([0, *boundaries[:-1]], boundaries, strict=True))


def pysbd_spans(
    transcript: str,
    detector: pysbd.Segmenter,
) -> list[tuple[int, int]]:
    """Use pySBD's non-cleaning native spans and verify every direct slice."""
    detected = detector.segment(transcript)
    if any(item.sent != transcript[item.start : item.end] for item in detected):
        raise AssertionError("pySBD returned a span that rewrites canonical text")
    return [(item.start, item.end) for item in detected]


def verify_integrity(
    segments: list[dict[str, str | float]],
    transcript: str,
    mappings: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    raw_spans: list[tuple[int, int]],
    repeated_raw_spans: list[tuple[int, int]],
) -> dict[str, bool]:
    """Check text, offsets, ordering, overlap mapping, and timestamp provenance."""
    source_text_preserved = all(
        transcript[item["char_start"] : item["char_end"]] == segment["text"]
        for segment, item in zip(segments, mappings, strict=True)
    )
    separator_preserved = transcript == SEPARATOR.join(
        str(segment["text"]) for segment in segments
    )
    raw_spans_cover_transcript = (
        bool(raw_spans)
        and raw_spans[0][0] == 0
        and raw_spans[-1][1] == len(transcript)
        and all(left[1] == right[0] for left, right in zip(raw_spans, raw_spans[1:]))
        and "".join(transcript[start:end] for start, end in raw_spans) == transcript
    )
    spans_ordered_non_overlapping = all(
        left["char_end"] <= right["char_start"]
        for left, right in zip(sentences, sentences[1:])
    )
    direct_sentence_slices = all(
        sentence["text"]
        == transcript[sentence["char_start"] : sentence["char_end"]]
        for sentence in sentences
    )
    every_sentence_mapped = all(sentence["source_indices"] for sentence in sentences)
    timestamps_from_sources = all(
        sentence["start"] == segments[sentence["source_indices"][0]]["start"]
        and sentence["end"] == segments[sentence["source_indices"][-1]]["end"]
        for sentence in sentences
    )
    valid_native_spans = (
        bool(raw_spans)
        and all(
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(transcript)
            for start, end in raw_spans
        )
    )
    return {
        "source_text_preserved": source_text_preserved,
        "separator_and_order_preserved": separator_preserved,
        "raw_spans_cover_transcript": raw_spans_cover_transcript,
        "spans_ordered_non_overlapping": spans_ordered_non_overlapping,
        "sentence_text_is_direct_slice": direct_sentence_slices,
        "every_sentence_maps_to_source": every_sentence_mapped,
        "timestamps_only_from_sources": timestamps_from_sources,
        "valid_native_spans": valid_native_spans,
        "repeat_run_identical": raw_spans == repeated_raw_spans,
    }


def artifact_sentence(sentence: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": sentence["start"],
        "end": sentence["end"],
        "text": sentence["text"],
        "source_segments": sentence["source_indices"],
    }


def write_artifact(
    path: Path,
    method: str,
    version: str,
    sentences: list[dict[str, Any]],
) -> None:
    artifact = {
        "method": method,
        "version": version,
        "source": FIXTURE_PATH.as_posix(),
        "sentence_count": len(sentences),
        "sentences": [artifact_sentence(sentence) for sentence in sentences],
    }
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def summarize_mapping(sentences: list[dict[str, Any]]) -> tuple[Counter[int], int]:
    span_counts = Counter(len(sentence["source_indices"]) for sentence in sentences)
    source_use = Counter(
        source_index
        for sentence in sentences
        for source_index in sentence["source_indices"]
    )
    return span_counts, sum(count > 1 for count in source_use.values())


def find_sentence_at_boundary(
    sentences: list[dict[str, Any]],
    offset: int,
) -> list[dict[str, Any]]:
    containing = [
        sentence
        for sentence in sentences
        if sentence["char_start"] < offset < sentence["char_end"]
    ]
    if containing:
        return containing
    before = [sentence for sentence in sentences if sentence["char_end"] <= offset]
    after = [sentence for sentence in sentences if sentence["char_start"] >= offset]
    return ([before[-1]] if before else []) + ([after[0]] if after else [])


def print_boundary_disagreement(
    transcript: str,
    offset: int,
    yasbd_sentences: list[dict[str, Any]],
    pysbd_sentences: list[dict[str, Any]],
) -> None:
    context_start = max(0, offset - 100)
    context_end = min(len(transcript), offset + 100)
    context = transcript[context_start:offset] + "<BOUNDARY>" + transcript[offset:context_end]
    print(f"\noffset {offset}")
    print(f"context: {context}")
    for method, sentences in (("yasbd", yasbd_sentences), ("pysbd", pysbd_sentences)):
        matching = find_sentence_at_boundary(sentences, offset)
        print(f"{method}:")
        for sentence in matching:
            print(
                f"  [{sentence['start']:.2f}-{sentence['end']:.2f}] "
                f"{sentence['text']}"
            )


def main() -> None:
    segments = load_segments(FIXTURE_PATH)
    transcript, mappings = build_canonical_transcript(segments)
    yasbd_detector = BoundaryDetector("en")
    pysbd_detector = pysbd.Segmenter(language="en", clean=False, char_span=True)

    started = perf_counter()
    yasbd_raw_spans = yasbd_spans(transcript, yasbd_detector)
    yasbd_seconds = perf_counter() - started
    started = perf_counter()
    pysbd_raw_spans = pysbd_spans(transcript, pysbd_detector)
    pysbd_seconds = perf_counter() - started

    yasbd_repeated_spans = yasbd_spans(transcript, yasbd_detector)
    pysbd_repeated_spans = pysbd_spans(transcript, pysbd_detector)
    yasbd_sentences = map_sentence_spans(transcript, yasbd_raw_spans, mappings)
    pysbd_sentences = map_sentence_spans(transcript, pysbd_raw_spans, mappings)

    yasbd_checks = verify_integrity(
        segments,
        transcript,
        mappings,
        yasbd_sentences,
        yasbd_raw_spans,
        yasbd_repeated_spans,
    )
    pysbd_checks = verify_integrity(
        segments,
        transcript,
        mappings,
        pysbd_sentences,
        pysbd_raw_spans,
        pysbd_repeated_spans,
    )
    if len(yasbd_sentences) != 444:
        raise AssertionError(f"Expected 444 yasbd sentences, got {len(yasbd_sentences)}")
    if not all(yasbd_checks.values()) or not all(pysbd_checks.values()):
        raise AssertionError("One or more integrity checks failed")

    yasbd_version = importlib.metadata.version("yasbd-lib")
    pysbd_version = importlib.metadata.version("pysbd")
    write_artifact(YASBD_OUTPUT_PATH, "yasbd", yasbd_version, yasbd_sentences)
    write_artifact(PYSBD_OUTPUT_PATH, "pysbd", pysbd_version, pysbd_sentences)

    yasbd_boundaries = {sentence["char_end"] for sentence in yasbd_sentences[:-1]}
    pysbd_boundaries = {sentence["char_end"] for sentence in pysbd_sentences[:-1]}
    identical = yasbd_boundaries & pysbd_boundaries
    yasbd_only = yasbd_boundaries - pysbd_boundaries
    pysbd_only = pysbd_boundaries - yasbd_boundaries
    disagreements = yasbd_only | pysbd_only
    yasbd_counts, yasbd_multi_source = summarize_mapping(yasbd_sentences)
    pysbd_counts, pysbd_multi_source = summarize_mapping(pysbd_sentences)

    print("SENTENCE RECONSTRUCTION COMPARISON")
    print(f"source segments: {len(segments)}")
    print(f"canonical characters: {len(transcript)}")
    print(f"yasbd version: {yasbd_version}")
    print(f"pysbd version: {pysbd_version}")
    print(f"yasbd sentences: {len(yasbd_sentences)}")
    print(f"pysbd sentences: {len(pysbd_sentences)}")
    print(f"identical internal boundaries: {len(identical)}")
    print(f"differing internal boundaries: {len(disagreements)}")
    print(f"yasbd-only boundaries: {len(yasbd_only)} {sorted(yasbd_only)}")
    print(f"pysbd-only boundaries: {len(pysbd_only)} {sorted(pysbd_only)}")
    for method, counts, multi_source, seconds, checks in (
        ("yasbd", yasbd_counts, yasbd_multi_source, yasbd_seconds, yasbd_checks),
        ("pysbd", pysbd_counts, pysbd_multi_source, pysbd_seconds, pysbd_checks),
    ):
        three_plus = sum(count for size, count in counts.items() if size >= 3)
        print(f"{method} mapping: 1={counts[1]} 2={counts[2]} 3+={three_plus}")
        print(f"{method} source segments used by multiple sentences: {multi_source}")
        print(f"{method} detection only: {seconds:.6f} seconds")
        print(f"{method} integrity: {'PASS' if all(checks.values()) else 'FAIL'}")
        for name, passed in checks.items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print("\nEVERY BOUNDARY DISAGREEMENT")
    for offset in sorted(disagreements):
        print_boundary_disagreement(
            transcript,
            offset,
            yasbd_sentences,
            pysbd_sentences,
        )

    print("\nKNOWN 'NO. IF ANYTHING' CASE")
    for method, sentences in (("yasbd", yasbd_sentences), ("pysbd", pysbd_sentences)):
        print(f"{method}:")
        for sentence in sentences:
            if "downplaying it" in sentence["text"] or sentence["text"] == "No.":
                print(
                    f"  [{sentence['start']:.2f}-{sentence['end']:.2f}] "
                    f"{sentence['text']}"
                )


if __name__ == "__main__":
    main()
