"""Local-video VideoMind command-line application."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from src.ingestion import compile_transcript_text, ingest_video
from src.retrieval import build_retriever


_NO_EVIDENCE_MESSAGE = "No relevant evidence found in the video."
_EVIDENCE_EXPANSION_WORD_CAP = 35
_TERMINAL_ENDING = re.compile(r"""[.?!](?:[\"'\u2019\u201d)\]}]+)?$""")


def _has_terminal_ending(text: str) -> bool:
    return bool(_TERMINAL_ENDING.search(text.rstrip()))


def _find_anchor_segment_span(
    anchor: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[int, int] | None:
    """Map one retrieved chunk exactly to its original contiguous segments."""
    matches: list[tuple[int, int]] = []
    anchor_start = float(anchor["start"])
    anchor_end = float(anchor["end"])
    anchor_text = str(anchor["text"])

    for start_index, segment in enumerate(segments):
        if float(segment["start"]) != anchor_start:
            continue
        for end_index in range(start_index, len(segments)):
            segment_end = float(segments[end_index]["end"])
            if segment_end > anchor_end:
                break
            if (
                segment_end == anchor_end
                and compile_transcript_text(segments[start_index : end_index + 1])
                == anchor_text
            ):
                matches.append((start_index, end_index))

    return matches[0] if len(matches) == 1 else None


def _segment_word_count(segments: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(str(segment["text"]).split()) for segment in segments)


def _expand_evidence(
    anchor: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """Expand an immutable retrieval anchor to bounded sentence boundaries."""
    anchor_text = str(anchor["text"])
    try:
        anchor_span = _find_anchor_segment_span(anchor, segments)
    except (KeyError, TypeError, ValueError):
        return anchor_text
    if anchor_span is None:
        return anchor_text

    anchor_start, anchor_end = anchor_span
    expanded_start = anchor_start
    expanded_end = anchor_end

    if (
        anchor_start > 0
        and not _has_terminal_ending(str(segments[anchor_start - 1]["text"]))
    ):
        cursor = anchor_start - 1
        while cursor >= 0 and not _has_terminal_ending(
            str(segments[cursor]["text"])
        ):
            cursor -= 1
        proposed_start = cursor + 1
        if (
            _segment_word_count(segments[proposed_start:anchor_start])
            <= _EVIDENCE_EXPANSION_WORD_CAP
        ):
            expanded_start = proposed_start

    if (
        anchor_end < len(segments) - 1
        and not _has_terminal_ending(str(segments[anchor_end]["text"]))
    ):
        cursor = anchor_end + 1
        while cursor < len(segments) - 1 and not _has_terminal_ending(
            str(segments[cursor]["text"])
        ):
            cursor += 1
        if (
            _segment_word_count(segments[anchor_end + 1 : cursor + 1])
            <= _EVIDENCE_EXPANSION_WORD_CAP
        ):
            expanded_end = cursor

    if expanded_start == anchor_start and expanded_end == anchor_end:
        return anchor_text
    return compile_transcript_text(segments[expanded_start : expanded_end + 1])


class _VideoMindSession:
    """Reuse one prepared transcript and BM25 index across questions."""

    def __init__(self, transcript: Mapping[str, Any]) -> None:
        self.transcript = transcript
        self.retriever = build_retriever(transcript)

    def query(self, question: str) -> dict[str, Any]:
        results = self.retriever.search(question, top_k=1)
        focused_evidence = None
        if results:
            focused_evidence = _expand_evidence(
                results[0],
                self.transcript["segments"],
            )
        return {
            "query": question,
            "focused_evidence": focused_evidence,
        }


def _print_result(
    result: Mapping[str, Any],
    *,
    json_output: bool,
    pretty: bool,
) -> None:
    if json_output:
        print(json.dumps(result, indent=2 if pretty else None, ensure_ascii=False))
        return
    print(result["focused_evidence"] or _NO_EVIDENCE_MESSAGE)


def _query_and_print(
    session: _VideoMindSession,
    question: str,
    *,
    json_output: bool,
    pretty: bool,
) -> None:
    try:
        _print_result(
            session.query(question),
            json_output=json_output,
            pretty=pretty,
        )
    except Exception as exc:
        print(f"VideoMind query failed: {exc}", file=sys.stderr)


def _run_interactive_session(
    session: _VideoMindSession,
    *,
    initial_question: str | None,
    json_output: bool,
    pretty: bool,
) -> int:
    """Read questions while reusing one transcript and one BM25 index."""
    try:
        if initial_question is not None:
            _query_and_print(
                session,
                initial_question,
                json_output=json_output,
                pretty=pretty,
            )

        print(
            "VideoMind ready. Enter a question, or use :help for commands.",
            file=sys.stderr,
        )
        while True:
            print("videomind> ", end="", file=sys.stderr, flush=True)
            question_input = sys.stdin.readline()
            if question_input == "":
                print("", file=sys.stderr)
                return 0

            question = question_input.strip()
            if not question:
                continue
            if question == ":help":
                print("Commands: :help, :quit, :exit", file=sys.stderr)
                continue
            if question in {":quit", ":exit"}:
                return 0
            _query_and_print(
                session,
                question,
                json_output=json_output,
                pretty=pretty,
            )
    except KeyboardInterrupt:
        print("\nVideoMind interactive session ended", file=sys.stderr)
        return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one local video and retrieve evidence for one "
            "or more questions."
        )
    )
    parser.add_argument("video", help="Path to one local media file.")
    parser.add_argument("question", nargs="?", help="Question about the video.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read questions from stdin while reusing one BM25 index.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured query result as JSON.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON output when used with --json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        if args.question is None and not args.interactive:
            raise ValueError("A question is required after the video path")

        transcript = ingest_video(args.video)
        session = _VideoMindSession(transcript)
        if args.interactive:
            return _run_interactive_session(
                session,
                initial_question=args.question,
                json_output=args.json,
                pretty=args.pretty,
            )

        _print_result(
            session.query(args.question),
            json_output=args.json,
            pretty=args.pretty,
        )
        return 0
    except KeyboardInterrupt:
        print("VideoMind interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"VideoMind failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
