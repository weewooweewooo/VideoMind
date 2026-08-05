"""Single-video VideoMind command-line application."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from src.ingestion import ingest_video
from src.retrieval import build_retriever


_NO_EVIDENCE_MESSAGE = "No relevant evidence found in the video."


class _VideoMindSession:
    """Reuse one prepared transcript and BM25 index across questions."""

    def __init__(self, transcript: Mapping[str, Any]) -> None:
        self.transcript = transcript
        self.retriever = build_retriever(transcript)

    def query(self, question: str) -> dict[str, Any]:
        focused_result = self.retriever.search_focused(question)
        return {
            "query": question,
            "focused_evidence": (
                focused_result["text"] if focused_result is not None else None
            ),
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
            "Transcribe one local video and retrieve transcript evidence "
            "for one or more questions."
        )
    )
    parser.add_argument("video", help="Path to one local video.")
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
