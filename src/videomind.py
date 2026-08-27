"""Transcribe one local video or chat about its retrieved evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from src.answering import VideoAnswerer
from src.ingestion import ingest_video
from src.retrieval import VideoRetriever


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one local video or chat about its evidence."
    )
    parser.add_argument("video", help="Path to one local media file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--chat",
        action="store_true",
        help="Start a local interactive chat grounded in the video.",
    )
    mode.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask one question using local video retrieval and inference.",
    )
    return parser


def _print_answer(result: dict[str, Any]) -> None:
    print(result["answer"])
    citations = result["citations"]
    if isinstance(citations, list) and citations:
        print("\nVideo evidence:")
        for citation in citations:
            print(f"[{citation['timestamp']}]")


def _answer_question(
    question: str,
    retriever: VideoRetriever,
    answerer: VideoAnswerer,
) -> None:
    evidence = retriever.retrieve(question)
    _print_answer(answerer.answer(question, evidence))


def _run_chat(retriever: VideoRetriever, answerer: VideoAnswerer) -> None:
    print('VideoMind ready. Type "exit" to quit.')
    while True:
        try:
            question = input("\nYou:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        print("\nVideoMind:")
        _answer_question(question, retriever, answerer)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        sentences = ingest_video(args.video)
        if args.chat or args.ask is not None:
            retriever = VideoRetriever(sentences)
            answerer = VideoAnswerer()
            answerer.check_availability()
            if args.ask is not None:
                _answer_question(args.ask, retriever, answerer)
            else:
                _run_chat(retriever, answerer)
            return 0
        print(json.dumps(sentences, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("VideoMind interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"VideoMind failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
