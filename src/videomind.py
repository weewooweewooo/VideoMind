"""Transcribe one local video into clean timestamped segments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from src.ingestion import ingest_video


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one local video into clean timestamped segments."
    )
    parser.add_argument("video", help="Path to one local media file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        segments = ingest_video(args.video)
        print(json.dumps(segments, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("VideoMind interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"VideoMind failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
