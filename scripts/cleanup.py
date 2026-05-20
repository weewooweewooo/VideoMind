from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

from src.utils.cleanup import clean_video, clean_all

logger = logging.getLogger(__name__)


def format_bytes(bytes_count: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} TB"


def print_cleanup_summary(
    deleted_items: list[str],
    bytes_freed: int,
    title: str = "Cleanup Summary",
) -> None:
    mb_freed = bytes_freed / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    if deleted_items:
        print(f"\nDeleted items: {len(deleted_items)}")
        for item in deleted_items:
            logger.debug("Deleted item: %s", item)
    else:
        print("\nNo items deleted")

    print(f"\nSpace freed: {format_bytes(bytes_freed)} ({mb_freed:.2f} MB)")
    print("=" * 60 + "\n")


def main() -> None:
    """Run the cleanup CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Clean up cached VideoMind data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup.py --video MIT6_034F10_lec01_300k --targets pairs
  python scripts/cleanup.py --all --targets pairs
  python scripts/cleanup.py --all --targets pairs redis
        """,
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Clean specific video by name (video stem)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean all videos",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["pairs"],
        help="What to clean: pairs, redis (default: pairs)",
    )

    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Must specify --video or --all")
    if args.video and args.all:
        parser.error("Cannot specify both --video and --all")

    valid_targets = {"pairs", "redis"}
    for target in args.targets:
        if target not in valid_targets:
            parser.error(
                f"Invalid target: {target}. Valid options: {', '.join(valid_targets)}"
            )

    try:
        if args.video:
            result = clean_video(args.video, args.targets)
            print_cleanup_summary(
                result["deleted"],
                result["bytes_freed"],
                title=f"Cleanup Summary for {args.video}",
            )
        else:
            result = clean_all(args.targets)
            print_cleanup_summary(
                result["deleted"],
                result["bytes_freed"],
                title="Cleanup Summary for All Videos",
            )

    except Exception as exc:
        print(f"Error during cleanup: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
