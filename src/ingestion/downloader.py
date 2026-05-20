"""Command-line entry point for video ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ingestion.archive_utils import resolve_direct_url
from src.ingestion.stream_processor import discover_and_download, process_video_from_url

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for the downloader."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stream and process lecture videos without downloading to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ingestion.downloader --discover "Artificial Intelligence"
  python -m src.ingestion.downloader --discover "Machine Learning" --limit 10
  python -m src.ingestion.downloader --url "https://archive.org/details/MIT6.034F10"
  python -m src.ingestion.downloader --batch urls.txt
  python -m src.ingestion.downloader --batch urls.txt --force

Note: All videos are streamed directly without saving to disk.
Only frames, transcripts, and training pairs are saved.
        """,
    )

    parser.add_argument("--discover", type=str, help="Discover and process videos by topic (interactive mode)")
    parser.add_argument("--url", type=str, help="Single video URL to process (archive.org, vimeo, youtube, or direct MP4)")
    parser.add_argument("--batch", type=str, help="Text file with one URL per line for batch processing")
    parser.add_argument("--output", type=str, default="data/videos", help="Output directory for frames/transcripts (default: data/videos prefix, actual: data/frames, data/transcripts)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum videos to discover/process (default: 5)")
    parser.add_argument("--force", action="store_true", help="Force re-process even if data exists")

    args = parser.parse_args()

    mode_count = sum([bool(args.discover), bool(args.url), bool(args.batch)])
    if mode_count == 0:
        parser.error("Must specify --discover, --url, or --batch")
    if mode_count > 1:
        parser.error("Cannot specify multiple modes (--discover, --url, --batch)")

    try:
        if args.discover:
            discover_and_download(args.discover, limit=args.limit, output_dir=args.output)
        elif args.url:
            print("Processing video via streaming")
            url = args.url
            video_name = Path(args.url.split("?")[0]).stem or "video"
            resolved_url = resolve_direct_url(url)
            if resolved_url != url:
                url = resolved_url
                logger.debug("Resolved direct URL: %s", url)
            if process_video_from_url(url, video_name):
                print(f"\n✓ Success")
            else:
                print("\n✗ Processing failed")
                sys.exit(1)
        else:
            batch_file = Path(args.batch)
            if not batch_file.exists():
                print(f"Error: Batch file not found: {args.batch}")
                sys.exit(1)

            urls = [
                line.strip()
                for line in batch_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

            if not urls:
                print("Error: No valid URLs in batch file")
                sys.exit(1)

            print(f"\n{'='*70}")
            print(f"Batch processing {len(urls)} video(s) via streaming")
            print(f"{'='*70}\n")

            success_count = 0
            failed_count = 0

            for i, url in enumerate(urls, 1):
                video_name = Path(url.split("?")[0]).stem or f"video_{i}"
                print(f"\n[{i}/{len(urls)}] {video_name}")
                print("-" * 70)

                resolved_url = resolve_direct_url(url)
                if resolved_url != url:
                    url = resolved_url
                    logger.debug("Resolved direct URL: %s", url)

                if process_video_from_url(url, video_name):
                    success_count += 1
                else:
                    failed_count += 1

            print(f"\n{'='*70}")
            print("Batch Processing Summary")
            print(f"{'='*70}")
            print(f"Successfully processed: {success_count}")
            print(f"Failed: {failed_count}")
            print(f"{'='*70}\n")

            if failed_count > 0:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
