"""Build timestamp-aligned lecture frame/text pairs from Phase 1 outputs."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


FrameTextPair = dict[str, str | int | float]
TIMESTAMP_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def load_transcript(transcript_path: str | Path) -> dict[str, Any]:
    """Load a Whisper transcript JSON file and validate its segment list."""
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    try:
        transcript = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON: {path}") from exc

    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Transcript has no usable segments: {path}")
    return transcript


def parse_frame_timestamp(frame_path: str | Path) -> float:
    """Parse a timestamp from filenames like frame_00000_0.00s.jpg."""
    match = TIMESTAMP_PATTERN.search(Path(frame_path).name)
    if not match:
        raise ValueError(f"Could not parse timestamp from frame filename: {frame_path}")
    return float(match.group(1))


def list_frame_files(frames_dir: str | Path) -> list[Path]:
    """Return frame files sorted by timestamp."""
    path = Path(frames_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {path}")

    frames = [
        frame
        for frame in path.iterdir()
        if frame.is_file() and frame.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not frames:
        raise ValueError(f"No frame images found in: {path}")
    return sorted(frames, key=parse_frame_timestamp)


def find_segment_index(timestamp: float, segments: list[dict[str, Any]]) -> int:
    """Find the transcript segment containing a timestamp, or the nearest segment."""
    nearest_index = 0
    nearest_distance = float("inf")

    for index, segment in enumerate(segments):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if start <= timestamp <= end:
            return index

        distance = min(abs(timestamp - start), abs(timestamp - end))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    return nearest_index


def merge_context_text(
    segments: list[dict[str, Any]],
    segment_index: int,
    context_window: int = 3,
) -> str:
    """Merge the matched segment with surrounding transcript context."""
    if context_window < 0:
        raise ValueError("context_window must be zero or greater")

    start_index = max(0, segment_index - context_window)
    end_index = min(len(segments), segment_index + context_window + 1)
    texts = [str(segment.get("text", "")).strip() for segment in segments[start_index:end_index]]
    return " ".join(text for text in texts if text)


def build_pairs(
    video_name: str,
    frames_dir: str | Path,
    transcript_path: str | Path,
    context_window: int = 3,
    min_words: int = 5,
) -> list[FrameTextPair]:
    """Align frames to transcript segments and produce contrastive pairs."""
    transcript = load_transcript(transcript_path)
    segments: list[dict[str, Any]] = transcript["segments"]
    pairs: list[FrameTextPair] = []

    for frame_path in list_frame_files(frames_dir):
        timestamp = parse_frame_timestamp(frame_path)
        segment_index = find_segment_index(timestamp, segments)
        segment = segments[segment_index]
        text = merge_context_text(segments, segment_index, context_window=context_window)

        if len(text.split()) < min_words:
            continue

        pairs.append(
            {
                "frame_path": str(frame_path),
                "timestamp": round(timestamp, 2),
                "text": text,
                "segment_id": int(segment.get("id", segment_index)),
                "video": video_name,
            }
        )

    if not pairs:
        raise ValueError(
            f"No frame/text pairs created for {video_name}. "
            f"Check transcript timestamps, frame names, and min_words={min_words}."
        )
    return pairs


def split_pairs(
    pairs: list[FrameTextPair],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[FrameTextPair]]:
    """Create deterministic train/validation/test splits."""
    if not pairs:
        raise ValueError("Cannot split an empty pair list")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train_ratio > 0, val_ratio >= 0, and sum < 1")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if total >= 3:
        train_end = max(1, train_end)
        val_end = min(total - 1, max(train_end + 1, val_end))

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def save_pairs(
    pairs: list[FrameTextPair],
    output_dir: str | Path,
    video_name: str,
    seed: int = 42,
) -> dict[str, Path]:
    """Save the full pair file and 80/10/10 split files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "all": output_path / f"{video_name}_pairs.json",
        "train": output_path / f"{video_name}_train.json",
        "val": output_path / f"{video_name}_val.json",
        "test": output_path / f"{video_name}_test.json",
    }
    files["all"].write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name, split_items in split_pairs(pairs, seed=seed).items():
        files[split_name].write_text(
            json.dumps(split_items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return files


def build_video_pairs(
    video_path: str | Path,
    frames_root: str | Path = "data/frames",
    transcripts_dir: str | Path = "data/transcripts",
    output_dir: str | Path = "data/pairs",
    context_window: int = 3,
    min_words: int = 5,
    seed: int = 42,
) -> dict[str, Path]:
    """Build and save frame/text pairs for one video."""
    video = Path(video_path)
    video_name = video.stem
    pairs = build_pairs(
        video_name=video_name,
        frames_dir=Path(frames_root) / video_name,
        transcript_path=Path(transcripts_dir) / f"{video_name}.json",
        context_window=context_window,
        min_words=min_words,
    )
    return save_pairs(pairs, output_dir=output_dir, video_name=video_name, seed=seed)


def build_all_videos(
    videos_dir: str | Path = "data/videos",
    frames_root: str | Path = "data/frames",
    transcripts_dir: str | Path = "data/transcripts",
    output_dir: str | Path = "data/pairs",
    context_window: int = 3,
    min_words: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, Path]]:
    """Build and save pairs for every video in a directory."""
    video_root = Path(videos_dir)
    if not video_root.exists() or not video_root.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {video_root}")

    video_files = sorted(
        path
        for path in video_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )
    if not video_files:
        raise ValueError(f"No video files found in: {video_root}")

    return {
        video.stem: build_video_pairs(
            video_path=video,
            frames_root=frames_root,
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            context_window=context_window,
            min_words=min_words,
            seed=seed,
        )
        for video in video_files
    }


def main() -> None:
    """Run the dataset builder from the command line."""
    parser = argparse.ArgumentParser(description="Build VideoMind frame/text pair datasets.")
    parser.add_argument("--video", default=None, help="Single video path to process.")
    parser.add_argument("--videos-dir", default="data/videos", help="Directory of videos.")
    parser.add_argument("--frames-root", default="data/frames", help="Root frame directory.")
    parser.add_argument("--transcripts-dir", default="data/transcripts", help="Transcript directory.")
    parser.add_argument("--output", default="data/pairs", help="Output pair directory.")
    parser.add_argument("--context-window", type=int, default=3, help="Segments to include on each side.")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum merged text word count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    args = parser.parse_args()

    if args.video:
        saved = build_video_pairs(
            video_path=args.video,
            frames_root=args.frames_root,
            transcripts_dir=args.transcripts_dir,
            output_dir=args.output,
            context_window=args.context_window,
            min_words=args.min_words,
            seed=args.seed,
        )
        print(json.dumps({key: str(value) for key, value in saved.items()}, indent=2))
        return

    saved_by_video = build_all_videos(
        videos_dir=args.videos_dir,
        frames_root=args.frames_root,
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output,
        context_window=args.context_window,
        min_words=args.min_words,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                video: {key: str(value) for key, value in saved.items()}
                for video, saved in saved_by_video.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
