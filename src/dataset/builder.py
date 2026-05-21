from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from pathlib import Path
from typing import Any

FrameTextPair = dict[str, str | float]
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
FILLER_WORDS = {
    "ah",
    "er",
    "hmm",
    "like",
    "mm",
    "okay",
    "right",
    "uh",
    "uhh",
    "um",
    "umm",
    "yeah",
}


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
        raise ValueError(
            f"Whisper returned empty transcript for {path} — "
            "audio may be unavailable or in unsupported format"
        )
    return transcript


def clean_text(text: str) -> str:
    """Strip whitespace, collapse spacing, and remove duplicate consecutive words."""
    words = text.strip().split()
    deduped: list[str] = []
    previous_word = ""
    for word in words:
        normalized = word.strip().lower().strip(".,!?;:\"'()[]{}")
        if normalized and normalized == previous_word:
            continue
        deduped.append(word)
        previous_word = normalized
    return re.sub(r"\s+", " ", " ".join(deduped)).strip()


def meaningful_word_count(text: str) -> int:
    """Count non-filler words in cleaned transcript text."""
    words = WORD_PATTERN.findall(text.lower())
    return sum(1 for word in words if word not in FILLER_WORDS)


def build_transcript_chunks(
    segments: list[dict[str, Any]],
    chunk_size: int,
    min_meaningful_words: int,
) -> list[dict[str, str | float]]:
    """Combine transcript segments into coherent non-overlapping text chunks."""
    chunks: list[dict[str, str | float]] = []
    current_texts: list[str] = []
    current_start: float | None = None
    current_end = 0.0
    current_count = 0

    def flush_current() -> None:
        if current_start is None:
            return
        text = clean_text(" ".join(current_texts))
        if meaningful_word_count(text) >= min_meaningful_words:
            chunks.append(
                {
                    "text": text,
                    "start": round(current_start, 2),
                    "end": round(current_end, 2),
                }
            )

    for segment in segments:
        segment_text = clean_text(str(segment.get("text", "")))
        if not segment_text or meaningful_word_count(segment_text) == 0:
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        overlaps_current = current_start is not None and start <= current_end
        if current_count >= chunk_size and not overlaps_current:
            flush_current()
            current_texts = []
            current_start = None
            current_end = 0.0
            current_count = 0

        if current_start is None:
            current_start = start
            current_end = end
        else:
            current_end = max(current_end, end)

        current_texts.append(segment_text)
        current_count += 1

    flush_current()
    return chunks


def build_pairs(
    video_name: str,
    transcript_path: str | Path,
    context_window: int = 3,
    min_words: int = 5,
) -> list[FrameTextPair]:
    """Build cleaned transcript chunks for retrieval indexing."""
    if context_window < 0:
        raise ValueError("context_window must be zero or greater")

    transcript = load_transcript(transcript_path)
    segments: list[dict[str, Any]] = transcript["segments"]
    chunks = build_transcript_chunks(
        segments,
        chunk_size=max(1, context_window * 2 + 1),
        min_meaningful_words=max(10, min_words),
    )
    pairs: list[FrameTextPair] = [
        {
            "id": str(uuid.uuid4()),
            "text": str(chunk["text"]),
            "video": video_name,
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
        }
        for chunk in chunks
    ]

    if not pairs:
        raise ValueError(
            f"No transcript chunks created for {video_name}. "
            f"Check transcript timestamps and min_words={min_words}."
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
        raise ValueError(
            "Ratios must satisfy train_ratio > 0, val_ratio >= 0, and sum < 1"
        )

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
    files["all"].write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for split_name, split_items in split_pairs(pairs, seed=seed).items():
        files[split_name].write_text(
            json.dumps(split_items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return files


def build_video_pairs(
    video_path: str | Path,
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
        transcript_path=Path(transcripts_dir) / f"{video_name}.json",
        context_window=context_window,
        min_words=min_words,
    )
    return save_pairs(pairs, output_dir=output_dir, video_name=video_name, seed=seed)


def build_all_videos(
    videos_dir: str | Path = "data/videos",
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
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )
    if not video_files:
        raise ValueError(f"No video files found in: {video_root}")

    return {
        video.stem: build_video_pairs(
            video_path=video,
            transcripts_dir=transcripts_dir,
            output_dir=output_dir,
            context_window=context_window,
            min_words=min_words,
            seed=seed,
        )
        for video in video_files
    }


def print_pair_statistics(pairs_file: str | Path) -> None:
    """Print comprehensive statistics about the pair dataset."""
    pairs_file = Path(pairs_file)
    if not pairs_file.exists():
        print(f"Warning: Pairs file not found: {pairs_file}")
        return

    with pairs_file.open(encoding="utf-8") as f:
        pairs = json.load(f)

    if not pairs:
        print("No pairs in dataset.")
        return

    total_pairs = len(pairs)
    print(f"\n=== Pair Statistics ===")
    print(f"Total pairs created: {total_pairs}")

    timestamps = [pair.get("start", 0) for pair in pairs] + [
        pair.get("end", 0) for pair in pairs
    ]
    texts = [pair.get("text", "") for pair in pairs]

    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    print(f"Timestamp range: {min_timestamp}s - {max_timestamp}s")

    word_counts = [len(text.split()) for text in texts]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"Average text length: {avg_words:.1f} words")


def print_split_statistics(output_dir: str | Path, video_name: str) -> None:
    """Print train/val/test split counts."""
    output_path = Path(output_dir)
    splits = {
        "train": output_path / f"{video_name}_train.json",
        "val": output_path / f"{video_name}_val.json",
        "test": output_path / f"{video_name}_test.json",
    }

    split_counts = {}
    for split_name, split_file in splits.items():
        if split_file.exists():
            with split_file.open(encoding="utf-8") as f:
                split_data = json.load(f)
                split_counts[split_name] = len(split_data)

    if split_counts:
        print(f"\nTrain/Val/Test split:")
        print(f"  Train: {split_counts.get('train', 0)} pairs")
        print(f"  Val: {split_counts.get('val', 0)} pairs")
        print(f"  Test: {split_counts.get('test', 0)} pairs")


def main() -> None:
    """Run the dataset builder from the command line."""
    parser = argparse.ArgumentParser(
        description="Build VideoMind frame/text pair datasets."
    )
    parser.add_argument("--video", default=None, help="Single video path to process.")
    parser.add_argument(
        "--videos-dir", default="data/videos", help="Directory of videos."
    )
    parser.add_argument(
        "--transcripts-dir", default="data/transcripts", help="Transcript directory."
    )
    parser.add_argument("--output", default="data/pairs", help="Output pair directory.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Segments to include on each side.",
    )
    parser.add_argument(
        "--min-words", type=int, default=5, help="Minimum merged text word count."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic split seed."
    )
    args = parser.parse_args()

    if args.video:
        saved = build_video_pairs(
            video_path=args.video,
            transcripts_dir=args.transcripts_dir,
            output_dir=args.output,
            context_window=args.context_window,
            min_words=args.min_words,
            seed=args.seed,
        )
        print(json.dumps({key: str(value) for key, value in saved.items()}, indent=2))

        video_name = Path(args.video).stem
        pairs_file = Path(args.output) / f"{video_name}_pairs.json"
        print_pair_statistics(pairs_file)
        print_split_statistics(args.output, video_name)
        return

    saved_by_video = build_all_videos(
        videos_dir=args.videos_dir,
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

    print("\n" + "=" * 50)
    for video_name in saved_by_video.keys():
        pairs_file = Path(args.output) / f"{video_name}_pairs.json"
        print_pair_statistics(pairs_file)
        print_split_statistics(args.output, video_name)
