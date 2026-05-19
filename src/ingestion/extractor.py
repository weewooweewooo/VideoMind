# src/ingestion/extractor.py
import cv2
import os
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str = "data/frames",
    fps: float = 1.0
) -> list[dict]:
    """
    Extract frames from a video at a given FPS rate.
    Default: 1 frame per second.
    Returns list of {frame_path, timestamp} dicts.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = int(video_fps / fps)

    print(f"Video: {video_name}")
    print(f"FPS: {video_fps:.2f} | Duration: {duration:.1f}s | Extracting every {frame_interval} frames")

    extracted = []
    frame_count = 0
    saved_count = 0

    with tqdm(total=int(duration), desc="Extracting frames", unit="s") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frame_filename = f"frame_{saved_count:05d}_{timestamp:.2f}s.jpg"
                frame_path = frame_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)
                extracted.append({
                    "frame_path": str(frame_path),
                    "timestamp": round(timestamp, 2),
                    "frame_index": saved_count
                })
                saved_count += 1
                pbar.update(1)

            frame_count += 1

    cap.release()
    print(f"\nExtracted {saved_count} frames → {frame_dir}")
    return extracted


if __name__ == "__main__":
    video_file = "data/videos/MIT6_034F10_lec01_300k.mp4"
    frames = extract_frames(video_file, fps=1.0)
    print(f"\nSample frame: {frames[0]}")
    print(f"Total frames extracted: {len(frames)}")