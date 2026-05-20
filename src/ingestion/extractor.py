"""In-memory video frame extraction using decord."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def compute_frame_hash(frame: np.ndarray) -> np.ndarray:
    """Compute a small grayscale perceptual hash for an RGB frame."""
    resized = Image.fromarray(frame).resize((16, 16)).convert("L")
    return np.array(resized).flatten().astype(np.float32)


def is_scene_change(
    prev_hash: np.ndarray | None,
    curr_hash: np.ndarray,
    threshold: float = 15.0,
) -> bool:
    """Return whether two frame hashes differ enough to count as a scene change."""
    if prev_hash is None:
        return True
    diff = np.mean(np.abs(curr_hash - prev_hash))
    return diff >= threshold


def is_valid_frame(frame: np.ndarray, min_brightness: float = 10.0) -> bool:
    """Return whether an RGB frame is bright enough to keep."""
    return float(np.mean(frame)) >= min_brightness


def extract_frames_to_memory(
    video_path_or_url: str,
    scene_threshold: float = 15.0,
    min_interval: float = 2.0,
    min_brightness: float = 10.0,
) -> list[dict[str, Any]]:
    """Extract unique video frames into memory without writing files to disk."""
    start_time = time.time()
    vr = VideoReader(video_path_or_url, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if fps <= 0:
        raise ValueError(f"Could not determine video FPS: {video_path_or_url}")
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video_path_or_url}")

    sample_step = max(1, int(fps * min_interval))
    sample_indices = list(range(0, total_frames, sample_step))
    frames_batch = vr.get_batch(sample_indices).asnumpy()

    extracted: list[dict[str, Any]] = []
    prev_hash: np.ndarray | None = None
    saved_count = 0

    for index, frame in enumerate(frames_batch):
        if not is_valid_frame(frame, min_brightness):
            continue

        curr_hash = compute_frame_hash(frame)
        if not is_scene_change(prev_hash, curr_hash, scene_threshold):
            continue

        pil_image = Image.fromarray(frame)
        timestamp = sample_indices[index] / fps
        extracted.append(
            {
                "image": pil_image,
                "timestamp": round(timestamp, 2),
                "frame_index": saved_count,
            }
        )
        prev_hash = curr_hash
        saved_count += 1

    elapsed = time.time() - start_time
    print(f"{len(extracted)} unique frames extracted in {elapsed:.1f}s")
    return extracted
