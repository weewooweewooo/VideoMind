"""In-memory video frame extraction using decord."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from decord import VideoReader, cpu, gpu
from PIL import Image

logger = logging.getLogger(__name__)


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


def _is_url(source: str) -> bool:
    """Return True when the source is an HTTP(S) URL."""
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def extract_frames_and_transcript_concurrent(
    video_path: str,
    whisper_model_size: str = "medium",
) -> tuple[list, dict]:
    """
    Runs frame extraction and transcription concurrently.
    Returns (frames, transcription)
    """
    from src.ingestion.transcriber import transcribe_to_memory

    video_name = Path(video_path.split("?")[0]).stem or "video"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        frames_future = executor.submit(extract_frames_to_memory, video_path)
        transcript_future = executor.submit(
            transcribe_to_memory,
            video_path,
            video_name,
            model_size=whisper_model_size,
        )
        return frames_future.result(), transcript_future.result()


def _extract_frames_with_decord(
    video_path_or_url: str,
    scene_threshold: float,
    min_interval: float,
    min_brightness: float,
) -> list[dict[str, Any]]:
    """Extract unique video frames from a decord-readable source."""
    vr = VideoReader(video_path_or_url, ctx=gpu(0) if torch.cuda.is_available() else cpu(0))
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

    return extracted


def _download_url_to_temp_file(url: str) -> str:
    """Download a URL to a temporary MP4 file and return its path."""
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            return tmp.name


def extract_frames_to_memory(
    video_path_or_url: str,
    scene_threshold: float = float(os.environ.get("SCENE_THRESHOLD", "15.0")),
    min_interval: float = 2.0,
    min_brightness: float = float(os.environ.get("BRIGHTNESS_THRESHOLD", "10.0")),
) -> list[dict[str, Any]]:
    """Extract unique video frames from a local file or URL into memory."""
    start_time = time.time()

    try:
        extracted = _extract_frames_with_decord(
            video_path_or_url,
            scene_threshold,
            min_interval,
            min_brightness,
        )
    except Exception:
        if not _is_url(video_path_or_url):
            raise

        tmp_path = _download_url_to_temp_file(video_path_or_url)
        try:
            extracted = _extract_frames_with_decord(
                tmp_path,
                scene_threshold,
                min_interval,
                min_brightness,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elapsed = time.time() - start_time
    logger.info("%s unique frames extracted in %.1fs", len(extracted), elapsed)
    return extracted
