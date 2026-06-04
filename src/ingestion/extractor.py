"""Content-aware in-memory video frame extraction using decord and OpenCV."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from decord import VideoReader, cpu, gpu


def _env_float(name: str, default: str) -> float:
    """Read a float environment value with a clear error on invalid input."""
    raw_value = os.environ.get(name, default)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _env_int(name: str, default: str) -> int:
    """Read an integer environment value with a clear error on invalid input."""
    raw_value = os.environ.get(name, default)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _resolve_decord_context() -> Any:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = os.environ.get("DEVICE", default_device).strip().lower()

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return gpu(0)
        logging.warning("DEVICE=cuda requested but CUDA is unavailable; using CPU")
        return cpu(0)

    if requested_device != "cpu":
        logging.warning(
            "Unsupported DEVICE=%r for decord; using %s",
            requested_device,
            default_device,
        )

    return gpu(0) if default_device == "cuda" and requested_device != "cpu" else cpu(0)


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _download_url_to_temp_file(url: str) -> str:
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            return tmp.name


def extract_frames_and_transcript_concurrent(
    video_path: str,
    whisper_model_size: str = "medium",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run frame extraction and transcription concurrently.
    Returns (frames, transcription).
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


def _frame_record(frame: np.ndarray, timestamp: float) -> dict[str, Any]:
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    return {
        "frame": frame,
        "timestamp": float(timestamp),
        "width": int(width),
        "height": int(height),
    }


def _extract_one_frame_per_second(video_path: str) -> list[dict[str, Any]]:
    vr = VideoReader(video_path, ctx=_resolve_decord_context())
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)

    if fps <= 0:
        raise ValueError(f"Could not determine video FPS: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video_path}")

    duration_seconds = total_frames / fps
    sample_seconds = range(max(1, math.ceil(duration_seconds)))
    sample_indices: list[int] = []

    for second in sample_seconds:
        frame_index = min(int(round(second * fps)), total_frames - 1)
        if not sample_indices or frame_index != sample_indices[-1]:
            sample_indices.append(frame_index)

    frames_batch = vr.get_batch(sample_indices).asnumpy()
    return [
        _frame_record(frame, frame_index / fps)
        for frame, frame_index in zip(frames_batch, sample_indices)
    ]


def _remove_blank_frames(
    frames: list[dict[str, Any]],
    brightness_threshold: float,
) -> list[dict[str, Any]]:
    return [
        frame_data
        for frame_data in frames
        if float(np.mean(frame_data["frame"])) >= brightness_threshold
    ]


def _load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def _detect_frame_content(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier,
    text_edge_threshold: float,
) -> tuple[bool, bool, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    has_face = len(faces) > 0
    has_text = edge_density > text_edge_threshold
    return has_face, has_text, faces


def _remove_presenter_only_frames(
    frames: list[dict[str, Any]],
    face_detector: cv2.CascadeClassifier,
    text_edge_threshold: float,
) -> list[dict[str, Any]]:
    kept_frames: list[dict[str, Any]] = []

    for frame_data in frames:
        has_face, has_text, faces = _detect_frame_content(
            frame_data["frame"],
            face_detector,
            text_edge_threshold,
        )
        if has_face and not has_text:
            continue

        kept_frame = dict(frame_data)
        kept_frame["_has_face"] = has_face
        kept_frame["_has_text"] = has_text
        kept_frame["_faces"] = faces
        kept_frames.append(kept_frame)

    return kept_frames


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second

    return not (
        first_x + first_w <= second_x
        or second_x + second_w <= first_x
        or first_y + first_h <= second_y
        or second_y + second_h <= first_y
    )


def _quadrants_for_frame(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    mid_x = width // 2
    mid_y = height // 2
    return [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, width - mid_x, mid_y),
        (0, mid_y, mid_x, height - mid_y),
        (mid_x, mid_y, width - mid_x, height - mid_y),
    ]


def _edge_density(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean())


def _crop_slide_region(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cropped_frames: list[dict[str, Any]] = []

    for frame_data in frames:
        frame = frame_data["frame"]
        faces = frame_data.get("_faces", [])
        has_face = bool(frame_data.get("_has_face", False))
        has_text = bool(frame_data.get("_has_text", False))

        if has_face and has_text:
            candidates: list[tuple[float, tuple[int, int, int, int]]] = []
            for quadrant in _quadrants_for_frame(frame):
                overlaps_face = any(
                    _rectangles_overlap(quadrant, tuple(int(value) for value in face))
                    for face in faces
                )
                if overlaps_face:
                    continue

                x, y, width, height = quadrant
                crop = frame[y : y + height, x : x + width]
                candidates.append((_edge_density(crop), quadrant))

            if candidates:
                _, best_quadrant = max(candidates, key=lambda item: item[0])
                x, y, width, height = best_quadrant
                frame = frame[y : y + height, x : x + width]

        cropped_frames.append(_frame_record(frame, frame_data["timestamp"]))

    return cropped_frames


def _create_phash_hasher() -> Any:
    try:
        return cv2.img_hash.PHash_create()
    except AttributeError as exc:
        raise RuntimeError(
            "OpenCV img_hash is required for duplicate removal; "
            "install opencv-contrib-python-headless."
        ) from exc


def _group_consecutive_duplicates(
    frames: list[dict[str, Any]],
    duplicate_threshold: int,
) -> list[list[dict[str, Any]]]:
    if not frames:
        return []

    hasher = _create_phash_hasher()
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    previous_hash: np.ndarray | None = None

    for frame_data in frames:
        frame_hash = hasher.compute(frame_data["frame"])
        if previous_hash is None:
            current_group = [frame_data]
        elif hasher.compare(previous_hash, frame_hash) < duplicate_threshold:
            current_group.append(frame_data)
        else:
            groups.append(current_group)
            current_group = [frame_data]

        previous_hash = frame_hash

    if current_group:
        groups.append(current_group)

    return groups


def _sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _keep_sharpest_from_groups(
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    final_frames: list[dict[str, Any]] = []

    for group in groups:
        best_frame = max(
            group,
            key=lambda frame_data: _sharpness_score(frame_data["frame"]),
        )
        final_frames.append(best_frame)

    return final_frames


def _extract_frames_from_decord_source(video_path: str) -> list[dict[str, Any]]:
    brightness_threshold = _env_float("BRIGHTNESS_THRESHOLD", "10.0")
    text_edge_threshold = _env_float("TEXT_EDGE_THRESHOLD", "500.0")
    duplicate_threshold = _env_int("DUPLICATE_THRESHOLD", "8")

    extracted_frames = _extract_one_frame_per_second(video_path)
    logging.info("Step 1: %d frames extracted", len(extracted_frames))

    bright_frames = _remove_blank_frames(extracted_frames, brightness_threshold)
    logging.info("Step 2: %d frames after blank removal", len(bright_frames))

    face_detector = _load_face_detector()
    content_frames = _remove_presenter_only_frames(
        bright_frames,
        face_detector,
        text_edge_threshold,
    )
    logging.info("Step 3: %d frames after presenter removal", len(content_frames))

    slide_frames = _crop_slide_region(content_frames)
    logging.info("Step 4: %d frames after slide crop", len(slide_frames))

    duplicate_groups = _group_consecutive_duplicates(slide_frames, duplicate_threshold)
    final_frames = _keep_sharpest_from_groups(duplicate_groups)
    logging.info("Step 5-6: %d frames after dedup+sharpness", len(final_frames))

    return final_frames


def extract_frames_to_memory(video_path: str) -> list[dict]:
    """Extract content-aware video frames from a local file or URL into memory."""
    try:
        return _extract_frames_from_decord_source(video_path)
    except Exception:
        if not _is_url(video_path):
            raise

        logging.info("Retrying frame extraction through temporary URL download")
        tmp_path = _download_url_to_temp_file(video_path)
        try:
            return _extract_frames_from_decord_source(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
