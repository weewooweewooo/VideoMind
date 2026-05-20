import cv2
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm


def compute_frame_hash(frame: np.ndarray) -> np.ndarray:
    """Compute perceptual hash of a frame for deduplication."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16))
    return resized.flatten().astype(np.float32)


def is_scene_change(
    prev_hash: np.ndarray, curr_hash: np.ndarray, threshold: float = 15.0
) -> bool:
    """Detect scene change using mean absolute difference of frame hashes."""
    if prev_hash is None:
        return True
    diff = np.mean(np.abs(curr_hash - prev_hash))
    return diff > threshold


def is_valid_frame(frame: np.ndarray, min_brightness: float = 10.0) -> bool:
    """
    Filter out black frames, near-black frames.
    Returns False for frames with very low brightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > min_brightness


def extract_frames(
    video_path: str,
    output_dir: str = "data/frames",
    scene_threshold: float = 15.0,
    min_brightness: float = 10.0,
    min_interval: float = 2.0,
    video_name: str | None = None,
) -> list[dict]:
    """
    Extract frames using scene change detection.
    Only saves frames when visual content changes significantly.

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        scene_threshold: Sensitivity of scene change detection (lower = more frames)
        min_brightness: Minimum frame brightness (filters black frames)
        min_interval: Minimum seconds between saved frames
        video_name: Optional output subfolder name

    Returns:
        List of {frame_path, timestamp, frame_index} dicts
    """
    video_path = Path(video_path)
    output_video_name = video_name or video_path.stem
    frame_dir = Path(output_dir) / output_video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    min_frame_interval = int(video_fps * min_interval)

    print(f"Video: {output_video_name}")
    print(f"FPS: {video_fps:.2f} | Duration: {duration:.1f}s")
    print(f"Scene threshold: {scene_threshold} | Min interval: {min_interval}s")

    extracted = []
    frame_count = 0
    saved_count = 0
    prev_hash = None
    last_saved_frame = -min_frame_interval

    with tqdm(total=total_frames, desc="Scanning frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count - last_saved_frame < min_frame_interval:
                frame_count += 1
                pbar.update(1)
                continue

            if not is_valid_frame(frame, min_brightness):
                frame_count += 1
                pbar.update(1)
                continue

            curr_hash = compute_frame_hash(frame)
            if is_scene_change(prev_hash, curr_hash, scene_threshold):
                timestamp = frame_count / video_fps
                frame_filename = f"frame_{saved_count:05d}_{timestamp:.2f}s.jpg"
                frame_path = frame_dir / frame_filename

                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                extracted.append(
                    {
                        "frame_path": str(frame_path),
                        "timestamp": round(timestamp, 2),
                        "frame_index": saved_count,
                    }
                )

                prev_hash = curr_hash
                last_saved_frame = frame_count
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()

    print(f"\nTotal frames scanned: {frame_count}")
    print(f"Frames saved (scene changes): {saved_count}")
    print(
        f"Reduction: {frame_count} → {saved_count} ({100 * (1 - saved_count/frame_count):.1f}% filtered)"
    )
    print(f"Saved → {frame_dir}")

    return extracted


def extract_frames_from_url(
    url: str,
    output_dir: str = "data/frames",
    video_name: str | None = None,
    scene_threshold: float = 15.0,
    min_brightness: float = 10.0,
    min_interval: float = 2.0,
) -> list[dict[str, Any]]:
    """Extract frames directly from a video URL using FFmpeg streaming.

    Streams video via FFmpeg without downloading the full file.
    Falls back to temporary file download if streaming fails.

    Args:
        url: Video URL (HTTP/HTTPS or archive.org)
        output_dir: Output directory for frames
        video_name: Name for the video (extracted from URL if not provided)
        scene_threshold: Sensitivity of scene change detection (lower = more frames)
        min_brightness: Minimum frame brightness (filters black frames)
        min_interval: Minimum seconds between saved frames

    Returns:
        List of {frame_path, timestamp, frame_index} dicts
    """
    if video_name is None:
        video_name = Path(url.split("?")[0]).stem or "video"

    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_name}")
    print(f"Source: {url[:80]}...")

    try:
        return _stream_frames_via_ffmpeg(
            url, frame_dir, video_name, scene_threshold, min_brightness, min_interval
        )
    except Exception as exc:
        print(f"FFmpeg streaming failed: {exc}")
        print("Falling back to temporary file download...")
        return _stream_frames_via_temp_file(
            url, frame_dir, video_name, scene_threshold, min_brightness, min_interval
        )


def _stream_frames_via_ffmpeg(
    url: str,
    frame_dir: Path,
    video_name: str,
    scene_threshold: float,
    min_brightness: float,
    min_interval: float,
) -> list[dict[str, Any]]:
    """Use FFmpeg to pipe frames directly from URL.

    Args:
        url: Video URL
        frame_dir: Output directory for frames
        video_name: Name for the video
        scene_threshold: Scene detection sensitivity
        min_brightness: Minimum frame brightness
        min_interval: Minimum seconds between frames

    Returns:
        List of extracted frames metadata
    """
    frame_pattern = str(frame_dir / "frame_%05d.ppm")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        url,
        "-vf",
        f"select=gt(scene\\,{scene_threshold/100})",
        "-vsync",
        "vfr",
        "-frame_pts",
        "true",
        frame_pattern,
    ]

    print(f"Streaming video with FFmpeg...")
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=7200)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg failed: {exc.stderr.decode() if exc.stderr else str(exc)}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg not found. Install FFmpeg and add to PATH.") from exc

    extracted = []
    ppm_files = sorted(frame_dir.glob("frame_*.ppm"))

    print(f"Converting {len(ppm_files)} PPM frames to JPEG...")
    for i, ppm_file in tqdm(
        enumerate(ppm_files), total=len(ppm_files), desc="Converting frames"
    ):
        try:
            frame = cv2.imread(str(ppm_file))
            if frame is None:
                ppm_file.unlink()
                continue

            if not is_valid_frame(frame, min_brightness):
                ppm_file.unlink()
                continue

            timestamp = i * min_interval
            jpg_filename = f"frame_{i:05d}_{timestamp:.2f}s.jpg"
            jpg_path = frame_dir / jpg_filename

            cv2.imwrite(str(jpg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            extracted.append(
                {
                    "frame_path": str(jpg_path),
                    "timestamp": round(timestamp, 2),
                    "frame_index": i,
                }
            )

            ppm_file.unlink()

        except Exception as exc:
            print(f"Error processing frame {i}: {exc}")
            if ppm_file.exists():
                ppm_file.unlink()
            continue

    print(f"\nTotal frames extracted: {len(extracted)}")
    print(f"Saved → {frame_dir}")

    return extracted


def _stream_frames_via_temp_file(
    url: str,
    frame_dir: Path,
    video_name: str,
    scene_threshold: float,
    min_brightness: float,
    min_interval: float,
) -> list[dict[str, Any]]:
    """Fallback: Download to temporary file, extract frames, then delete temp file.

    Args:
        url: Video URL
        frame_dir: Output directory for frames
        video_name: Name for the video
        scene_threshold: Scene detection sensitivity
        min_brightness: Minimum frame brightness
        min_interval: Minimum seconds between frames

    Returns:
        List of extracted frames metadata
    """
    import requests

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        print(f"Downloading to temporary file: {temp_path}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print("Extracting frames from temporary file...")
        return extract_frames(
            str(temp_path),
            str(frame_dir.parent),
            scene_threshold,
            min_brightness,
            min_interval,
            video_name=video_name,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()
            print(f"Deleted temporary file: {temp_path}")

