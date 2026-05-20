import whisper
import json
import os
import requests
import tempfile
from pathlib import Path
from typing import Any
from tqdm import tqdm

from src.ingestion.archive_utils import fetch_archive_metadata, resolve_direct_url


def parse_srt_time(time_str: str) -> float:
    """Convert an SRT or VTT timestamp to seconds.

    Args:
        time_str: Timestamp in HH:MM:SS,mmm or HH:MM:SS.mmm format

    Returns:
        Timestamp in seconds
    """
    time_part = time_str.strip().replace(",", ".")
    hours_text, minutes_text, seconds_text = time_part.split(":")
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + float(seconds_text)
    )


def get_archive_transcript(identifier: str) -> dict | None:
    """Fetch and parse an existing archive.org SRT or VTT transcript.

    Args:
        identifier: Archive.org item identifier

    Returns:
        Whisper-compatible transcript dictionary, or None if none exists
    """
    try:
        data = fetch_archive_metadata(identifier)
        files = data.get("files", [])

        transcript_extensions = [".srt", ".vtt"]
        transcript_file = None
        for file_info in files:
            name = file_info.get("name", "")
            if any(name.endswith(ext) for ext in transcript_extensions):
                transcript_file = name
                break

        if not transcript_file:
            return None

        file_url = f"https://archive.org/download/{identifier}/{transcript_file}"
        transcript_response = requests.get(file_url, timeout=10)
        transcript_response.raise_for_status()
        content = transcript_response.text

        if transcript_file.endswith(".vtt"):
            lines = content.strip().splitlines()
            if lines and lines[0].strip() == "WEBVTT":
                content = "\n".join(lines[1:])

        segments = []
        blocks = content.strip().split("\n\n")
        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                times = lines[1].split(" --> ")
                start = parse_srt_time(times[0])
                end = parse_srt_time(times[1])
                text = " ".join(lines[2:])
                segments.append(
                    {"id": i, "start": start, "end": end, "text": text, "words": []}
                )

        return {
            "video": identifier,
            "duration": segments[-1]["end"] if segments else 0,
            "language": "en",
            "segments": segments,
        }

    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Could not fetch archive.org transcript for {identifier}: {exc}")
        return None


def transcribe_video(
    video_path: str, output_dir: str = "data/transcripts", model_size: str = "base"
) -> dict:
    """
    Transcribe a video using Whisper.
    Returns transcript with word-level timestamps.
    Saves output as JSON.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{video_name}.json"

    print(f"Loading Whisper {model_size}...")
    model = whisper.load_model(model_size, device="cpu")

    print(f"Transcribing: {video_path.name}")
    print("This may take a few minutes on CPU...")
    result = model.transcribe(
        str(video_path),
        verbose=False,
        language="en",
        task="transcribe",
        word_timestamps=True,
    )

    transcript = {
        "video": video_path.name,
        "duration": result.get("duration", 0),
        "language": result.get("language", "en"),
        "segments": [],
    }

    for seg in tqdm(result["segments"], desc="Processing segments"):
        transcript["segments"].append(
            {
                "id": seg["id"],
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "words": [
                    {
                        "word": w["word"].strip(),
                        "start": round(w["start"], 2),
                        "end": round(w["end"], 2),
                    }
                    for w in seg.get("words", [])
                ],
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved → {output_file}")
    print(f"Total segments: {len(transcript['segments'])}")
    print(f"\nSample segment:")
    print(
        f"  [{transcript['segments'][0]['start']}s → {transcript['segments'][0]['end']}s]"
    )
    print(f"  {transcript['segments'][0]['text']}")

    return transcript


def transcribe_url(
    url: str,
    output_dir: str = "data/transcripts",
    video_name: str | None = None,
    model_size: str = "base",
) -> dict[str, Any]:
    """Transcribe a video directly from a URL using Whisper.

    Whisper natively supports URLs. Falls back to temporary file
    download if URL streaming fails.

    Args:
        url: Video URL (HTTP/HTTPS or archive.org)
        output_dir: Output directory for transcripts
        video_name: Name for the video (extracted from URL if not provided)
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Transcript dictionary with segments and word-level timestamps
    """
    if video_name is None:
        video_name = Path(url.split("?")[0]).stem or "video"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{video_name}.json"

    resolved_url = resolve_direct_url(url)
    if resolved_url != url:
        url = resolved_url
        print(f"Resolved direct URL: {url}")

    if "archive.org/download/" in url:
        identifier = url.split("/download/")[1].split("/")[0]
        print("Checking for existing transcript on archive.org...")
        existing = get_archive_transcript(identifier)
        if existing:
            print(
                f"Found existing transcript with {len(existing['segments'])} segments — skipping Whisper"
            )
            existing["video"] = video_name
            with open(output_file, "w") as f:
                json.dump(existing, f, indent=2)
            return existing

    print(f"Loading Whisper {model_size}...")
    model = whisper.load_model(model_size, device="cpu")

    print(f"Transcribing from URL: {url[:80]}...")
    print("This may take a few minutes on CPU...")

    try:
        result = model.transcribe(
            url,
            verbose=False,
            language="en",
            task="transcribe",
            word_timestamps=True,
        )
    except Exception as exc:
        print(f"URL transcription failed: {exc}")
        print("Falling back to temporary file download...")
        result = _transcribe_via_temp_file(url, model)

    segments = result.get("segments", [])
    if len(segments) < 50:
        print(
            f"Only {len(segments)} segments from URL, trying yt-dlp fallback..."
        )
        result = _transcribe_via_ytdlp_audio(url, model)

    transcript = {
        "video": video_name,
        "duration": result.get("duration", 0),
        "language": result.get("language", "en"),
        "segments": [],
    }

    for seg in tqdm(result["segments"], desc="Processing segments"):
        transcript["segments"].append(
            {
                "id": seg["id"],
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "words": [
                    {
                        "word": w["word"].strip(),
                        "start": round(w["start"], 2),
                        "end": round(w["end"], 2),
                    }
                    for w in seg.get("words", [])
                ],
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved → {output_file}")
    print(f"Total segments: {len(transcript['segments'])}")
    if transcript['segments']:
        print(f"\nSample segment:")
        print(
            f"  [{transcript['segments'][0]['start']}s → {transcript['segments'][0]['end']}s]"
        )
        print(f"  {transcript['segments'][0]['text']}")

    return transcript


def _transcribe_via_ytdlp_audio(url: str, model: Any) -> dict[str, Any]:
    """Download best audio with yt-dlp, transcribe it, then delete the temp file."""
    import yt_dlp

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ydl_opts = {
            "format": "bestaudio",
            "outtmpl": tmp_path,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return model.transcribe(
            tmp_path,
            verbose=False,
            language="en",
            task="transcribe",
            word_timestamps=True,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _transcribe_via_temp_file(url: str, model: Any) -> dict[str, Any]:
    """Fallback: Download to temporary file, transcribe, then delete.

    Args:
        url: Video URL
        model: Loaded Whisper model

    Returns:
        Transcription result dictionary
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

        print("Transcribing from temporary file...")
        result = model.transcribe(
            str(temp_path),
            verbose=False,
            language="en",
            task="transcribe",
            word_timestamps=True,
        )
        return result

    finally:
        if temp_path.exists():
            temp_path.unlink()
            print(f"Deleted temporary file: {temp_path}")

