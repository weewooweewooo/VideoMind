from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
from src.ingestion import (
    _normalize_segments,
    _prepare_transcript,
    compile_transcript,
)


OUTPUT_ROOT = Path(__file__).resolve().parent
VIDEO_PATHS = [ROOT / "data" / "test1.mp4", ROOT / "data" / "test2.mp4"]
MODEL_REPOSITORIES = {
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
PROFILE = {
    "device": "cpu",
    "compute_type": "int8",
    "beam_size": 5,
    "num_workers": 4,
    "cpu_threads": 8,
    "vad_filter": False,
    "word_timestamps": False,
    "hotwords": None,
    "initial_prompt": None,
}


class PeakRssSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        self.peak = self.process.memory_info().rss

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak


def _snapshot_path(model: str, *, local_only: bool) -> Path | None:
    try:
        return Path(download_model(model, local_files_only=local_only))
    except Exception:
        return None


def _snapshot_size(path: Path | None) -> int | None:
    if path is None:
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _context(text: str, pattern: str, radius: int = 110) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end]


def _write_cache(model: str, video_path: Path, transcript: dict[str, Any]) -> Path:
    cache_dir = OUTPUT_ROOT / "cache" / model.replace(".", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = video_path.stat()
    record = {
        "source": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "profile": {
            "model": model,
            "device": PROFILE["device"],
            "compute_type": PROFILE["compute_type"],
            "beam_size": PROFILE["beam_size"],
        },
        "language": transcript["language"],
        "duration": transcript["duration"],
        "segments": transcript["segments"],
    }
    path = cache_dir / f"{video_path.stem}.json"
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return path


def _transcribe(model: WhisperModel, video_path: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    raw_segments, info = model.transcribe(
        str(video_path),
        beam_size=PROFILE["beam_size"],
        word_timestamps=PROFILE["word_timestamps"],
        vad_filter=PROFILE["vad_filter"],
        hotwords=PROFILE["hotwords"],
        initial_prompt=PROFILE["initial_prompt"],
    )
    segments = _normalize_segments(raw_segments)
    elapsed = time.perf_counter() - started
    duration = float(getattr(info, "duration", 0) or segments[-1]["end"])
    language = getattr(info, "language", None)
    return {
        "language": language if isinstance(language, str) and language else None,
        "duration": duration,
        "segments": segments,
    }, elapsed


def _run(model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_REPOSITORIES:
        raise ValueError(f"Unsupported benchmark model: {model_name}")

    sampler = PeakRssSampler()
    sampler.start()
    preexisting_path = _snapshot_path(model_name, local_only=True)
    print(f"[{model_name}] preparing model (preexisting={preexisting_path is not None})", flush=True)
    download_seconds: float | None = None
    model_reference = model_name
    snapshot_path = preexisting_path
    if preexisting_path is None:
        isolated_model_path = OUTPUT_ROOT / "models" / model_name.replace(".", "_")
        download_started = time.perf_counter()
        snapshot_path = Path(
            download_model(model_name, output_dir=str(isolated_model_path))
        )
        download_seconds = time.perf_counter() - download_started
        model_reference = str(snapshot_path)

    first_load_started = time.perf_counter()
    first_model = WhisperModel(
        model_reference,
        device=PROFILE["device"],
        compute_type=PROFILE["compute_type"],
        num_workers=PROFILE["num_workers"],
        cpu_threads=PROFILE["cpu_threads"],
    )
    first_load_seconds = time.perf_counter() - first_load_started
    snapshot_bytes = _snapshot_size(snapshot_path)
    del first_model
    gc.collect()

    cached_load_started = time.perf_counter()
    whisper = WhisperModel(
        model_reference,
        device=PROFILE["device"],
        compute_type=PROFILE["compute_type"],
        num_workers=PROFILE["num_workers"],
        cpu_threads=PROFILE["cpu_threads"],
    )
    cached_load_seconds = time.perf_counter() - cached_load_started

    result: dict[str, Any] = {
        "model": model_name,
        "repository": MODEL_REPOSITORIES[model_name],
        "profile": PROFILE,
        "model_preexisting": preexisting_path is not None,
        "download_seconds": download_seconds,
        "first_load_seconds": first_load_seconds,
        "cached_load_seconds": cached_load_seconds,
        "model_snapshot_path": str(snapshot_path) if snapshot_path else None,
        "model_snapshot_bytes": snapshot_bytes,
        "videos": {},
    }

    transcripts: dict[str, dict[str, Any]] = {}
    for video_path in VIDEO_PATHS:
        sampler.reset()
        print(f"[{model_name}] transcribing {video_path.name}", flush=True)
        transcript, elapsed = _transcribe(whisper, video_path)
        peak_rss = sampler.peak
        cache_path = _write_cache(model_name, video_path, transcript)
        compiled = compile_transcript(transcript["segments"])
        prepared = _prepare_transcript(transcript, video_path)
        text = str(compiled["text"])
        result["videos"][video_path.name] = {
            "transcription_seconds": elapsed,
            "duration_seconds": transcript["duration"],
            "realtime_factor": elapsed / transcript["duration"],
            "peak_rss_bytes": peak_rss,
            "language": transcript["language"],
            "segment_count": len(transcript["segments"]),
            "chunk_count": prepared["chunk_count"],
            "compiled_word_count": len(text.split()),
            "compiled_character_count": len(text),
            "compiled_start": compiled["start"],
            "compiled_end": compiled["end"],
            "transcript_start": text[:500],
            "transcript_end": text[-500:],
            "term_contexts": {
                "med": _context(text, r"\b(?:med\s*[- ]?l\s*m|medalam|medellem|metlm)\b"),
                "hca": _context(text, r"\bHCA\b"),
                "michael": _context(text, r"\bMichael\b"),
                "schlosser": _context(text, r"\bSchlosser\b"),
                "ashma": _context(text, r"\bAshma\b"),
                "google_cloud": _context(text, r"\bGoogle Cloud\b"),
            },
            "full_text": text,
            "experimental_cache": str(cache_path),
        }
        transcripts[video_path.name] = prepared
        print(
            f"[{model_name}] {video_path.name} finished in {elapsed:.1f}s "
            f"with {len(transcript['segments'])} segments",
            flush=True,
        )

    evaluation = evaluate._load_evaluation()
    retrieval = evaluate._evaluate(
        transcripts["test2.mp4"], evaluation["queries"]
    )
    result["retrieval"] = {
        "rank_one": retrieval["rank_one"],
        "top_three": retrieval["top_three"],
        "mrr": retrieval["mrr"],
        "negative_false_positives": retrieval["negative_false_positives"],
        "positive_count": retrieval["positive_count"],
        "negative_count": retrieval["negative_count"],
        "rows": [
            {
                "type": row["item"]["type"],
                "query": row["item"]["query"],
                "rank": row.get("rank"),
                "false_positive": row.get("false_positive"),
            }
            for row in retrieval["rows"]
        ],
    }
    result["peak_process_rss_bytes"] = sampler.stop()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REPOSITORIES)
    args = parser.parse_args()
    output_path = OUTPUT_ROOT / f"{args.model.replace('.', '_')}.json"
    try:
        result = _run(args.model)
    except BaseException as exc:
        failure = {
            "model": args.model,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output_path.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise
    result["status"] = "completed"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    retrieval = result["retrieval"]
    print(
        f"[{args.model}] completed: Rank-1 {retrieval['rank_one']}/"
        f"{retrieval['positive_count']}, Top-3 {retrieval['top_three']}/"
        f"{retrieval['positive_count']}, MRR {retrieval['mrr']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
