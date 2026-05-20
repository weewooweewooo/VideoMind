"""FastAPI entrypoint for the local VideoMind service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.dataset.builder import build_video_pairs
from src.ingestion.extractor import extract_frames
from src.ingestion.archive_utils import resolve_direct_url
from src.ingestion.transcriber import transcribe_video
from src.retrieval.pipeline import VideoMindPipeline
from src.retrieval.store import VideoMindStore
from src.utils.cleanup import clean_video, clean_all

app = FastAPI(title="VideoMind", version="0.1.0")


class IngestRequest(BaseModel):
    """Request body for ingesting a video file or URL."""

    video_path: str | None = Field(
        default=None, description="Path to a local video file."
    )
    url: str | None = Field(
        default=None,
        description="URL to stream and process (archive.org, youtube, vimeo, etc.).",
    )
    video_name: str | None = Field(
        default=None,
        description="Optional name for the video (extracted from URL/path if not provided).",
    )
    force: bool = Field(
        default=False, description="Force re-run all steps even if data exists."
    )

    @model_validator(mode="after")
    def validate_video_source(self) -> IngestRequest:
        """Ensure exactly one of video_path or url is provided."""
        if not self.video_path and not self.url:
            raise ValueError("Either video_path or url must be provided")
        if self.video_path and self.url:
            raise ValueError("Cannot provide both video_path and url")
        return self


class IngestResponse(BaseModel):
    """Response returned after a video is ingested and indexed."""

    video: str
    frames_extracted: int
    transcript_segments: int
    pairs_indexed: int
    pairs_path: str


class QueryRequest(BaseModel):
    """Request body for querying indexed video content."""

    question: str
    video_name: str | None = None


class SourceResponse(BaseModel):
    """Retrieved source returned with an answer."""

    timestamp: float
    text: str
    frame_path: str
    score: float
    video: str


class QueryResponse(BaseModel):
    """Structured RAG response."""

    answer: str
    sources: list[SourceResponse]


class VideosResponse(BaseModel):
    """Indexed video list response."""

    videos: list[str]


class DeleteVideoResponse(BaseModel):
    """Video deletion response."""

    video: str
    deleted: bool


class CleanupRequest(BaseModel):
    """Request body for cache cleanup operations."""

    targets: list[str] = Field(
        default=["frames", "transcripts", "pairs", "chroma"],
        description="Types to clean: frames, transcripts, pairs, chroma",
    )


class CleanupResponse(BaseModel):
    """Response returned after cache cleanup."""

    deleted: list[str]
    bytes_freed: float
    mb_freed: float


def get_store() -> VideoMindStore:
    """Create a Chroma store instance for the current request."""
    return VideoMindStore()


def get_pipeline() -> VideoMindPipeline:
    """Create a RAG pipeline instance for the current request."""
    return VideoMindPipeline(store=get_store())


def frames_exist(video_name: str) -> bool:
    """Return whether extracted frames already exist for a video."""
    frames_dir = Path("data/frames") / video_name
    return (
        frames_dir.exists()
        and (
            any(frames_dir.glob("*.jpg"))
            or any(frames_dir.glob("*.jpeg"))
            or any(frames_dir.glob("*.png"))
        )
    )


def transcript_exists(video_name: str) -> bool:
    """Return whether a transcript JSON already exists for a video."""
    return (Path("data/transcripts") / f"{video_name}.json").exists()


def pairs_exist(video_name: str) -> bool:
    """Return whether the all-pairs JSON already exists for a video."""
    return (Path("data/pairs") / f"{video_name}_pairs.json").exists()


def cleanup_cache(targets: list[str], video_name: str | None = None) -> dict[str, Any]:
    """Clean selected cache targets and remove Chroma records when requested."""
    store = get_store()
    if video_name is None:
        result = clean_all(targets)
        if "chroma" in targets:
            for indexed_video in store.list_videos():
                store.delete_video(indexed_video)
    else:
        result = clean_video(video_name, targets)
        if "chroma" in targets:
            store.delete_video(video_name)
    return result


def cleanup_response(result: dict[str, Any]) -> CleanupResponse:
    """Convert cleanup result data to an API response."""
    mb_freed = result["bytes_freed"] / (1024 * 1024)
    return CleanupResponse(
        deleted=result["deleted"],
        bytes_freed=float(result["bytes_freed"]),
        mb_freed=mb_freed,
    )


def ingest_video_sync(
    video_path: str | None = None,
    url: str | None = None,
    video_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run extraction, transcription, pair building, and Chroma indexing.

    Supports both local files and streaming from URLs.

    Args:
        video_path: Path to local video file
        url: URL to stream and process
        video_name: Optional name for the video
        force: Force re-run all steps

    Returns:
        Dictionary with ingestion results
    """
    if url:
        return _ingest_video_from_url(url, video_name, force)
    elif video_path:
        return _ingest_video_from_path(video_path, force)
    else:
        raise ValueError("Either video_path or url must be provided")


def _ingest_video_from_path(video_path: str, force: bool = False) -> dict[str, Any]:
    """Ingest a local video file."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    video_name = path.stem

    frames = None
    frames_dir = Path("data/frames") / video_name
    if not force and frames_exist(video_name):
        print(f"Frames already exist, skipping extraction")
    else:
        frames = extract_frames(str(path))

    transcript = None
    transcript_path = Path("data/transcripts") / f"{video_name}.json"
    if not force and transcript_exists(video_name):
        print(f"Transcript already exists, skipping transcription")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    else:
        transcript = transcribe_video(str(path))

    pairs_path = None
    pairs_all_path = Path("data/pairs") / f"{video_name}_pairs.json"
    if not force and pairs_exist(video_name):
        print(f"Dataset already exists, skipping building")
        pairs_path = pairs_all_path
    else:
        saved_files = build_video_pairs(path)
        pairs_path = saved_files["all"]

    store = get_store()
    indexed = 0
    if not force and video_name in store.list_videos():
        print(f"Video already indexed, skipping ChromaDB indexing")
    else:
        indexed = store.index_video(pairs_path)

    if frames is None:
        frames = list(frames_dir.glob("*.jp*g")) if frames_dir.exists() else []
    if transcript is None:
        transcript = (
            json.loads(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.exists()
            else {"segments": []}
        )

    return {
        "video": video_name,
        "frames_extracted": len(frames),
        "transcript_segments": len(transcript.get("segments", [])),
        "pairs_indexed": indexed,
        "pairs_path": str(pairs_path),
    }


def _ingest_video_from_url(
    url: str, video_name: str | None = None, force: bool = False
) -> dict[str, Any]:
    """Ingest a video from URL via streaming."""
    from src.ingestion.extractor import extract_frames_from_url
    from src.ingestion.transcriber import transcribe_url

    if video_name is None:
        video_name = Path(url.split("?")[0]).stem or "video"

    resolved_url = resolve_direct_url(url)
    if resolved_url != url:
        url = resolved_url
        print(f"Resolved direct URL: {url}")

    print(f"Ingesting from URL: {url[:80]}...")

    frames = None
    frames_dir = Path("data/frames") / video_name
    if not force and frames_exist(video_name):
        print(f"Frames already exist, skipping extraction")
        frames = list(frames_dir.glob("*.jp*g"))
    else:
        frames = extract_frames_from_url(url, video_name=video_name)

    transcript = None
    transcript_path = Path("data/transcripts") / f"{video_name}.json"
    if not force and transcript_exists(video_name):
        print(f"Transcript already exists, skipping transcription")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    else:
        transcript = transcribe_url(url, video_name=video_name)

    pairs_path = None
    pairs_all_path = Path("data/pairs") / f"{video_name}_pairs.json"
    if not force and pairs_exist(video_name):
        print(f"Dataset already exists, skipping building")
        pairs_path = pairs_all_path
    else:
        saved_files = build_video_pairs(frames_dir)
        pairs_path = saved_files["all"]

    store = get_store()
    indexed = 0
    if not force and video_name in store.list_videos():
        print(f"Video already indexed, skipping ChromaDB indexing")
    else:
        indexed = store.index_video(pairs_path)

    if frames is None:
        frames = list(frames_dir.glob("*.jp*g")) if frames_dir.exists() else []
    if transcript is None:
        transcript = (
            json.loads(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.exists()
            else {"segments": []}
        )

    return {
        "video": video_name,
        "frames_extracted": len(frames),
        "transcript_segments": len(transcript.get("segments", [])),
        "pairs_indexed": indexed,
        "pairs_path": str(pairs_path),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_video(request: IngestRequest) -> IngestResponse:
    """Ingest a video from local file or URL and index it for retrieval."""
    try:
        result = await asyncio.to_thread(
            ingest_video_sync,
            request.video_path,
            request.url,
            request.video_name,
            request.force,
        )
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest) -> QueryResponse:
    """Answer a question against indexed video content."""
    try:
        result = await asyncio.to_thread(
            lambda: get_pipeline().query(request.question, request.video_name)
        )
        return QueryResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.get("/videos", response_model=VideosResponse)
async def list_videos() -> VideosResponse:
    """List indexed videos without loading the embedding model."""
    try:
        videos = await asyncio.to_thread(lambda: get_store().list_videos())
        return VideosResponse(videos=videos)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not list videos: {exc}"
        ) from exc


@app.delete("/videos/{video_name}", response_model=DeleteVideoResponse)
async def delete_video(video_name: str) -> DeleteVideoResponse:
    """Delete all indexed records for a video."""
    try:
        await asyncio.to_thread(lambda: get_store().delete_video(video_name))
        return DeleteVideoResponse(video=video_name, deleted=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not delete video: {exc}"
        ) from exc


@app.delete("/videos/{video_name}/cache", response_model=CleanupResponse)
async def cleanup_video_cache(
    video_name: str, request: CleanupRequest
) -> CleanupResponse:
    """Delete selected cached data files for a specific video."""
    try:
        result = await asyncio.to_thread(cleanup_cache, request.targets, video_name)
        return cleanup_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Cache cleanup failed: {exc}"
        ) from exc


@app.delete("/cache/all", response_model=CleanupResponse)
async def cleanup_all_cache(request: CleanupRequest) -> CleanupResponse:
    """Delete selected cached data files for all videos."""
    try:
        result = await asyncio.to_thread(cleanup_cache, request.targets)
        return cleanup_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Cache cleanup failed: {exc}"
        ) from exc
