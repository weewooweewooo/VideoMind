"""FastAPI entrypoint for the local VideoMind service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import redis
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

load_dotenv()

from src.ingestion.archive_utils import resolve_direct_url
from src.ingestion.extractor import extract_frames_and_transcript_concurrent
from src.retrieval.embedder import CLIPEmbedder
from src.retrieval.pipeline import VideoMindPipeline
from src.retrieval.store import REDIS_URL, VideoMindStore
from src.utils.cleanup import clean_video, clean_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VideoMind", version="0.1.0")
sessions: dict[str, VideoMindPipeline] = {}
session_timestamps: dict[str, float] = {}
SESSION_TTL_SECONDS = float(os.environ.get("SESSION_TTL", "3600"))
EMBEDDER = CLIPEmbedder()
STORE = VideoMindStore()


@app.on_event("startup")
def warn_if_ollama_model_missing() -> None:
    """Warn when the configured Ollama model has not been pulled locally."""
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:
        logger.warning("Could not check Ollama model %s: %s", model, exc)
        return

    available_models = {
        item.get("name") or item.get("model")
        for item in models
        if isinstance(item, dict)
    }
    if model not in available_models:
        logger.warning("Ollama model %s not found. Run: ollama pull %s", model, model)


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
    frames_indexed: int


class QueryRequest(BaseModel):
    """Request body for querying indexed video content."""

    question: str
    session_id: str | None = None
    video_name: str | None = None


class QueryResponse(BaseModel):
    """Structured RAG response."""

    answer: str
    session_id: str
    conversation_turn: int


class SessionHistoryResponse(BaseModel):
    """Conversation history response for a session."""

    session_id: str
    history: list[dict[str, str]]


class DeleteSessionResponse(BaseModel):
    """Response returned after clearing a session."""

    session_id: str
    cleared: bool


class ClearQueryCacheResponse(BaseModel):
    """Response returned after clearing cached query answers."""

    cleared: bool


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
        default=["pairs", "redis"],
        description="Types to clean: pairs, redis",
    )


class CleanupResponse(BaseModel):
    """Response returned after cache cleanup."""

    deleted: list[str]
    bytes_freed: float
    mb_freed: float


def get_store() -> VideoMindStore:
    """Create a Redis vector store instance for the current request."""
    return STORE


def get_pipeline() -> VideoMindPipeline:
    """Create a RAG pipeline instance for the current request."""
    pipeline = VideoMindPipeline(store=get_store())
    pipeline.embedder = EMBEDDER
    return pipeline


def get_session_pipeline(session_id: str) -> VideoMindPipeline:
    """Return the pipeline for a session, creating it if needed."""
    if session_id not in sessions:
        sessions[session_id] = get_pipeline()
    return sessions[session_id]


def clear_expired_sessions(now: float | None = None) -> None:
    """Remove sessions that have not been used within the expiry window."""
    current_time = time.time() if now is None else now
    expired_session_ids = [
        session_id
        for session_id, timestamp in session_timestamps.items()
        if current_time - timestamp > SESSION_TTL_SECONDS
    ]
    for session_id in expired_session_ids:
        sessions.pop(session_id, None)
        session_timestamps.pop(session_id, None)


def cleanup_cache(targets: list[str], video_name: str | None = None) -> dict[str, Any]:
    """Clean selected cache targets and remove Redis records when requested."""
    invalid_targets = set(targets) - {"pairs", "redis"}
    if invalid_targets:
        raise ValueError(
            f"Invalid cleanup targets: {', '.join(sorted(invalid_targets))}"
        )

    if video_name is None:
        return clean_all(targets)
    return clean_video(video_name, targets)


def cleanup_response(result: dict[str, Any]) -> CleanupResponse:
    """Convert cleanup result data to an API response."""
    mb_freed = result["bytes_freed"] / (1024 * 1024)
    return CleanupResponse(
        deleted=result["deleted"],
        bytes_freed=float(result["bytes_freed"]),
        mb_freed=mb_freed,
    )


def health_check_status(func) -> str:
    """Return ok or an error string for one dependency check."""
    try:
        func()
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def check_redis_stack() -> None:
    """Verify Redis Stack is reachable and RediSearch commands are available."""
    client = redis.from_url(REDIS_URL)
    client.execute_command("FT._LIST")


def check_ollama() -> None:
    """Verify the configured Ollama server responds to its HTTP API."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    response = requests.get(f"{ollama_host}/api/tags", timeout=5)
    response.raise_for_status()


def check_openclip_model() -> None:
    """Verify the global OpenCLIP embedder has an initialized model."""
    if getattr(EMBEDDER, "model", None) is None:
        raise RuntimeError("OpenCLIP model is not initialized")
    if getattr(EMBEDDER, "tokenizer", None) is None:
        raise RuntimeError("OpenCLIP tokenizer is not initialized")
    next(EMBEDDER.model.parameters())


@app.get("/health")
async def health_check() -> JSONResponse:
    """Return dependency health for Redis Stack, Ollama, and OpenCLIP."""
    checks = {
        "redis": health_check_status(check_redis_stack),
        "ollama": health_check_status(check_ollama),
        "model": health_check_status(check_openclip_model),
    }
    healthy = all(value == "ok" for value in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "degraded",
            **checks,
        },
    )


def _check_already_indexed(
    video_name: str, store: VideoMindStore | None = None
) -> bool:
    """Return whether a video already has indexed Redis documents."""
    target_store = store or get_store()
    return video_name in target_store.list_videos()


def ingest_video_sync(
    video_path: str | None = None,
    url: str | None = None,
    video_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run extraction, transcription, embedding, and Redis indexing.

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
        return _ingest_video_source(url, video_name, force, is_url=True)
    elif video_path:
        return _ingest_video_source(video_path, video_name, force, is_url=False)
    else:
        raise ValueError("Either video_path or url must be provided")


def _resolve_ingest_source(
    source: str,
    video_name: str | None,
    is_url: bool,
) -> tuple[str, str]:
    """Resolve a local path or URL to the source and video name used for ingestion."""
    if is_url:
        resolved_source = resolve_direct_url(source)
        if video_name is None:
            video_name = Path(source.split("?")[0]).stem or "video"
        return resolved_source, video_name

    path = Path(source)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return str(path), video_name or path.stem


def _extract_video_assets(
    resolved_source: str, video_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract frames and transcript concurrently for an ingest source."""
    frames, transcript = extract_frames_and_transcript_concurrent(resolved_source)
    transcript["video"] = video_name
    return frames, transcript


def _ingest_video_source(
    source: str,
    video_name: str | None,
    force: bool,
    is_url: bool,
) -> dict[str, Any]:
    """Ingest a local path or URL through the in-memory pipeline."""
    resolved_source, video_name = _resolve_ingest_source(source, video_name, is_url)

    logger.info("Ingesting video in memory")
    start = time.time()
    frames, transcript = _extract_video_assets(resolved_source, video_name)
    elapsed = time.time() - start
    logger.info("In-memory extraction and transcription finished in %.1fs", elapsed)

    embeddings = EMBEDDER.embed_frames_from_memory(frames)
    text_segments = EMBEDDER.embed_transcript_segments(transcript["segments"])
    store = get_store()
    if force and _check_already_indexed(video_name, store):
        store.delete_video(video_name)
    indexed = store.index_frames_from_memory(embeddings, transcript, video_name)
    store.index_transcript_segments(video_name, text_segments)

    return {
        "video": video_name,
        "frames_extracted": len(frames),
        "transcript_segments": len(transcript.get("segments", [])),
        "frames_indexed": indexed,
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
        current_time = time.time()
        clear_expired_sessions(current_time)
        session_id = request.session_id or str(uuid.uuid4())
        pipeline = get_session_pipeline(session_id)
        session_timestamps[session_id] = current_time
        result = await asyncio.to_thread(
            lambda: pipeline.query(request.question, video_name=request.video_name)
        )
        return QueryResponse(
            answer=result["answer"],
            session_id=session_id,
            conversation_turn=len(pipeline.get_history()) // 2,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/query/stream")
async def query_video_stream(request: QueryRequest) -> StreamingResponse:
    """Stream an answer against indexed video content as SSE events."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    current_time = time.time()
    clear_expired_sessions(current_time)
    session_id = request.session_id or str(uuid.uuid4())
    pipeline = get_session_pipeline(session_id)
    session_timestamps[session_id] = current_time

    def event_stream():
        try:
            for event in pipeline.query_stream(
                request.question,
                session_id=session_id,
                video_name=request.video_name,
            ):
                if "sources" in event:
                    continue
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'conversation_turn': len(pipeline.get_history()) // 2})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Query failed: {exc}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def clear_session(session_id: str) -> DeleteSessionResponse:
    """Clear conversation history for a session."""
    try:
        pipeline = sessions.get(session_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Session not found")
        pipeline.clear_history()
        session_timestamps[session_id] = time.time()
        return DeleteSessionResponse(session_id=session_id, cleared=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not clear session: {exc}"
        ) from exc


@app.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Return conversation history for a session."""
    try:
        pipeline = sessions.get(session_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session_timestamps[session_id] = time.time()
        return SessionHistoryResponse(
            session_id=session_id,
            history=pipeline.get_history(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not get session history: {exc}"
        ) from exc


@app.delete("/cache/queries", response_model=ClearQueryCacheResponse)
async def clear_query_cache() -> ClearQueryCacheResponse:
    """Clear cached query responses."""
    try:
        pipeline = next(iter(sessions.values()), None) or get_pipeline()
        await asyncio.to_thread(pipeline.clear_cache)
        return ClearQueryCacheResponse(cleared=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not clear query cache: {exc}"
        ) from exc


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
