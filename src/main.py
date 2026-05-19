"""FastAPI entrypoint for the local VideoMind service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.dataset.builder import build_video_pairs
from src.ingestion.extractor import extract_frames
from src.ingestion.transcriber import transcribe_video
from src.retrieval.pipeline import VideoMindPipeline
from src.retrieval.store import VideoMindStore


app = FastAPI(title="VideoMind", version="0.1.0")


class IngestRequest(BaseModel):
    """Request body for ingesting a local video file."""

    video_path: str = Field(..., description="Path to a local video file.")


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


def get_store() -> VideoMindStore:
    """Create a Chroma store instance for the current request."""
    return VideoMindStore()


def get_pipeline() -> VideoMindPipeline:
    """Create a RAG pipeline instance for the current request."""
    return VideoMindPipeline(store=get_store())


def ingest_video_sync(video_path: str) -> dict[str, Any]:
    """Run extraction, transcription, pair building, and Chroma indexing."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    frames = extract_frames(str(path))
    transcript = transcribe_video(str(path))
    saved_files = build_video_pairs(path)
    pairs_path = saved_files["all"]
    indexed = get_store().index_video(pairs_path)

    return {
        "video": path.stem,
        "frames_extracted": len(frames),
        "transcript_segments": len(transcript.get("segments", [])),
        "pairs_indexed": indexed,
        "pairs_path": str(pairs_path),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_video(request: IngestRequest) -> IngestResponse:
    """Ingest a local video file and index it for retrieval."""
    try:
        result = await asyncio.to_thread(ingest_video_sync, request.video_path)
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
        raise HTTPException(status_code=500, detail=f"Could not list videos: {exc}") from exc


@app.delete("/videos/{video_name}", response_model=DeleteVideoResponse)
async def delete_video(video_name: str) -> DeleteVideoResponse:
    """Delete all indexed records for a video."""
    try:
        await asyncio.to_thread(lambda: get_store().delete_video(video_name))
        return DeleteVideoResponse(video=video_name, deleted=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete video: {exc}") from exc
