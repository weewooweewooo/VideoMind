"""Redis Stack vector storage for VideoMind retrieval embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
INDEX_NAME = "videomind"
VECTOR_DIM = 512
logger = logging.getLogger(__name__)

schema = {
    "index": {
        "name": INDEX_NAME,
        "prefix": "videomind:doc",
    },
    "fields": [
        {"name": "id", "type": "tag"},
        {"name": "video", "type": "tag"},
        {"name": "text", "type": "text"},
        {"name": "start", "type": "numeric"},
        {"name": "end", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "datatype": "float32",
                "dims": VECTOR_DIM,
                "distance_metric": "cosine",
            },
        },
    ],
}


class VideoMindStore:
    """Persist and query VideoMind vectors in Redis Stack."""

    def __init__(self) -> None:
        """Initialize the RedisVL search index."""
        self.index = SearchIndex.from_dict(schema)
        self.index.connect(REDIS_URL)
        self.index.create(overwrite=False)

    def index_video(self, pairs: list[dict[str, Any]]) -> int:
        """Index video pairs into Redis."""
        docs = [
            {
                "id": pair.get("id", str(uuid.uuid4())),
                "video": pair["video"],
                "text": pair["text"],
                "start": float(pair.get("start", 0)),
                "end": float(pair.get("end", 0)),
                "embedding": np.array(pair["embedding"], dtype=np.float32).tobytes(),
            }
            for pair in pairs
        ]

        if docs:
            self.index.load(docs, id_field="id")
        return len(docs)

    def _nearest_segment(
        self, timestamp: float, segments: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Return the transcript segment closest to a frame timestamp."""
        best_segment = None
        best_distance = float("inf")
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            if start <= timestamp <= end:
                return segment
            distance = min(abs(timestamp - start), abs(timestamp - end))
            if distance < best_distance:
                best_distance = distance
                best_segment = segment
        return best_segment

    def _segment_context(
        self, segment: dict[str, Any], segments: list[dict[str, Any]]
    ) -> str:
        """Build local transcript context around a matched segment."""
        segment_index = segments.index(segment)
        start_index = max(0, segment_index - 3)
        end_index = min(len(segments), segment_index + 4)
        return " ".join(
            str(item["text"]) for item in segments[start_index:end_index]
        ).strip()

    def index_frames_from_memory(
        self,
        frames_with_embeddings: list[dict[str, Any]],
        transcript: dict[str, Any],
        video_name: str,
    ) -> int:
        """Align embeddings with transcript segments and index them to Redis."""
        segments = transcript.get("segments", [])
        if not isinstance(segments, list) or not segments:
            raise ValueError("Transcript contains no segments to index")

        docs: list[dict[str, Any]] = []
        for frame in frames_with_embeddings:
            segment = self._nearest_segment(float(frame["timestamp"]), segments)
            if segment is None:
                continue

            context = self._segment_context(segment, segments)
            if len(context.split()) < 5:
                continue

            doc = {
                "id": str(uuid.uuid4()),
                "video": video_name,
                "text": context,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "embedding": np.array(frame["embedding"], dtype=np.float32).tobytes(),
            }
            docs.append(doc)

        if docs:
            self.index.load(docs, id_field="id")
        return len(docs)

    def query(
        self,
        text_embedding: np.ndarray,
        top_k: int = 5,
        video_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query Redis for similar vectors."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = np.array(text_embedding, dtype=np.float32)
        logger.debug(
            "Redis vector query embedding shape=%s dtype=%s",
            query_vector.shape,
            query_vector.dtype,
        )

        query = VectorQuery(
            vector=query_vector.tobytes(),
            vector_field_name="embedding",
            return_fields=["id", "video", "text", "start", "end"],
            num_results=top_k,
        )
        query.dialect(2)

        if video_name:
            query.set_filter(f"@video:{{{video_name}}}")

        results = self.index.query(query)
        logger.debug("Redis vector query raw results=%d", len(results))
        if results:
            logger.debug("Redis vector query first result=%s", results[0])

        return [self._parse_query_result(result) for result in results]

    def _parse_query_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convert one RedisVL result into the API source shape."""
        vector_distance = result.get("vector_distance", 1)
        return {
            "video": result["video"],
            "text": result["text"],
            "start": float(result["start"]),
            "end": float(result["end"]),
            "score": 1 - float(vector_distance),
        }

    def list_videos(self) -> list[str]:
        """List all indexed video names."""
        client = redis.from_url(REDIS_URL)
        keys = client.keys("videomind:doc:*")
        videos = set()
        for key in keys:
            doc = client.hgetall(key)
            if b"video" in doc:
                videos.add(doc[b"video"].decode())
        return sorted(videos)

    def delete_video(self, video_name: str) -> int:
        """Delete all documents for a video."""
        if not video_name.strip():
            raise ValueError("video_name must not be empty")

        client = redis.from_url(REDIS_URL)
        keys = client.keys("videomind:doc:*")
        deleted = 0
        for key in keys:
            doc = client.hgetall(key)
            if doc.get(b"video", b"").decode() == video_name:
                client.delete(key)
                deleted += 1
        return deleted


def main() -> None:
    """Run simple Redis indexing from a JSON file containing embedded pairs."""
    parser = argparse.ArgumentParser(
        description="Index embedded VideoMind pair JSON files into Redis Stack."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Pairs JSON file or directory of *_pairs.json files.",
    )
    args = parser.parse_args()

    store = VideoMindStore()
    pairs_path = Path(args.pairs)
    files = sorted(pairs_path.glob("*_pairs.json")) if pairs_path.is_dir() else [pairs_path]
    total = 0
    for file in files:
        try:
            pairs = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid pairs JSON: {file}") from exc
        if not isinstance(pairs, list):
            raise ValueError(f"Pairs JSON must contain a list: {file}")
        total += store.index_video(pairs)
    logging.info("Indexed %d pairs", total)


if __name__ == "__main__":
    main()
