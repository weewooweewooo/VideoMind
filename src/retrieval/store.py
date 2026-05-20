"""Redis Stack vector storage for VideoMind retrieval embeddings."""

from __future__ import annotations

import argparse
import json
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
        docs: list[dict[str, Any]] = []
        for pair in pairs:
            doc = {
                "id": pair.get("id", str(uuid.uuid4())),
                "video": pair["video"],
                "text": pair["text"],
                "start": float(pair.get("start", 0)),
                "end": float(pair.get("end", 0)),
                "embedding": np.array(pair["embedding"], dtype=np.float32).tobytes(),
            }
            docs.append(doc)

        if docs:
            self.index.load(docs, id_field="id")
        return len(docs)

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
            ts = float(frame["timestamp"])
            embedding = frame["embedding"]

            best_seg = None
            best_dist = float("inf")
            for seg in segments:
                start = float(seg["start"])
                end = float(seg["end"])
                if start <= ts <= end:
                    best_seg = seg
                    break
                dist = min(abs(ts - start), abs(ts - end))
                if dist < best_dist:
                    best_dist = dist
                    best_seg = seg

            if best_seg is None:
                continue

            seg_idx = segments.index(best_seg)
            start_idx = max(0, seg_idx - 3)
            end_idx = min(len(segments), seg_idx + 4)
            context = " ".join(
                str(segment["text"]) for segment in segments[start_idx:end_idx]
            ).strip()

            if len(context.split()) < 5:
                continue

            doc = {
                "id": str(uuid.uuid4()),
                "video": video_name,
                "text": context,
                "start": float(best_seg["start"]),
                "end": float(best_seg["end"]),
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),
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

        query = VectorQuery(
            vector=text_embedding.astype(np.float32).tobytes(),
            vector_field_name="embedding",
            return_fields=["id", "video", "text", "start", "end"],
            num_results=top_k,
        )

        if video_name:
            query.set_filter(f"@video:{{{video_name}}}")

        results = self.index.query(query)
        return [
            {
                "video": result["video"],
                "text": result["text"],
                "start": float(result["start"]),
                "end": float(result["end"]),
                "score": 1 - float(result.get("vector_distance", 1)),
            }
            for result in results
        ]

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
    print(f"Indexed {total} pairs")


if __name__ == "__main__":
    main()
