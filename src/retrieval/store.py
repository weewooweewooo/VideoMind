"""ChromaDB storage wrapper for VideoMind frame embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb

from src.retrieval.embedder import CLIPEmbedder


class VideoMindStore:
    """Persist and query OpenCLIP frame embeddings in local ChromaDB."""

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "videomind_frames",
        embedder: CLIPEmbedder | None = None,
        checkpoint: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize a local persistent Chroma collection."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self.checkpoint = checkpoint
        self.device = device
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @property
    def embedder(self) -> CLIPEmbedder:
        """Return the OpenCLIP embedder, loading it only when embeddings are needed."""
        if self._embedder is None:
            self._embedder = CLIPEmbedder(checkpoint=self.checkpoint, device=self.device)
        return self._embedder

    def _make_id(self, pair: dict[str, Any]) -> str:
        """Build a stable Chroma id for a pair record."""
        raw = f"{pair.get('video')}|{pair.get('segment_id')}|{pair.get('timestamp')}|{pair.get('frame_path')}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"{pair.get('video', 'video')}_{digest}"

    def index_video(self, pairs_json_path: str | Path, batch_size: int = 32) -> int:
        """Embed all frame pairs from a JSON file and upsert them into ChromaDB."""
        path = Path(pairs_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Pairs JSON not found: {path}")

        try:
            pairs = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid pairs JSON: {path}") from exc
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"Pairs JSON must contain a non-empty list: {path}")

        count = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            frame_paths = [str(pair["frame_path"]) for pair in batch]
            embeddings = self.embedder.embed_batch_images(frame_paths, batch_size=batch_size)
            ids = [self._make_id(pair) for pair in batch]
            documents = [str(pair.get("text", "")) for pair in batch]
            metadatas = [
                {
                    "timestamp": float(pair.get("timestamp", 0.0)),
                    "video": str(pair.get("video", "")),
                    "text": str(pair.get("text", "")),
                    "frame_path": str(pair.get("frame_path", "")),
                    "segment_id": int(pair.get("segment_id", -1)),
                }
                for pair in batch
            ]
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            count += len(batch)
        return count

    def query(
        self,
        text: str,
        top_k: int = 5,
        video_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar frame/text records for a text query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed_text(text)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if video_name:
            query_kwargs["where"] = {"video": video_name}

        results = self.collection.query(**query_kwargs)
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        records: list[dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            records.append(
                {
                    "frame_path": str(metadata.get("frame_path", "")),
                    "timestamp": float(metadata.get("timestamp", 0.0)),
                    "video": str(metadata.get("video", "")),
                    "text": str(metadata.get("text", "")),
                    "score": 1.0 - float(distance),
                    "segment_id": int(metadata.get("segment_id", -1)),
                }
            )
        return records

    def delete_video(self, video_name: str) -> None:
        """Delete all indexed records for a video name."""
        if not video_name.strip():
            raise ValueError("video_name must not be empty")
        self.collection.delete(where={"video": video_name})

    def list_videos(self) -> list[str]:
        """Return sorted unique video names currently indexed."""
        results = self.collection.get(include=["metadatas"])
        videos = {
            str(metadata.get("video", ""))
            for metadata in results.get("metadatas", [])
            if metadata.get("video")
        }
        return sorted(videos)


def main() -> None:
    """Run simple Chroma indexing from the command line."""
    parser = argparse.ArgumentParser(description="Index VideoMind pair JSON files into ChromaDB.")
    parser.add_argument("--pairs", required=True, help="Pairs JSON file or directory of *_pairs.json files.")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned OpenCLIP checkpoint path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    args = parser.parse_args()

    store = VideoMindStore(checkpoint=args.checkpoint, device=args.device)
    pairs_path = Path(args.pairs)
    files = sorted(pairs_path.glob("*_pairs.json")) if pairs_path.is_dir() else [pairs_path]
    total = 0
    for file in files:
        total += store.index_video(file)
        print(f"Indexed {file}")
    print(f"Indexed {total} pairs")


if __name__ == "__main__":
    main()
