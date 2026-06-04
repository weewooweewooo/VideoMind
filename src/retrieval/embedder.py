"""OpenCLIP embedding wrapper for VideoMind retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F

from src.dataset.loader import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED
from src.utils.model_utils import load_openclip_with_checkpoint, resolve_device


class CLIPEmbedder:
    """Embed images and text with a local OpenCLIP model."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        device: str = "auto",
    ) -> None:
        """Load vanilla OpenCLIP and optional fine-tuned checkpoint weights."""
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)

        checkpoint_path = (
            str(checkpoint)
            if checkpoint is not None and Path(checkpoint).exists()
            else None
        )
        model, _, preprocess_val = load_openclip_with_checkpoint(
            model_name,
            pretrained,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model: nn.Module = model
        self.model.eval()
        self.preprocess = preprocess_val
        self.tokenizer = tokenizer

    def _open_image(self, image_path: str | Path) -> Image.Image:
        """Open an image as RGB with a clear error on failure."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            return Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image: {path}") from exc

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one text query as a normalized numpy vector."""
        if not text.strip():
            raise ValueError("Text must not be empty")

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            embeddings = F.normalize(self.model.encode_text(tokens), dim=-1)
        return embeddings.detach().cpu().numpy()[0].astype(np.float32)

    def query_embedding(self, text: str) -> np.ndarray:
        """Embed one retrieval query as a normalized float32 numpy vector."""
        return self.embed_text(text)

    def embed_transcript_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Embed transcript segments and preserve text/timestamp metadata."""
        chunked_segments = self.chunk_transcript_segments(segments)
        embedded_segments: list[dict[str, Any]] = []
        for segment in chunked_segments:
            text = str(segment["text"]).strip()
            if not text:
                continue
            embedded_segments.append(
                {
                    "embedding": self.embed_text(text),
                    "text": text,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                }
            )
        return embedded_segments

    def chunk_transcript_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge short transcript segments into larger overlapping chunks."""
        target_words = int(os.environ.get("CHUNK_SIZE", "250"))
        overlap_segments = int(os.environ.get("CHUNK_OVERLAP", "2"))
        max_words = 400

        chunks: list[dict[str, Any]] = []
        valid_segments = [
            segment
            for segment in segments
            if str(segment.get("text", "")).strip()
        ]
        index = 0

        while index < len(valid_segments):
            chunk_segments: list[dict[str, Any]] = []
            word_count = 0
            cursor = index

            while cursor < len(valid_segments):
                segment = valid_segments[cursor]
                segment_words = str(segment["text"]).split()
                next_count = word_count + len(segment_words)
                if chunk_segments and next_count > max_words:
                    break
                chunk_segments.append(segment)
                word_count = next_count
                cursor += 1
                if word_count >= target_words:
                    break

            chunks.append(
                {
                    "start": float(chunk_segments[0]["start"]),
                    "end": float(chunk_segments[-1]["end"]),
                    "text": " ".join(
                        str(segment["text"]).strip() for segment in chunk_segments
                    ),
                }
            )

            if cursor >= len(valid_segments):
                break
            index = max(index + 1, cursor - max(0, overlap_segments))

        return chunks

    def embed_batch_images(self, paths: list[str | Path], batch_size: int = 32) -> np.ndarray:
        """Embed image paths as normalized numpy vectors."""
        if not paths:
            raise ValueError("paths must contain at least one image")

        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [self.preprocess(self._open_image(path)) for path in batch_paths]
            image_tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                embeddings = F.normalize(self.model.encode_image(image_tensor), dim=-1)
            batches.append(embeddings.detach().cpu().numpy())

        return np.vstack(batches)

    def embed_frames_from_memory(
        self,
        frames: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Embed in-memory frames and preserve their timestamps."""
        if not frames:
            return []

        embedded_frames: list[dict[str, Any]] = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            image_tensor = torch.stack(
                [self.preprocess(self._memory_frame_to_image(frame)) for frame in batch]
            ).to(self.device)
            with torch.no_grad():
                embeddings = F.normalize(self.model.encode_image(image_tensor), dim=-1)
            for frame, embedding in zip(batch, embeddings.detach().cpu().numpy()):
                embedded_frames.append(
                    {
                        "embedding": embedding,
                        "timestamp": float(frame["timestamp"]),
                    }
                )

        return embedded_frames

    def _memory_frame_to_image(self, frame: dict[str, Any]) -> Image.Image:
        if "frame" in frame:
            return Image.fromarray(frame["frame"]).convert("RGB")
        return frame["image"].convert("RGB")
