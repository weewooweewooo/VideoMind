"""OpenCLIP embedding wrapper for VideoMind retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F

from src.dataset.loader import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED


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
        self.device = self._resolve_device(device)

        try:
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Could not create OpenCLIP model {model_name!r} with pretrained={pretrained!r}") from exc

        if checkpoint is not None and Path(checkpoint).exists():
            self._load_checkpoint(model, checkpoint)

        self.model: nn.Module = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess_val
        self.tokenizer = tokenizer

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve cpu/cuda/auto into a PyTorch device."""
        requested = device.lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "npu":
            raise RuntimeError("OpenCLIP retrieval uses PyTorch and supports cpu/cuda in this project")
        return torch.device(requested)

    def _resolve_checkpoint_file(self, checkpoint: str | Path) -> Path:
        """Resolve a checkpoint directory or file to model.pt."""
        path = Path(checkpoint)
        if path.is_dir():
            path = path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"OpenCLIP checkpoint not found: {path}")
        return path

    def _load_checkpoint(self, model: nn.Module, checkpoint: str | Path) -> None:
        """Load fine-tuned OpenCLIP weights into the model."""
        checkpoint_file = self._resolve_checkpoint_file(checkpoint)
        state = torch.load(checkpoint_file, map_location="cpu")
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)

    def _open_image(self, image_path: str | Path) -> Image.Image:
        """Open an image as RGB with a clear error on failure."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            return Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image: {path}") from exc

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Embed one image path as a normalized numpy vector."""
        return self.embed_batch_images([image_path])[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one text query as a normalized numpy vector."""
        if not text.strip():
            raise ValueError("Text must not be empty")

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            embeddings = F.normalize(self.model.encode_text(tokens), dim=-1)
        return embeddings.detach().cpu().numpy()[0]

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
