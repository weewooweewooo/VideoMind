"""PyTorch dataset for OpenCLIP lecture frame/text pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"
ImageTransform = Callable[[Image.Image], torch.Tensor]
TextTokenizer = Callable[[list[str]], torch.Tensor]


def load_pair_records(dataset_path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Load one pair JSON file or all matching split JSON files from a directory."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if path.is_file():
        files = [path]
    else:
        pattern = f"*_{split}.json" if split else "*.json"
        files = sorted(path.glob(pattern))
        if split is None:
            files = [file for file in files if file.name.endswith("_pairs.json")]

    if not files:
        raise FileNotFoundError(f"No dataset JSON files found in {path} for split={split!r}")

    records: list[dict[str, Any]] = []
    for file in files:
        try:
            loaded = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in dataset file {file}: {exc}") from exc
        if not isinstance(loaded, list):
            raise ValueError(f"Dataset file must contain a list: {file}")
        records.extend(loaded)

    if not records:
        raise ValueError(f"Dataset contains no records: {path}")
    return records


def create_open_clip_preprocessors(
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> tuple[ImageTransform, ImageTransform, TextTokenizer]:
    """Create OpenCLIP train/validation image transforms and tokenizer."""
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return preprocess_train, preprocess_val, tokenizer


class CLIPLectureDataset(Dataset):
    """Load aligned lecture frame/text pairs for OpenCLIP fine-tuning."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str | None = None,
        preprocess: ImageTransform | None = None,
        tokenizer: TextTokenizer | None = None,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        is_train: bool | None = None,
    ) -> None:
        """Initialize the dataset and OpenCLIP preprocessing components."""
        self.records = load_pair_records(dataset_path, split=split)
        self.split = split

        if preprocess is None or tokenizer is None:
            preprocess_train, preprocess_val, created_tokenizer = create_open_clip_preprocessors(
                model_name=model_name,
                pretrained=pretrained,
            )
            if preprocess is None:
                train_mode = split == "train" if is_train is None else is_train
                preprocess = preprocess_train if train_mode else preprocess_val
            if tokenizer is None:
                tokenizer = created_tokenizer

        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the number of pair records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one processed OpenCLIP image/text item."""
        record = self.records[index]
        frame_path = Path(str(record.get("frame_path", "")))
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame image not found: {frame_path}")

        try:
            image = Image.open(frame_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read frame image: {frame_path}") from exc

        text = str(record.get("text", ""))
        text_tokens = self.tokenizer([text]).squeeze(0)

        return {
            "image": self.preprocess(image),
            "text_tokens": text_tokens,
            "timestamp": float(record.get("timestamp", 0.0)),
            "video": str(record.get("video", "")),
            "segment_id": int(record.get("segment_id", -1)),
            "frame_path": str(frame_path),
            "text": text,
        }


def collate_clip_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate CLIPLectureDataset items into a batch."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "text_tokens": torch.stack([item["text_tokens"] for item in batch]).long(),
        "timestamp": torch.tensor([item["timestamp"] for item in batch], dtype=torch.float32),
        "video": [item["video"] for item in batch],
        "segment_id": torch.tensor([item["segment_id"] for item in batch], dtype=torch.long),
        "frame_path": [item["frame_path"] for item in batch],
        "text": [item["text"] for item in batch],
    }
