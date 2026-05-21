from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import open_clip
import torch
from torch.utils.data import Dataset

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"
TextTokenizer = Callable[[list[str]], torch.Tensor]


def load_pair_records(
    dataset_path: str | Path, split: str | None = None
) -> list[dict[str, Any]]:
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
        raise FileNotFoundError(
            f"No dataset JSON files found in {path} for split={split!r}"
        )

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


def create_open_clip_tokenizer(
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> TextTokenizer:
    """Create OpenCLIP tokenizer."""
    tokenizer = open_clip.get_tokenizer(model_name)
    return tokenizer


class CLIPLectureDataset(Dataset):
    """Load lecture text pairs for text-only contrastive learning.

    Note: Image encoding disabled until frame pipeline is integrated.
    Returns blank image tensors as placeholders for now.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str | None = None,
        tokenizer: TextTokenizer | None = None,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
    ) -> None:
        """Initialize the dataset and OpenCLIP tokenizer."""
        self.records = load_pair_records(dataset_path, split=split)
        self.split = split

        if tokenizer is None:
            tokenizer = create_open_clip_tokenizer(
                model_name=model_name,
                pretrained=pretrained,
            )

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the number of pair records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one processed text-only item (image frames not yet available)."""
        record = self.records[index]
        text = str(record.get("text", ""))
        if not text:
            raise ValueError(f"Record at index {index} has no text field")

        text_tokens = self.tokenizer([text]).squeeze(0)
        image_tensor = torch.zeros(3, 224, 224)

        return {
            "image": image_tensor,
            "text_input_ids": text_tokens,
            "text_attention_mask": torch.ones_like(text_tokens),
            "video": str(record.get("video", "")),
            "start": float(record.get("start", 0.0)),
        }


def collate_clip_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate CLIPLectureDataset items into a batch."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "text_input_ids": torch.stack(
            [item["text_input_ids"] for item in batch]
        ).long(),
        "text_attention_mask": torch.stack(
            [item["text_attention_mask"] for item in batch]
        ).long(),
        "video": [item["video"] for item in batch],
        "start": torch.tensor([item["start"] for item in batch], dtype=torch.float32),
    }
