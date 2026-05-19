"""Evaluate vanilla and fine-tuned OpenCLIP retrieval quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.loader import (
    CLIPLectureDataset,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    collate_clip_batch,
)


def resolve_device(device: str) -> torch.device:
    """Resolve an evaluation device."""
    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "npu":
        raise RuntimeError("OpenCLIP evaluation uses PyTorch and supports cpu/cuda in this project")
    return torch.device(requested)


def resolve_checkpoint_file(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint directory or file to a saved model.pt path."""
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"OpenCLIP checkpoint not found: {path}")
    return path


def load_open_clip_model(
    model_name: str,
    pretrained: str,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    """Create an OpenCLIP model and optionally load fine-tuned weights."""
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    except Exception as exc:
        raise RuntimeError(f"Could not create OpenCLIP model {model_name!r} with pretrained={pretrained!r}") from exc

    if checkpoint is not None:
        checkpoint_file = resolve_checkpoint_file(checkpoint)
        state = torch.load(checkpoint_file, map_location="cpu")
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def compute_embeddings(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
    """Compute normalized image/text embeddings and segment keys for a split."""
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    keys: list[tuple[str, int]] = []

    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="embedding", leave=False):
            images = raw_batch["image"].to(device)
            text_tokens = raw_batch["text_tokens"].to(device)

            image_embeddings = F.normalize(model.encode_image(images), dim=-1)
            text_embeddings = F.normalize(model.encode_text(text_tokens), dim=-1)
            image_batches.append(image_embeddings.cpu())
            text_batches.append(text_embeddings.cpu())
            keys.extend(
                (video, int(segment_id))
                for video, segment_id in zip(raw_batch["video"], raw_batch["segment_id"].tolist())
            )

    if not image_batches:
        raise ValueError("Evaluation dataloader produced no batches")
    return torch.cat(image_batches, dim=0), torch.cat(text_batches, dim=0), keys


def recall_at_k(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    segment_keys: list[tuple[str, int]],
    k: int,
) -> float:
    """Compute Recall@K where a match is correct if video and segment id match."""
    if image_embeddings.shape[0] != text_embeddings.shape[0] or image_embeddings.shape[0] != len(segment_keys):
        raise ValueError("Embedding counts and segment key counts must match")

    max_k = min(k, text_embeddings.shape[0])
    similarities = image_embeddings @ text_embeddings.T
    top_indices = similarities.topk(k=max_k, dim=1).indices
    correct = 0

    for row_index, candidates in enumerate(top_indices):
        expected = segment_keys[row_index]
        if any(segment_keys[int(candidate)] == expected for candidate in candidates):
            correct += 1
    return correct / len(segment_keys)


def evaluate(
    checkpoint: str | Path,
    dataset: str | Path = "data/pairs",
    output_path: str | Path = "data/eval_results.json",
    batch_size: int = 32,
    device: str = "auto",
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> dict[str, dict[str, float]]:
    """Evaluate vanilla and fine-tuned OpenCLIP on the test split."""
    torch_device = resolve_device(device)
    _, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    test_dataset = CLIPLectureDataset(
        dataset,
        split="test",
        preprocess=preprocess_val,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
        is_train=False,
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch,
    )

    vanilla_model = load_open_clip_model(model_name, pretrained, torch_device)
    fine_tuned_model = load_open_clip_model(model_name, pretrained, torch_device, checkpoint=checkpoint)

    vanilla_images, vanilla_texts, keys = compute_embeddings(vanilla_model, dataloader, torch_device)
    fine_images, fine_texts, fine_keys = compute_embeddings(fine_tuned_model, dataloader, torch_device)
    if fine_keys != keys:
        raise RuntimeError("Evaluation keys changed between vanilla and fine-tuned embedding passes")

    results: dict[str, dict[str, float]] = {}
    for k in (1, 5, 10):
        vanilla = recall_at_k(vanilla_images, vanilla_texts, keys, k)
        fine_tuned = recall_at_k(fine_images, fine_texts, keys, k)
        results[f"Recall@{k}"] = {
            "vanilla_clip": vanilla,
            "fine_tuned_clip": fine_tuned,
            "improvement": fine_tuned - vanilla,
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print evaluation results as a compact table."""
    print("metric | vanilla CLIP | fine-tuned CLIP | improvement")
    print("--- | ---: | ---: | ---:")
    for metric, values in results.items():
        print(
            f"{metric} | {values['vanilla_clip']:.4f} | "
            f"{values['fine_tuned_clip']:.4f} | {values['improvement']:+.4f}"
        )


def main() -> None:
    """Run retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate VideoMind OpenCLIP retrieval.")
    parser.add_argument("--model", "--checkpoint", dest="checkpoint", required=True, help="Fine-tuned checkpoint path.")
    parser.add_argument("--dataset", "--test-split", dest="dataset", default="data/pairs", help="Pair directory or test JSON.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--output", default="data/eval_results.json", help="Results JSON path.")
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL, help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default=DEFAULT_CLIP_PRETRAINED, help="OpenCLIP pretrained tag.")
    args = parser.parse_args()

    results = evaluate(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
