"""Fine-tune OpenCLIP ViT-B/32 on lecture frame/transcript pairs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import open_clip
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.loader import (
    CLIPLectureDataset,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    collate_clip_batch,
)
from src.training.loss import InfoNCELoss
from src.utils.model_utils import resolve_device

logger = logging.getLogger(__name__)


def create_cosine_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Create a cosine learning-rate scheduler with linear warmup."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

    return LambdaLR(optimizer, lr_lambda)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a batch to the target device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def encode_batch(model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run OpenCLIP encoders and return image/text embeddings."""
    image_embeddings = model.encode_image(batch["image"])
    text_embeddings = model.encode_text(batch["text_tokens"])
    return image_embeddings, text_embeddings


def run_epoch(
    model: nn.Module,
    criterion: InfoNCELoss,
    dataloader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    epoch: int = 0,
) -> float:
    """Run one train or validation epoch and return mean loss."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(dataloader, desc=f"{'train' if is_training else 'val'} epoch {epoch}", leave=False)

    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(raw_batch, device)

        with torch.set_grad_enabled(is_training):
            image_embeddings, text_embeddings = encode_batch(model, batch)
            loss = criterion(image_embeddings, text_embeddings)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            logger.debug(
                "epoch=%s step=%s loss=%.6f lr=%.8f",
                epoch,
                step,
                loss.item(),
                learning_rate,
            )

        total_loss += float(loss.detach().cpu())
        total_steps += 1
        progress.set_postfix(loss=f"{total_loss / total_steps:.4f}")

    if total_steps == 0:
        raise ValueError("Dataloader produced no batches")
    return total_loss / total_steps


def save_checkpoint(
    model: nn.Module,
    criterion: InfoNCELoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    output_dir: str | Path,
    epoch: int,
    metrics: dict[str, float],
    model_name: str,
    pretrained: str,
) -> Path:
    """Save an OpenCLIP checkpoint plus training state."""
    checkpoint_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "model.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "pretrained": pretrained,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint_file,
    )
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return checkpoint_dir


def train(
    dataset: str | Path,
    epochs: int,
    batch_size: int,
    output: str | Path,
    device: str,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    warmup_ratio: float = 0.1,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> Path:
    """Fine-tune OpenCLIP and return the best checkpoint path."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for in-batch negatives")

    torch_device = resolve_device(device)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Could not create OpenCLIP model {model_name!r} with pretrained={pretrained!r}") from exc

    model.to(torch_device)
    criterion = InfoNCELoss().to(torch_device)

    train_dataset = CLIPLectureDataset(
        dataset,
        split="train",
        preprocess=preprocess_train,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
        is_train=True,
    )
    val_dataset = CLIPLectureDataset(
        dataset,
        split="val",
        preprocess=preprocess_val,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
        is_train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_clip_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch,
        drop_last=False,
    )

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = create_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best_val_loss = float("inf")
    best_checkpoint: Path | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, criterion, train_loader, torch_device, optimizer, scheduler, epoch)
        with torch.no_grad():
            val_loss = run_epoch(model, criterion, val_loader, torch_device, optimizer=None, epoch=epoch)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "temperature": float(criterion.temperature.detach().cpu()),
        }
        checkpoint = save_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_path,
            epoch=epoch,
            metrics=metrics,
            model_name=model_name,
            pretrained=pretrained,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} checkpoint={checkpoint}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = checkpoint
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs; best_val_loss={best_val_loss:.6f}")
                break

    if best_checkpoint is None:
        raise RuntimeError("Training finished without creating a checkpoint")
    return best_checkpoint


def main() -> None:
    """Run OpenCLIP fine-tuning from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP for VideoMind lecture retrieval.")
    parser.add_argument("--dataset", default="data/pairs", help="Pair directory or JSON dataset path.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--output", default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL, help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default=DEFAULT_CLIP_PRETRAINED, help="OpenCLIP pretrained tag.")
    args = parser.parse_args()

    best = train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output=args.output,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
