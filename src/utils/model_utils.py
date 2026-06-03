"""Shared model loading utility helpers."""

from __future__ import annotations

from pathlib import Path

import open_clip
import torch


def load_openclip_with_checkpoint(
    model_name: str,
    pretrained: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> tuple:
    """
    Creates OpenCLIP model and transforms.
    Loads checkpoint weights if checkpoint_path provided.
    Returns (model, preprocess_train, preprocess_val)
    """
    try:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not create OpenCLIP model {model_name!r} with pretrained={pretrained!r}"
        ) from exc

    if checkpoint_path is not None:
        checkpoint_file = resolve_checkpoint_file(checkpoint_path)
        state = torch.load(checkpoint_file, map_location="cpu")
        state_dict = (
            state.get("model_state_dict", state) if isinstance(state, dict) else state
        )
        model.load_state_dict(state_dict)

    model.to(device)
    return model, preprocess_train, preprocess_val


def resolve_device(device: str) -> str:
    """Resolve cpu/cuda/auto into a device string accepted by PyTorch."""
    requested = device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "npu":
        raise RuntimeError("OpenCLIP uses PyTorch and supports cpu/cuda in this project")
    return requested


def resolve_checkpoint_file(checkpoint_dir: str) -> str:
    """Resolve a checkpoint directory or file path to a saved model.pt path."""
    path = Path(checkpoint_dir)
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"OpenCLIP checkpoint not found: {path}")
    return str(path)
