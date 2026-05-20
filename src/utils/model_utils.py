"""Shared model loading utility helpers."""

from __future__ import annotations

from pathlib import Path

import torch


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
