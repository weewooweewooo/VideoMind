"""Contrastive losses for OpenCLIP lecture fine-tuning."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE loss for image/text embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        """Create a learnable positive temperature initialized to the given value."""
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive learned temperature scalar."""
        return self.log_temperature.exp().clamp(min=1e-6)

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute symmetric image-to-text and text-to-image contrastive loss."""
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("image_embeddings and text_embeddings must be rank-2 tensors")
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                "image_embeddings and text_embeddings must have identical shape, "
                f"got {tuple(image_embeddings.shape)} and {tuple(text_embeddings.shape)}"
            )
        if image_embeddings.shape[0] < 2:
            raise ValueError("InfoNCE requires at least two samples for in-batch negatives")

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
