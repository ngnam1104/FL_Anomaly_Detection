"""Lightweight symmetric autoencoder used by every sensor."""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dims: tuple[int, int] = (16, 8)):
        super().__init__()
        h1, h2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def reconstruction_errors(model: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.sum((samples - model(samples)) ** 2, dim=1)
