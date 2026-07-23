"""Top-K sparsification, 8-bit quantisation, and error feedback."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import torch


@dataclass
class SparseQuantizedUpdate:
    indices: torch.Tensor
    values: torch.Tensor
    scale: float
    length: int

    @property
    def payload_bits(self) -> int:
        index_bits = max(1, ceil(log2(max(2, self.length))))
        return int(self.indices.numel() * (8 + index_bits))

    def decompress(self) -> torch.Tensor:
        dense = torch.zeros(self.length, dtype=torch.float32)
        if self.indices.numel():
            dense[self.indices.long()] = self.values.float() * self.scale
        return dense


class ErrorFeedbackTopK:
    def __init__(self, length: int, ratio: float = 0.05):
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Top-K ratio must be in (0, 1]")
        self.length = int(length)
        self.ratio = float(ratio)
        self.error = torch.zeros(self.length, dtype=torch.float32)

    def compress(self, update: torch.Tensor) -> SparseQuantizedUpdate:
        vector = update.detach().cpu().float().reshape(-1)
        if vector.numel() != self.length:
            raise ValueError(f"Expected {self.length} values, got {vector.numel()}")
        corrected = vector + self.error
        k = min(self.length, max(1, ceil(self.ratio * self.length)))
        indices = torch.topk(corrected.abs(), k, sorted=False).indices
        selected = corrected[indices]
        max_abs = float(selected.abs().max()) if selected.numel() else 0.0
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        quantized = torch.clamp(torch.round(selected / scale), -127, 127).to(torch.int8)
        payload = SparseQuantizedUpdate(indices.int(), quantized, scale, self.length)
        self.error = corrected - payload.decompress()
        return payload


def flatten_state(state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    metadata = []
    vectors = []
    for name, value in state.items():
        flat = value.detach().cpu().float().reshape(-1)
        vectors.append(flat)
        metadata.append((name, value.shape, flat.numel()))
    return torch.cat(vectors), metadata


def unflatten_state(
    vector: torch.Tensor, metadata: list[tuple[str, torch.Size, int]]
) -> dict[str, torch.Tensor]:
    state = {}
    offset = 0
    for name, shape, count in metadata:
        state[name] = vector[offset : offset + count].reshape(shape).clone()
        offset += count
    return state
