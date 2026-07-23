"""Population-weighted aggregation for flat and hierarchical anomaly FL."""

from __future__ import annotations

from typing import Dict, Iterable

import torch


State = Dict[str, torch.Tensor]


def weighted_average(states: Iterable[State], weights: Iterable[float]) -> State:
    states = list(states)
    weights = [float(weight) for weight in weights]
    if not states:
        raise ValueError("At least one state is required")
    total = sum(weights)
    weights = (
        [weight / total for weight in weights]
        if total > 0.0
        else [1.0 / len(states)] * len(states)
    )
    output: State = {}
    for name in states[0]:
        reference = states[0][name]
        value = torch.zeros_like(reference, dtype=torch.float32)
        for state, weight in zip(states, weights):
            value.add_(state[name].detach().float(), alpha=weight)
        output[name] = value.to(reference.dtype)
    return output


def apply_weighted_deltas(global_state: State, deltas: list[State], counts: list[int]) -> State:
    if not deltas:
        return {name: value.detach().clone() for name, value in global_state.items()}
    mean_delta = weighted_average(deltas, counts)
    return {
        name: global_state[name].detach().clone() + mean_delta[name]
        for name in global_state
    }


def blend_states(own: State, neighbour: State, neighbour_weight: float) -> State:
    return weighted_average([own, neighbour], [1.0 - neighbour_weight, neighbour_weight])
