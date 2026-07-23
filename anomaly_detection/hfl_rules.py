"""Deterministic anomaly-HFL fog cooperation rules from Section V-B."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_mean_cluster_size(cluster_sizes: Dict[int, int]) -> float:
    non_empty = [size for size in cluster_sizes.values() if size > 0]
    return float(np.mean(non_empty)) if non_empty else 0.0


def compute_q1_fog_distance(feasibility_graph: Dict) -> float:
    distances = [
        link.distance
        for (kind_u, _, kind_v, _), link in feasibility_graph.items()
        if kind_u == "fog" and kind_v == "fog"
    ]
    return float(np.percentile(distances, 25)) if distances else float("inf")


# Backwards-compatible name retained for external notebooks.
compute_q1_relay_distance = compute_q1_fog_distance


def should_cooperate(
    cluster_size: int, mean_cluster_size: float, threshold_multiplier: float = 0.75
) -> bool:
    return cluster_size > 0 and cluster_size <= max(
        2.0, threshold_multiplier * mean_cluster_size
    )


def find_coop_partner(
    fog_id: int,
    cluster_sizes: Dict[int, int],
    feasibility_graph: Dict,
    *,
    q1_distance: Optional[float] = None,
    require_larger_cluster: bool,
) -> Optional[int]:
    candidates: list[tuple[float, int]] = []
    own_size = cluster_sizes.get(fog_id, 0)
    for other_id, other_size in cluster_sizes.items():
        if other_id == fog_id or other_size <= 0:
            continue
        if require_larger_cluster and other_size <= own_size:
            continue
        key = ("fog", fog_id, "fog", other_id)
        if key not in feasibility_graph:
            continue
        distance = feasibility_graph[key].distance
        if q1_distance is not None and distance > q1_distance:
            continue
        candidates.append((distance, other_id))
    return min(candidates)[1] if candidates else None


def select_cooperation(
    rule: str,
    cluster_sizes: Dict[int, int],
    feasibility_graph: Dict,
    threshold_multiplier: float = 0.75,
) -> Dict[int, int]:
    """Return receiver-fog -> donor-fog mappings."""

    if rule == "nocoop":
        return {}
    if rule not in {"nearest", "selective"}:
        raise ValueError(f"Unknown cooperation rule: {rule}")
    mean_size = compute_mean_cluster_size(cluster_sizes)
    q1 = compute_q1_fog_distance(feasibility_graph)
    partners: Dict[int, int] = {}
    for fog_id in sorted(cluster_sizes):
        if cluster_sizes[fog_id] <= 0:
            continue
        if rule == "selective" and not should_cooperate(
            cluster_sizes[fog_id], mean_size, threshold_multiplier
        ):
            continue
        partner = find_coop_partner(
            fog_id,
            cluster_sizes,
            feasibility_graph,
            q1_distance=q1 if rule == "selective" else None,
            require_larger_cluster=rule == "selective",
        )
        if partner is not None:
            partners[fog_id] = partner
    return partners
