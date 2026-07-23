"""Three-tier IoUT topology and capped-source-level feasibility graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from physics_models.communication import (
    is_link_feasible,
    shannon_capacity,
    transmission_loss,
    wenz_noise_level,
)


NodeKey = tuple[str, int, str, int]


@dataclass(frozen=True)
class LinkInfo:
    distance: float
    SL_min: float
    TL: float
    NL: float
    R_bps: float


class Topology3D:
    """Sensors are fixed; fog nodes are quasi-static within a round."""

    def __init__(self, net_cfg, acoustic_cfg, seed: int = 42):
        self.net_cfg = net_cfg
        self.acoustic_cfg = acoustic_cfg
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.N = int(net_cfg.N_SENSORS)
        self.M = int(net_cfg.M_FOGS)
        self.sensor_positions = self._place(self.N, net_cfg.SENSOR_DEPTH)
        self.fog_positions = self._place(self.M, net_cfg.FOG_DEPTH)
        self.gateway_position = np.array(
            [net_cfg.AREA_X / 2.0, net_cfg.AREA_Y / 2.0, net_cfg.SURFACE_Z],
            dtype=float,
        )
        mean_speed = max(float(net_cfg.GM_MEAN_SPEED), 0.0)
        self.fog_speeds = self.rng.uniform(0.0, mean_speed, self.M)
        self.fog_headings = self.rng.uniform(-np.pi, np.pi, self.M)
        self.fog_pitches = self.rng.uniform(-0.2, 0.2, self.M)

    def _place(self, count: int, depth: tuple[float, float]) -> np.ndarray:
        return np.column_stack(
            [
                self.rng.uniform(0.0, self.net_cfg.AREA_X, count),
                self.rng.uniform(0.0, self.net_cfg.AREA_Y, count),
                self.rng.uniform(depth[0], depth[1], count),
            ]
        )

    def step_mobile_fogs(self) -> dict:
        """Advance fog/AUV positions by one configurable Gauss-Markov slot."""

        cfg = self.net_cfg
        mu = float(cfg.GM_ALPHA)
        scale = np.sqrt(max(0.0, 1.0 - mu**2))
        self.fog_speeds = np.clip(
            mu * self.fog_speeds
            + (1.0 - mu) * cfg.GM_MEAN_SPEED
            + scale * self.rng.normal(0.0, cfg.GM_SIGMA_SPEED, self.M),
            0.0,
            cfg.GM_MAX_SPEED,
        )
        self.fog_headings = (
            mu * self.fog_headings
            + (1.0 - mu) * cfg.GM_MEAN_HEADING
            + scale * self.rng.normal(0.0, cfg.GM_SIGMA_HEADING, self.M)
        )
        self.fog_pitches = np.clip(
            mu * self.fog_pitches
            + (1.0 - mu) * cfg.GM_MEAN_PITCH
            + scale * self.rng.normal(0.0, cfg.GM_SIGMA_PITCH, self.M),
            -np.pi / 2.0,
            np.pi / 2.0,
        )
        old = self.fog_positions.copy()
        dt = float(cfg.MOBILITY_DT_PER_ROUND)
        cos_pitch = np.cos(self.fog_pitches)
        self.fog_positions += np.column_stack(
            [
                dt * self.fog_speeds * cos_pitch * np.cos(self.fog_headings),
                dt * self.fog_speeds * cos_pitch * np.sin(self.fog_headings),
                dt * self.fog_speeds * np.sin(self.fog_pitches),
            ]
        )
        for idx in range(self.M):
            x, y, z = self.fog_positions[idx]
            if x < 0.0 or x > cfg.AREA_X:
                self.fog_headings[idx] = np.pi - self.fog_headings[idx]
            if y < 0.0 or y > cfg.AREA_Y:
                self.fog_headings[idx] = -self.fog_headings[idx]
            if z < cfg.FOG_DEPTH[0] or z > cfg.FOG_DEPTH[1]:
                self.fog_pitches[idx] *= -1.0
            self.fog_positions[idx] = np.clip(
                self.fog_positions[idx],
                [0.0, 0.0, cfg.FOG_DEPTH[0]],
                [cfg.AREA_X, cfg.AREA_Y, cfg.FOG_DEPTH[1]],
            )
        distances = np.linalg.norm(self.fog_positions - old, axis=1)
        return {
            "avg_move_m": float(distances.mean()) if self.M else 0.0,
            "max_move_m": float(distances.max()) if self.M else 0.0,
            "avg_speed_mps": float(self.fog_speeds.mean()) if self.M else 0.0,
        }


def build_feasibility_graph(topology: Topology3D, acoustic_cfg) -> Dict[NodeKey, LinkInfo]:
    """Build directed edges whose required source level does not exceed SLmax."""

    graph: Dict[NodeKey, LinkInfo] = {}

    def add(kind_u: str, id_u: int, pos_u, kind_v: str, id_v: int, pos_v) -> None:
        distance = float(np.linalg.norm(np.asarray(pos_u) - np.asarray(pos_v)))
        if distance <= 0.0:
            return
        feasible, sl_min = is_link_feasible(
            distance,
            acoustic_cfg.CARRIER_FREQ,
            acoustic_cfg.BANDWIDTH,
            acoustic_cfg.TARGET_SNR,
            acoustic_cfg.SL_MAX,
            acoustic_cfg.IL_LOSS,
            acoustic_cfg.SPREADING_FACTOR,
            acoustic_cfg.WIND_SPEED,
            acoustic_cfg.SHIPPING_FACTOR,
        )
        if not feasible:
            return
        graph[(kind_u, id_u, kind_v, id_v)] = LinkInfo(
            distance=distance,
            SL_min=sl_min,
            TL=transmission_loss(
                distance, acoustic_cfg.CARRIER_FREQ, acoustic_cfg.SPREADING_FACTOR
            ),
            NL=wenz_noise_level(
                acoustic_cfg.CARRIER_FREQ,
                acoustic_cfg.BANDWIDTH,
                acoustic_cfg.WIND_SPEED,
                acoustic_cfg.SHIPPING_FACTOR,
            ),
            R_bps=shannon_capacity(
                acoustic_cfg.BANDWIDTH, acoustic_cfg.TARGET_SNR
            ),
        )

    for sensor_id, sensor_pos in enumerate(topology.sensor_positions):
        for fog_id, fog_pos in enumerate(topology.fog_positions):
            add("sensor", sensor_id, sensor_pos, "fog", fog_id, fog_pos)
        add("sensor", sensor_id, sensor_pos, "gateway", 0, topology.gateway_position)
    for fog_id, fog_pos in enumerate(topology.fog_positions):
        add("fog", fog_id, fog_pos, "gateway", 0, topology.gateway_position)
        for other_id, other_pos in enumerate(topology.fog_positions):
            if fog_id != other_id:
                add("fog", fog_id, fog_pos, "fog", other_id, other_pos)
    return graph


def nearest_feasible_association(
    topology: Topology3D, graph: Dict[NodeKey, LinkInfo]
) -> Dict[int, int]:
    """Associate each sensor with its nearest fog on a feasible path to gateway."""

    association: Dict[int, int] = {}
    for sensor_id in range(topology.N):
        candidates = [
            (graph[("sensor", sensor_id, "fog", fog_id)].distance, fog_id)
            for fog_id in range(topology.M)
            if ("sensor", sensor_id, "fog", fog_id) in graph
            and ("fog", fog_id, "gateway", 0) in graph
        ]
        if candidates:
            association[sensor_id] = min(candidates)[1]
    return association


def flat_feasible_sensors(
    topology: Topology3D, graph: Dict[NodeKey, LinkInfo]
) -> List[int]:
    return [
        sensor_id
        for sensor_id in range(topology.N)
        if ("sensor", sensor_id, "gateway", 0) in graph
    ]


def build_clusters(association: Dict[int, int], fog_count: int) -> Dict[int, List[int]]:
    clusters = {fog_id: [] for fog_id in range(fog_count)}
    for sensor_id, fog_id in association.items():
        clusters[fog_id].append(sensor_id)
    return clusters


def topology_stats(topology: Topology3D, graph: Dict[NodeKey, LinkInfo]) -> dict:
    hfl = nearest_feasible_association(topology, graph)
    flat = flat_feasible_sensors(topology, graph)
    return {
        "sensors": topology.N,
        "fogs": topology.M,
        "direct_gateway_reachability": len(flat) / max(1, topology.N),
        "feasible_fog_reachability": len(hfl) / max(1, topology.N),
        "feasible_edges": len(graph),
    }
