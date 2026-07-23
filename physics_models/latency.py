"""Per-link and synchronous-round latency from Eq. (21)."""

from __future__ import annotations


def comm_delay(bits: float, rate_bps: float, distance_m: float, sound_speed: float = 1500.0) -> float:
    if rate_bps <= 0.0:
        return float("inf")
    return float(bits / rate_bps + distance_m / sound_speed)


def comp_delay(
    total_flops: float,
    f_cpu: float,
    cores: int,
    flops_per_cycle: float,
) -> float:
    throughput = f_cpu * cores * flops_per_cycle
    return float(total_flops / throughput) if throughput > 0.0 else float("inf")


def round_latency(link_delays: list[float], max_local_compute: float) -> float:
    """Paper Eq. (21): slowest parallel link plus slowest local computation."""

    return (max(link_delays) if link_delays else 0.0) + max_local_compute
