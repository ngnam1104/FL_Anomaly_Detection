"""SNR-driven communication and computation energy models."""

from __future__ import annotations

import numpy as np


def acoustic_power_watts(
    source_level_db: float, rho_w: float = 1025.0, sound_speed: float = 1500.0
) -> float:
    p_ref = 1e-6
    return (
        4.0
        * np.pi
        * p_ref**2
        / (rho_w * sound_speed)
        * 10.0 ** (source_level_db / 10.0)
    )


def e_tx(
    bits: float,
    rate_bps: float,
    source_level_db: float,
    eta_ea: float = 0.25,
    circuit_power: float = 0.05,
    rho_w: float = 1025.0,
    sound_speed: float = 1500.0,
) -> float:
    if bits <= 0.0 or rate_bps <= 0.0:
        return 0.0
    power = acoustic_power_watts(source_level_db, rho_w, sound_speed) / eta_ea
    return float((power + circuit_power) * bits / rate_bps)


def e_rx(bits: float, rate_bps: float, circuit_power: float = 0.03) -> float:
    return float(circuit_power * bits / rate_bps) if rate_bps > 0.0 else 0.0


def e_comp(total_flops: float, epsilon_op: float) -> float:
    """Eq. (8): computation energy is epsilon_op times operation count."""

    return float(epsilon_op * total_flops)
