"""Underwater acoustic propagation, ambient noise, and receiver SNR."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def thorp_absorption(f_khz: float) -> float:
    """Thorp absorption coefficient alpha(f), in dB/km."""

    f2 = float(f_khz) ** 2
    return float(
        0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 2.75e-4 * f2
        + 0.003
    )


def transmission_loss(d_m: float, f_khz: float, spreading: float = 1.5) -> float:
    """Eq. (1): practical spreading plus frequency-dependent absorption."""

    if d_m <= 0.0:
        return 0.0
    return float(
        10.0 * spreading * np.log10(d_m)
        + thorp_absorption(f_khz) * d_m / 1000.0
    )


def wenz_noise_components(
    f_khz: float, wind_speed: float = 5.0, shipping_factor: float = 0.5
) -> dict[str, float]:
    """Wenz turbulence, shipping, wind, and thermal PSD components."""

    if f_khz <= 0.0:
        raise ValueError("Carrier frequency must be positive")
    f = float(f_khz)
    return {
        "turbulence": 17.0 - 30.0 * np.log10(f),
        "shipping": (
            40.0
            + 20.0 * (shipping_factor - 0.5)
            + 26.0 * np.log10(f)
            - 60.0 * np.log10(f + 0.03)
        ),
        "wind": (
            50.0
            + 7.5 * np.sqrt(wind_speed)
            + 20.0 * np.log10(f)
            - 40.0 * np.log10(f + 0.4)
        ),
        "thermal": -15.0 + 20.0 * np.log10(f),
    }


def wenz_noise_level(
    f_khz: float,
    bandwidth_hz: float,
    wind_speed: float = 5.0,
    shipping_factor: float = 0.5,
) -> float:
    """Eq. (3): integrate the total Wenz PSD over receiver bandwidth."""

    if bandwidth_hz <= 0.0:
        raise ValueError("Receiver bandwidth must be positive")
    components = wenz_noise_components(f_khz, wind_speed, shipping_factor)
    total_psd = sum(10.0 ** (value / 10.0) for value in components.values())
    return float(10.0 * np.log10(total_psd) + 10.0 * np.log10(bandwidth_hz))


def snr_passive(source_level: float, loss: float, noise: float, implementation_loss: float = 2.0) -> float:
    """Eq. (4): passive-sonar receiver SNR in dB."""

    return float(source_level - loss - noise - implementation_loss)


def shannon_capacity(bandwidth_hz: float, target_snr_db: float) -> float:
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    return float(bandwidth_hz * np.log2(1.0 + snr_linear))


def min_source_level(
    d_m: float,
    f_khz: float,
    bandwidth_hz: float,
    target_snr_db: float,
    implementation_loss: float = 2.0,
    spreading: float = 1.5,
    wind_speed: float = 5.0,
    shipping_factor: float = 0.5,
) -> float:
    """Eq. (5): source level needed to operate at the target SNR."""

    return float(
        target_snr_db
        + transmission_loss(d_m, f_khz, spreading)
        + wenz_noise_level(f_khz, bandwidth_hz, wind_speed, shipping_factor)
        + implementation_loss
    )


def is_link_feasible(
    d_m: float,
    f_khz: float,
    bandwidth_hz: float,
    target_snr_db: float,
    source_level_cap: float,
    implementation_loss: float = 2.0,
    spreading: float = 1.5,
    wind_speed: float = 5.0,
    shipping_factor: float = 0.5,
) -> Tuple[bool, float]:
    """Eq. (6): a directed edge exists iff SLmin does not exceed SLmax."""

    required = min_source_level(
        d_m,
        f_khz,
        bandwidth_hz,
        target_snr_db,
        implementation_loss,
        spreading,
        wind_speed,
        shipping_factor,
    )
    return required <= source_level_cap, required
