"""Baseline configuration from Table II of Omeke et al. (2026)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NetworkConfig:
    """Three-tier gateway--fog(AUV)--sensor deployment."""

    N_SENSORS: int = 100
    M_FOGS: int = 10
    AREA_X: float = 2000.0
    AREA_Y: float = 2000.0
    MAX_DEPTH: float = 1000.0
    SENSOR_DEPTH: tuple[float, float] = (500.0, 1000.0)
    FOG_DEPTH: tuple[float, float] = (100.0, 400.0)
    SURFACE_Z: float = 0.0

    # Fog/AUV Gauss-Markov mobility; sensors remain stationary.
    MOBILITY_ENABLED: bool = True
    MOBILITY_DT_PER_ROUND: float = 1.0
    GM_ALPHA: float = 0.7
    GM_MEAN_SPEED: float = 1.5
    GM_MAX_SPEED: float = 5.0
    GM_MEAN_HEADING: float = 0.0
    GM_MEAN_PITCH: float = 0.0
    GM_SIGMA_SPEED: float = 0.5
    GM_SIGMA_HEADING: float = 0.3
    GM_SIGMA_PITCH: float = 0.1


@dataclass
class AcousticChannelConfig:
    SOUND_SPEED: float = 1500.0
    CARRIER_FREQ: float = 12.0
    BANDWIDTH: float = 4000.0
    SPREADING_FACTOR: float = 1.5
    WIND_SPEED: float = 5.0
    SHIPPING_FACTOR: float = 0.5
    TARGET_SNR: float = 10.0
    IL_LOSS: float = 2.0
    SL_MAX: float = 140.0


@dataclass
class EnergyConfig:
    E_INIT: float = 500.0
    E_MIN: float = 0.0
    ETA_EA: float = 0.25
    P_C_TX: float = 0.05
    P_C_RX: float = 0.03
    RHO_WATER: float = 1025.0
    # Effective device-level compute energy. Calibrated from a 10 W embedded
    # module budget divided by the configured 36 GFLOP/s CPU throughput:
    # 10 / (1.5e9 * 6 * 4) ~= 2.78e-10 J/FLOP.
    EPSILON_OP: float = 2.8e-10
    F_CPU: float = 1.5e9
    N_CORES: int = 6
    FLOPS_PER_CYCLE: float = 4.0


@dataclass
class LearningConfig:
    FEATURE_DIM: int = 32
    HIDDEN_DIMS: tuple[int, int] = (16, 8)
    LOCAL_EPOCHS: int = 5
    LOCAL_BATCH_SIZE: int = 32
    LOCAL_LR: float = 0.01
    # Keep the paper's SGD learning rate while preventing rare, very large
    # standardized telemetry values from producing non-finite local updates.
    MAX_GRAD_NORM: float = 5.0
    FEDPROX_MU: float = 0.02
    RHO_S: float = 0.05
    QUANTIZATION_BITS: int = 8
    ANOMALY_PERCENTILE: float = 99.0
    COOP_THRESHOLD_MULTIPLIER: float = 0.75
    COOP_NEIGHBOR_WEIGHT_NEAREST: float = 0.30
    COOP_NEIGHBOR_WEIGHT_SELECTIVE: float = 0.20
    LAMBDA_E: float = 5e-4
    LAMBDA_TAU: float = 1e-3


network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
learning_cfg = LearningConfig()

# Compatibility alias for small reusable helpers that previously imported fed_cfg.
fed_cfg = learning_cfg


def baseline_parameters() -> dict:
    """Return a JSON-serialisable Table-II configuration snapshot."""

    return {
        "network": asdict(network_cfg),
        "acoustic": asdict(acoustic_cfg),
        "energy": asdict(energy_cfg),
        "learning": asdict(learning_cfg),
    }


def table_ii_rows() -> list[dict[str, str]]:
    """Exact presentation rows from Table II (not every simulator option)."""

    return [
        {"Parameter": "Lx, Ly", "Meaning": "Horizontal area", "Baseline": "2000 x 2000 m"},
        {"Parameter": "H", "Meaning": "Max depth", "Baseline": "1000 m"},
        {"Parameter": "N", "Meaning": "Sensor nodes", "Baseline": "100 (varied: 50-200)"},
        {"Parameter": "M", "Meaning": "Fog aggregators", "Baseline": "10 (varied: 5-20)"},
        {"Parameter": "zs_min, zs_max", "Meaning": "Sensor depth range", "Baseline": "500-1000 m"},
        {"Parameter": "zf_min, zf_max", "Meaning": "Fog depth range", "Baseline": "100-400 m"},
        {"Parameter": "f", "Meaning": "Carrier frequency", "Baseline": "12 kHz"},
        {"Parameter": "B", "Meaning": "Receiver bandwidth", "Baseline": "4 kHz"},
        {"Parameter": "k", "Meaning": "Spreading factor", "Baseline": "1.5"},
        {"Parameter": "cs", "Meaning": "Sound speed", "Baseline": "1500 m/s"},
        {"Parameter": "w, s", "Meaning": "Wind speed / shipping", "Baseline": "5 m/s, 0.5"},
        {"Parameter": "gamma_tgt", "Meaning": "Target SNR", "Baseline": "10 dB"},
        {"Parameter": "IL", "Meaning": "Implementation loss", "Baseline": "2 dB"},
        {"Parameter": "SLmax", "Meaning": "Source-level cap", "Baseline": "140 dB re 1 uPa @ 1 m"},
        {"Parameter": "eta_ea", "Meaning": "Electro-acoustic efficiency", "Baseline": "0.25"},
        {"Parameter": "Pc,tx, Pc,rx", "Meaning": "Circuit power", "Baseline": "50 mW, 30 mW"},
        {"Parameter": "Einit", "Meaning": "Initial battery / sensor", "Baseline": "500 J"},
        {"Parameter": "D", "Meaning": "Feature dimension", "Baseline": "32"},
        {"Parameter": "AE structure", "Meaning": "Hidden layers", "Baseline": "[32, 16, 8, 16, 32]"},
        {"Parameter": "d", "Meaning": "Model parameters", "Baseline": "about 1350 (exactly 1352)"},
        {"Parameter": "E", "Meaning": "Local epochs / round", "Baseline": "5"},
        {"Parameter": "T", "Meaning": "FL rounds", "Baseline": "20 synthetic, 30 real"},
        {"Parameter": "eta", "Meaning": "Learning rate", "Baseline": "0.01"},
    ]
