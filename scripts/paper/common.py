"""Shared result loading and plotting helpers for paper figures."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("results/.matplotlib").resolve())
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = [
    "centralized",
    "fedavg",
    "fedprox",
    "hfl-nocoop",
    "hfl-selective",
    "hfl-nearest",
]
METHOD_LABELS = {
    "centralized": "Centralised",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "hfl-nocoop": "HFL-NoCoop",
    "hfl-selective": "HFL-Selective",
    "hfl-nearest": "HFL-Nearest",
}
METHOD_COLORS = {
    "centralized": "#7f7f7f",
    "fedavg": "#4c78a8",
    "fedprox": "#f58518",
    "hfl-nocoop": "#54a24b",
    "hfl-selective": "#e45756",
    "hfl-nearest": "#b279a2",
}


def load_runs(results_root: Path, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[dict] = []
    curves: list[dict] = []
    for path in results_root.rglob("metrics.json"):
        with path.open(encoding="utf-8") as handle:
            bundle = json.load(handle)
        metadata = bundle["metadata"]
        if metadata.get("scenario") != scenario:
            continue
        rounds = bundle.get("rounds", [])
        if not rounds:
            continue
        topology = metadata["topology"]
        final = rounds[-1]
        base = {
            "dataset": metadata["dataset"],
            "method": metadata["baseline"],
            "seed": int(metadata["seed"]),
            "N": int(topology["sensors"]),
            "M": int(topology["fogs"]),
            "rho_s": float(metadata["learning_config"]["RHO_S"]),
            "alpha": metadata.get("partition_alpha"),
        }
        runs.append(
            {
                **base,
                "participation": final["participation"],
                "f1": final["f1"],
                "pa_f1": final["pa_f1"],
                "energy_j": final.get(
                    "e_cumulative_total_j", final["e_cumulative_comm_j"]
                ),
                "communication_energy_j": final["e_cumulative_comm_j"],
                "energy_per_sensor_j": final.get(
                    "e_cumulative_total_j", final["e_cumulative_comm_j"]
                )
                / max(1, topology["sensors"]),
                "latency_s": final["latency_cumulative_s"],
                "direct_reachability": topology["direct_gateway_reachability"],
                "fog_reachability": topology["feasible_fog_reachability"],
                "e_sensor_upload_j": sum(
                    row["e_sensor_upload_j"] for row in rounds
                ),
                "e_f2f_j": sum(row["e_f2f_j"] for row in rounds),
                "e_f2g_j": sum(row["e_f2g_j"] for row in rounds),
            }
        )
        for row in rounds:
            curves.append({**base, **row})
    if not runs:
        raise FileNotFoundError(
            f"No completed scenario={scenario!r} runs under {results_root}"
        )
    return pd.DataFrame(runs), pd.DataFrame(curves)


def ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"])
    return [method for method in METHOD_ORDER if method in present]


def save_figure(fig, output: Path, stem: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
