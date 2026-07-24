from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.paper import (
    fig4_convergence,
    fig5_scalability,
    fig6_engineering,
    fig7_noniid,
    fig8_real,
)
from scripts.paper.tables import table_ii, table_iii, table_iv


HFL_METHODS = ("fedprox", "hfl-nocoop", "hfl-selective", "hfl-nearest")
ALL_METHODS = (
    "centralized",
    "fedavg",
    "fedprox",
    "hfl-nocoop",
    "hfl-selective",
    "hfl-nearest",
)


def _write_run(
    root: Path,
    *,
    scenario: str,
    dataset: str,
    method: str,
    sensors: int,
    seed: int,
    rho_s: float = 0.05,
    alpha: float | None = 1.0,
) -> None:
    method_index = ALL_METHODS.index(method)
    base_energy = sensors * (0.15 + 0.04 * method_index)
    if rho_s == 1.0:
        base_energy *= 8.0
    rounds = []
    for round_index in (1, 2, 3):
        rounds.append(
            {
                "round": round_index,
                "train_loss": 100.0 / (round_index + 0.25 * method_index + 1.0),
                "participation": 0.49 if method in {"fedavg", "fedprox"} else 1.0,
                "f1": 0.84 + 0.004 * method_index + 0.001 * round_index,
                "pa_f1": 0.72 + 0.012 * method_index + 0.001 * round_index,
                "e_cumulative_comm_j": base_energy * round_index / 3.0,
                "e_cumulative_total_j": base_energy * round_index / 2.8,
                "latency_cumulative_s": 0.7 * round_index,
                "e_sensor_upload_j": base_energy * 0.12,
                "e_f2f_j": (
                    base_energy * 0.08
                    if method in {"hfl-selective", "hfl-nearest"}
                    else 0.0
                ),
                "e_f2g_j": base_energy * 0.18 if method.startswith("hfl-") else 0.0,
            }
        )
    bundle = {
        "metadata": {
            "scenario": scenario,
            "dataset": dataset,
            "baseline": method,
            "seed": seed,
            "partition_alpha": alpha,
            "topology": {
                "sensors": sensors,
                "fogs": max(1, sensors // 10),
                "direct_gateway_reachability": 0.49,
                "feasible_fog_reachability": 1.0,
            },
            "learning_config": {"RHO_S": rho_s},
        },
        "rounds": rounds,
    }
    path = (
        root
        / scenario
        / dataset.lower()
        / f"N_{sensors}"
        / method
        / f"rho_{rho_s}_alpha_{alpha}"
        / f"seed_{seed}"
        / "metrics.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle), encoding="utf-8")


def _write_fixture(root: Path) -> None:
    for sensors in (50, 100, 150, 200):
        for method in HFL_METHODS:
            for seed in (42, 43):
                _write_run(
                    root,
                    scenario="scalability",
                    dataset="synthetic",
                    method=method,
                    sensors=sensors,
                    seed=seed,
                )
    for sensors in (150, 200):
        for method in ("centralized", "fedavg"):
            for seed in (42, 43):
                _write_run(
                    root,
                    scenario="convergence",
                    dataset="synthetic",
                    method=method,
                    sensors=sensors,
                    seed=seed,
                )
    for method in ("fedavg", "fedprox", "hfl-nocoop", "hfl-nearest"):
        for rho_s in (0.05, 1.0):
            for seed in (42, 43):
                _write_run(
                    root,
                    scenario="compression",
                    dataset="synthetic",
                    method=method,
                    sensors=100,
                    seed=seed,
                    rho_s=rho_s,
                )
    for method in HFL_METHODS:
        for alpha in (0.1, 1.0e4):
            for seed in (42, 43):
                _write_run(
                    root,
                    scenario="noniid",
                    dataset="synthetic",
                    method=method,
                    sensors=100,
                    seed=seed,
                    alpha=alpha,
                )
    for dataset in ("SMD", "SMAP", "MSL"):
        for method in ALL_METHODS:
            for seed in (42, 43):
                _write_run(
                    root,
                    scenario="real",
                    dataset=dataset,
                    method=method,
                    sensors=100,
                    seed=seed,
                    alpha=None,
                )


def _invoke(monkeypatch, entrypoint, results: Path, output: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper-script",
            "--results",
            str(results),
            "--output",
            str(output),
        ],
    )
    entrypoint()


def test_all_paper_figures_and_tables_render(tmp_path, monkeypatch):
    results = tmp_path / "runs"
    output = tmp_path / "paper"
    _write_fixture(results)

    for entrypoint in (
        fig4_convergence.main,
        fig5_scalability.main,
        fig6_engineering.main,
        fig7_noniid.main,
        fig8_real.main,
    ):
        _invoke(monkeypatch, entrypoint, results, output)

    table_ii(output)
    table_iii(results, output)
    table_iv(results, output)

    expected = (
        "fig4_convergence.png",
        "fig5_scalability.png",
        "fig6_engineering.png",
        "fig7_noniid.png",
        "fig8_real_benchmarks.png",
        "table_ii_parameters.csv",
        "table_ii_parameters.md",
        "table_iii_scalability.csv",
        "table_iii_scalability.md",
        "table_iv_real.csv",
        "table_iv_real.md",
    )
    for filename in expected:
        artifact = output / filename
        assert artifact.exists()
        assert artifact.stat().st_size > 100

    table_text = (output / "table_iii_scalability.md").read_text(encoding="utf-8")
    assert table_text.index("FedProx") < table_text.index("HFL-NoCoop")
    assert "±" in table_text
