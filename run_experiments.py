"""Experiment matrix for Fig. 4-8 and Table III-IV of Omeke et al. (2026)."""

from __future__ import annotations

import argparse
import os

from main import parse_args as parse_single_args
from main import run_experiment


SEEDS = (42, 43, 44)
HFL_METHODS = ("hfl-nocoop", "hfl-selective", "hfl-nearest")
SCALABILITY_METHODS = ("fedprox",) + HFL_METHODS
COMPRESSION_METHODS = ("fedavg", "fedprox", "hfl-nocoop", "hfl-nearest")
REAL_METHODS = ("centralized", "fedavg", "fedprox") + HFL_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper experiment scenarios")
    parser.add_argument(
        "--suite",
        choices=("scalability", "compression", "noniid", "real", "all"),
        default="scalability",
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--quick", action="store_true", help="One seed, two rounds and small data"
    )
    return parser.parse_args()


def _base_args(cli: argparse.Namespace, scenario: str):
    args = parse_single_args([])
    args.data_root = type(args.data_root)(cli.data_root)
    args.output_root = type(args.output_root)(cli.output_root)
    args.workers = cli.workers
    args.verbose = False
    args.scenario = scenario
    if cli.quick:
        args.local_epochs = 1
        args.batch_size = 256
    return args


def _run_synthetic(
    cli: argparse.Namespace,
    *,
    scenario: str,
    sensors: int,
    method: str,
    seed: int,
    rho_s: float = 0.05,
    alpha: float = 1.0,
) -> None:
    args = _base_args(cli, scenario)
    args.dataset = "synthetic"
    args.sensors = sensors
    args.fogs = max(1, sensors // 10)
    args.baseline = method
    args.seed = seed
    args.rounds = 2 if cli.quick else 20
    args.samples_per_sensor = 24 if cli.quick else 128
    args.rho_s = rho_s
    args.dirichlet_alpha = alpha
    run_experiment(args)


def run_scalability(cli: argparse.Namespace) -> None:
    """Fig. 4, Fig. 5, Fig. 6(a), and Table III."""

    seeds = SEEDS[:1] if cli.quick else SEEDS
    # N=100 is included because it appears in Fig. 5 and Table III even though
    # the surrounding prose highlights N={50,150,200}.
    scales = (50,) if cli.quick else (50, 100, 150, 200)
    methods = ("hfl-selective",) if cli.quick else SCALABILITY_METHODS
    for sensors in scales:
        for method in methods:
            for seed in seeds:
                _run_synthetic(
                    cli,
                    scenario="scalability",
                    sensors=sensors,
                    method=method,
                    seed=seed,
                )


def run_compression(cli: argparse.Namespace) -> None:
    """Fig. 6(b): compressed rho=0.05 versus uncompressed FP32 rho=1."""

    seeds = SEEDS[:1] if cli.quick else SEEDS
    methods = ("hfl-nocoop",) if cli.quick else COMPRESSION_METHODS
    for method in methods:
        for rho_s in (0.05, 1.0):
            for seed in seeds:
                _run_synthetic(
                    cli,
                    scenario="compression",
                    sensors=100,
                    method=method,
                    seed=seed,
                    rho_s=rho_s,
                )


def run_noniid(cli: argparse.Namespace) -> None:
    """Fig. 7: Dirichlet alpha=0.1 and near-IID alpha=1e4."""

    seeds = SEEDS[:1] if cli.quick else SEEDS
    methods = ("hfl-selective",) if cli.quick else SCALABILITY_METHODS
    for method in methods:
        for alpha in (0.1, 1.0e4):
            for seed in seeds:
                _run_synthetic(
                    cli,
                    scenario="noniid",
                    sensors=100,
                    method=method,
                    seed=seed,
                    alpha=alpha,
                )


def run_real(cli: argparse.Namespace) -> None:
    """Fig. 8 and Table IV."""

    seeds = SEEDS[:1] if cli.quick else SEEDS
    for dataset in ("SMAP", "MSL"):
        for method in REAL_METHODS:
            for seed in seeds:
                args = _base_args(cli, "real")
                args.dataset = dataset
                args.sensors = 100
                args.fogs = 10
                args.baseline = method
                args.seed = seed
                args.rounds = 2 if cli.quick else 30
                run_experiment(args)


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.suite in {"scalability", "all"}:
        run_scalability(cli_args)
    if cli_args.suite in {"compression", "all"}:
        run_compression(cli_args)
    if cli_args.suite in {"noniid", "all"}:
        run_noniid(cli_args)
    if cli_args.suite in {"real", "all"}:
        run_real(cli_args)
