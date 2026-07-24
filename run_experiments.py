"""Experiment matrix for Fig. 4-8 and Table III-IV of Omeke et al. (2026)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from main import parse_args as parse_single_args
from main import run_experiment
from anomaly_detection.simulator import FLAT_FAILED_UPLOAD_POLICY


SEEDS = (42, 43, 44)
HFL_METHODS = ("hfl-nocoop", "hfl-selective", "hfl-nearest")
SCALABILITY_METHODS = ("fedprox",) + HFL_METHODS
CONVERGENCE_EXTRA_METHODS = ("centralized", "fedavg")
COMPRESSION_METHODS = ("fedavg", "fedprox", "hfl-nocoop", "hfl-nearest")
# Run the federated methods first; the centralized oracle is intentionally last.
REAL_METHODS = ("fedavg", "fedprox") + HFL_METHODS + ("centralized",)
REAL_DATASETS = ("SMD", "SMAP", "MSL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper experiment scenarios")
    parser.add_argument(
        "--suite",
        choices=(
            "convergence",
            "scalability",
            "compression",
            "noniid",
            "real",
            "all",
        ),
        default="scalability",
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--centralized-torch-threads",
        type=int,
        default=None,
        help="PyTorch CPU threads for centralized training only.",
    )
    parser.add_argument(
        "--parallel-backend", choices=("auto", "process", "thread"), default="auto"
    )
    parser.add_argument(
        "--quick", action="store_true", help="One seed, two rounds and small data"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run even when a matching completed summary exists.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _base_args(cli: argparse.Namespace, scenario: str):
    args = parse_single_args([])
    args.data_root = type(args.data_root)(cli.data_root)
    args.output_root = type(args.output_root)(cli.output_root)
    args.workers = cli.workers
    args.torch_threads = getattr(cli, "torch_threads", 1)
    args.parallel_backend = getattr(cli, "parallel_backend", "auto")
    args.verbose = False
    args.scenario = scenario
    if cli.quick:
        args.local_epochs = 1
        args.batch_size = 256
    return args


def _completed_run_path(args) -> Path:
    """Return the deterministic result path without loading the dataset."""

    fogs = int(args.fogs if args.fogs is not None else max(1, args.sensors // 10))
    rho_s = 0.05 if args.rho_s is None else float(args.rho_s)
    alpha = (
        "na"
        if args.dirichlet_alpha is None
        else str(float(args.dirichlet_alpha))
    )
    return (
        args.output_root
        / args.scenario
        / args.dataset.lower()
        / f"N_{args.sensors}_M_{fogs}"
        / args.baseline
        / f"rho_{rho_s:g}_alpha_{alpha}"
        / f"seed_{args.seed}"
    )


def _is_completed(args) -> bool:
    """A run is resumable only when every recorded metric is finite."""

    run_path = _completed_run_path(args)
    summary_path = run_path / "summary.json"
    metrics_path = run_path / "metrics.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        bundle = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    final = summary.get("final", {})
    history = bundle.get("rounds", [])
    finite_round_fields = (
        "train_loss",
        "f1",
        "pa_f1",
        "threshold",
        "test_reconstruction_loss",
        "e_cumulative_total_j",
        "latency_cumulative_s",
    )
    finite_summary_fields = (
        "best_f1",
        "best_pa_f1",
        "total_communication_energy_j",
        "total_modelled_energy_j",
        "total_latency_s",
    )
    structurally_complete = (
        summary.get("rounds") == args.rounds
        and str(summary.get("dataset", "")).upper() == str(args.dataset).upper()
        and summary.get("baseline") == args.baseline
        and summary.get("seed") == args.seed
        and final.get("round") == args.rounds
        and len(history) == args.rounds
        and bundle.get("metadata", {}).get("scenario") == args.scenario
    )
    if args.baseline in {"fedavg", "fedprox"}:
        structurally_complete = (
            structurally_complete
            and summary.get("flat_failed_upload_policy")
            == FLAT_FAILED_UPLOAD_POLICY
            and bundle.get("metadata", {}).get("flat_failed_upload_policy")
            == FLAT_FAILED_UPLOAD_POLICY
        )
    if not structurally_complete:
        return False
    values = [summary.get(field) for field in finite_summary_fields]
    values.extend(
        row.get(field) for row in history for field in finite_round_fields
    )
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _run_or_skip(cli: argparse.Namespace, args) -> None:
    if getattr(cli, "resume", True) and _is_completed(args):
        print(f"SKIP completed: {_completed_run_path(args)}")
        return
    run_experiment(args)


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
    if method == "centralized":
        centralized_threads = getattr(cli, "centralized_torch_threads", None)
        if centralized_threads:
            args.torch_threads = centralized_threads
    _run_or_skip(cli, args)


def run_convergence(cli: argparse.Namespace) -> None:
    """Fig. 4 additions: Centralised and FedAvg at N={150, 200}."""

    seeds = SEEDS[:1] if cli.quick else SEEDS
    scales = (150,) if cli.quick else (150, 200)
    for sensors in scales:
        for method in CONVERGENCE_EXTRA_METHODS:
            for seed in seeds:
                _run_synthetic(
                    cli,
                    scenario="convergence",
                    sensors=sensors,
                    method=method,
                    seed=seed,
                )


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
    for dataset in REAL_DATASETS:
        for method in REAL_METHODS:
            for seed in seeds:
                args = _base_args(cli, "real")
                args.dataset = dataset
                # The paper maps each benchmark's source entities (10 SMD
                # machines, 55 SMAP channels, 27 MSL channels) onto the same
                # N=100, M=10 physical topology.
                args.sensors = 100
                args.fogs = 10
                args.baseline = method
                centralized_threads = getattr(cli, "centralized_torch_threads", None)
                if method == "centralized" and centralized_threads:
                    args.torch_threads = centralized_threads
                args.seed = seed
                args.dirichlet_alpha = None
                args.rounds = 2 if cli.quick else 30
                _run_or_skip(cli, args)


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.suite in {"convergence", "all"}:
        run_convergence(cli_args)
    if cli_args.suite in {"scalability", "all"}:
        run_scalability(cli_args)
    if cli_args.suite in {"compression", "all"}:
        run_compression(cli_args)
    if cli_args.suite in {"noniid", "all"}:
        run_noniid(cli_args)
    if cli_args.suite in {"real", "all"}:
        run_real(cli_args)
