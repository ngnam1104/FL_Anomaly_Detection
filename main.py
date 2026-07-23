"""Windows-friendly entry point for one federated anomaly-detection run."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from anomaly_detection.data import load_real_benchmark, make_synthetic
from anomaly_detection.simulator import BASELINES, AnomalyFLSimulator
from config.settings import acoustic_cfg, energy_cfg, learning_cfg, network_cfg


class JsonEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-tier IoUT federated anomaly-detection simulator"
    )
    parser.add_argument("--dataset", choices=("synthetic", "SMD", "SMAP", "MSL"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--baseline", choices=BASELINES, default="hfl-selective")
    parser.add_argument("--sensors", type=int, default=50)
    parser.add_argument("--fogs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--parallel-backend",
        choices=("auto", "process", "thread"),
        default="auto",
        help="Local FL training backend; auto uses processes on Linux.",
    )
    parser.add_argument("--samples-per-sensor", type=int, default=128)
    parser.add_argument("--heterogeneity", type=float, default=0.35)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rho-s", type=float, default=None)
    parser.add_argument("--mobility-speed", type=float, default=None)
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--scenario", default="single")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _configure_logger(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"iout.{log_path.parent.as_posix()}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> Path:
    sensors = int(args.sensors)
    fogs = int(args.fogs if args.fogs is not None else max(1, sensors // 10))
    rounds = int(args.rounds if args.rounds is not None else (20 if args.dataset == "synthetic" else 30))
    if sensors <= 0 or fogs <= 0 or rounds <= 0:
        raise ValueError("sensors, fogs and rounds must be positive")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    net = type(network_cfg)(**asdict(network_cfg))
    learn = type(learning_cfg)(**asdict(learning_cfg))
    net.N_SENSORS = sensors
    net.M_FOGS = fogs
    net.MOBILITY_ENABLED = not args.no_mobility
    if args.mobility_speed is not None:
        net.GM_MEAN_SPEED = float(args.mobility_speed)
        net.GM_MAX_SPEED = max(net.GM_MAX_SPEED, net.GM_MEAN_SPEED)
    if args.local_epochs is not None:
        learn.LOCAL_EPOCHS = int(args.local_epochs)
    if args.batch_size is not None:
        learn.LOCAL_BATCH_SIZE = int(args.batch_size)
    if args.learning_rate is not None:
        learn.LOCAL_LR = float(args.learning_rate)
    if args.rho_s is not None:
        learn.RHO_S = float(args.rho_s)

    if args.dataset == "synthetic":
        data = make_synthetic(
            sensors,
            feature_dim=learn.FEATURE_DIM,
            samples_per_sensor=args.samples_per_sensor,
            heterogeneity=args.heterogeneity,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=args.seed,
        )
    else:
        data = load_real_benchmark(
            args.dataset,
            args.data_root,
            sensors,
            seed=args.seed,
            dirichlet_alpha=(
                args.dirichlet_alpha
                if args.dataset in {"SMAP", "MSL"}
                else None
            ),
        )

    run_dir = (
        args.output_root
        / args.scenario
        / data.name.lower()
        / f"N_{sensors}_M_{fogs}"
        / args.baseline
        / f"rho_{learn.RHO_S:g}_alpha_{data.partition_alpha if data.partition_alpha is not None else 'na'}"
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(run_dir / "training.log", args.verbose)
    logger.info(
        "start dataset=%s baseline=%s rounds=%d sensors=%d fogs=%d workers=%d backend=%s torch_threads=%d",
        data.name,
        args.baseline,
        rounds,
        sensors,
        fogs,
        args.workers,
        args.parallel_backend,
        args.torch_threads,
    )
    simulator = AnomalyFLSimulator(
        data,
        net,
        acoustic_cfg,
        energy_cfg,
        learn,
        baseline=args.baseline,
        seed=args.seed,
        workers=args.workers,
        parallel_backend=args.parallel_backend,
        torch_threads=args.torch_threads,
        logger=logger,
    )
    history = simulator.run(rounds)
    metadata = simulator.metadata()
    metadata["rounds"] = rounds
    metadata["scenario"] = args.scenario
    bundle = {"metadata": metadata, "rounds": history}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, cls=JsonEncoder)
    _write_csv(run_dir / "rounds.csv", history)
    summary = {
        **metadata,
        "final": history[-1],
        "best_f1": max(row["f1"] for row in history),
        "best_pa_f1": max(row["pa_f1"] for row in history),
        "total_communication_energy_j": sum(row["e_round_comm_j"] for row in history),
        "total_modelled_energy_j": sum(row["e_round_total_j"] for row in history),
        "total_latency_s": sum(row["latency_round_s"] for row in history),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, cls=JsonEncoder)
    logger.info("complete artifacts=%s", run_dir.resolve())
    return run_dir


if __name__ == "__main__":
    run_experiment(parse_args())
