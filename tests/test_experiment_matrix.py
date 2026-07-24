from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import run_experiments


def test_real_matrix_uses_three_datasets_six_methods_and_fixed_topology(
    tmp_path, monkeypatch
):
    captured = []
    monkeypatch.setattr(
        run_experiments,
        "run_experiment",
        lambda args: captured.append(args),
    )
    cli = Namespace(
        quick=True,
        data_root=tmp_path / "datasets",
        output_root=tmp_path / "results",
        workers=4,
    )

    run_experiments.run_real(cli)

    expected_methods = list(run_experiments.REAL_METHODS)
    assert [
        (args.dataset, args.dirichlet_alpha, args.baseline) for args in captured
    ] == [
        (dataset, None, method)
        for dataset in ("SMD", "SMAP", "MSL")
        for method in expected_methods
    ]
    assert all(args.sensors == 100 for args in captured)
    assert all(args.fogs == 10 for args in captured)
    assert all(args.rounds == 2 for args in captured)
    assert all(args.seed == 42 for args in captured)


def test_quick_all_suite_covers_every_paper_scenario(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(
        run_experiments,
        "run_experiment",
        lambda args: captured.append(args),
    )
    cli = Namespace(
        quick=True,
        data_root=tmp_path / "datasets",
        output_root=tmp_path / "results",
        workers=4,
        torch_threads=1,
        centralized_torch_threads=4,
        parallel_backend="process",
        resume=False,
    )

    run_experiments.run_convergence(cli)
    run_experiments.run_scalability(cli)
    run_experiments.run_compression(cli)
    run_experiments.run_noniid(cli)
    run_experiments.run_real(cli)

    assert len(captured) == 25
    assert {args.scenario for args in captured} == {
        "convergence",
        "scalability",
        "compression",
        "noniid",
        "real",
    }
    assert {args.dataset for args in captured if args.scenario == "real"} == {
        "SMD",
        "SMAP",
        "MSL",
    }


def test_ubuntu_setup_and_runner_have_separate_responsibilities():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "setup_ubuntu.sh").read_text(encoding="utf-8")
    runner = (root / "run_scenarios.sh").read_text(encoding="utf-8")

    assert "pip install" in setup
    assert "scripts.prepare_benchmarks" in setup
    assert "partition_manifest.json" in setup
    assert "run_experiments.py" not in setup

    assert "run_experiments.py" in runner
    assert "--suite all" in runner
    assert "scripts.paper.plot_all" in runner
    assert "partition_manifest.json" in runner
    assert "pip install" not in runner
    assert "scripts.prepare_benchmarks" not in runner
    assert "--dataset all" in setup
    assert not (root / "run_ubuntu.sh").exists()


def test_resume_rejects_stale_flat_energy_accounting(tmp_path):
    args = Namespace(
        output_root=tmp_path,
        scenario="scalability",
        dataset="synthetic",
        sensors=10,
        fogs=1,
        baseline="fedprox",
        rho_s=0.05,
        dirichlet_alpha=1.0,
        seed=42,
        rounds=2,
    )
    run_path = run_experiments._completed_run_path(args)
    run_path.mkdir(parents=True)
    rounds = [
        {
            "round": round_index,
            "train_loss": 1.0,
            "f1": 0.8,
            "pa_f1": 0.8,
            "threshold": 1.0,
            "test_reconstruction_loss": 1.0,
            "e_cumulative_total_j": float(round_index),
            "latency_cumulative_s": float(round_index),
        }
        for round_index in (1, 2)
    ]
    summary = {
        "rounds": 2,
        "dataset": "synthetic",
        "baseline": "fedprox",
        "seed": 42,
        "final": rounds[-1],
        "best_f1": 0.8,
        "best_pa_f1": 0.8,
        "total_communication_energy_j": 1.0,
        "total_modelled_energy_j": 2.0,
        "total_latency_s": 2.0,
    }
    bundle = {"metadata": {"scenario": "scalability"}, "rounds": rounds}
    (run_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_path / "metrics.json").write_text(json.dumps(bundle), encoding="utf-8")

    assert run_experiments._is_completed(args) is False

    policy = run_experiments.FLAT_FAILED_UPLOAD_POLICY
    summary["flat_failed_upload_policy"] = policy
    bundle["metadata"]["flat_failed_upload_policy"] = policy
    (run_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_path / "metrics.json").write_text(json.dumps(bundle), encoding="utf-8")

    assert run_experiments._is_completed(args) is True
