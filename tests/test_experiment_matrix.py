from __future__ import annotations

from argparse import Namespace

import run_experiments


def test_real_matrix_uses_two_datasets_six_methods_and_fixed_topology(
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
        (dataset, alpha, method)
        for dataset in ("SMAP", "MSL")
        for alpha in (0.1, 1.0e4)
        for method in expected_methods
    ]
    assert all(args.sensors == 100 for args in captured)
    assert all(args.fogs == 10 for args in captured)
    assert all(args.rounds == 2 for args in captured)
    assert all(args.seed == 42 for args in captured)
