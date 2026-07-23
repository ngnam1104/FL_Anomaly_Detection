"""Fig. 8: detection quality and modelled total energy on real benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.paper.common import (
    METHOD_COLORS,
    METHOD_LABELS,
    load_runs,
    ordered_methods,
    plt,
    save_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    runs, _ = load_runs(args.results, "real")
    runs = runs[runs["alpha"].isna()]
    regimes = (
        ("SMD", "SMD"),
        ("SMAP", "SMAP"),
        ("MSL", "MSL"),
    )
    if runs.empty:
        raise ValueError(
            "Fig. 8 requires paper real-data runs with partition_alpha=None"
        )
    methods = ordered_methods(runs)
    x = np.arange(len(regimes))
    width = 0.82 / max(1, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for method_index, method in enumerate(methods):
        rows = runs[runs["method"] == method]
        offset = (method_index - (len(methods) - 1) / 2) * width
        for axis, metric in zip(axes, ("pa_f1", "energy_j")):
            means = []
            deviations = []
            for dataset, _ in regimes:
                group = rows[rows["dataset"] == dataset][metric]
                means.append(float(group.mean()) if len(group) else 0.0)
                deviations.append(
                    float(group.std(ddof=0)) if len(group) else 0.0
                )
            axis.bar(
                x + offset,
                means,
                width,
                yerr=deviations,
                capsize=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
    axes[0].set(title="(a) Detection quality across real benchmarks", ylabel="PA-F1")
    axes[1].set(
        title="(b) Modelled total energy across real benchmarks",
        ylabel="Total energy (J)",
        yscale="log",
    )
    for axis in axes:
        axis.set_xticks(x, [label for _, label in regimes])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    save_figure(fig, args.output, "fig8_real_benchmarks")


if __name__ == "__main__":
    main()
