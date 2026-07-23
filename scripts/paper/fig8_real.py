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
    datasets = ("SMAP", "MSL")
    methods = ordered_methods(runs)
    x = np.arange(len(datasets))
    width = 0.82 / max(1, len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for method_index, method in enumerate(methods):
        rows = runs[runs["method"] == method]
        offset = (method_index - (len(methods) - 1) / 2) * width
        for axis, metric in zip(axes, ("pa_f1", "energy_j")):
            stats = (
                rows.groupby("dataset")[metric]
                .agg(["mean", "std"])
                .reindex(datasets)
                .fillna(0.0)
            )
            axis.bar(
                x + offset,
                stats["mean"],
                width,
                yerr=stats["std"],
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
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    save_figure(fig, args.output, "fig8_real_benchmarks")


if __name__ == "__main__":
    main()
