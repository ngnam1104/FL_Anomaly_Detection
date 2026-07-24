"""Fig. 7: sensitivity to Dirichlet data heterogeneity."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    runs, _ = load_runs(args.results, "noniid")
    alpha_order = (0.1, 1.0e4)
    alpha_labels = (
        "Strong non-IID\n" + r"$\alpha=0.1$",
        "Near-IID\n" + r"$\alpha=10^4$",
    )
    x = (0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    methods = ordered_methods(runs)
    markers = ("o", "s", "^", "D", "P", "X")
    for method_index, method in enumerate(methods):
        rows = runs[runs["method"] == method]
        for axis, metric in zip(axes, ("f1", "energy_j")):
            stats = (
                rows.groupby("alpha")[metric]
                .agg(["mean", "std"])
                .reindex(alpha_order)
                .fillna(0.0)
            )
            axis.errorbar(
                x,
                stats["mean"],
                yerr=stats["std"],
                marker=markers[method_index % len(markers)],
                linewidth=1.8,
                markersize=5,
                capsize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
    axes[0].set(title="(a) Detection quality", ylabel="F1 score")
    axes[1].set(
        title="(b) Modelled total energy",
        ylabel="Total energy (J)",
        yscale="log",
    )
    for axis in axes:
        axis.set_xticks(x, alpha_labels)
        axis.set_xlabel("Dirichlet concentration")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    save_figure(fig, args.output, "fig7_noniid")


if __name__ == "__main__":
    main()
