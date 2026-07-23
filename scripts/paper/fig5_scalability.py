"""Fig. 5: reachability, F1, and per-sensor energy versus network scale."""

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
    runs, _ = load_runs(args.results, "scalability")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    reach = (
        runs.groupby(["seed", "N"], as_index=False)[
            ["direct_reachability", "fog_reachability"]
        ]
        .first()
        .groupby("N")
        .agg(["mean", "std"])
        .fillna(0.0)
    )
    for metric, label, marker in (
        ("direct_reachability", "Direct gateway reachability", "o"),
        ("fog_reachability", "Feasible fog reachability", "s"),
    ):
        axes[0].errorbar(
            reach.index,
            reach[(metric, "mean")],
            yerr=reach[(metric, "std")],
            marker=marker,
            capsize=3,
            label=label,
        )
    axes[0].set(title="(a) Reachability", xlabel="Number of sensors (N)", ylabel="Reachability fraction", ylim=(0, 1.05))
    axes[0].legend(fontsize=7)

    for axis, metric, title, ylabel in (
        (axes[1], "f1", "(b) F1 vs scale", "F1 score"),
        (axes[2], "energy_per_sensor_j", "(c) Energy vs scale", "Energy per sensor (J)"),
    ):
        for method in ordered_methods(runs):
            rows = runs[runs["method"] == method]
            stats = rows.groupby("N")[metric].agg(["mean", "std"]).fillna(0.0)
            axis.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set(title=title, xlabel="Number of sensors (N)", ylabel=ylabel)
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(fig, args.output, "fig5_scalability")


if __name__ == "__main__":
    main()
