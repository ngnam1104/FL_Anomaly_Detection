"""Fig. 6: selective-cooperation trade-off and compression savings."""

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
    scale, _ = load_runs(args.results, "scalability")
    compression, _ = load_runs(args.results, "compression")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    hfl = scale[
        scale["method"].isin(("hfl-nocoop", "hfl-selective", "hfl-nearest"))
        & scale["N"].isin((150, 200))
    ]
    methods = ordered_methods(hfl)
    x = np.arange(2)
    width = 0.24
    for index, method in enumerate(methods):
        rows = hfl[hfl["method"] == method]
        energy = rows.groupby("N")["energy_j"].agg(["mean", "std"]).reindex((150, 200)).fillna(0.0)
        bars = axes[0].bar(
            x + (index - 1) * width,
            energy["mean"],
            width,
            yerr=energy["std"],
            capsize=3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        f1 = rows.groupby("N")["f1"].mean().reindex((150, 200))
        for bar, score in zip(bars, f1):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"F1={score:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axes[0].set(
        title="(a) Selective cooperation trade-off",
        ylabel="Total energy (J)",
        xticks=x,
        xticklabels=("N=150", "N=200"),
    )
    max_hfl_energy = float(hfl["energy_j"].max())
    axes[0].set_ylim(0.0, max_hfl_energy * 1.22)
    axes[0].legend(fontsize=7)

    savings = []
    labels = []
    colors = []
    for method in ordered_methods(compression):
        rows = compression[compression["method"] == method]
        low = rows[np.isclose(rows["rho_s"], 0.05)]["energy_j"].mean()
        full = rows[np.isclose(rows["rho_s"], 1.0)]["energy_j"].mean()
        if np.isnan(low) or np.isnan(full) or full <= 0:
            continue
        savings.append(100.0 * (full - low) / full)
        labels.append(METHOD_LABELS[method])
        colors.append(METHOD_COLORS[method])
    bars = axes[1].bar(labels, savings, color=colors)
    for bar, saving in zip(bars, savings):
        axes[1].text(bar.get_x() + bar.get_width() / 2, saving, f"{saving:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set(title="(b) Effect of compressed uploads", ylabel="Energy saving (%)")
    if savings:
        axes[1].set_ylim(0.0, min(100.0, max(savings) * 1.12))
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save_figure(fig, args.output, "fig6_engineering")


if __name__ == "__main__":
    main()
