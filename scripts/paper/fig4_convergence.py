"""Fig. 4: convergence at N=150 and N=200."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.paper.common import (
    METHOD_ORDER,
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
    _, scalability_curves = load_runs(args.results, "scalability")
    _, extra_curves = load_runs(args.results, "convergence")
    curves = pd.concat(
        [scalability_curves, extra_curves],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=False)
    for axis, size in zip(axes, (150, 200)):
        subset = curves[curves["N"] == size]
        if subset.empty:
            raise ValueError(f"Fig. 4 requires convergence data at N={size}")
        missing = [
            method for method in METHOD_ORDER if method not in set(subset["method"])
        ]
        if missing:
            raise ValueError(
                f"Fig. 4 requires every algorithm at N={size}; missing={missing}"
            )
        for method in ordered_methods(subset):
            rows = subset[subset["method"] == method]
            stats = rows.groupby("round")["train_loss"].agg(["mean", "std"]).fillna(0.0)
            x = stats.index.to_numpy()
            mean = stats["mean"].to_numpy()
            std = stats["std"].to_numpy()
            axis.plot(x, mean, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            axis.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.15)
        axis.set_title(f"({chr(97 + list((150, 200)).index(size))}) N={size}")
        axis.set_xlabel("Communication round")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Training loss")
    axes[0].legend(fontsize=7, ncol=2)
    save_figure(fig, args.output, "fig4_convergence")


if __name__ == "__main__":
    main()
